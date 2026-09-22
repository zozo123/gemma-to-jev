use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::quantized_gemma3::ModelWeights;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

pub const MODEL_ID: &str = "google/gemma-3-4b-it";
const WEIGHTS_REPO: &str = "unsloth/gemma-3-4b-it-GGUF";
const WEIGHTS_FILE: &str = "gemma-3-4b-it-Q4_K_M.gguf";
const TOKENIZER_REPO: &str = "unsloth/gemma-3-4b-it";
const TOKENIZER_FILE: &str = "tokenizer.json";

#[derive(Clone, Debug)]
pub struct Question<'a> {
    pub text: &'a str,
    pub choices: &'a [&'a str],
}

#[derive(Clone, Debug)]
pub struct Decision {
    pub selected: String,
    pub probabilities: Vec<(String, f32)>,
    pub latency: Duration,
}

pub struct SystemOne {
    model: ModelWeights,
    tokenizer: Tokenizer,
    device: Device,
    pub device_name: String,
    pub load_time: Duration,
}

impl SystemOne {
    pub fn load(
        model_path: Option<&Path>,
        tokenizer_path: Option<&Path>,
        cpu: bool,
    ) -> Result<Self> {
        let started = Instant::now();
        let model_path = match model_path {
            Some(path) => path.to_path_buf(),
            None => download(WEIGHTS_REPO, WEIGHTS_FILE)?,
        };
        let tokenizer_path = match tokenizer_path {
            Some(path) => path.to_path_buf(),
            None => download(TOKENIZER_REPO, TOKENIZER_FILE)?,
        };

        let (device, device_name) = if cpu {
            (Device::Cpu, "CPU".to_string())
        } else {
            match Device::new_metal(0) {
                Ok(device) => (device, "Apple Metal".to_string()),
                Err(_) => (Device::Cpu, "CPU (Metal unavailable)".to_string()),
            }
        };

        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(anyhow::Error::msg)?;
        let mut file =
            File::open(&model_path).with_context(|| format!("opening {}", model_path.display()))?;
        let content =
            gguf_file::Content::read(&mut file).map_err(|error| error.with_path(&model_path))?;
        let model = ModelWeights::from_gguf(content, &mut file, &device)?;

        Ok(Self {
            model,
            tokenizer,
            device,
            device_name,
            load_time: started.elapsed(),
        })
    }

    pub fn decide(
        &self,
        state: &str,
        question: &str,
        choices: &[&str],
        temperature: f64,
    ) -> Result<Decision> {
        if !(temperature.is_finite() && temperature > 0.0) {
            bail!("temperature must be finite and greater than zero");
        }
        if choices.len() < 2 || choices.len() > 26 {
            bail!("choices must contain between 2 and 26 items");
        }

        let prompt = prompt(state, question, choices);
        let prompt_ids = self.encode(&prompt)?;
        let candidate_ids = self.candidate_token_ids(&prompt, &prompt_ids, choices.len())?;
        let input = Tensor::new(prompt_ids.as_slice(), &self.device)?.unsqueeze(0)?;

        // Clone starts with an empty KV cache. The only model operation in this
        // path is this single forward pass: no generate/decode loop exists.
        let mut model = self.model.clone();
        let started = Instant::now();
        let logits = model.forward(&input, 0)?.squeeze(0)?;
        let candidate_index = Tensor::new(candidate_ids.as_slice(), &self.device)?;
        let selected_logits = logits
            .index_select(&candidate_index, 0)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        let probabilities = softmax(&selected_logits, temperature);
        let latency = started.elapsed();

        let selected_index = probabilities
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index)
            .context("empty probability vector")?;

        Ok(Decision {
            selected: choices[selected_index].to_string(),
            probabilities: choices
                .iter()
                .zip(probabilities)
                .map(|(choice, probability)| ((*choice).to_string(), probability))
                .collect(),
            latency,
        })
    }

    pub fn decide_batch(
        &self,
        state: &str,
        questions: &[Question<'_>],
        temperature: f64,
    ) -> Result<Vec<Decision>> {
        // Candle's quantized Gemma cache has no public reset API. Cheap model
        // clones give each row an empty cache and preserve exact, unpadded
        // prompts; this is a sequential batch API, not a fused tensor batch.
        questions
            .iter()
            .map(|question| self.decide(state, question.text, question.choices, temperature))
            .collect()
    }

    pub fn verify_labels(&self) -> Result<Vec<(char, u32)>> {
        let choices = ["one", "two", "three", "four", "five"];
        let prompt = prompt("state", "question", &choices);
        let prompt_ids = self.encode(&prompt)?;
        let ids = self.candidate_token_ids(&prompt, &prompt_ids, choices.len())?;
        Ok(ids
            .into_iter()
            .enumerate()
            .map(|(index, id)| ((b'A' + index as u8) as char, id))
            .collect())
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec())
    }

    fn candidate_token_ids(
        &self,
        prompt: &str,
        prompt_ids: &[u32],
        count: usize,
    ) -> Result<Vec<u32>> {
        (0..count)
            .map(|index| {
                let label = (b'A' + index as u8) as char;
                let with_label = self.encode(&format!("{prompt}{label}"))?;
                if !with_label.starts_with(prompt_ids) || with_label.len() != prompt_ids.len() + 1 {
                    bail!("answer label {label:?} is not exactly one token at the answer position");
                }
                Ok(with_label[prompt_ids.len()])
            })
            .collect()
    }
}

fn download(repo: &str, file: &str) -> Result<PathBuf> {
    Ok(Api::new()?.model(repo.to_string()).get(file)?)
}

fn prompt(state: &str, question: &str, choices: &[&str]) -> String {
    let choices = choices
        .iter()
        .enumerate()
        .map(|(index, choice)| format!("{}) {choice}", (b'A' + index as u8) as char))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "<start_of_turn>user\n\
You are a software incident decision classifier.\n\
Read the state and question, compare every choice, and select the best answer.\n\
Policy: exact retries are only useful for transient failures. A deterministic missing\n\
library or dependency persists until the environment or dependency changes.\n\
Reply with exactly one choice letter and nothing else.\n\n\
State:\n{state}\n\n\
Question:\n{question}\n\n\
Choices:\n{choices}\n\n\
Answer:<end_of_turn>\n<start_of_turn>model\n"
    )
}

fn softmax(logits: &[f32], temperature: f64) -> Vec<f32> {
    let scaled = logits
        .iter()
        .map(|value| *value as f64 / temperature)
        .collect::<Vec<_>>();
    let maximum = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exponentials = scaled
        .iter()
        .map(|value| (value - maximum).exp())
        .collect::<Vec<_>>();
    let total = exponentials.iter().sum::<f64>();
    exponentials
        .into_iter()
        .map(|value| (value / total) as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::softmax;

    #[test]
    fn softmax_is_normalized() {
        let probabilities = softmax(&[1.0, 2.0, -1.0, 0.5, 4.0], 1.0);
        assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(probabilities.len(), 5);
    }
}
