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
const END_OF_TURN: &str = "<end_of_turn>";

/// A yes/no question is answered on these two labels, so the "yes" mass is
/// read straight off the restricted distribution.
const NOUL_LABELS: [&str; 2] = ["no", "yes"];

/// Jev exposes three question primitives. Noul is a yes/no probability, Choice
/// selects one of N labels, Score places the state on an ordered scale.
#[derive(Clone, Copy, Debug)]
pub enum Primitive<'a> {
    Noul,
    Choice(&'a [&'a str]),
    Score(&'a [&'a str]),
}

impl<'a> Primitive<'a> {
    pub fn kind(&self) -> &'static str {
        match self {
            Primitive::Noul => "noul",
            Primitive::Choice(_) => "choice",
            Primitive::Score(_) => "score",
        }
    }

    fn labels(&self) -> &'a [&'a str] {
        match self {
            Primitive::Noul => &NOUL_LABELS,
            Primitive::Choice(labels) | Primitive::Score(labels) => labels,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Question<'a> {
    pub id: &'a str,
    pub text: &'a str,
    pub primitive: Primitive<'a>,
}

#[derive(Clone, Debug)]
pub enum Value {
    /// Probability that the answer is yes.
    Noul(f32),
    Choice(String),
    /// Expected level on the ordered scale.
    Score(f32),
}

#[derive(Clone, Debug)]
pub struct Answer {
    pub id: String,
    pub text: String,
    pub kind: &'static str,
    pub value: Value,
    pub probabilities: Vec<(String, f32)>,
    /// Largest probability in the restricted distribution. This is our own
    /// shape statistic, not Jev's undisclosed calibrated confidence.
    pub confidence: f32,
    pub latency: Duration,
}

impl Answer {
    pub fn rendered_value(&self) -> String {
        match &self.value {
            Value::Noul(probability) => {
                let verdict = if *probability >= 0.5 { "yes" } else { "no" };
                format!("{verdict} · p={probability:.3}")
            }
            Value::Choice(choice) => choice.clone(),
            Value::Score(score) => format!("{score:.2} / {}", self.probabilities.len() - 1),
        }
    }
}

pub struct Baseline {
    pub text: String,
    pub tokens: usize,
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

    /// Restricted next-token distribution over the legal labels. This is the
    /// whole System-One path: one forward pass, gather, softmax, stop.
    pub fn label_probabilities(
        &self,
        state: &str,
        question: &str,
        labels: &[&str],
        temperature: f64,
    ) -> Result<(Vec<f32>, Duration)> {
        if !(temperature.is_finite() && temperature > 0.0) {
            bail!("temperature must be finite and greater than zero");
        }
        if labels.len() < 2 || labels.len() > 26 {
            bail!("a question needs between 2 and 26 labels");
        }

        let prompt = prompt(state, question, labels);
        let prompt_ids = self.encode(&prompt)?;
        let candidate_ids = self.candidate_token_ids(&prompt, &prompt_ids, labels.len())?;
        let input = Tensor::new(prompt_ids.as_slice(), &self.device)?.unsqueeze(0)?;

        // Cloning gives this call an empty KV cache. The clone shares quantized
        // weights, so it copies metadata rather than the 2.3 GB of tensors.
        let mut model = self.model.clone();
        let started = Instant::now();
        let logits = model.forward(&input, 0)?.squeeze(0)?;
        let candidate_index = Tensor::new(candidate_ids.as_slice(), &self.device)?;
        let label_logits = logits
            .index_select(&candidate_index, 0)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        let probabilities = softmax(&label_logits, temperature);
        Ok((probabilities, started.elapsed()))
    }

    pub fn answer(&self, state: &str, question: &Question<'_>, temperature: f64) -> Result<Answer> {
        let labels = question.primitive.labels();
        let (probabilities, latency) =
            self.label_probabilities(state, question.text, labels, temperature)?;

        let best = probabilities
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index)
            .context("empty probability vector")?;
        let confidence = probabilities[best];

        let value = match question.primitive {
            Primitive::Noul => Value::Noul(probabilities[1]),
            Primitive::Choice(_) => Value::Choice(labels[best].to_string()),
            // Fractional level, so a split between adjacent levels lands
            // between them instead of snapping to one.
            Primitive::Score(_) => Value::Score(
                probabilities
                    .iter()
                    .enumerate()
                    .map(|(level, probability)| level as f32 * probability)
                    .sum(),
            ),
        };

        Ok(Answer {
            id: question.id.to_string(),
            text: question.text.to_string(),
            kind: question.primitive.kind(),
            value,
            probabilities: labels
                .iter()
                .zip(probabilities)
                .map(|(label, probability)| ((*label).to_string(), probability))
                .collect(),
            confidence,
            latency,
        })
    }

    /// Evaluates every question against one shared state.
    pub fn evaluate(
        &self,
        state: &str,
        questions: &[Question<'_>],
        temperature: f64,
    ) -> Result<Vec<Answer>> {
        // Candle's quantized Gemma exposes no cache reset and no attention mask
        // for ragged rows, so questions run sequentially. Each still costs
        // exactly one forward pass.
        questions
            .iter()
            .map(|question| self.answer(state, question, temperature))
            .collect()
    }

    /// Ordinary autoregressive decoding, kept only as a comparison baseline.
    /// Nothing in the System-One path calls this.
    pub fn generate_baseline(
        &self,
        state: &str,
        question: &str,
        labels: &[&str],
        max_tokens: usize,
    ) -> Result<Baseline> {
        let prompt = prompt(state, question, labels);
        let prompt_ids = self.encode(&prompt)?;
        let end_of_turn = self
            .tokenizer
            .get_vocab(true)
            .get(END_OF_TURN)
            .copied()
            .context("tokenizer is missing the end-of-turn token")?;

        let mut model = self.model.clone();
        let started = Instant::now();
        let input = Tensor::new(prompt_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut logits = model.forward(&input, 0)?.squeeze(0)?;
        let mut generated = Vec::new();

        for step in 0..max_tokens {
            let next = logits
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?
                .into_iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(token, _)| token as u32)
                .context("empty logit vector")?;
            if next == end_of_turn {
                break;
            }
            generated.push(next);
            let input = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;
            logits = model.forward(&input, prompt_ids.len() + step)?.squeeze(0)?;
        }

        let text = self
            .tokenizer
            .decode(&generated, true)
            .map_err(anyhow::Error::msg)?;
        Ok(Baseline {
            text: text.trim().to_string(),
            tokens: generated.len(),
            latency: started.elapsed(),
        })
    }

    /// Confirms each label is exactly one token at the real answer boundary.
    pub fn verify_labels(&self) -> Result<Vec<(char, u32)>> {
        let labels = ["one", "two", "three", "four", "five"];
        let prompt = prompt("state", "question", &labels);
        let prompt_ids = self.encode(&prompt)?;
        let ids = self.candidate_token_ids(&prompt, &prompt_ids, labels.len())?;
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

fn prompt(state: &str, question: &str, labels: &[&str]) -> String {
    let choices = labels
        .iter()
        .enumerate()
        .map(|(index, label)| format!("{}) {label}", (b'A' + index as u8) as char))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "<start_of_turn>user\n\
You are a software incident decision classifier.\n\
Read the state and question, compare every choice, and select the best answer.\n\
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
    use super::{NOUL_LABELS, Primitive, prompt, softmax};

    #[test]
    fn softmax_is_normalized() {
        let probabilities = softmax(&[1.0, 2.0, -1.0, 0.5, 4.0], 1.0);
        assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(probabilities.len(), 5);
    }

    #[test]
    fn temperature_flattens_the_distribution() {
        let sharp = softmax(&[4.0, 1.0], 0.5);
        let flat = softmax(&[4.0, 1.0], 4.0);
        assert!(sharp[0] > flat[0]);
        assert!((flat.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn noul_orders_no_before_yes() {
        // Value::Noul reads index 1, so "yes" must stay second.
        assert_eq!(NOUL_LABELS, ["no", "yes"]);
    }

    #[test]
    fn every_primitive_exposes_labels() {
        let levels = ["low", "medium", "high"];
        assert_eq!(Primitive::Noul.labels().len(), 2);
        assert_eq!(Primitive::Choice(&levels).labels().len(), 3);
        assert_eq!(Primitive::Score(&levels).labels().len(), 3);
    }

    #[test]
    fn prompt_ends_at_the_answer_boundary() {
        let rendered = prompt("state", "question", &["no", "yes"]);
        assert!(rendered.ends_with("<start_of_turn>model\n"));
        assert!(rendered.contains("A) no"));
        assert!(rendered.contains("B) yes"));
    }
}
