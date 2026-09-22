mod system_one;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::Parser;
use system_one::{Decision, MODEL_ID, Question, SystemOne};

const STATE: &str = r#"Command:

cargo test

Result:

48 tests passed.
1 test failed.

Linker output:

error: linking with `cc` failed: exit status: 1
/usr/bin/ld: cannot find -lssl
collect2: error: ld returned 1 exit status"#;

const FAILURE: &[&str] = &["compilation", "dependency", "infrastructure", "test"];
const YES_NO: &[&str] = &["no", "yes"];
const OWNER: &[&str] = &["application team", "platform team", "security team"];
const RISK: &[&str] = &["very low", "low", "medium", "high", "very high"];

#[derive(Parser, Debug)]
#[command(about = "Gemma as a direct-logit System-One-style decision function")]
struct Args {
    /// Softmax temperature. This is not calibration.
    #[arg(long, default_value_t = 1.0)]
    temperature: f64,

    /// Repeat the five-question benchmark and report warm latency.
    #[arg(long, default_value_t = 1)]
    repeat: usize,

    /// Force CPU instead of Apple Metal.
    #[arg(long)]
    cpu: bool,

    /// Use an already downloaded GGUF file.
    #[arg(long)]
    model: Option<PathBuf>,

    /// Use an already downloaded tokenizer.json file.
    #[arg(long)]
    tokenizer: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    banner();

    println!("Loading {MODEL_ID} (Q4_K_M)...");
    let engine = SystemOne::load(args.model.as_deref(), args.tokenizer.as_deref(), args.cpu)?;
    println!(
        "Device: {} | cold load: {:.2}s",
        engine.device_name,
        engine.load_time.as_secs_f64()
    );
    let labels = engine
        .verify_labels()?
        .into_iter()
        .map(|(label, id)| format!("{label}={id}"))
        .collect::<Vec<_>>()
        .join(", ");
    println!("Verified one-token labels: {labels}\n");

    let questions = questions();
    let batch_started = Instant::now();
    let decisions = engine.decide_batch(STATE, &questions, args.temperature)?;
    let batch_latency = batch_started.elapsed();

    println!("State: cargo test; 48 passed, 1 failed; linker cannot find -lssl\n");
    for (index, (question, decision)) in questions.iter().zip(&decisions).enumerate() {
        print_decision(index + 1, question.text, decision);
    }
    println!("BATCH");
    println!("{} questions", decisions.len());
    println!("total latency: {:.1} ms", ms(batch_latency));
    println!(
        "average equivalent: {:.1} ms/question\n",
        ms(batch_latency) / decisions.len() as f64
    );

    let ambiguous = engine.decide(
        STATE,
        "Which team should look at this after the first owner investigates?",
        OWNER,
        args.temperature,
    )?;
    print_decision(6, "Ambiguous confidence check", &ambiguous);

    if args.repeat > 1 {
        benchmark(&engine, &questions, args.temperature, args.repeat)?;
    }

    println!("generate() calls: 0");
    println!("generated tokens: 0");
    Ok(())
}

fn questions() -> Vec<Question<'static>> {
    vec![
        Question {
            text: "What kind of failure is this?",
            choices: FAILURE,
        },
        Question {
            text: "Should we retry the exact same command without changing anything?",
            choices: YES_NO,
        },
        Question {
            text: "Who should initially handle this?",
            choices: OWNER,
        },
        Question {
            text: "Did the command mostly make progress before failing?",
            choices: YES_NO,
        },
        Question {
            text: "How risky would it be to automatically retry this exact command?",
            choices: RISK,
        },
    ]
}

fn banner() {
    println!("==============================================================");
    println!(" GEMMA 3 4B -> SYSTEM ONE (native Rust)");
    println!(" prompt -> one forward pass -> choice logits -> softmax");
    println!(" no generate(), no JSON, no parser");
    println!("==============================================================\n");
    println!("Gemma is NOT generating text.\n");
    println!("prompt");
    println!("  -> single forward pass");
    println!("  -> legal-choice logits only");
    println!("  -> softmax");
    println!("  -> decision probabilities\n");
    println!("generate() calls: 0\n");
}

fn print_decision(number: usize, question: &str, decision: &Decision) {
    println!(
        "Question {number}: {question}                  {:.1} ms",
        ms(decision.latency)
    );
    let mut rows = decision.probabilities.clone();
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (choice, probability) in rows {
        let bars = (probability * 30.0).round() as usize;
        println!(
            "{choice:<20} {:<30} {:>6.2}%",
            "█".repeat(bars),
            probability * 100.0
        );
    }
    println!("DECISION -> {}\n", decision.selected);
}

fn benchmark(
    engine: &SystemOne,
    questions: &[Question<'_>],
    temperature: f64,
    repeat: usize,
) -> Result<()> {
    let mut samples = Vec::with_capacity(repeat);
    for _ in 0..repeat {
        let started = Instant::now();
        let decisions = engine.decide_batch(STATE, questions, temperature)?;
        assert_eq!(decisions.len(), questions.len());
        samples.push(started.elapsed());
    }
    samples.sort();
    let median = samples[samples.len() / 2];
    let p95_index = ((samples.len() as f64 * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(samples.len() - 1);
    println!("BENCHMARK ({repeat} warm batches)");
    println!("median: {:.1} ms", ms(median));
    println!("p95: {:.1} ms", ms(samples[p95_index]));
    println!(
        "median/question: {:.1} ms\n",
        ms(median) / questions.len() as f64
    );
    Ok(())
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}
