mod system_one;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::Parser;
use system_one::{Answer, MODEL_ID, Primitive, Question, SystemOne, Value};

const STATE: &str = r#"Command:

cargo test

Result:

48 tests passed.
1 test failed.

Linker output:

error: linking with `cc` failed: exit status: 1
/usr/bin/ld: cannot find -lssl
collect2: error: ld returned 1 exit status"#;

const STATE_SUMMARY: &str = "cargo test; 48 passed, 1 failed; linker cannot find -lssl";

const FAILURE: &[&str] = &["compilation", "dependency", "infrastructure", "test"];
const OWNER: &[&str] = &["application team", "platform team", "security team"];
const RISK: &[&str] = &["very low", "low", "medium", "high", "very high"];

/// Above this the decision can be acted on automatically, below 0.5 it should
/// go to a human. These are demo thresholds, not calibrated guarantees.
const ACT_THRESHOLD: f32 = 0.90;
const ESCALATE_THRESHOLD: f32 = 0.50;

/// Incidents with a known failure class, used to spot-check the decision layer.
const CORPUS: &[(&str, &str)] = &[
    (
        "cargo test\n48 passed, 1 failed\n/usr/bin/ld: cannot find -lssl",
        "dependency",
    ),
    (
        "cargo build\nerror[E0308]: mismatched types\nexpected `u32`, found `String`",
        "compilation",
    ),
    (
        "cargo test\n12 passed, 3 failed\nassertion `left == right` failed in parser::tests",
        "test",
    ),
    (
        "cargo test\nrunner lost connection to the build cluster\nDNS resolution failed for cache.internal",
        "infrastructure",
    ),
];

#[derive(Parser, Debug)]
#[command(about = "Gemma as a direct-logit, Jev-style System One decision function")]
struct Args {
    /// Softmax temperature. This is an inference control, not calibration.
    #[arg(long, default_value_t = 1.0)]
    temperature: f64,

    /// Repeat the typed evaluation and report warm latency percentiles.
    #[arg(long, default_value_t = 1)]
    repeat: usize,

    /// Run the stress suite: determinism, option-order stability, spot-check.
    #[arg(long)]
    stress: bool,

    /// Also decode an answer autoregressively to compare against System One.
    #[arg(long)]
    baseline: bool,

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
        "Device: {} | model load: {:.2}s",
        engine.device_name,
        engine.load_time.as_secs_f64()
    );
    let labels = engine
        .verify_labels()?
        .into_iter()
        .map(|(label, id)| format!("{label}={id}"))
        .collect::<Vec<_>>()
        .join(", ");
    println!("Answer labels verified as single tokens: {labels}\n");

    let questions = questions();
    let started = Instant::now();
    let answers = engine.evaluate(STATE, &questions, args.temperature)?;
    let elapsed = started.elapsed();

    println!("STATE\n{STATE_SUMMARY}\n");
    println!("TYPED ANSWERS");
    for answer in &answers {
        print_answer(answer);
    }
    println!(
        "{} questions in {:.0} ms · {:.0} ms per decision · 0 tokens generated\n",
        answers.len(),
        ms(elapsed),
        ms(elapsed) / answers.len() as f64
    );

    confidence_gate(&answers);

    let ambiguous = engine.answer(
        STATE,
        &Question {
            id: "second_owner",
            text: "Which team should look at this after the first owner investigates?",
            primitive: Primitive::Choice(OWNER),
        },
        args.temperature,
    )?;
    println!("AMBIGUOUS QUESTION");
    print_answer(&ambiguous);
    println!("{}\n", gate_for(ambiguous.confidence));

    if args.repeat > 1 {
        percentiles(&engine, &questions, args.temperature, args.repeat)?;
    }
    if args.baseline {
        baseline(&engine)?;
    }
    if args.stress {
        stress(&engine, args.temperature)?;
    }

    println!("generate() calls in the System One path: 0");
    Ok(())
}

fn questions() -> Vec<Question<'static>> {
    vec![
        Question {
            id: "failure_kind",
            text: "What kind of failure is this?",
            primitive: Primitive::Choice(FAILURE),
        },
        Question {
            id: "retry_unchanged",
            text: "Should we retry the exact same command without changing anything?",
            primitive: Primitive::Noul,
        },
        Question {
            id: "first_owner",
            text: "Who should initially handle this?",
            primitive: Primitive::Choice(OWNER),
        },
        Question {
            id: "made_progress",
            text: "Did the command mostly make progress before failing?",
            primitive: Primitive::Noul,
        },
        Question {
            id: "retry_risk",
            text: "How risky would it be to automatically retry this exact command?",
            primitive: Primitive::Score(RISK),
        },
    ]
}

fn banner() {
    println!("==============================================================");
    println!(" GEMMA 3 4B -> SYSTEM ONE (native Rust)");
    println!(" prompt -> one forward pass -> legal-label logits -> softmax");
    println!(" typed answers: choice / noul / score");
    println!("==============================================================\n");
    println!("Gemma is NOT generating text.\n");
    println!("prompt");
    println!("  -> single forward pass");
    println!("  -> legal-label logits only");
    println!("  -> softmax");
    println!("  -> typed decision + probabilities\n");
    println!("generate() calls: 0\n");
}

fn print_answer(answer: &Answer) {
    println!("  {:<7} {:<16} {}", answer.kind, answer.id, answer.text);
    println!(
        "  {:<24} {:<22} {:>6.0} ms",
        "",
        answer.rendered_value(),
        ms(answer.latency)
    );
    let mut rows = answer.probabilities.clone();
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (label, probability) in rows {
        println!(
            "      {label:<18} {:<24} {:>6.2}%",
            "█".repeat((probability * 24.0).round() as usize),
            probability * 100.0
        );
    }
    if !matches!(answer.value, Value::Noul(_)) {
        println!("      confidence {:.2}", answer.confidence);
    }
    println!();
}

fn gate_for(confidence: f32) -> String {
    if confidence >= ACT_THRESHOLD {
        format!("confidence {confidence:.2} -> act automatically")
    } else if confidence >= ESCALATE_THRESHOLD {
        format!("confidence {confidence:.2} -> act, but log for review")
    } else {
        format!("confidence {confidence:.2} -> escalate to a human")
    }
}

fn confidence_gate(answers: &[Answer]) {
    println!("CONFIDENCE GATING");
    for answer in answers {
        println!("  {:<16} {}", answer.id, gate_for(answer.confidence));
    }
    println!();
}

fn percentiles(
    engine: &SystemOne,
    questions: &[Question<'_>],
    temperature: f64,
    repeat: usize,
) -> Result<()> {
    let mut batches = Vec::with_capacity(repeat);
    let mut singles = Vec::new();
    for _ in 0..repeat {
        let started = Instant::now();
        let answers = engine.evaluate(STATE, questions, temperature)?;
        assert_eq!(answers.len(), questions.len());
        batches.push(started.elapsed());
        singles.extend(answers.iter().map(|answer| answer.latency));
    }
    println!(
        "LATENCY ({repeat} warm evaluations, {} decisions)",
        singles.len()
    );
    report("  per evaluation", &mut batches);
    report("  per decision  ", &mut singles);
    println!();
    Ok(())
}

fn baseline(engine: &SystemOne) -> Result<()> {
    let question = questions()[0];
    let labels = FAILURE;
    let system_one = engine.answer(STATE, &question, 1.0)?;
    let generated = engine.generate_baseline(STATE, question.text, labels, 64)?;

    println!("BASELINE COMPARISON (same model, same prompt)");
    println!(
        "  system one   {:>7.0} ms   0 tokens    -> {}",
        ms(system_one.latency),
        system_one.rendered_value()
    );
    println!(
        "  generate()   {:>7.0} ms   {} tokens   -> {:?} (still needs parsing)",
        ms(generated.latency),
        generated.tokens,
        generated.text
    );
    println!();
    Ok(())
}

fn stress(engine: &SystemOne, temperature: f64) -> Result<()> {
    println!("STRESS SUITE");
    let started = Instant::now();
    let mut latencies = Vec::new();
    let mut decisions = 0usize;

    // 1. Invariants and a labelled spot-check across distinct incidents.
    let mut correct = 0usize;
    for (state, expected) in CORPUS {
        let answer = engine.answer(
            state,
            &Question {
                id: "failure_kind",
                text: "What kind of failure is this?",
                primitive: Primitive::Choice(FAILURE),
            },
            temperature,
        )?;
        decisions += 1;
        latencies.push(answer.latency);

        let total: f32 = answer.probabilities.iter().map(|(_, p)| p).sum();
        assert!((total - 1.0).abs() < 1e-4, "probabilities must sum to 1");
        assert_eq!(answer.probabilities.len(), FAILURE.len());
        let selected = answer.rendered_value();
        if selected == *expected {
            correct += 1;
        }
        println!(
            "  spot-check  {:<14} expected {:<14} got {:<14} {}",
            state.split('\n').next().unwrap_or(state),
            expected,
            selected,
            if selected == *expected { "ok" } else { "MISS" }
        );
    }
    println!("  labelled accuracy: {correct}/{}\n", CORPUS.len());

    // 2. Determinism: the same call must return the same distribution.
    let mut identical = 0usize;
    for _ in 0..5 {
        let (first, first_latency) =
            engine.label_probabilities(STATE, questions()[0].text, FAILURE, temperature)?;
        let (second, second_latency) =
            engine.label_probabilities(STATE, questions()[0].text, FAILURE, temperature)?;
        decisions += 2;
        latencies.push(first_latency);
        latencies.push(second_latency);
        if first == second {
            identical += 1;
        }
    }
    println!("  determinism: {identical}/5 repeated calls returned identical probabilities");

    // 3. Option-order stability. A real decision function should not change its
    //    mind when the same options are listed in a different order.
    let mut stable = 0usize;
    let mut rotations = 0usize;
    for question in questions() {
        // Score levels are ordinal, so rotating them would change the meaning
        // of the scale rather than just the presentation order.
        let labels: Vec<&str> = match question.primitive {
            Primitive::Choice(labels) => labels.to_vec(),
            Primitive::Noul => vec!["no", "yes"],
            Primitive::Score(_) => continue,
        };
        let mut picks = Vec::new();
        let mut letters = Vec::new();
        for shift in 0..labels.len() {
            let rotated: Vec<&str> = labels
                .iter()
                .cycle()
                .skip(shift)
                .take(labels.len())
                .copied()
                .collect();
            let (pick, index) = argmax_label(engine, &rotated, question.text, temperature)?;
            decisions += 1;
            letters.push((b'A' + index as u8) as char);
            picks.push(pick);
        }
        for pick in picks.iter().skip(1) {
            rotations += 1;
            if *pick == picks[0] {
                stable += 1;
            }
        }
        println!(
            "  order  {:<16} picks: {:<44} letters: {}",
            question.id,
            picks.join(", "),
            letters.iter().collect::<String>()
        );
    }
    println!(
        "  option-order stability: {stable}/{rotations} rotations kept the same decision ({:.0}%)",
        100.0 * stable as f64 / rotations.max(1) as f64
    );

    let wall = started.elapsed();
    println!("  decisions: {decisions} in {:.2}s", wall.as_secs_f64());
    println!(
        "  throughput: {:.1} decisions/s",
        decisions as f64 / wall.as_secs_f64()
    );
    report("  latency      ", &mut latencies);
    println!();
    Ok(())
}

fn argmax_label(
    engine: &SystemOne,
    labels: &[&str],
    question: &str,
    temperature: f64,
) -> Result<(String, usize)> {
    let (probabilities, _) = engine.label_probabilities(STATE, question, labels, temperature)?;
    let best = probabilities
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
        .unwrap_or_default();
    Ok((labels[best].to_string(), best))
}

fn report(label: &str, samples: &mut [Duration]) {
    samples.sort();
    println!(
        "{label} p50 {:.0} ms · p90 {:.0} ms · p99 {:.0} ms · max {:.0} ms",
        ms(quantile(samples, 0.50)),
        ms(quantile(samples, 0.90)),
        ms(quantile(samples, 0.99)),
        ms(samples[samples.len() - 1])
    );
}

fn quantile(sorted: &[Duration], q: f64) -> Duration {
    let index = ((sorted.len() as f64 * q).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[index]
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}
