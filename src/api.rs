use std::collections::BTreeMap;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tiny_http::{Header, Method, Response, Server, StatusCode};

use crate::system_one::{Primitive, Question, SystemOne, Value as DecisionValue};

#[derive(Deserialize)]
struct SystemOneRequest {
    state: Value,
    #[serde(default)]
    model: Option<String>,
    questions: BTreeMap<String, ApiQuestion>,
}

#[derive(Deserialize)]
struct ApiQuestion {
    #[serde(rename = "type")]
    kind: String,
    instructions: String,
    #[serde(default)]
    criteria: Option<Value>,
}

#[derive(Serialize)]
struct SystemOneResponse {
    model: String,
    answers: BTreeMap<String, Value>,
    usage: BTreeMap<String, usize>,
}

pub fn serve(engine: &SystemOne, listen: &str, temperature: f64) -> Result<()> {
    let server = Server::http(listen).map_err(|error| anyhow::anyhow!(error.to_string()))?;
    println!("TypeSafe-compatible API: http://{listen}/v1/systemone");
    println!("Health check:             http://{listen}/health");

    for mut request in server.incoming_requests() {
        let response = match (request.method(), request.url()) {
            (&Method::Get, "/health") => json_response(StatusCode(200), &json!({"ok": true}))?,
            (&Method::Post, "/v1/systemone") => {
                let mut body = String::new();
                request.as_reader().read_to_string(&mut body)?;
                match handle(engine, &body, temperature) {
                    Ok(value) => json_response(StatusCode(200), &value)?,
                    Err(error) => {
                        json_response(StatusCode(400), &json!({"error": error.to_string()}))?
                    }
                }
            }
            _ => json_response(StatusCode(404), &json!({"error": "not found"}))?,
        };
        request.respond(response)?;
    }
    Ok(())
}

fn handle(engine: &SystemOne, body: &str, temperature: f64) -> Result<Value> {
    let request: SystemOneRequest = serde_json::from_str(body).context("invalid request JSON")?;
    if request.questions.is_empty() {
        bail!("questions must not be empty");
    }

    let state = match request.state {
        Value::String(text) => text,
        value => serde_json::to_string_pretty(&value)?,
    };
    let owned = request
        .questions
        .iter()
        .map(|(id, definition)| OwnedQuestion::new(id, definition))
        .collect::<Result<Vec<_>>>()?;
    let label_refs = owned
        .iter()
        .map(|question| match &question.kind {
            OwnedKind::Noul => Vec::new(),
            OwnedKind::Choice(labels) | OwnedKind::Score(labels) => {
                labels.iter().map(String::as_str).collect()
            }
        })
        .collect::<Vec<Vec<&str>>>();
    let questions = owned
        .iter()
        .enumerate()
        .map(|(index, question)| Question {
            id: &question.id,
            text: &question.text,
            primitive: match &question.kind {
                OwnedKind::Noul => Primitive::Noul,
                OwnedKind::Choice(_) => Primitive::Choice(&label_refs[index]),
                OwnedKind::Score(_) => Primitive::Score(&label_refs[index]),
            },
        })
        .collect::<Vec<_>>();

    let started = Instant::now();
    let decisions = if questions.len() == 1 {
        vec![engine.answer(&state, &questions[0], temperature)?]
    } else {
        // Production API default: share only the state prefix, then evaluate each
        // question in its own batch row. This preserves question isolation: no
        // question text is placed in another question's context.
        //
        // The faster question-sheet pointer path remains available to benchmarks
        // as an explicitly experimental topology because it can change answers.
        let mut cache = engine.prefill_batched(&state, questions.len())?;
        engine.evaluate_batched_cached(&mut cache, &questions, temperature)?
    };
    let elapsed_ms = started.elapsed().as_millis() as usize;

    let mut answers = BTreeMap::new();
    for ((id, definition), answer) in request.questions.iter().zip(decisions) {
        let probabilities = answer
            .probabilities
            .iter()
            .map(|(label, probability)| (label.clone(), *probability))
            .collect::<BTreeMap<_, _>>();
        let value = match (&definition.kind[..], answer.value) {
            ("noul", DecisionValue::Noul(probability)) => json!({
                "type": "noul",
                "noul": probability,
                "confidence": answer.confidence
            }),
            ("choice", DecisionValue::Choice(choice)) => json!({
                "type": "choice",
                "choice": choice,
                "probabilities": probabilities,
                "confidence": answer.confidence
            }),
            ("score", DecisionValue::Score(score)) => json!({
                "type": "score",
                "score": score,
                "probabilities": probabilities,
                "confidence": answer.confidence
            }),
            _ => bail!("internal answer type mismatch for {id}"),
        };
        answers.insert(id.clone(), value);
    }

    Ok(serde_json::to_value(SystemOneResponse {
        model: request
            .model
            .unwrap_or_else(|| "gemma-system-one-4b-q4".to_string()),
        answers,
        usage: BTreeMap::from([("latency_ms".to_string(), elapsed_ms)]),
    })?)
}

struct OwnedQuestion {
    id: String,
    text: String,
    kind: OwnedKind,
}

enum OwnedKind {
    Noul,
    Choice(Vec<String>),
    Score(Vec<String>),
}

impl OwnedQuestion {
    fn new(id: &str, question: &ApiQuestion) -> Result<Self> {
        let mut text = question.instructions.clone();
        let kind = match question.kind.as_str() {
            "noul" => {
                append_rubric(&mut text, question.criteria.as_ref());
                OwnedKind::Noul
            }
            "choice" => {
                let criteria = question
                    .criteria
                    .as_ref()
                    .and_then(Value::as_object)
                    .context("choice criteria must be an object")?;
                append_rubric(&mut text, question.criteria.as_ref());
                OwnedKind::Choice(criteria.keys().cloned().collect())
            }
            "score" => {
                let criteria = question
                    .criteria
                    .as_ref()
                    .and_then(Value::as_array)
                    .context("score criteria must be an array")?;
                append_rubric(&mut text, question.criteria.as_ref());
                OwnedKind::Score((0..criteria.len()).map(|level| level.to_string()).collect())
            }
            other => bail!("unsupported question type {other:?}"),
        };
        Ok(Self {
            id: id.to_string(),
            text,
            kind,
        })
    }
}

fn append_rubric(text: &mut String, criteria: Option<&Value>) {
    let Some(criteria) = criteria else {
        return;
    };
    text.push_str("\n\nRubric:");
    match criteria {
        Value::Object(items) => {
            for (label, description) in items {
                text.push_str(&format!(
                    "\n- {label}: {}",
                    description.as_str().unwrap_or_default()
                ));
            }
        }
        Value::Array(items) => {
            for (index, description) in items.iter().enumerate() {
                text.push_str(&format!(
                    "\n- {index}: {}",
                    description.as_str().unwrap_or_default()
                ));
            }
        }
        _ => {}
    }
}

fn json_response(status: StatusCode, value: &Value) -> Result<Response<std::io::Cursor<Vec<u8>>>> {
    let body = serde_json::to_vec(value)?;
    let content_type = Header::from_bytes("Content-Type", "application/json")
        .map_err(|_| anyhow::anyhow!("invalid content-type header"))?;
    Ok(Response::from_data(body)
        .with_status_code(status)
        .with_header(content_type))
}
