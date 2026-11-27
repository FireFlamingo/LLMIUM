//! Local Qwen2 forward evaluation with deterministic decoding and finite-logit checks.
use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::{Config, ModelForCausalLM};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, time::Instant};
use tokenizers::Tokenizer;

enum RuntimeModel {
    Standard(ModelForCausalLM),
    Traced(super::traced_qwen::ModelForCausalLM),
}

impl RuntimeModel {
    fn clear_kv_cache(&mut self) {
        match self {
            Self::Standard(model) => model.clear_kv_cache(),
            Self::Traced(model) => model.clear_kv_cache(),
        }
    }

    fn forward(&mut self, input: &Tensor, offset: usize) -> Result<(Tensor, Option<Vec<Value>>)> {
        match self {
            Self::Standard(model) => Ok((model.forward(input, offset)?, None)),
            Self::Traced(model) => {
                let mut trace = super::traced_qwen::Trace::default();
                let output = model.forward(input, offset, &mut trace)?;
                Ok((output, Some(trace.records)))
            }
        }
    }
}

pub fn evaluate(
    model_path: &Path,
    assets: &Path,
    prompts: &[String],
    max_new_tokens: usize,
    dtype: &str,
    trace: bool,
) -> Result<Value> {
    ensure!(!prompts.is_empty(), "Provide at least one prompt");
    ensure!(
        (1..=256).contains(&max_new_tokens),
        "Generate between 1 and 256 tokens"
    );
    ensure!(
        !trace || max_new_tokens <= 16,
        "Activation tracing supports at most 16 generation steps"
    );
    let started = Instant::now();
    let config_path = assets.join("config.json");
    let config_json: Value = serde_json::from_slice(&fs::read(&config_path)?)?;
    ensure!(
        config_json["model_type"] == "qwen2",
        "This runtime adapter supports Qwen2 and Qwen2.5"
    );
    let config: Config = serde_json::from_value(config_json)?;
    ensure!(
        config.num_attention_heads > 0
            && config.num_key_value_heads > 0
            && config.hidden_size > 0
            && config
                .hidden_size
                .is_multiple_of(config.num_attention_heads)
            && config
                .num_attention_heads
                .is_multiple_of(config.num_key_value_heads)
            && (config.hidden_size / config.num_attention_heads).is_multiple_of(2)
            && config.max_position_embeddings > 0
            && config.max_position_embeddings <= u32::MAX as usize,
        "Invalid Qwen2 attention or position dimensions"
    );
    let tokenizer =
        Tokenizer::from_file(assets.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
    let checkpoint = super::Checkpoint::open(model_path)?;
    let files = checkpoint.files.iter().map(|path| Ok(json!({"name":path.file_name().unwrap().to_string_lossy(),"sha256":super::file_hash(path)?}))).collect::<Result<Vec<_>>>()?;
    let device = Device::Cpu;
    let dtype = if dtype == "bf16" {
        DType::BF16
    } else {
        DType::F32
    };
    eprintln!("Loading Qwen model on CPU using {dtype:?}");
    // The same immutable-file contract applies to the checkpoint scanner and this loader.
    let builder =
        unsafe { VarBuilder::from_mmaped_safetensors(&checkpoint.files, dtype, &device)? };
    let mut model = if trace {
        RuntimeModel::Traced(super::traced_qwen::ModelForCausalLM::new(&config, builder)?)
    } else {
        RuntimeModel::Standard(ModelForCausalLM::new(&config, builder)?)
    };
    let load_seconds = started.elapsed().as_secs_f64();
    let eos = [
        tokenizer.token_to_id("<|endoftext|>"),
        tokenizer.token_to_id("<|im_end|>"),
    ];
    let mut evaluations = Vec::new();
    for prompt in prompts {
        model.clear_kv_cache();
        let encoding = tokenizer
            .encode(prompt.as_str(), true)
            .map_err(anyhow::Error::msg)?;
        let input_ids = encoding.get_ids().to_vec();
        ensure!(!input_ids.is_empty(), "Prompt must contain tokens");
        ensure!(
            !trace || input_ids.len() <= 256,
            "Activation tracing supports at most 256 input tokens per prompt"
        );
        ensure!(
            input_ids.len() + max_new_tokens <= config.max_position_embeddings,
            "Prompt and generation exceed context limit"
        );
        let mut tokens = input_ids.clone();
        let mut generated = Vec::new();
        let mut steps = Vec::new();
        let mut stop_reason = "token_limit";
        let evaluation_started = Instant::now();
        for step in 0..max_new_tokens {
            let offset = if step == 0 { 0 } else { tokens.len() - 1 };
            let context = &tokens[offset..];
            let input = Tensor::new(context, &device)?.unsqueeze(0)?;
            let (output, activations) = model.forward(&input, offset)?;
            let logits = output
                .squeeze(0)?
                .squeeze(0)?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            let nonfinite = logits.iter().filter(|value| !value.is_finite()).count();
            let mut hasher = Sha256::new();
            for value in &logits {
                hasher.update(value.to_le_bytes());
            }
            let hash = format!("{:x}", hasher.finalize());
            let mut step_report = json!({"step":step,"logits_sha256":hash,"nonfinite_logits":nonfinite,"next_token_id":null});
            if let Some(activations) = activations {
                step_report["first_nonfinite_activation"] = activations
                    .iter()
                    .find(|record| record["nonfinite"].as_u64().unwrap_or(0) > 0)
                    .cloned()
                    .unwrap_or(Value::Null);
                step_report["activation_trace"] = json!(activations);
            }
            if nonfinite != 0 {
                steps.push(step_report);
                stop_reason = "nonfinite_logits";
                break;
            }
            let token = logits
                .iter()
                .enumerate()
                .max_by(|(index_a, a), (index_b, b)| {
                    a.partial_cmp(b).unwrap().then_with(|| index_b.cmp(index_a))
                })
                .context("Model returned empty logits")?
                .0 as u32;
            step_report["next_token_id"] = json!(token);
            steps.push(step_report);
            generated.push(token);
            tokens.push(token);
            if eos.contains(&Some(token)) {
                stop_reason = "eos";
                break;
            }
        }
        let text = tokenizer
            .decode(&generated, true)
            .map_err(anyhow::Error::msg)?;
        evaluations.push(json!({"prompt":prompt,"input_token_ids":input_ids,"generated_token_ids":generated,"generated_text":text,
            "steps":steps,"stop_reason":stop_reason,"elapsed_seconds":evaluation_started.elapsed().as_secs_f64()}));
        eprintln!(
            "Evaluated prompt using {} generation steps, stop {stop_reason}",
            steps.len()
        );
    }
    Ok(
        json!({"schema_version":1,"report_type":"runtime_evaluation","model":model_path,"model_canonical":fs::canonicalize(model_path)?,"assets":assets,
        "runtime":if trace{"llmium-instrumented-candle-qwen2-cpu"}else{"candle-qwen2-cpu"},"runtime_version":"0.11.1","dtype":format!("{dtype:?}"),"decoding":"greedy, lowest-token-id tie break",
        "activation_tracing":trace,"trace_coverage":if trace{Some("Ordered observations of embedding, each decoder module and residual, final normalization and logits. Statistics cover all elements in each observed output; hashes use row-major F32 conversions. Intentional attention-mask infinities are excluded. First observed nonfinite output identifies a numerical boundary, not by itself a unique causal weight or semantic defect.")}else{None},
        "files":files,"config_sha256":super::file_hash(&config_path)?,"tokenizer_sha256":super::file_hash(&assets.join("tokenizer.json"))?,
        "max_new_tokens":max_new_tokens,"load_seconds":load_seconds,"elapsed_seconds":started.elapsed().as_secs_f64(),"evaluations":evaluations,
        "interpretation":"Local full-model observations cover only these prompts and runtime settings. Numerical failure requires attribution to a controlled checkpoint comparison."}),
    )
}

pub fn compare(
    stages: &[(&str, &Path)],
    assets: &Path,
    prompts: &[String],
    max_new_tokens: usize,
    dtype: &str,
    trace: bool,
) -> Result<Value> {
    ensure!(
        stages
            .iter()
            .map(|(label, _)| *label)
            .eq(["reference", "candidate", "repaired"]),
        "Compare reference, candidate and repaired checkpoints in that order"
    );
    let mut reports = serde_json::Map::new();
    for (label, path) in stages {
        eprintln!("Full-model evaluation for {label}");
        reports.insert(
            (*label).into(),
            evaluate(path, assets, prompts, max_new_tokens, dtype, trace)?,
        );
    }
    Ok(comparison_report(reports))
}

fn comparison_report(reports: serde_json::Map<String, Value>) -> Value {
    let reference = reports["reference"]["evaluations"].as_array().unwrap();
    let repaired = reports["repaired"]["evaluations"].as_array().unwrap();
    let evidence_complete = !reference.is_empty()
        && reference.len() == repaired.len()
        && reference.iter().zip(repaired).all(|(a, b)| {
            a["input_token_ids"] == b["input_token_ids"]
                && a["steps"].as_array().is_some_and(|steps| !steps.is_empty())
                && b["steps"].as_array().is_some_and(|steps| !steps.is_empty())
        });
    let tokens_recovered = evidence_complete
        && reference
            .iter()
            .zip(repaired)
            .all(|(a, b)| a["generated_token_ids"] == b["generated_token_ids"]);
    let logits_recovered = evidence_complete
        && reference
            .iter()
            .zip(repaired)
            .all(|(a, b)| a["steps"] == b["steps"]);
    let nonfinite = |stage: &str| {
        reports[stage]["evaluations"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|evaluation| evaluation["steps"].as_array().unwrap())
            .map(|step| step["nonfinite_logits"].as_u64().unwrap())
            .sum::<u64>()
    };
    let candidate_nonfinite = nonfinite("candidate");
    let reference_nonfinite = nonfinite("reference");
    let repaired_nonfinite = nonfinite("repaired");
    let checkpoint_hash_recovery = reports["reference"]["files"] == reports["repaired"]["files"];
    let settings_match = [
        "config_sha256",
        "tokenizer_sha256",
        "runtime",
        "runtime_version",
        "dtype",
        "decoding",
        "max_new_tokens",
    ]
    .iter()
    .all(|key| {
        !reports["reference"][key].is_null()
            && reports["reference"][key] == reports["candidate"][key]
            && reports["reference"][key] == reports["repaired"][key]
    });
    let recovery_verified = evidence_complete
        && settings_match
        && logits_recovered
        && tokens_recovered
        && checkpoint_hash_recovery
        && reference_nonfinite == 0
        && repaired_nonfinite == 0;
    let activation_comparison = compare_activations(&reports);
    json!({"schema_version":1,"report_type":"runtime_comparison","stages":reports,
        "exact_logit_recovery":logits_recovered,"token_sequence_recovery":tokens_recovered,
        "checkpoint_hash_recovery":checkpoint_hash_recovery,"recovery_verified":recovery_verified,
        "settings_match":settings_match,
        "activation_comparison":activation_comparison,
        "reference_nonfinite_logits":reference_nonfinite,"candidate_nonfinite_logits":candidate_nonfinite,"repaired_nonfinite_logits":repaired_nonfinite,
        "interpretation":"Controlled full-model observations for the supplied checkpoints, prompts and CPU runtime. Interpret numerical failures and recovery from the recorded measurements. Broader security and behavioral claims require separate evidence."})
}

fn compare_activations(reports: &serde_json::Map<String, Value>) -> Option<Value> {
    if !["reference", "candidate", "repaired"]
        .iter()
        .all(|stage| reports[*stage]["activation_tracing"] == true)
    {
        return None;
    }
    let reference = reports["reference"]["evaluations"].as_array()?;
    let candidate = reports["candidate"]["evaluations"].as_array()?;
    let repaired = reports["repaired"]["evaluations"].as_array()?;
    let mut exact_recovery = !reference.is_empty() && reference.len() == repaired.len();
    let mut observations = Vec::new();
    for (prompt_index, (reference, candidate)) in reference.iter().zip(candidate).enumerate() {
        for (step_index, (a, b)) in reference["steps"]
            .as_array()?
            .iter()
            .zip(candidate["steps"].as_array()?)
            .enumerate()
        {
            let baseline = a["activation_trace"].as_array()?;
            let changed = b["activation_trace"].as_array()?;
            let aligned = baseline.len() == changed.len()
                && baseline.iter().zip(changed).all(|(a, b)| {
                    a["module"] == b["module"]
                        && a["operation"] == b["operation"]
                        && a["shape"] == b["shape"]
                });
            let first_changed = if aligned {
                baseline.iter().zip(changed).find(|(a,b)| a["f32_sha256"] != b["f32_sha256"]).map(|(a,b)|json!({"sequence":a["sequence"],"module":a["module"],"operation":a["operation"],"reference_sha256":a["f32_sha256"],"candidate_sha256":b["f32_sha256"],"reference_nonfinite":a["nonfinite"],"candidate_nonfinite":b["nonfinite"]}))
            } else {
                None
            };
            observations.push(json!({"prompt_index":prompt_index,"step":step_index,"records_aligned":aligned,"first_changed_output":first_changed,"first_nonfinite_output":b["first_nonfinite_activation"]}));
        }
    }
    for (reference, repaired) in reference.iter().zip(repaired) {
        let a = reference["steps"].as_array()?;
        let b = repaired["steps"].as_array()?;
        exact_recovery &= a.len() == b.len() && !a.is_empty();
        for (a, b) in a.iter().zip(b) {
            let a = a["activation_trace"].as_array()?;
            let b = b["activation_trace"].as_array()?;
            exact_recovery &= !a.is_empty() && a == b;
        }
    }
    Some(
        json!({"exact_recorded_activation_recovery":exact_recovery,"candidate_observations":observations,
        "interpretation":"The first changed recorded output localizes an observed execution boundary for the controlled inputs. Intermediate values between observation points, unique causal parameter identity, semantic correctness and untested inputs require separate evidence."}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reports(nonfinite: u64) -> serde_json::Map<String, Value> {
        ["reference", "candidate", "repaired"].into_iter().map(|label| {
            (label.into(), json!({"files":[{"name":"model.safetensors","sha256":"a"}],
                "config_sha256":"config","tokenizer_sha256":"tokenizer","runtime":"test","runtime_version":"1","dtype":"F32","decoding":"greedy","max_new_tokens":1,
                "evaluations":[{"generated_token_ids":[1],"steps":[{"logits_sha256":"a","nonfinite_logits":nonfinite,"next_token_id":1}]}]}))
        }).collect()
    }

    #[test]
    fn matching_failed_logits_do_not_verify_recovery() {
        let report = comparison_report(reports(1));
        assert_eq!(report["exact_logit_recovery"], true);
        assert_eq!(report["recovery_verified"], false);
    }

    #[test]
    fn recovery_requires_both_checkpoint_and_runtime_evidence() {
        assert_eq!(comparison_report(reports(0))["recovery_verified"], true);
        let mut changed = reports(0);
        changed["repaired"]["evaluations"][0]["steps"][0]["logits_sha256"] = json!("b");
        assert_eq!(comparison_report(changed)["recovery_verified"], false);
        let mut changed = reports(0);
        changed["repaired"]["files"][0]["sha256"] = json!("b");
        assert_eq!(comparison_report(changed)["recovery_verified"], false);
        let mut changed = reports(0);
        changed["repaired"]["evaluations"] = json!([]);
        assert_eq!(comparison_report(changed)["recovery_verified"], false);
        let mut changed = reports(0);
        changed["candidate"]["tokenizer_sha256"] = json!("different-assets");
        assert_eq!(comparison_report(changed)["recovery_verified"], false);
    }

    #[test]
    fn trace_comparison_localizes_first_changed_output_and_checks_all_restored_records() {
        let mut stages = reports(0);
        for stage in stages.values_mut() {
            stage["activation_tracing"] = json!(true);
            stage["evaluations"][0]["steps"][0]["activation_trace"] = json!([
                {"sequence":0,"module":"embedding","operation":"embedding","shape":[1,2,4],"f32_sha256":"a","nonfinite":0},
                {"sequence":1,"module":"q_proj","operation":"linear","shape":[1,2,4],"f32_sha256":"b","nonfinite":0}
            ]);
        }
        stages["candidate"]["evaluations"][0]["steps"][0]["activation_trace"][1]["f32_sha256"] =
            json!("fault");
        let result = comparison_report(stages.clone());
        assert_eq!(
            result["activation_comparison"]["candidate_observations"][0]["first_changed_output"]
                ["module"],
            "q_proj"
        );
        assert_eq!(
            result["activation_comparison"]["exact_recorded_activation_recovery"],
            true
        );
        stages["repaired"]["evaluations"][0]["steps"][0]["activation_trace"][0]["f32_sha256"] =
            json!("drift");
        assert_eq!(
            comparison_report(stages)["activation_comparison"]
                ["exact_recorded_activation_recovery"],
            false
        );
    }
}
