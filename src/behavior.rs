//! Descriptive, bounded-output signals. These are not semantic or security verdicts.
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub fn analyze(tokens: &[u32], stop_reason: &str) -> Value {
    let mut frequencies = BTreeMap::<u32, usize>::new();
    let (mut longest_run, mut run) = (0, 0);
    for (index, &token) in tokens.iter().enumerate() {
        *frequencies.entry(token).or_default() += 1;
        run = if index > 0 && tokens[index - 1] == token {
            run + 1
        } else {
            1
        };
        longest_run = longest_run.max(run);
    }
    let most_common = frequencies
        .iter()
        .max_by(|(a_id, a), (b_id, b)| a.cmp(b).then_with(|| b_id.cmp(a_id)));
    let mut suffix = None;
    let mut best_span = 0;
    for period in 1..=32.min(tokens.len() / 3) {
        let pattern = &tokens[tokens.len() - period..];
        let mut repeats = 1;
        while (repeats + 1) * period <= tokens.len()
            && &tokens[tokens.len() - (repeats + 1) * period..tokens.len() - repeats * period]
                == pattern
        {
            repeats += 1;
        }
        let span = repeats * period;
        if repeats >= 3 && span >= 8 && span > best_span {
            best_span = span;
            suffix = Some(
                json!({"period":period,"repeats":repeats,"span":span,"start":tokens.len()-span,"pattern_token_ids":pattern}),
            );
        }
    }
    let count = tokens.len();
    let entropy = frequencies
        .values()
        .map(|&frequency| {
            let p = frequency as f64 / count as f64;
            -p * p.log2()
        })
        .sum::<f64>();
    let repetition_signal =
        stop_reason == "token_limit" && best_span >= 8 && best_span * 2 >= count;
    let concentration_signal =
        stop_reason == "token_limit" && count >= 16 && frequencies.len() <= 2;
    json!({"generated_tokens":count,"unique_tokens":frequencies.len(),"empirical_token_entropy_bits":entropy,
        "longest_identical_token_run":longest_run,"most_common_token":most_common.map(|(&id,&frequency)|json!({"token_id":id,"count":frequency,"fraction":frequency as f64/count as f64})),
        "repeated_suffix":suffix,"bounded_repetition_signal":repetition_signal,"low_diversity_signal":concentration_signal,
        "thresholds":{"minimum_repeated_span":8,"minimum_repeats":3,"maximum_period":32,"minimum_suffix_fraction":0.5,"low_diversity_minimum_tokens":16,"low_diversity_maximum_unique_tokens":2,"required_stop_reason":"token_limit"},
        "interpretation":"Descriptive bounded-output signals, not evidence of an infinite reasoning loop, malicious weights or a security vulnerability. Legitimate requested repetition can trigger them. Compare controlled prompts, checkpoints and recovery before attributing degradation."})
}

fn evaluation(evaluation: &Value) -> Result<Value> {
    let ids = evaluation["generated_token_ids"]
        .as_array()
        .context("Missing generated token IDs")?;
    ensure!(
        ids.len() <= 65536,
        "Behavior assessment accepts at most 65536 generated tokens per evaluation"
    );
    let ids = ids
        .iter()
        .map(|id| {
            u32::try_from(
                id.as_u64()
                    .context("Token IDs must be nonnegative integers")?,
            )
            .map_err(Into::into)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(analyze(
        &ids,
        evaluation["stop_reason"]
            .as_str()
            .context("Missing stop reason")?,
    ))
}

pub fn comparison(stages: &serde_json::Map<String, Value>) -> Result<Value> {
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
        let a = &stages.get("reference").unwrap_or(&Value::Null)[*key];
        !a.is_null()
            && ["candidate", "repaired"]
                .iter()
                .all(|stage| stages.get(*stage).is_some_and(|value| value[*key] == *a))
    });
    let reference = stages.get("reference").context("Missing reference stage")?["evaluations"]
        .as_array()
        .context("Missing reference evaluations")?;
    let candidate = stages.get("candidate").context("Missing candidate stage")?["evaluations"]
        .as_array()
        .context("Missing candidate evaluations")?;
    let repaired = stages.get("repaired").context("Missing repaired stage")?["evaluations"]
        .as_array()
        .context("Missing repaired evaluations")?;
    ensure!(
        reference.len() == candidate.len() && reference.len() == repaired.len(),
        "Behavior comparison requires matching evaluation counts"
    );
    let mut observations = Vec::new();
    for (index, ((a, b), c)) in reference.iter().zip(candidate).zip(repaired).enumerate() {
        let reference = evaluation(a)?;
        let candidate = evaluation(b)?;
        let repaired = evaluation(c)?;
        let same_inputs = a["input_token_ids"].is_array()
            && a["prompt"].is_string()
            && a["input_token_ids"] == b["input_token_ids"]
            && a["input_token_ids"] == c["input_token_ids"]
            && a["prompt"] == b["prompt"]
            && a["prompt"] == c["prompt"];
        observations.push(json!({"evaluation":index,"prompt":a["prompt"],"input_tokens_match":same_inputs,
            "candidate_only_repetition_signal":same_inputs && settings_match && candidate["bounded_repetition_signal"]==true && reference["bounded_repetition_signal"]==false,
            "candidate_only_low_diversity_signal":same_inputs && settings_match && candidate["low_diversity_signal"]==true && reference["low_diversity_signal"]==false,
            "recorded_behavior_recovered":reference==repaired && a["generated_token_ids"]==c["generated_token_ids"] && a["stop_reason"]==c["stop_reason"],
            "reference":reference,"candidate":candidate,"repaired":repaired}));
    }
    Ok(
        json!({"settings_match":settings_match,"observations":observations,"interpretation":"Controlled descriptive comparison. A candidate-only signal requires matching inputs and recorded runtime settings. It can prioritize review but does not by itself establish semantic incorrectness, infinite reasoning or exploitability."}),
    )
}

pub fn report(source: &Value) -> Result<Value> {
    match source["report_type"].as_str() {
        Some("runtime_comparison") => {
            comparison(source["stages"].as_object().context("Missing stages")?)
        }
        Some("runtime_evaluation") => {
            let observations=source["evaluations"].as_array().context("Missing evaluations")?.iter().enumerate().map(|(index,item)|Ok(json!({"evaluation":index,"prompt":item["prompt"],"behavior":evaluation(item)?}))).collect::<Result<Vec<_>>>()?;
            Ok(json!({"observations":observations}))
        }
        _ => anyhow::bail!("Choose a runtime evaluation or comparison report"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn detects_constant_periodic_and_prefixed_suffixes_with_exact_spans() {
        let constant = analyze(&[7; 16], "token_limit");
        assert_eq!(constant["repeated_suffix"]["period"], 1);
        assert_eq!(constant["bounded_repetition_signal"], true);
        let alternating = analyze(
            &[1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "token_limit",
        );
        assert_eq!(alternating["repeated_suffix"]["period"], 2);
        assert_eq!(alternating["low_diversity_signal"], true);
        let prefix = analyze(&[9, 8, 7, 1, 2, 3, 1, 2, 3, 1, 2, 3], "token_limit");
        assert_eq!(prefix["repeated_suffix"]["start"], 3);
        assert_eq!(prefix["repeated_suffix"]["span"], 9);
    }
    #[test]
    fn short_diverse_and_eos_outputs_do_not_claim_bounded_loop_signals() {
        assert_eq!(
            analyze(&[], "token_limit")["empirical_token_entropy_bits"],
            0.0
        );
        assert_eq!(
            analyze(&[1, 1, 1, 1], "token_limit")["bounded_repetition_signal"],
            false
        );
        assert_eq!(
            analyze(&(0..32).collect::<Vec<_>>(), "token_limit")["bounded_repetition_signal"],
            false
        );
        assert_eq!(analyze(&[1; 16], "eos")["bounded_repetition_signal"], false);
        // Requested repetition is intentionally still reported as a descriptive signal.
        assert_eq!(
            analyze(&[42; 16], "token_limit")["low_diversity_signal"],
            true
        );
    }
    #[test]
    fn comparisons_require_same_inputs_for_candidate_only_attribution() {
        assert!(comparison(&serde_json::Map::new()).is_err());
        let clean = json!({"prompt":"question","input_token_ids":[5],"generated_token_ids":(0..16).collect::<Vec<_>>(),"stop_reason":"token_limit"});
        let changed = json!({"prompt":"question","input_token_ids":[5],"generated_token_ids":vec![2u32;16],"stop_reason":"token_limit"});
        let mut stages = serde_json::Map::new();
        stages.insert("reference".into(), json!({"evaluations":[clean.clone()]}));
        stages.insert("candidate".into(), json!({"evaluations":[changed]}));
        stages.insert("repaired".into(), json!({"evaluations":[clean]}));
        for stage in stages.values_mut() {
            for key in [
                "config_sha256",
                "tokenizer_sha256",
                "runtime",
                "runtime_version",
                "dtype",
                "decoding",
                "max_new_tokens",
            ] {
                stage[key] = json!("same");
            }
        }
        assert_eq!(
            comparison(&stages).unwrap()["observations"][0]["candidate_only_repetition_signal"],
            true
        );
        stages["candidate"]["evaluations"][0]["input_token_ids"] = json!([6]);
        assert_eq!(
            comparison(&stages).unwrap()["observations"][0]["candidate_only_repetition_signal"],
            false
        );
    }
}
