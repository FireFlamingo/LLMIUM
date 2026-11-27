//! Bounded agent and CLI access to recorded activation evidence; no model loading.
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{fs::File, io::Read, path::Path};

const MAX_REPORT_BYTES: u64 = 64 * 1024 * 1024;

fn index(args: &Value, name: &str, default: usize) -> Result<usize> {
    args.get(name)
        .map(|value| {
            usize::try_from(
                value
                    .as_u64()
                    .with_context(|| format!("{name} must be a nonnegative integer"))?,
            )
            .map_err(Into::into)
        })
        .unwrap_or(Ok(default))
}

pub fn read(args: &Value) -> Result<Value> {
    let path = Path::new(args["path"].as_str().context("Missing trace report path")?);
    select(&load(path)?, args)
}

pub fn load(path: &Path) -> Result<Value> {
    let file = File::open(path)?;
    let metadata = file.metadata()?;
    ensure!(
        metadata.is_file() && metadata.len() <= MAX_REPORT_BYTES,
        "Trace report must be a regular file no larger than 64 MiB"
    );
    let mut bytes = Vec::new();
    file.take(MAX_REPORT_BYTES + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= MAX_REPORT_BYTES,
        "Trace report exceeds 64 MiB"
    );
    serde_json::from_slice(&bytes).context("Invalid trace report JSON")
}

fn select(report: &Value, args: &Value) -> Result<Value> {
    let (stage_name, stage) = match report["report_type"].as_str() {
        Some("runtime_comparison") => {
            let name = args
                .get("stage")
                .map(|value| value.as_str().context("Stage must be a string"))
                .transpose()?
                .unwrap_or("candidate");
            ensure!(
                ["reference", "candidate", "repaired"].contains(&name),
                "Unknown comparison stage"
            );
            (Some(name), &report["stages"][name])
        }
        Some("runtime_evaluation") => {
            ensure!(
                args.get("stage").is_none(),
                "Single evaluation reports do not have named stages"
            );
            (None, report)
        }
        _ => anyhow::bail!("Choose a runtime evaluation or comparison report"),
    };
    let evaluation_index = index(args, "evaluation", 0)?;
    let step_index = index(args, "step", 0)?;
    let offset = index(args, "offset", 0)?;
    let count = index(args, "count", 32)?;
    ensure!(
        (1..=128).contains(&count),
        "Read between 1 and 128 trace records per call"
    );
    let prefix = args
        .get("module_prefix")
        .map(|value| value.as_str().context("module_prefix must be a string"))
        .transpose()?
        .unwrap_or("");
    let nonfinite_only = args
        .get("nonfinite_only")
        .map(|value| value.as_bool().context("nonfinite_only must be a boolean"))
        .transpose()?
        .unwrap_or(false);
    let changed_only = args
        .get("changed_only")
        .map(|value| value.as_bool().context("changed_only must be a boolean"))
        .transpose()?
        .unwrap_or(false);
    let evaluation = stage["evaluations"]
        .as_array()
        .and_then(|items| items.get(evaluation_index))
        .context("Evaluation index outside report")?;
    let step = evaluation["steps"]
        .as_array()
        .and_then(|items| items.get(step_index))
        .context("Step index outside evaluation")?;
    let records = step["activation_trace"]
        .as_array()
        .context("This step has no recorded activation trace")?;
    ensure!(
        records.iter().all(Value::is_object),
        "Activation observations must be objects"
    );
    let reference = stage_name.and_then(|_| {
        report["stages"]["reference"]["evaluations"]
            .as_array()?
            .get(evaluation_index)
    });
    let baseline = reference
        .and_then(|evaluation| evaluation["steps"].as_array()?.get(step_index))
        .and_then(|step| step["activation_trace"].as_array());
    ensure!(
        !changed_only || baseline.is_some(),
        "Changed-only reads require a comparison report with a recorded reference step"
    );
    let input_context_matches = reference.and_then(|reference| {
        let reference_ids = reference["input_token_ids"].as_array()?;
        let candidate_ids = evaluation["input_token_ids"].as_array()?;
        let a = reference["generated_token_ids"].as_array()?;
        let b = evaluation["generated_token_ids"].as_array()?;
        Some(
            reference_ids == candidate_ids
                && a.len() >= step_index
                && b.len() >= step_index
                && a[..step_index] == b[..step_index],
        )
    });
    let annotated = records.iter().enumerate().map(|(index,record)| {
        let mut record=record.clone();
        if let Some(baseline)=baseline {
            let original=baseline.get(index).unwrap_or(&Value::Null);
            let aligned=record["f32_sha256"].is_string() && original["f32_sha256"].is_string() &&
                ["module","operation","shape"].iter().all(|key| record[*key]==original[*key]);
            record["reference_comparison"]=json!({"aligned":aligned,"changed":if aligned{Some(record["f32_sha256"]!=original["f32_sha256"])}else{None},
                "reference_f32_sha256":if aligned{original["f32_sha256"].clone()}else{Value::Null}});
        }
        record
    }).collect::<Vec<_>>();
    let filtered = annotated
        .iter()
        .filter(|record| {
            record["module"]
                .as_str()
                .is_some_and(|module| module.starts_with(prefix))
                && (!nonfinite_only || record["nonfinite"].as_u64().unwrap_or(0) > 0)
                && (!changed_only || record["reference_comparison"]["changed"] == true)
        })
        .collect::<Vec<_>>();
    ensure!(
        offset <= filtered.len(),
        "Trace offset outside filtered observations"
    );
    let selected = filtered.iter().skip(offset).take(count).collect::<Vec<_>>();
    let end = offset + selected.len();
    Ok(
        json!({"report":args["path"],"stage":stage_name,"model":stage["model"],"files":stage["files"],
        "config_sha256":stage["config_sha256"],"tokenizer_sha256":stage["tokenizer_sha256"],
        "evaluation":evaluation_index,"prompt":evaluation["prompt"],"step":step_index,"input_context_matches":input_context_matches,
        "observations_total":records.len(),"matching_observations":filtered.len(),"offset":offset,
        "next_offset":if end<filtered.len(){Some(end)}else{None},"records":selected,
        "first_nonfinite_activation":step["first_nonfinite_activation"],"coverage":stage["trace_coverage"]}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn report() -> Value {
        json!({"report_type":"runtime_comparison","stages":{"candidate":{"evaluations":[{"steps":[{"activation_trace":[
            {"module":"embedding","nonfinite":0},{"module":"layer.q","nonfinite":1},{"module":"layer.k","nonfinite":0},{"module":"layer.attention","nonfinite":8}
        ]}]}]}}})
    }
    #[test]
    fn trace_pages_preserve_order_and_filter_before_paging() {
        let first = select(
            &report(),
            &json!({"module_prefix":"layer.","nonfinite_only":true,"count":1}),
        )
        .unwrap();
        assert_eq!(first["records"][0]["module"], "layer.q");
        assert_eq!(first["matching_observations"], 2);
        assert_eq!(first["next_offset"], 1);
        let second = select(
            &report(),
            &json!({"module_prefix":"layer.","nonfinite_only":true,"count":1,"offset":1}),
        )
        .unwrap();
        assert_eq!(second["records"][0]["module"], "layer.attention");
        assert!(second["next_offset"].is_null());
    }
    #[test]
    fn trace_selection_rejects_invalid_types_bounds_and_untraced_steps() {
        for args in [
            json!({"count":129}),
            json!({"count":0}),
            json!({"offset":5}),
            json!({"step":1}),
            json!({"evaluation":-1}),
            json!({"stage":"unknown"}),
            json!({"nonfinite_only":"true"}),
            json!({"changed_only":"true"}),
            json!({"changed_only":true}),
        ] {
            assert!(select(&report(), &args).is_err(), "{args}");
        }
        let mut untraced = report();
        untraced["stages"]["candidate"]["evaluations"][0]["steps"][0]["activation_trace"] =
            Value::Null;
        assert!(select(&untraced, &json!({})).is_err());
        let single = json!({"report_type":"runtime_evaluation","evaluations":report()["stages"]["candidate"]["evaluations"]});
        assert!(select(&single, &json!({})).is_ok());
        assert!(select(&single, &json!({"stage":"candidate"})).is_err());
    }
    #[test]
    fn finite_differences_filter_before_paging_and_identify_context_changes() {
        let records = json!([
            {"module":"embedding","operation":"lookup","shape":[1,2],"nonfinite":0,"f32_sha256":"a"},
            {"module":"norm","operation":"norm","shape":[1,2],"nonfinite":0,"f32_sha256":"b"},
            {"module":"head","operation":"linear","shape":[1,2],"nonfinite":0,"f32_sha256":"c"}]);
        let evaluation = json!({"input_token_ids":[9],"generated_token_ids":[10,11],"steps":[{"activation_trace":records},{"activation_trace":records}]});
        let mut report = json!({"report_type":"runtime_comparison","stages":{"reference":{"evaluations":[evaluation]},"candidate":{"evaluations":[evaluation]}}});
        report["stages"]["candidate"]["evaluations"][0]["steps"][0]["activation_trace"][1]
            ["f32_sha256"] = json!("changed");
        report["stages"]["candidate"]["evaluations"][0]["steps"][0]["activation_trace"][2]
            ["f32_sha256"] = json!("changed2");
        let result = select(&report, &json!({"changed_only":true,"count":1})).unwrap();
        assert_eq!(result["matching_observations"], 2);
        assert_eq!(result["records"][0]["module"], "norm");
        assert_eq!(result["records"][0]["nonfinite"], 0);
        assert_eq!(result["next_offset"], 1);
        assert_eq!(result["input_context_matches"], true);
        report["stages"]["candidate"]["evaluations"][0]["generated_token_ids"][0] = json!(99);
        assert_eq!(
            select(&report, &json!({"step":1})).unwrap()["input_context_matches"],
            false
        );
    }
    #[test]
    fn trace_files_are_size_limited_before_json_parsing() {
        let file = tempfile::NamedTempFile::new().unwrap();
        file.as_file().set_len(MAX_REPORT_BYTES + 1).unwrap();
        assert!(read(&json!({"path":file.path()}))
            .unwrap_err()
            .to_string()
            .contains("64 MiB"));
    }
}
