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
    let report: Value = serde_json::from_slice(&bytes).context("Invalid trace report JSON")?;
    select(&report, args)
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
    let filtered = records
        .iter()
        .filter(|record| {
            record["module"]
                .as_str()
                .is_some_and(|module| module.starts_with(prefix))
                && (!nonfinite_only || record["nonfinite"].as_u64().unwrap_or(0) > 0)
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
        "evaluation":evaluation_index,"prompt":evaluation["prompt"],"step":step_index,
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
    fn trace_files_are_size_limited_before_json_parsing() {
        let file = tempfile::NamedTempFile::new().unwrap();
        file.as_file().set_len(MAX_REPORT_BYTES + 1).unwrap();
        assert!(read(&json!({"path":file.path()}))
            .unwrap_err()
            .to_string()
            .contains("64 MiB"));
    }
}
