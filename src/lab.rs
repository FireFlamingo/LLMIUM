//! Explicit local experiments. Writes are confined to fresh directories under the report root.
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{SystemTime, UNIX_EPOCH},
};

pub struct Lab {
    pub checkpoints: Mutex<Vec<PathBuf>>,
    pub enabled: bool,
    root: PathBuf,
    jobs: Mutex<BTreeMap<String, Value>>,
    busy: AtomicBool,
    serial: AtomicU64,
}

impl Lab {
    pub fn new(root: PathBuf, checkpoints: Vec<PathBuf>, enabled: bool) -> Arc<Self> {
        Arc::new(Self {
            root,
            checkpoints: Mutex::new(checkpoints),
            enabled,
            jobs: Mutex::new(BTreeMap::new()),
            busy: AtomicBool::new(false),
            serial: AtomicU64::new(0),
        })
    }
    pub fn status(&self, id: Option<&str>) -> Result<Value> {
        let jobs = self.jobs.lock().unwrap();
        Ok(match id {
            Some(id) => jobs.get(id).context("Unknown experiment")?.clone(),
            None => {
                json!({"editing_enabled":self.enabled,"busy":self.busy.load(Ordering::SeqCst),"jobs":jobs.values().collect::<Vec<_>>()})
            }
        })
    }
    fn update(&self, id: &str, phase: &str, extra: Value) {
        let mut jobs = self.jobs.lock().unwrap();
        let job = jobs.get_mut(id).unwrap();
        job["phase"] = json!(phase);
        for (key, value) in extra.as_object().unwrap() {
            job[key] = value.clone();
        }
        let _ = super::write_json(&self.root.join("lab").join(id).join("job-state.json"), job);
    }
    pub fn start(self: &Arc<Self>, args: &Value) -> Result<Value> {
        ensure!(
            self.enabled,
            "Restart the viewer with --allow-edits to enable checkpoint-copy experiments"
        );
        let model = usize::try_from(
            args["model"]
                .as_u64()
                .context("Choose a registered checkpoint")?,
        )?;
        let reference = self
            .checkpoints
            .lock()
            .unwrap()
            .get(model)
            .context("Unknown checkpoint")?
            .clone();
        let name = args["tensor"]
            .as_str()
            .context("Choose a tensor")?
            .to_owned();
        let index = usize::try_from(
            args["index"]
                .as_u64()
                .context("Index must be a nonnegative integer")?,
        )?;
        let value = args["value"]
            .as_f64()
            .context("Enter a finite parameter value")?;
        ensure!(value.is_finite(), "Enter a finite parameter value");
        let mode = args["mode"].as_str().unwrap_or("probe").to_owned();
        ensure!(
            ["probe", "generation"].contains(&mode.as_str()),
            "Unknown experiment mode"
        );
        let prompt = args["prompt"]
            .as_str()
            .unwrap_or("The capital of France is")
            .to_owned();
        ensure!(
            !prompt.trim().is_empty() && prompt.len() <= 4096,
            "Prompt must contain 1 to 4096 bytes"
        );
        let steps = args
            .get("steps")
            .map(|v| v.as_u64().context("Steps must be an integer"))
            .transpose()?
            .unwrap_or(4);
        ensure!((1..=16).contains(&steps), "Choose 1 to 16 generated tokens");
        #[cfg(not(feature = "inference"))]
        ensure!(
            mode != "generation",
            "Build with --features inference for model generation"
        );
        if mode == "generation" {
            let assets = if reference.is_dir() {
                reference.clone()
            } else {
                reference
                    .parent()
                    .context("Model has no assets directory")?
                    .to_owned()
            };
            let config: Value = serde_json::from_slice(
                &fs::read(assets.join("config.json"))
                    .context("Full generation requires model config.json beside the checkpoint")?,
            )?;
            ensure!(
                config["model_type"] == "qwen2" && assets.join("tokenizer.json").is_file(),
                "Full generation requires a Qwen2/2.5 checkpoint with tokenizer assets"
            );
        }
        // Validate before copying any files or accepting a job.
        {
            let cp = super::Checkpoint::open(&reference)?;
            let tensors = cp.tensors()?;
            let tensor = tensors.get(&name).context("Unknown tensor")?;
            ensure!(
                matches!(
                    tensor.dtype(),
                    safetensors::Dtype::F32 | safetensors::Dtype::F16 | safetensors::Dtype::BF16
                ),
                "Editing supports F32, F16 and BF16"
            );
            let width = tensor.dtype().bitsize() / 8;
            ensure!(
                index < tensor.data().len() / width,
                "Parameter index outside tensor"
            );
            let encoded = match tensor.dtype() {
                safetensors::Dtype::F32 => (value as f32).to_le_bytes().to_vec(),
                safetensors::Dtype::F16 => half::f16::from_f64(value).to_le_bytes().to_vec(),
                _ => half::bf16::from_f64(value).to_le_bytes().to_vec(),
            };
            ensure!(
                super::number(&encoded, tensor.dtype())?.is_finite(),
                "Value overflows the tensor storage dtype"
            );
            ensure!(
                encoded != tensor.data()[index * width..(index + 1) * width],
                "Value rounds to the current stored parameter"
            );
            if mode == "probe" {
                ensure!((1..=2).contains(&tensor.shape().len()) && tensor.data().len()/width<=20_000_000,"Probe mode supports vectors and matrices up to 20 million parameters; choose full generation for other tensors");
            }
        }
        ensure!(
            self.busy
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok(),
            "Another experiment is running; wait for it to finish"
        );
        let id = format!(
            "{}-{}",
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),
            self.serial.fetch_add(1, Ordering::SeqCst)
        );
        let output = self.root.join("lab").join(&id);
        let create = (|| -> Result<()> {
            fs::create_dir_all(self.root.join("lab"))?;
            ensure!(
                fs::canonicalize(self.root.join("lab"))?.starts_with(&self.root),
                "Experiment output must remain under the report root"
            );
            fs::create_dir(&output)?;
            Ok(())
        })();
        if let Err(error) = create {
            self.busy.store(false, Ordering::SeqCst);
            return Err(error);
        }
        let job = json!({"id":id,"phase":"queued","state":"running","source_model":model,"tensor":name,"flat_index":index,"requested_value":value,"mode":mode,"prompt":prompt,"steps":steps});
        self.jobs.lock().unwrap().insert(id.clone(), job.clone());
        let lab = Arc::clone(self);
        std::thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                lab.run(
                    &id,
                    &reference,
                    &name,
                    index,
                    value,
                    &mode,
                    &prompt,
                    steps as usize,
                    &output,
                )
            }));
            match result {
                Ok(Ok(result))=>lab.update(&id,"complete",result),
                Ok(Err(error))=>lab.update(&id,"failed",json!({"state":"failed","error":format!("{error:#}")})),
                Err(_)=>lab.update(&id,"failed",json!({"state":"failed","error":"Experiment worker stopped unexpectedly; inspect the saved job directory."})),
            }
            lab.busy.store(false, Ordering::SeqCst);
        });
        Ok(job)
    }
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        id: &str,
        reference: &Path,
        name: &str,
        index: usize,
        value: f64,
        mode: &str,
        prompt: &str,
        steps: usize,
        output: &Path,
    ) -> Result<Value> {
        self.update(id, "copying_and_editing", json!({}));
        let candidate = output.join("candidate");
        let seed = super::perturb::set(reference, name, index, value, &candidate)?;
        self.update(id, "assessing_change", json!({"seed":seed}));
        let before = super::scan(&candidate, Some(reference))?;
        super::write_json(&output.join("before.json"), &before)?;
        self.update(id, "restoring_and_verifying", json!({}));
        let repaired = output.join("repaired");
        let repair = super::repair(&candidate, reference, &repaired, &[name.to_owned()])?;
        let after = super::scan(&repaired, Some(reference))?;
        let mut demonstration = json!({"experiment":"User-selected finite parameter experiment","seeded_fault":true,"target":name,"flat_index":index,"seed":seed,"repair":repair,"before":before,"after":after,
            "limits":"Deliberate checkpoint-copy edit. Selected-parameter probes use deterministic synthetic inputs and do not measure language-model quality. Full generation is measured only when explicitly selected."});
        self.update(id, "measuring_outputs", json!({}));
        let report_name = if mode == "generation" {
            #[cfg(feature = "inference")]
            {
                super::write_json(&output.join("demo.json"), &demonstration)?;
                let assets = if reference.is_dir() {
                    reference
                } else {
                    reference
                        .parent()
                        .context("Model has no assets directory")?
                };
                let report = super::inference::compare(
                    &[
                        ("reference", reference),
                        ("candidate", candidate.as_path()),
                        ("repaired", repaired.as_path()),
                    ],
                    assets,
                    &[prompt.to_owned()],
                    steps,
                    "bf16",
                    true,
                )?;
                super::write_json(&output.join("runtime.json"), &report)?;
                "runtime.json"
            }
            #[cfg(not(feature = "inference"))]
            {
                let _ = (prompt, steps);
                anyhow::bail!("Inference feature is unavailable");
            }
        } else {
            let a = probe(reference, name)?;
            let b = probe(&candidate, name)?;
            let c = probe(&repaired, name)?;
            let stats = |values: &[f32]| json!({"elements":values.len(),"nonfinite_outputs":values.iter().filter(|x|!x.is_finite()).count(),"first_values":values.iter().take(8).map(|x|if x.is_finite(){json!(x)}else{json!(x.to_string())}).collect::<Vec<_>>(),"f32_sha256":format!("{:x}",Sha256::digest(values.iter().flat_map(|x|x.to_le_bytes()).collect::<Vec<_>>()))});
            demonstration["forward_pass"] = json!({"reference":stats(&a),"candidate":stats(&b),"repaired":stats(&c),"changed_outputs":a.iter().zip(&b).filter(|(a,b)|a.to_bits()!=b.to_bits()).count(),"maximum_absolute_delta":a.iter().zip(&b).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max)});
            demonstration["exact_forward_recovery"] = json!(a
                .iter()
                .map(|x| x.to_bits())
                .eq(c.iter().map(|x| x.to_bits())));
            demonstration["recovery_verified"] = json!(
                demonstration["exact_forward_recovery"] == true
                    && a.iter().chain(&c).all(|value| value.is_finite())
            );
            "demo.json"
        };
        super::write_json(&output.join("demo.json"), &demonstration)?;
        let mut checkpoints = self.checkpoints.lock().unwrap();
        let candidate_id = checkpoints.len();
        checkpoints.push(fs::canonicalize(candidate)?);
        let repaired_id = checkpoints.len();
        checkpoints.push(fs::canonicalize(repaired)?);
        Ok(
            json!({"state":"complete","report":format!("lab/{id}/{report_name}"),"assessment_report":format!("lab/{id}/demo.json"),"candidate_model":candidate_id,"repaired_model":repaired_id}),
        )
    }
}

fn probe(path: &Path, name: &str) -> Result<Vec<f32>> {
    let cp = super::Checkpoint::open(path)?;
    let tensors = cp.tensors()?;
    let tensor = tensors.get(name).context("Unknown probe tensor")?;
    if tensor.shape().len() == 2 {
        return super::forward(path, name);
    }
    ensure!(
        tensor.shape().len() == 1,
        "Probe requires a vector or matrix"
    );
    tensor
        .data()
        .chunks_exact(tensor.dtype().bitsize() / 8)
        .enumerate()
        .map(|(i, bytes)| Ok(super::number(bytes, tensor.dtype())? as f32 * (i as f32).cos()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selected_parameter_job_is_confined_and_recovers_probe() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("reference.safetensors");
        super::super::fixture(&source, false).unwrap();
        let lab = Lab::new(dir.path().to_owned(), vec![source.clone()], true);
        let original = super::super::file_hash(&source).unwrap();
        let args = json!({"model":0,"tensor":"model.layers.2.mlp.down_proj.weight","index":459,"value":100.0});
        let job = lab.start(&args).unwrap();
        let id = job["id"].as_str().unwrap();
        for _ in 0..300 {
            if lab.status(Some(id)).unwrap()["state"] != "running" {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let result = lab.status(Some(id)).unwrap();
        assert_eq!(result["state"], "complete", "{result}");
        let saved: Value = serde_json::from_slice(
            &fs::read(dir.path().join(result["report"].as_str().unwrap())).unwrap(),
        )
        .unwrap();
        assert_eq!(saved["exact_forward_recovery"], true);
        assert_eq!(saved["forward_pass"]["changed_outputs"], 8);
        assert_eq!(super::super::file_hash(&source).unwrap(), original);
        assert_eq!(lab.checkpoints.lock().unwrap().len(), 3);
        assert!(lab
            .start(&json!({"model":999,"tensor":"x","index":0,"value":1}))
            .is_err());
        assert!(lab
            .start(&json!({"model":0,"tensor":args["tensor"],"index":999999,"value":1}))
            .is_err());
        let locked = Lab::new(dir.path().to_owned(), vec![source], false);
        assert!(locked.start(&args).is_err());
    }
}
