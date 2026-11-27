//! Explicit, reproducible finite parameter perturbations in new checkpoint copies.
use anyhow::{ensure, Context, Result};
use safetensors::{Dtype, SafeTensors};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Seek, SeekFrom, Write},
    path::Path,
};

pub fn scale(
    reference: &Path,
    name: &str,
    index: usize,
    factor: f64,
    output: &Path,
) -> Result<Value> {
    ensure!(!output.exists(), "Seed output must be a new directory");
    ensure!(factor.is_finite(), "Scale factor must be finite");
    let checkpoint = super::Checkpoint::open(reference)?;
    let tensors = checkpoint.tensors()?;
    let tensor = tensors.get(name).context("Unknown seed tensor")?;
    let width = tensor.dtype().bitsize() / 8;
    ensure!(
        width > 0 && index < tensor.data().len() / width,
        "Seed index outside tensor"
    );
    let old_bytes = &tensor.data()[index * width..(index + 1) * width];
    let old = super::number(old_bytes, tensor.dtype())?;
    ensure!(old.is_finite(), "Seed source parameter must be finite");
    let desired = old * factor;
    ensure!(desired.is_finite(), "Scaled parameter overflow");
    let replacement = match tensor.dtype() {
        Dtype::F32 => (desired as f32).to_le_bytes().to_vec(),
        Dtype::F16 => half::f16::from_f64(desired).to_le_bytes().to_vec(),
        Dtype::BF16 => half::bf16::from_f64(desired).to_le_bytes().to_vec(),
        _ => anyhow::bail!("Finite scaling supports F32, F16 and BF16"),
    };
    let actual = super::number(&replacement, tensor.dtype())?;
    ensure!(
        actual.is_finite(),
        "Scaled parameter is nonfinite in the storage dtype"
    );
    ensure!(
        replacement != old_bytes,
        "Scaling does not change the stored parameter"
    );
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let staging = tempfile::Builder::new()
        .prefix(".llmium-scale-")
        .tempdir_in(parent)?;
    for file in &checkpoint.files {
        fs::copy(file, staging.path().join(file.file_name().unwrap()))?;
    }
    super::copy_checkpoint_json(reference, staging.path())?;
    for (file, map) in checkpoint.files.iter().zip(&checkpoint.maps) {
        if let Ok(view) = SafeTensors::deserialize(map)?.tensor(name) {
            let offset = view.data().as_ptr() as usize - map.as_ptr() as usize + index * width;
            let mut target = fs::OpenOptions::new()
                .write(true)
                .open(staging.path().join(file.file_name().unwrap()))?;
            target.seek(SeekFrom::Start(offset as u64))?;
            target.write_all(&replacement)?;
            target.sync_all()?;
        }
    }
    let seeded = super::Checkpoint::open(staging.path())?;
    let seeded_tensors = seeded.tensors()?;
    ensure!(
        seeded_tensors[name].data()[index * width..(index + 1) * width] == replacement,
        "Seed readback mismatch"
    );
    let hex = |bytes: &[u8]| {
        bytes
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    };
    let manifest = json!({"schema_version":1,"seeded_fault":true,"operation":"finite_parameter_scale","reference":reference,"output":output,"tensor":name,"flat_index":index,"shape":tensor.shape(),"dtype":format!("{:?}",tensor.dtype()),"factor":factor,"original_value":old,"stored_value":actual,"original_raw_le_hex":hex(old_bytes),"stored_raw_le_hex":hex(&replacement),"source_tensor_sha256":format!("{:x}",Sha256::digest(tensor.data())),"seeded_tensor_sha256":format!("{:x}",Sha256::digest(seeded_tensors[name].data())),"interpretation":"Deliberately seeded finite parameter change in a new checkpoint copy. Behavioral impact must be measured separately; this operation does not establish a naturally occurring vulnerability."});
    super::write_json(&staging.path().join("llmium-seed.json"), &manifest)?;
    drop(seeded_tensors);
    drop(seeded);
    fs::rename(staging.path(), output)?;
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn finite_seed_changes_exactly_one_parameter_and_preserves_source() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.safetensors");
        super::super::fixture(&source, false).unwrap();
        let hash = super::super::file_hash(&source).unwrap();
        let output = dir.path().join("candidate");
        let name = "model.layers.0.self_attn.q_proj.weight";
        let result = scale(&source, name, 1, 1000.0, &output).unwrap();
        assert!(result["stored_value"].as_f64().unwrap().is_finite());
        assert_eq!(super::super::file_hash(&source).unwrap(), hash);
        let assessed = super::super::scan(&output, Some(&source)).unwrap();
        let target = assessed["tensors"]
            .as_array()
            .unwrap()
            .iter()
            .find(|tensor| tensor["name"] == name)
            .unwrap();
        assert_eq!(target["changed_elements"], 1);
        assert_eq!(target["first_changed_indices"], json!([1]));
        assert_eq!(target["nonfinite"], 0);
        assert!(scale(&source, name, 1, 1000.0, &output).is_err());
        assert!(scale(&source, name, usize::MAX, 2.0, &dir.path().join("bad")).is_err());
        assert!(scale(&source, name, 1, f64::NAN, &dir.path().join("bad")).is_err());
    }
}
