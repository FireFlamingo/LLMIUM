use anyhow::{bail, ensure, Context, Result};
use clap::{Parser, Subcommand};
use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{self, BufRead, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

mod atlas;
mod behavior;
#[cfg(feature = "inference")]
mod inference;
mod lab;
mod perturb;
mod trace_report;
#[cfg(feature = "inference")]
mod traced_qwen;

#[derive(Parser)]
#[command(
    version,
    about = "Open-weight checkpoint assessment and verified reference repair"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    SeedScale {
        reference: PathBuf,
        #[arg(long)]
        tensor: String,
        #[arg(long)]
        index: usize,
        #[arg(long)]
        factor: f64,
        #[arg(long)]
        output: PathBuf,
    },
    AssessBehavior {
        path: PathBuf,
    },
    TraceSlice {
        path: PathBuf,
        #[arg(long)]
        stage: Option<String>,
        #[arg(long, default_value_t = 0)]
        evaluation: usize,
        #[arg(long, default_value_t = 0)]
        step: usize,
        #[arg(long, default_value = "")]
        module_prefix: String,
        #[arg(long)]
        nonfinite_only: bool,
        #[arg(long)]
        changed_only: bool,
        #[arg(long, default_value_t = 0)]
        offset: usize,
        #[arg(long, default_value_t = 32)]
        count: usize,
    },
    Graph {
        path: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    TensorTile {
        path: PathBuf,
        #[arg(long)]
        reference: Option<PathBuf>,
        #[arg(long)]
        tensor: String,
        #[arg(long, default_value_t = 0)]
        row: usize,
        #[arg(long, default_value_t = 0)]
        column: usize,
        #[arg(long, default_value_t = 16)]
        rows: usize,
        #[arg(long, default_value_t = 16)]
        columns: usize,
    },
    TensorSlice {
        path: PathBuf,
        #[arg(long)]
        tensor: String,
        #[arg(long, default_value_t = 0)]
        offset: usize,
        #[arg(long, default_value_t = 64)]
        count: usize,
    },
    Scan {
        path: PathBuf,
        #[arg(long)]
        reference: Option<PathBuf>,
        #[arg(long)]
        output: PathBuf,
    },
    Repair {
        path: PathBuf,
        #[arg(long)]
        reference: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, required = true)]
        tensor: Vec<String>,
    },
    Demo {
        #[arg(long, default_value = "runs/rust-demo")]
        output: PathBuf,
    },
    Experiment {
        reference: PathBuf,
        #[arg(long)]
        tensor: String,
        #[arg(long)]
        output: PathBuf,
    },
    Evaluate {
        model: PathBuf,
        #[arg(long)]
        assets: PathBuf,
        #[arg(long, required = true)]
        prompt: Vec<String>,
        #[arg(long, default_value_t = 8)]
        max_new_tokens: usize,
        #[arg(long, default_value = "bf16", value_parser = ["bf16", "f32"])]
        dtype: String,
        /// Record bounded intermediate activations from an instrumented Qwen2 forward pass.
        #[arg(long)]
        trace: bool,
        #[arg(long)]
        output: PathBuf,
    },
    RuntimeCompare {
        #[arg(long)]
        reference: PathBuf,
        #[arg(long)]
        candidate: PathBuf,
        #[arg(long)]
        repaired: PathBuf,
        #[arg(long)]
        assets: PathBuf,
        #[arg(long, required = true)]
        prompt: Vec<String>,
        #[arg(long, default_value_t = 8)]
        max_new_tokens: usize,
        #[arg(long, default_value = "bf16", value_parser = ["bf16", "f32"])]
        dtype: String,
        /// Record intermediate activation traces for each comparison stage.
        #[arg(long)]
        trace: bool,
        #[arg(long)]
        output: PathBuf,
    },
    Serve {
        #[arg(long, default_value = "runs")]
        root: PathBuf,
        #[arg(long, default_value_t = 8765)]
        port: u16,
        #[arg(long)]
        checkpoint: Vec<PathBuf>,
        /// Allow explicit parameter experiments in fresh checkpoint copies under the report root.
        #[arg(long)]
        allow_edits: bool,
    },
    Mcp,
}

struct Checkpoint {
    files: Vec<PathBuf>,
    maps: Vec<Mmap>,
}
impl Checkpoint {
    fn reader(&self, tensor: &safetensors::tensor::TensorView<'_>) -> Result<File> {
        let pointer = tensor.data().as_ptr() as usize;
        for (path, map) in self.files.iter().zip(&self.maps) {
            let base = map.as_ptr() as usize;
            if pointer >= base && pointer + tensor.data().len() <= base + map.len() {
                let mut file = File::open(path)?;
                file.seek(SeekFrom::Start((pointer - base) as u64))?;
                return Ok(file);
            }
        }
        bail!("Tensor storage unavailable")
    }
    fn open(path: &Path) -> Result<Self> {
        let mut files = if path.is_dir() {
            fs::read_dir(path)?
                .map(|e| Ok(e?.path()))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
                .collect()
        } else {
            vec![path.to_path_buf()]
        };
        files.sort();
        ensure!(!files.is_empty(), "No safetensors files found");
        let mut maps = Vec::new();
        let mut names = std::collections::HashSet::new();
        for file in &files {
            ensure!(
                file.extension().is_some_and(|e| e == "safetensors"),
                "Only safetensors files are supported"
            );
            let input = File::open(file)?;
            // Files must remain immutable during assessment. Mapping is read-only.
            let map = unsafe { Mmap::map(&input)? };
            let tensors =
                SafeTensors::deserialize(&map).context("Invalid safetensors structure")?;
            for name in tensors.names() {
                ensure!(
                    names.insert(name.to_string()),
                    "Duplicate tensor across shards: {name}"
                );
            }
            maps.push(map);
        }
        Ok(Self { files, maps })
    }
    fn tensors(&self) -> Result<BTreeMap<String, safetensors::tensor::TensorView<'_>>> {
        let mut result = BTreeMap::new();
        for map in &self.maps {
            for (name, view) in SafeTensors::deserialize(map)?.tensors() {
                result.insert(name, view);
            }
        }
        Ok(result)
    }
}
fn file_hash(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut hasher = Sha256::new();
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}
fn number(bytes: &[u8], dtype: Dtype) -> Result<f64> {
    Ok(match dtype {
        Dtype::F64 => f64::from_le_bytes(bytes.try_into()?),
        Dtype::F32 => f32::from_le_bytes(bytes.try_into()?) as f64,
        Dtype::F16 => f16::from_bits(u16::from_le_bytes(bytes.try_into()?)).to_f64(),
        Dtype::BF16 => bf16::from_bits(u16::from_le_bytes(bytes.try_into()?)).to_f64(),
        Dtype::I64 => i64::from_le_bytes(bytes.try_into()?) as f64,
        Dtype::I32 => i32::from_le_bytes(bytes.try_into()?) as f64,
        Dtype::I16 => i16::from_le_bytes(bytes.try_into()?) as f64,
        Dtype::I8 => bytes[0] as i8 as f64,
        Dtype::U64 => u64::from_le_bytes(bytes.try_into()?) as f64,
        Dtype::U32 => u32::from_le_bytes(bytes.try_into()?) as f64,
        Dtype::U16 => u16::from_le_bytes(bytes.try_into()?) as f64,
        Dtype::U8 | Dtype::BOOL => bytes[0] as f64,
        other => bail!("Unsupported numerical dtype {other:?}"),
    })
}
fn scan(path: &Path, reference: Option<&Path>) -> Result<Value> {
    let cp = Checkpoint::open(path)?;
    let baseline = reference.map(Checkpoint::open).transpose()?;
    let tensors = cp.tensors()?;
    let refs = baseline
        .as_ref()
        .map(Checkpoint::tensors)
        .transpose()?
        .unwrap_or_default();
    let mut rows = Vec::new();
    let mut findings = Vec::new();
    let mut parameters = 0usize;
    for (name, tensor) in &tensors {
        let dtype = tensor.dtype();
        ensure!(dtype.bitsize() % 8 == 0, "Unsupported packed dtype");
        let width = dtype.bitsize() / 8;
        ensure!(width > 0, "Unsupported packed dtype");
        let compatible = refs
            .get(name)
            .filter(|r| r.shape() == tensor.shape() && r.dtype() == dtype);
        let (mut finite, mut nonfinite, mut zeros, mut changed) = (0usize, 0usize, 0usize, 0usize);
        let (mut sum, mut squares, mut maximum, mut delta_square) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let mut indices = Vec::new();
        let mut reader = cp.reader(tensor)?;
        let mut ref_reader = compatible
            .map(|t| baseline.as_ref().unwrap().reader(t))
            .transpose()?;
        let mut buffer = vec![0u8; 1024 * 1024];
        let mut ref_buffer = vec![0u8; buffer.len()];
        let mut byte_offset = 0;
        let mut tensor_hasher = Sha256::new();
        while byte_offset < tensor.data().len() {
            let length = buffer.len().min(tensor.data().len() - byte_offset);
            reader.read_exact(&mut buffer[..length])?;
            tensor_hasher.update(&buffer[..length]);
            if let Some(reader) = &mut ref_reader {
                reader.read_exact(&mut ref_buffer[..length])?;
            }
            for (local_index, bytes) in buffer[..length].chunks_exact(width).enumerate() {
                let index = byte_offset / width + local_index;
                let value = number(bytes, dtype)?;
                if value.is_finite() {
                    finite += 1;
                    zeros += usize::from(value == 0.0);
                    sum += value;
                    squares += value * value;
                    maximum = maximum.max(value.abs());
                } else {
                    nonfinite += 1;
                }
                if compatible.is_some() {
                    let other_bytes = &ref_buffer[local_index * width..(local_index + 1) * width];
                    if bytes != other_bytes {
                        changed += 1;
                        if indices.len() < 8 {
                            indices.push(index);
                        }
                    }
                    let delta = value - number(other_bytes, dtype)?;
                    if delta.is_finite() {
                        delta_square += delta * delta;
                    }
                }
            }
            byte_offset += length;
        }
        let count = tensor.data().len() / width;
        parameters += count;
        rows.push(json!({"name":name,"shape":tensor.shape(),"dtype":format!("{dtype:?}"),"parameters":count,
            "bytes":tensor.data().len(),"sha256":format!("{:x}",tensor_hasher.finalize()),"finite":finite,"nonfinite":nonfinite,"zeros":zeros,
            "mean":if finite>0 {Some(sum/finite as f64)} else {None},"rms":if finite>0 {Some((squares/finite as f64).sqrt())}else{None},
            "max_abs":maximum,"changed_elements":compatible.map(|_|changed),"first_changed_indices":indices,
            "finite_delta_l2":compatible.map(|_|delta_square.sqrt())}));
        if nonfinite > 0 {
            findings.push(json!({"severity":"high","rule":"NONFINITE","tensor":name,"evidence":format!("{nonfinite} NaN or infinity values"),"confidence":"measured"}));
        }
        if count > 0 && zeros == count {
            findings.push(json!({"severity":"info","rule":"ALL_ZERO","tensor":name,"evidence":"All elements zero. Some bias tensors are intentionally zero.","confidence":"measured"}));
        }
        if compatible.is_some() && changed > 0 {
            findings.push(json!({"severity":"review","rule":"REFERENCE_DRIFT","tensor":name,"evidence":format!("{changed} elements differ from supplied reference"),"confidence":"measured"}));
        }
        if reference.is_some() && compatible.is_none() {
            findings.push(json!({"severity":"review","rule":"SCHEMA_DRIFT","tensor":name,"evidence":"Reference tensor missing or shape/dtype differs","confidence":"measured"}));
        }
    }
    for name in refs.keys().filter(|n| !tensors.contains_key(*n)) {
        findings.push(json!({"severity":"review","rule":"MISSING_TENSOR","tensor":name,"evidence":"Reference tensor absent from candidate"}));
    }
    let files: Vec<_> = cp.files.iter().map(|p| Ok(json!({"name":p.file_name().unwrap().to_string_lossy(),"bytes":p.metadata()?.len(),"sha256":file_hash(p)?}))).collect::<Result<_>>()?;
    Ok(
        json!({"schema_version":1,"checkpoint":path,"reference":reference,"parameters":parameters,"tensors":rows,"findings":findings,"files":files,
        "coverage":"All supported tensor elements scanned","interpretation":"Numerical health and reference differences require behavioral validation. Reference trust is supplied by the operator."}),
    )
}
fn write_json(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
fn checkpoint_json_asset(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            [
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "added_tokens.json",
                "special_tokens_map.json",
                "preprocessor_config.json",
                "processor_config.json",
                "image_processor_config.json",
                "feature_extractor_config.json",
                "adapter_config.json",
                "quantization_config.json",
                "model.safetensors.index.json",
            ]
            .contains(&name)
        })
}
fn copy_checkpoint_json(path: &Path, output: &Path) -> Result<()> {
    let root = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or(Path::new("."))
    };
    for entry in fs::read_dir(root)? {
        let file = entry?.path();
        if file.is_file() && checkpoint_json_asset(&file) {
            fs::copy(&file, output.join(file.file_name().unwrap()))?;
        }
    }
    Ok(())
}
fn repair(path: &Path, reference: &Path, output: &Path, names: &[String]) -> Result<Value> {
    ensure!(!output.exists(), "Output must be a new directory");
    ensure!(!names.is_empty(), "Select at least one tensor");
    let cp = Checkpoint::open(path)?;
    let baseline = Checkpoint::open(reference)?;
    let tensors = cp.tensors()?;
    let refs = baseline.tensors()?;
    for name in names {
        let a = tensors.get(name).context("Unknown candidate tensor")?;
        let b = refs.get(name).context("Unknown reference tensor")?;
        ensure!(
            a.shape() == b.shape() && a.dtype() == b.dtype(),
            "Incompatible reference tensor"
        );
    }
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let staging = tempfile::Builder::new()
        .prefix(".llmium-")
        .tempdir_in(parent)?;
    for file in &cp.files {
        fs::copy(file, staging.path().join(file.file_name().unwrap()))?;
    }
    for (file, map) in cp.files.iter().zip(&cp.maps) {
        let source_tensors = SafeTensors::deserialize(map)?;
        for name in names {
            if let Ok(view) = source_tensors.tensor(name) {
                let offset = view.data().as_ptr() as usize - map.as_ptr() as usize;
                let mut target = fs::OpenOptions::new()
                    .write(true)
                    .open(staging.path().join(file.file_name().unwrap()))?;
                target.seek(SeekFrom::Start(offset as u64))?;
                target.write_all(refs[name].data())?;
                target.sync_all()?;
            }
        }
    }
    copy_checkpoint_json(path, staging.path())?;
    let verified = scan(staging.path(), Some(reference))?;
    for name in names {
        ensure!(
            verified["tensors"]
                .as_array()
                .unwrap()
                .iter()
                .any(|r| r["name"] == *name && r["changed_elements"] == 0),
            "Repair verification failed"
        );
    }
    let result = json!({"output":output,"restored":names,"verified":true,"reference":reference,"reference_files":scan(reference,None)?["files"]});
    write_json(&staging.path().join("llmium-repair.json"), &result)?;
    fs::rename(staging.path(), output)?;
    Ok(result)
}
fn fixture(path: &Path, fault: bool) -> Result<()> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for layer in 0..4 {
        for part in ["self_attn.q_proj", "self_attn.v_proj", "mlp.down_proj"] {
            let start = data.len();
            for index in 0..4096 {
                let value = if fault && layer == 2 && part == "mlp.down_proj" && index == 459 {
                    f32::NAN
                } else {
                    ((index * 17 + layer * 23) as f32).sin() * 0.03
                };
                data.extend_from_slice(&value.to_le_bytes());
            }
            header.insert(
                format!("model.layers.{layer}.{part}.weight"),
                json!({"dtype":"F32","shape":[64,64],"data_offsets":[start,data.len()]}),
            );
        }
    }
    let mut bytes = serde_json::to_vec(&header)?;
    while bytes.len() % 8 != 0 {
        bytes.push(b' ');
    }
    let mut file = File::create(path)?;
    file.write_all(&(bytes.len() as u64).to_le_bytes())?;
    file.write_all(&bytes)?;
    file.write_all(&data)?;
    Ok(())
}
fn forward(path: &Path, target: &str) -> Result<Vec<f32>> {
    let cp = Checkpoint::open(path)?;
    let tensors = cp.tensors()?;
    let tensor = tensors.get(target).context("Unknown forward tensor")?;
    ensure!(
        tensor.shape().len() == 2,
        "Forward experiment requires a matrix"
    );
    let rows = tensor.shape()[0];
    let cols = tensor.shape()[1];
    ensure!(
        rows.checked_mul(cols).is_some_and(|n| n <= 20_000_000),
        "Matrix exceeds experiment limit"
    );
    let weights = tensor
        .data()
        .chunks_exact(tensor.dtype().bitsize() / 8)
        .map(|b| number(b, tensor.dtype()).map(|v| v as f32))
        .collect::<Result<Vec<_>>>()?;
    let mut output = Vec::new();
    for sample in 0..8 {
        for row in 0..rows {
            let mut sum = 0.0;
            for col in 0..cols {
                sum += ((sample * cols + col) as f32).cos() * weights[row * cols + col];
            }
            output.push(sum);
        }
    }
    Ok(output)
}
fn demo(output: &Path) -> Result<Value> {
    fs::create_dir_all(output)?;
    ensure!(
        fs::read_dir(output)?.next().is_none(),
        "Demo requires an empty directory"
    );
    let reference = output.join("reference.safetensors");
    let candidate = output.join("candidate.safetensors");
    let restored = output.join("repaired");
    fixture(&reference, false)?;
    fixture(&candidate, true)?;
    let target = "model.layers.2.mlp.down_proj.weight";
    let before = scan(&candidate, Some(&reference))?;
    repair(&candidate, &reference, &restored, &[target.to_string()])?;
    let after = scan(&restored, Some(&reference))?;
    let mut evidence = serde_json::Map::new();
    let mut outputs = Vec::new();
    for (label, path) in [
        ("reference", &reference),
        ("candidate", &candidate),
        ("repaired", &restored),
    ] {
        let values = forward(path, target)?;
        evidence.insert(label.into(),json!({"nonfinite_outputs":values.iter().filter(|v| !v.is_finite()).count(),"output_shape":[8,64]}));
        outputs.push(values);
    }
    let result = json!({"experiment":"Seeded numerical fault in a synthetic linear module","synthetic":true,"target":target,"flat_index":459,
        "before":before,"after":after,"forward_pass":evidence,"exact_forward_recovery":outputs[0]==outputs[2],"limits":"Synthetic module experiment. Full language-model generation requires separate evaluation."});
    write_json(&output.join("demo.json"), &result)?;
    Ok(result)
}
fn experiment(reference: &Path, target: &str, output: &Path) -> Result<Value> {
    let cp = Checkpoint::open(reference)?;
    let tensors = cp.tensors()?;
    let tensor = tensors.get(target).context("Unknown experiment tensor")?;
    ensure!(
        tensor.shape().len() == 2 && !tensor.data().is_empty(),
        "Select a nonempty matrix"
    );
    let fault = match tensor.dtype() {
        Dtype::F32 => f32::NAN.to_le_bytes().to_vec(),
        Dtype::F16 => 0x7e00u16.to_le_bytes().to_vec(),
        Dtype::BF16 => 0x7fc0u16.to_le_bytes().to_vec(),
        _ => bail!("Experiment supports F32, F16 and BF16"),
    };
    fs::create_dir_all(output)?;
    ensure!(
        fs::read_dir(output)?.next().is_none(),
        "Use an empty experiment directory"
    );
    let candidate = output.join("candidate");
    fs::create_dir(&candidate)?;
    for file in &cp.files {
        fs::copy(file, candidate.join(file.file_name().unwrap()))?;
    }
    copy_checkpoint_json(reference, &candidate)?;
    for (file, map) in cp.files.iter().zip(&cp.maps) {
        if let Ok(view) = SafeTensors::deserialize(map)?.tensor(target) {
            let offset = view.data().as_ptr() as usize - map.as_ptr() as usize;
            let mut file = fs::OpenOptions::new()
                .write(true)
                .open(candidate.join(file.file_name().unwrap()))?;
            file.seek(SeekFrom::Start(offset as u64))?;
            file.write_all(&fault)?;
        }
    }
    eprintln!("Assessing candidate against reference");
    let before = scan(&candidate, Some(reference))?;
    write_json(&output.join("before.json"), &before)?;
    let restored = output.join("repaired");
    eprintln!("Restoring selected tensor and verifying");
    repair(&candidate, reference, &restored, &[target.to_string()])?;
    let after = scan(&restored, Some(reference))?;
    let mut evidence = serde_json::Map::new();
    let mut outputs = Vec::new();
    for (label, path) in [
        ("reference", reference),
        ("candidate", candidate.as_path()),
        ("repaired", restored.as_path()),
    ] {
        let values = forward(path, target)?;
        evidence.insert(label.into(),json!({"nonfinite_outputs":values.iter().filter(|v|!v.is_finite()).count(),"output_shape":[8,tensor.shape()[0]]}));
        outputs.push(values);
    }
    let result = json!({"experiment":"Seeded numerical fault in a real checkpoint matrix","synthetic":false,"seeded_fault":true,
        "target":target,"flat_index":0,"before":before,"after":after,"forward_pass":evidence,"exact_forward_recovery":outputs[0]==outputs[2],
        "limits":"Actual checkpoint matrix computation. Full model generation requires a separate runtime experiment."});
    write_json(&output.join("demo.json"), &result)?;
    Ok(result)
}
fn validate_viewer_origin(host: Option<&str>, origin: Option<&str>, port: u16) -> Result<()> {
    let host = host
        .context("Missing viewer Host header")?
        .to_ascii_lowercase();
    let local = [format!("127.0.0.1:{port}"), format!("localhost:{port}")];
    ensure!(
        local.contains(&host)
            || (port == 80 && ["127.0.0.1", "localhost"].contains(&host.as_str())),
        "Viewer requires a loopback Host header"
    );
    if let Some(origin) = origin {
        ensure!(
            origin.eq_ignore_ascii_case(&format!("http://{host}")),
            "Cross-origin viewer request rejected"
        );
    }
    Ok(())
}

fn serve(root: &Path, port: u16, checkpoints: &[PathBuf], allow_edits: bool) -> Result<()> {
    let root = fs::canonicalize(root)?;
    let checkpoints = checkpoints
        .iter()
        .map(fs::canonicalize)
        .collect::<io::Result<Vec<_>>>()?;
    for checkpoint in &checkpoints {
        Checkpoint::open(checkpoint)?;
    }
    let lab = lab::Lab::new(root.clone(), checkpoints, allow_edits);
    let server =
        tiny_http::Server::http(("127.0.0.1", port)).map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("LLMIUM viewer http://127.0.0.1:{port}");
    for mut request in server.incoming_requests() {
        let response = (|| -> Result<(Vec<u8>, &str)> {
            let header = |name| {
                request
                    .headers()
                    .iter()
                    .find(|header| header.field.equiv(name))
                    .map(|header| header.value.as_str())
            };
            validate_viewer_origin(header("Host"), header("Origin"), port)?;
            let url = url::Url::parse(&format!("http://localhost{}", request.url()))?;
            if request.method() == &tiny_http::Method::Post {
                ensure!(url.path() == "/api/experiments", "Unknown write endpoint");
                ensure!(
                    header("Origin").is_some(),
                    "Experiment requests require a same-origin Origin header"
                );
                ensure!(
                    header("Content-Type")
                        .is_some_and(|value| value.split(';').next() == Some("application/json")),
                    "Experiment requests require application/json"
                );
                let mut bytes = Vec::new();
                request.as_reader().take(16385).read_to_end(&mut bytes)?;
                ensure!(bytes.len() <= 16384, "Experiment request exceeds 16 KiB");
                return Ok((
                    serde_json::to_vec(&lab.start(&serde_json::from_slice(&bytes)?)?)?,
                    "application/json",
                ));
            }
            ensure!(
                request.method() == &tiny_http::Method::Get,
                "Use GET for viewer reads"
            );
            match url.path() {
                "/" => Ok((
                    include_bytes!("../web/index.html").to_vec(),
                    "text/html; charset=utf-8",
                )),
                "/atlas.js" => Ok((
                    include_bytes!("../web/atlas.js").to_vec(),
                    "text/javascript; charset=utf-8",
                )),
                "/trace.js" => Ok((
                    include_bytes!("../web/trace.js").to_vec(),
                    "text/javascript; charset=utf-8",
                )),
                "/workbench.css" => Ok((
                    include_bytes!("../web/workbench.css").to_vec(),
                    "text/css; charset=utf-8",
                )),
                "/workbench.js" => Ok((
                    include_bytes!("../web/workbench.js").to_vec(),
                    "text/javascript; charset=utf-8",
                )),
                "/fonts/Manrope.ttf" => Ok((
                    include_bytes!("../web/fonts/Manrope.ttf").to_vec(),
                    "font/ttf",
                )),
                "/fonts/IBMPlexMono-Regular.ttf" => Ok((
                    include_bytes!("../web/fonts/IBMPlexMono-Regular.ttf").to_vec(),
                    "font/ttf",
                )),
                "/documents/demo-guide.pdf" | "/documents/project-overview.pdf" => {
                    let name = url.path().rsplit('/').next().unwrap();
                    Ok((
                        fs::read(
                            root.parent()
                                .context("Report root has no parent")?
                                .join("output/pdf")
                                .join(name),
                        )
                        .context("PDF is not generated yet")?,
                        "application/pdf",
                    ))
                }
                "/api/jobs" => Ok((
                    serde_json::to_vec(
                        &lab.status(
                            url.query_pairs()
                                .find(|(key, _)| key == "id")
                                .as_ref()
                                .map(|(_, value)| value.as_ref()),
                        )?,
                    )?,
                    "application/json",
                )),
                "/api/checkpoints" | "/api/checkpoint" | "/api/tensor-tile" => Ok((
                    serde_json::to_vec(&atlas::http_request(
                        &url,
                        &lab.checkpoints.lock().unwrap(),
                    )?)?,
                    "application/json",
                )),
                "/api/reports" => {
                    let mut files = Vec::new();
                    list_reports(&root, &root, &mut files)?;
                    Ok((serde_json::to_vec(&files)?, "application/json"))
                }
                "/api/report" => {
                    let relative = url
                        .query_pairs()
                        .find(|(k, _)| k == "path")
                        .context("Missing path")?
                        .1
                        .into_owned();
                    let path = fs::canonicalize(root.join(relative))?;
                    ensure!(
                        path.starts_with(&root) && path.extension().is_some_and(|e| e == "json"),
                        "Invalid report path"
                    );
                    Ok((fs::read(path)?, "application/json"))
                }
                _ => bail!("Unknown endpoint"),
            }
        })();
        let response = match response {
            Ok((body, kind)) => tiny_http::Response::from_data(body)
                .with_header(tiny_http::Header::from_bytes("Content-Type", kind).unwrap()),
            Err(e) => tiny_http::Response::from_string(e.to_string()).with_status_code(400),
        };
        let response = response
            .with_header(tiny_http::Header::from_bytes("Cache-Control", "no-store").unwrap())
            .with_header(tiny_http::Header::from_bytes("X-Content-Type-Options", "nosniff").unwrap())
            .with_header(tiny_http::Header::from_bytes("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; object-src 'none'; frame-ancestors 'none'; form-action 'none'; base-uri 'none'").unwrap());
        let _ = request.respond(response);
    }
    Ok(())
}
fn list_reports(root: &Path, dir: &Path, result: &mut Vec<String>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if entry.file_type()?.is_symlink() {
            continue;
        }
        let p = entry.path();
        if p.is_dir() {
            // Older copies can contain inherited reports. Their provenance does not
            // describe the edited checkpoint; experiment reports live above it.
            if !p.join("llmium-seed.json").is_file() && !p.join("llmium-repair.json").is_file() {
                list_reports(root, &p, result)?;
            }
        } else if p.extension().is_some_and(|e| e == "json")
            && !checkpoint_json_asset(&p)
            && !["llmium-repair.json", "llmium-seed.json", "job-state.json"]
                .iter()
                .any(|name| p.file_name().unwrap() == *name)
        {
            result.push(p.strip_prefix(root)?.to_string_lossy().replace('\\', "/"));
        }
    }
    result.sort();
    Ok(())
}
fn tensor_slice(args: &Value) -> Result<Value> {
    let path = Path::new(args["path"].as_str().context("Missing path")?);
    let cp = Checkpoint::open(path)?;
    let tensors = cp.tensors()?;
    let name = args["tensor"].as_str().context("Missing tensor")?;
    let tensor = tensors.get(name).context("Unknown tensor")?;
    let width = tensor.dtype().bitsize() / 8;
    ensure!(width > 0, "Unsupported packed dtype");
    let offset = usize::try_from(
        args.get("offset")
            .map(|v| v.as_u64().context("Offset must be a nonnegative integer"))
            .transpose()?
            .unwrap_or(0),
    )?;
    let count = usize::try_from(
        args.get("count")
            .map(|v| v.as_u64().context("Count must be a nonnegative integer"))
            .transpose()?
            .unwrap_or(64),
    )?;
    ensure!(count <= 4096, "Read at most 4096 elements per call");
    let end = offset.checked_add(count).context("Slice overflow")?;
    ensure!(end <= tensor.data().len() / width, "Slice outside tensor");
    let mut values = Vec::new();
    for bytes in tensor.data()[offset * width..end * width].chunks_exact(width) {
        let value = number(bytes, tensor.dtype())?;
        values.push(if value.is_finite() {
            json!(value)
        } else {
            json!(value.to_string())
        });
    }
    Ok(
        json!({"tensor":name,"shape":tensor.shape(),"dtype":format!("{:?}",tensor.dtype()),"offset":offset,"values":values}),
    )
}
fn mcp_result(request: &Value) -> Result<Value> {
    let params = &request["params"];
    match request["method"].as_str().unwrap_or("") {
        "initialize" => Ok(
            json!({"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"llmium","version":"0.1.0"}}),
        ),
        "ping" => Ok(json!({})),
        "tools/list" => Ok(
            json!({"tools":[{"name":"assess_checkpoint","description":"Full numerical health and optional reference comparison","inputSchema":{"type":"object","properties":{"path":{"type":"string"},"reference":{"type":"string"}},"required":["path"]}},
                {"name":"tensor_slice","description":"Read a bounded flattened slice of actual tensor values","inputSchema":{"type":"object","properties":{"path":{"type":"string"},"tensor":{"type":"string"},"offset":{"type":"integer","minimum":0},"count":{"type":"integer","minimum":0,"maximum":4096}},"required":["path","tensor"]}},
                {"name":"checkpoint_graph","description":"Inspect every stored tensor and validated Qwen2 architecture topology","inputSchema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}},
                {"name":"activation_trace","description":"Read a bounded, filtered page of recorded runtime activation observations without loading a model","inputSchema":{"type":"object","properties":{"path":{"type":"string"},"stage":{"type":"string","enum":["reference","candidate","repaired"]},"evaluation":{"type":"integer","minimum":0},"step":{"type":"integer","minimum":0},"module_prefix":{"type":"string"},"nonfinite_only":{"type":"boolean"},"changed_only":{"type":"boolean"},"offset":{"type":"integer","minimum":0},"count":{"type":"integer","minimum":1,"maximum":128}},"required":["path"]}},
                {"name":"assess_behavior","description":"Measure bounded repetition and token concentration in a saved runtime report; descriptive signals, not semantic or security verdicts","inputSchema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}},
                {"name":"tensor_tile","description":"Read a bounded matrix tile with exact raw bytes and optional reference differences","inputSchema":{"type":"object","properties":{"path":{"type":"string"},"reference":{"type":"string"},"tensor":{"type":"string"},"row":{"type":"integer","minimum":0},"column":{"type":"integer","minimum":0},"rows":{"type":"integer","minimum":1,"maximum":64},"columns":{"type":"integer","minimum":1,"maximum":64}},"required":["path","tensor"]}}]}),
        ),
        "tools/call" => {
            let result = (|| -> Result<Value> {
                let a = &params["arguments"];
                if params["name"] == "assess_behavior" {
                    return behavior::report(&trace_report::load(Path::new(
                        a["path"].as_str().context("Missing path")?,
                    ))?);
                }
                if params["name"] == "activation_trace" {
                    return trace_report::read(a);
                }
                if params["name"] == "checkpoint_graph" {
                    return atlas::graph(Path::new(a["path"].as_str().context("Missing path")?));
                }
                if params["name"] == "tensor_tile" {
                    return atlas::tile(
                        Path::new(a["path"].as_str().context("Missing path")?),
                        a["reference"].as_str().map(Path::new),
                        a,
                    );
                }
                if params["name"] == "tensor_slice" {
                    return tensor_slice(a);
                }
                ensure!(params["name"] == "assess_checkpoint", "Unknown tool");
                scan(
                    Path::new(a["path"].as_str().context("Missing path")?),
                    a["reference"].as_str().map(Path::new),
                )
            })();
            match result {
                Ok(v) => Ok(json!({"content":[{"type":"text","text":serde_json::to_string(&v)?}]})),
                Err(e) => {
                    Ok(json!({"isError":true,"content":[{"type":"text","text":e.to_string()}]}))
                }
            }
        }
        _ => bail!("Unsupported method"),
    }
}
fn main() -> Result<()> {
    match Cli::parse().command {
        Command::SeedScale {
            reference,
            tensor,
            index,
            factor,
            output,
        } => println!(
            "{}",
            perturb::scale(&reference, &tensor, index, factor, &output)?
        ),
        Command::AssessBehavior { path } => {
            println!("{}", behavior::report(&trace_report::load(&path)?)?)
        }
        Command::TraceSlice {
            path,
            stage,
            evaluation,
            step,
            module_prefix,
            nonfinite_only,
            changed_only,
            offset,
            count,
        } => {
            let mut args = json!({"path":path,"evaluation":evaluation,"step":step,"module_prefix":module_prefix,"nonfinite_only":nonfinite_only,"changed_only":changed_only,"offset":offset,"count":count});
            if let Some(stage) = stage {
                args["stage"] = json!(stage);
            }
            println!("{}", trace_report::read(&args)?);
        }
        Command::Graph { path, output } => {
            ensure!(!output.exists(), "Graph output must be a new file");
            let report = atlas::graph(&path)?;
            write_json(&output, &report)?;
            println!(
                "{}",
                json!({"report":output,"tensors":report["tensors"].as_array().unwrap().len(),"topology":report["topology"]["adapter"]})
            );
        }
        Command::TensorTile {
            path,
            reference,
            tensor,
            row,
            column,
            rows,
            columns,
        } => {
            println!(
                "{}",
                atlas::tile(
                    &path,
                    reference.as_deref(),
                    &json!({"tensor":tensor,"row":row,"column":column,"rows":rows,"columns":columns})
                )?
            );
        }
        Command::TensorSlice {
            path,
            tensor,
            offset,
            count,
        } => {
            println!(
                "{}",
                tensor_slice(&json!({"path":path,"tensor":tensor,"offset":offset,"count":count}))?
            );
        }
        Command::RuntimeCompare {
            reference,
            candidate,
            repaired,
            assets,
            prompt,
            max_new_tokens,
            dtype,
            trace,
            output,
        } => {
            ensure!(
                !output.exists(),
                "Runtime comparison output must be a new file"
            );
            #[cfg(feature = "inference")]
            {
                let stages = [
                    ("reference", reference.as_path()),
                    ("candidate", candidate.as_path()),
                    ("repaired", repaired.as_path()),
                ];
                let result =
                    inference::compare(&stages, &assets, &prompt, max_new_tokens, &dtype, trace)?;
                write_json(&output, &result)?;
                println!(
                    "{}",
                    json!({"report":output,"recovery_verified":result["recovery_verified"],"candidate_nonfinite_logits":result["candidate_nonfinite_logits"]})
                );
            }
            #[cfg(not(feature = "inference"))]
            {
                let _ = (
                    reference,
                    candidate,
                    repaired,
                    assets,
                    prompt,
                    max_new_tokens,
                    dtype,
                    trace,
                );
                bail!(
                    "Build full-model evaluation with cargo build --release --features inference"
                );
            }
        }
        Command::Evaluate {
            model,
            assets,
            prompt,
            max_new_tokens,
            dtype,
            trace,
            output,
        } => {
            ensure!(!output.exists(), "Evaluation output must be a new file");
            #[cfg(feature = "inference")]
            {
                let report =
                    inference::evaluate(&model, &assets, &prompt, max_new_tokens, &dtype, trace)?;
                write_json(&output, &report)?;
                println!(
                    "{}",
                    json!({"report":output,"evaluations":report["evaluations"].as_array().unwrap().len()})
                );
            }
            #[cfg(not(feature = "inference"))]
            {
                let _ = (model, assets, prompt, max_new_tokens, dtype, trace);
                bail!(
                    "Build full-model evaluation with cargo build --release --features inference"
                );
            }
        }
        Command::Scan {
            path,
            reference,
            output,
        } => {
            let r = scan(&path, reference.as_deref())?;
            write_json(&output, &r)?;
            println!(
                "{}",
                json!({"report":output,"tensors":r["tensors"].as_array().unwrap().len(),"findings":r["findings"].as_array().unwrap().len()})
            );
        }
        Command::Repair {
            path,
            reference,
            output,
            tensor,
        } => println!("{}", repair(&path, &reference, &output, &tensor)?),
        Command::Experiment {
            reference,
            tensor,
            output,
        } => {
            let result = experiment(&reference, &tensor, &output)?;
            println!(
                "{}",
                json!({"report":output.join("demo.json"),"recovered":result["exact_forward_recovery"]})
            );
        }
        Command::Demo { output } => {
            let r = demo(&output)?;
            println!(
                "{}",
                json!({"report":output.join("demo.json"),"recovered":r["exact_forward_recovery"]})
            );
        }
        Command::Serve {
            root,
            port,
            checkpoint,
            allow_edits,
        } => serve(&root, port, &checkpoint, allow_edits)?,
        Command::Mcp => {
            for line in io::stdin().lock().lines() {
                let line = line?;
                let request: Value = match serde_json::from_str(&line) {
                    Ok(v) => v,
                    Err(e) => {
                        println!(
                            "{}",
                            json!({"jsonrpc":"2.0","id":null,"error":{"code":-32700,"message":e.to_string()}})
                        );
                        continue;
                    }
                };
                if request.get("id").is_none() {
                    continue;
                }
                let response = match mcp_result(&request) {
                    Ok(v) => json!({"jsonrpc":"2.0","id":request["id"],"result":v}),
                    Err(e) => {
                        json!({"jsonrpc":"2.0","id":request["id"],"error":{"code":-32601,"message":e.to_string()}})
                    }
                };
                println!("{response}");
                io::stdout().flush()?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn viewer_rejects_rebinding_hosts_and_cross_origin_requests() {
        assert!(validate_viewer_origin(Some("127.0.0.1:8765"), None, 8765).is_ok());
        assert!(validate_viewer_origin(
            Some("localhost:8765"),
            Some("http://localhost:8765"),
            8765
        )
        .is_ok());
        assert!(validate_viewer_origin(Some("localhost"), None, 80).is_ok());
        assert!(validate_viewer_origin(None, None, 8765).is_err());
        assert!(validate_viewer_origin(Some("rebind.example:8765"), None, 8765).is_err());
        assert!(validate_viewer_origin(Some("localhost:9999"), None, 8765).is_err());
        assert!(validate_viewer_origin(
            Some("localhost:8765"),
            Some("https://attacker.example"),
            8765
        )
        .is_err());
        assert!(validate_viewer_origin(Some("localhost:8765"), Some("null"), 8765).is_err());
    }
    #[test]
    fn fault_localization_and_forward_recovery() {
        let dir = tempfile::tempdir().unwrap();
        let r = demo(dir.path()).unwrap();
        assert_eq!(r["exact_forward_recovery"], true);
        assert_eq!(r["forward_pass"]["candidate"]["nonfinite_outputs"], 8);
        let row = r["before"]["tensors"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["name"] == "model.layers.2.mlp.down_proj.weight")
            .unwrap();
        assert_eq!(row["changed_elements"], 1);
        assert_eq!(row["first_changed_indices"], json!([459]));
        assert_eq!(r["after"]["findings"], json!([]));
    }
    #[test]
    fn reject_corrupt_container() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.safetensors");
        fs::write(&path, b"broken").unwrap();
        assert!(Checkpoint::open(&path).is_err());
    }
    #[test]
    fn repair_rejects_overwrite() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.safetensors");
        fixture(&path, false).unwrap();
        assert!(repair(&path, &path, dir.path(), &["x".into()]).is_err());
    }
    #[test]
    fn exact_integer_decoding() {
        assert_eq!(
            number(&u64::MAX.to_le_bytes(), Dtype::U64).unwrap(),
            u64::MAX as f64
        );
    }

    #[test]
    fn bf16_and_f16_nonfinite_decoding() {
        assert!(number(&0x7fc0u16.to_le_bytes(), Dtype::BF16)
            .unwrap()
            .is_nan());
        assert!(number(&0x7e00u16.to_le_bytes(), Dtype::F16)
            .unwrap()
            .is_nan());
        assert_eq!(number(&0x3f80u16.to_le_bytes(), Dtype::BF16).unwrap(), 1.0);
    }

    #[test]
    fn checkpoint_matrix_experiment_recovers_exactly() {
        let dir = tempfile::tempdir().unwrap();
        let reference = dir.path().join("reference.safetensors");
        fixture(&reference, false).unwrap();
        let config = br#"{"model_type":"fixture","hidden_size":64}"#;
        fs::write(dir.path().join("config.json"), config).unwrap();
        fs::write(
            dir.path().join("demo.json"),
            br#"{"experiment":"old evidence"}"#,
        )
        .unwrap();
        fs::write(dir.path().join("unrelated.json"), br#"{"local_only":true}"#).unwrap();
        let result = experiment(
            &reference,
            "model.layers.0.self_attn.q_proj.weight",
            &dir.path().join("experiment"),
        )
        .unwrap();
        assert_eq!(result["seeded_fault"], true);
        assert_eq!(result["exact_forward_recovery"], true);
        assert_eq!(result["forward_pass"]["candidate"]["nonfinite_outputs"], 8);
        assert_eq!(result["after"]["findings"], json!([]));
        for stage in ["candidate", "repaired"] {
            let copied = dir.path().join("experiment").join(stage);
            assert!(!copied.join("demo.json").exists());
            assert!(!copied.join("unrelated.json").exists());
            assert_eq!(
                fs::read(
                    dir.path()
                        .join("experiment")
                        .join(stage)
                        .join("config.json")
                )
                .unwrap(),
                config
            );
        }
    }

    #[test]
    fn report_picker_excludes_assets_and_inherited_checkpoint_reports() {
        let dir = tempfile::tempdir().unwrap();
        let job = dir.path().join("lab/job");
        fs::create_dir_all(job.join("candidate")).unwrap();
        fs::create_dir_all(job.join("repaired")).unwrap();
        fs::write(job.join("demo.json"), b"{}").unwrap();
        fs::write(job.join("processor_config.json"), b"{}").unwrap();
        fs::write(job.join("candidate/llmium-seed.json"), b"{}").unwrap();
        fs::write(job.join("candidate/demo.json"), b"{}").unwrap();
        fs::write(job.join("repaired/llmium-repair.json"), b"{}").unwrap();
        fs::write(job.join("repaired/demo.json"), b"{}").unwrap();
        let mut reports = Vec::new();
        list_reports(dir.path(), dir.path(), &mut reports).unwrap();
        assert_eq!(reports, vec!["lab/job/demo.json"]);
    }

    #[test]
    fn tensor_reads_validate_bounds_and_integer_arguments() {
        let dir = tempfile::tempdir().unwrap();
        let reference = dir.path().join("reference.safetensors");
        fixture(&reference, false).unwrap();
        let mut args = json!({"path":reference,"tensor":"model.layers.0.self_attn.q_proj.weight","offset":0,"count":3});
        assert_eq!(
            tensor_slice(&args).unwrap()["values"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
        args["offset"] = json!(-1);
        assert!(tensor_slice(&args).is_err());
        args["offset"] = json!(4096);
        assert!(tensor_slice(&args).is_err());
    }
}
