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
    io::{self, BufRead, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

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
    Serve {
        #[arg(long, default_value = "runs")]
        root: PathBuf,
        #[arg(long, default_value_t = 8765)]
        port: u16,
    },
    Mcp,
}

struct Checkpoint {
    files: Vec<PathBuf>,
    maps: Vec<Mmap>,
}
impl Checkpoint {
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
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
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
        for (index, bytes) in tensor.data().chunks_exact(width).enumerate() {
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
            if let Some(other) = compatible {
                let other_bytes = &other.data()[index * width..(index + 1) * width];
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
        let count = tensor.data().len() / width;
        parameters += count;
        rows.push(json!({"name":name,"shape":tensor.shape(),"dtype":format!("{dtype:?}"),"parameters":count,
            "bytes":tensor.data().len(),"sha256":hash(tensor.data()),"finite":finite,"nonfinite":nonfinite,"zeros":zeros,
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
    let files: Vec<_> = cp.files.iter().zip(&cp.maps).map(|(p,m)| json!({"name":p.file_name().unwrap().to_string_lossy(),"bytes":m.len(),"sha256":hash(m)})).collect();
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
    let root = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or(Path::new("."))
    };
    for entry in fs::read_dir(root)? {
        let file = entry?.path();
        if file.is_file() && file.extension().is_some_and(|e| e == "json") {
            fs::copy(&file, staging.path().join(file.file_name().unwrap()))?;
        }
    }
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
fn serve(root: &Path, port: u16) -> Result<()> {
    let root = fs::canonicalize(root)?;
    let server =
        tiny_http::Server::http(("127.0.0.1", port)).map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("LLMIUM viewer http://127.0.0.1:{port}");
    for request in server.incoming_requests() {
        let response = (|| -> Result<(Vec<u8>, &str)> {
            ensure!(
                request.method() == &tiny_http::Method::Get,
                "Read-only viewer"
            );
            let url = url::Url::parse(&format!("http://localhost{}", request.url()))?;
            match url.path() {
                "/" => Ok((
                    include_bytes!("../web/index.html").to_vec(),
                    "text/html; charset=utf-8",
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
            list_reports(root, &p, result)?;
        } else if p.extension().is_some_and(|e| e == "json")
            && p.file_name().unwrap() != "llmium-repair.json"
        {
            result.push(p.strip_prefix(root)?.to_string_lossy().replace('\\', "/"));
        }
    }
    result.sort();
    Ok(())
}
fn mcp_result(request: &Value) -> Result<Value> {
    let params = &request["params"];
    match request["method"].as_str().unwrap_or("") {
        "initialize" => Ok(
            json!({"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"llmium","version":"0.1.0"}}),
        ),
        "ping" => Ok(json!({})),
        "tools/list" => Ok(
            json!({"tools":[{"name":"assess_checkpoint","description":"Full numerical health and optional reference comparison","inputSchema":{"type":"object","properties":{"path":{"type":"string"},"reference":{"type":"string"}},"required":["path"]}}]}),
        ),
        "tools/call" => {
            let result = (|| -> Result<Value> {
                ensure!(params["name"] == "assess_checkpoint", "Unknown tool");
                let a = &params["arguments"];
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
        Command::Serve { root, port } => serve(&root, port)?,
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
}
