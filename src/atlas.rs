//! Header-derived hierarchy, an explicitly selected Qwen2 topology adapter, and bounded raw tiles.
use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::Read,
    path::{Path, PathBuf},
};

pub fn http_request(url: &url::Url, checkpoints: &[PathBuf]) -> Result<Value> {
    if url.path() == "/api/checkpoints" {
        return Ok(json!(checkpoints
            .iter()
            .enumerate()
            .map(|(index, path)| json!({"id":index,"path":path}))
            .collect::<Vec<_>>()));
    }
    let query = url.query_pairs().collect::<BTreeMap<_, _>>();
    let model = query
        .get("model")
        .context("Missing registered model ID")?
        .parse::<usize>()
        .context("Invalid model ID")?;
    let path = checkpoints.get(model).context("Unregistered checkpoint")?;
    match url.path() {
        "/api/checkpoint" => graph(path),
        "/api/tensor-tile" => {
            let reference = query
                .get("reference")
                .map(|index| -> Result<&Path> {
                    Ok(checkpoints
                        .get(index.parse::<usize>().context("Invalid reference ID")?)
                        .context("Unregistered reference checkpoint")?
                        .as_path())
                })
                .transpose()?;
            let mut args = json!({"tensor":query.get("tensor").context("Missing tensor")?});
            for name in ["row", "column", "rows", "columns"] {
                if let Some(value) = query.get(name) {
                    args[name] = json!(value
                        .parse::<usize>()
                        .with_context(|| format!("Invalid {name}"))?);
                }
            }
            tile(path, reference, &args)
        }
        _ => bail!("Unknown checkpoint endpoint"),
    }
}

fn layer_index(name: &str) -> Option<usize> {
    let parts = name.split('.').collect::<Vec<_>>();
    parts.windows(2).find_map(|pair| {
        ["layers", "blocks", "h"]
            .contains(&pair[0])
            .then(|| pair[1].parse().ok())
            .flatten()
    })
}

pub fn graph(path: &Path) -> Result<Value> {
    let checkpoint = super::Checkpoint::open(path)?;
    let tensors = checkpoint.tensors()?;
    let mut layers: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    let mut inventory = Vec::new();
    let mut parameters = 0usize;
    for (name, tensor) in &tensors {
        let count = tensor.shape().iter().product::<usize>();
        parameters = parameters
            .checked_add(count)
            .context("Parameter count overflow")?;
        let layer = layer_index(name);
        if let Some(index) = layer {
            layers.entry(index).or_default().push(name.clone());
        }
        inventory.push(
            json!({"name":name,"shape":tensor.shape(),"dtype":format!("{:?}",tensor.dtype()),
            "parameters":count,"bytes":tensor.data().len(),"layer":layer}),
        );
    }
    let assets = if path.is_dir() {
        path
    } else {
        path.parent()
            .context("Checkpoint has no parent directory")?
    };
    let config_path = assets.join("config.json");
    let config: Option<Value> = if config_path.exists() {
        Some(serde_json::from_slice(&fs::read(&config_path)?).context("Invalid config.json")?)
    } else {
        None
    };
    let mut topology = Value::Null;
    let mut topology_error = Value::Null;
    if let Some(config) = &config {
        if config["model_type"] == "qwen2" {
            match qwen_topology(config, &tensors) {
                Ok(value) => topology = value,
                Err(error) => topology_error = json!(error.to_string()),
            }
        }
    }
    Ok(
        json!({"schema_version":1,"report_type":"checkpoint_atlas","checkpoint":path,
        "parameters":parameters,"tensors":inventory,
        "layers":layers.into_iter().map(|(index,names)|json!({"index":index,"tensors":names})).collect::<Vec<_>>(),
        "config":config,"config_sha256":if config_path.exists(){Some(super::file_hash(&config_path)?)}else{None},
        "topology":topology,"topology_error":topology_error,
        "interpretation":"The inventory covers every stored tensor. Qwen2 dataflow is derived from a known architecture adapter and validated tensor shapes. It is not an execution trace or a semantic interpretation of individual neurons. Live matrix tiles read bounded actual parameter bytes."}),
    )
}

fn qwen_topology(
    config: &Value,
    tensors: &BTreeMap<String, safetensors::tensor::TensorView<'_>>,
) -> Result<Value> {
    ensure!(
        config["hidden_act"] == "silu",
        "Qwen2 topology adapter requires the SiLU gated MLP"
    );
    let dimension = |name: &str| -> Result<usize> {
        let value = usize::try_from(
            config[name]
                .as_u64()
                .with_context(|| format!("Missing integer {name}"))?,
        )?;
        ensure!(value > 0, "Invalid dimension {name}");
        Ok(value)
    };
    let hidden = dimension("hidden_size")?;
    let intermediate = dimension("intermediate_size")?;
    let heads = dimension("num_attention_heads")?;
    let kv_heads = dimension("num_key_value_heads")?;
    let count = dimension("num_hidden_layers")?;
    let vocab = dimension("vocab_size")?;
    ensure!(
        count <= 4096 && count <= tensors.len(),
        "Layer count exceeds checkpoint inventory"
    );
    ensure!(
        hidden % heads == 0 && heads % kv_heads == 0,
        "Incompatible Qwen attention dimensions"
    );
    let kv_width = (hidden / heads)
        .checked_mul(kv_heads)
        .context("Attention width overflow")?;
    let expect_shape = |name: &str, expected: &[usize]| -> Result<()> {
        let tensor = tensors
            .get(name)
            .with_context(|| format!("Topology requires {name}"))?;
        ensure!(tensor.shape() == expected, "Shape mismatch for {name}");
        Ok(())
    };
    expect_shape("model.embed_tokens.weight", &[vocab, hidden])?;
    expect_shape("model.norm.weight", &[hidden])?;
    let head = if tensors.contains_key("lm_head.weight") {
        expect_shape("lm_head.weight", &[vocab, hidden])?;
        "lm_head.weight"
    } else {
        ensure!(
            config["tie_word_embeddings"] == true,
            "Missing untied language-model head"
        );
        "model.embed_tokens.weight"
    };
    let mut blocks = Vec::new();
    for index in 0..count {
        let prefix = format!("model.layers.{index}");
        for (suffix, shape) in [
            ("input_layernorm.weight", vec![hidden]),
            ("post_attention_layernorm.weight", vec![hidden]),
            ("self_attn.q_proj.weight", vec![hidden, hidden]),
            ("self_attn.k_proj.weight", vec![kv_width, hidden]),
            ("self_attn.v_proj.weight", vec![kv_width, hidden]),
            ("self_attn.o_proj.weight", vec![hidden, hidden]),
            ("mlp.gate_proj.weight", vec![intermediate, hidden]),
            ("mlp.up_proj.weight", vec![intermediate, hidden]),
            ("mlp.down_proj.weight", vec![hidden, intermediate]),
        ] {
            expect_shape(&format!("{prefix}.{suffix}"), &shape)?;
        }
        for (projection, width) in [
            ("q_proj", hidden),
            ("k_proj", kv_width),
            ("v_proj", kv_width),
        ] {
            let bias = format!("{prefix}.self_attn.{projection}.bias");
            if tensors.contains_key(&bias) {
                expect_shape(&bias, &[width])?;
            }
        }
        let mut nodes = Vec::new();
        let mut add = |id: &str, label: String, suffix: Option<&str>, x: usize, y: usize| {
            let tensor = suffix.map(|suffix| format!("{prefix}.{suffix}"));
            let parameters = tensor
                .as_ref()
                .map(|name| {
                    let module = name.rsplit_once('.').unwrap().0;
                    tensors
                        .keys()
                        .filter(|name| {
                            name.rsplit_once('.')
                                .is_some_and(|(parent, _)| parent == module)
                        })
                        .cloned()
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            nodes.push(
                json!({"id":id,"label":label,"tensor":tensor,"parameters":parameters,"x":x,"y":y}),
            );
        };
        add("input", format!("Hidden state · {hidden}"), None, 450, 45);
        add(
            "norm1",
            "Input RMSNorm".into(),
            Some("input_layernorm.weight"),
            450,
            140,
        );
        add(
            "q",
            format!("Q projection + RoPE · {heads} heads"),
            Some("self_attn.q_proj.weight"),
            165,
            260,
        );
        add(
            "k",
            format!("K projection + RoPE · {kv_heads} heads"),
            Some("self_attn.k_proj.weight"),
            450,
            260,
        );
        add(
            "v",
            format!("V projection · {kv_heads} heads"),
            Some("self_attn.v_proj.weight"),
            735,
            260,
        );
        add(
            "attention",
            "Causal attention · softmax(QKᵀ/√d)V".into(),
            None,
            450,
            385,
        );
        add(
            "o",
            "Attention output projection".into(),
            Some("self_attn.o_proj.weight"),
            450,
            485,
        );
        add("add1", "Residual addition".into(), None, 450, 585);
        add(
            "norm2",
            "Post-attention RMSNorm".into(),
            Some("post_attention_layernorm.weight"),
            450,
            685,
        );
        add(
            "gate",
            format!("Gate projection · {intermediate}"),
            Some("mlp.gate_proj.weight"),
            265,
            795,
        );
        add(
            "up",
            format!("Up projection · {intermediate}"),
            Some("mlp.up_proj.weight"),
            635,
            795,
        );
        add("gated", "SiLU(gate) × up".into(), None, 450, 905);
        add(
            "down",
            format!("Down projection · {hidden}"),
            Some("mlp.down_proj.weight"),
            450,
            1005,
        );
        add("add2", "Residual addition".into(), None, 450, 1105);
        add(
            "output",
            format!("Block output · {hidden}"),
            None,
            450,
            1205,
        );
        let edges = [
            ("input", "norm1", false),
            ("norm1", "q", false),
            ("norm1", "k", false),
            ("norm1", "v", false),
            ("q", "attention", false),
            ("k", "attention", false),
            ("v", "attention", false),
            ("attention", "o", false),
            ("o", "add1", false),
            ("input", "add1", true),
            ("add1", "norm2", false),
            ("norm2", "gate", false),
            ("norm2", "up", false),
            ("gate", "gated", false),
            ("up", "gated", false),
            ("gated", "down", false),
            ("down", "add2", false),
            ("add1", "add2", true),
            ("add2", "output", false),
        ]
        .map(|(from, to, residual)| json!({"from":from,"to":to,"residual":residual}));
        blocks.push(json!({"index":index,"nodes":nodes,"edges":edges}));
    }
    Ok(
        json!({"adapter":"qwen2","basis":"Architecture-derived module dataflow with validated weight shapes",
        "input_embedding":"model.embed_tokens.weight","final_norm":"model.norm.weight","output_head":head,
        "tied_output_head":head=="model.embed_tokens.weight","layers":blocks,"hidden_size":hidden,
        "attention_heads":heads,"key_value_heads":kv_heads,"head_dimension":hidden/heads}),
    )
}

fn integer(args: &Value, name: &str, default: usize) -> Result<usize> {
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

pub fn tile(path: &Path, reference: Option<&Path>, args: &Value) -> Result<Value> {
    let name = args["tensor"].as_str().context("Missing tensor")?;
    let row = integer(args, "row", 0)?;
    let column = integer(args, "column", 0)?;
    let rows = integer(args, "rows", 16)?;
    let columns = integer(args, "columns", 16)?;
    ensure!(
        (1..=64).contains(&rows) && (1..=64).contains(&columns),
        "Tile dimensions must be between 1 and 64"
    );
    let checkpoint = super::Checkpoint::open(path)?;
    let tensors = checkpoint.tensors()?;
    let tensor = tensors.get(name).context("Unknown tensor")?;
    let (height, width) = match tensor.shape() {
        [] => (1, 1),
        [width] => (1, *width),
        [height, width] => (*height, *width),
        _ => bail!("Matrix tiles support scalar, vector and rank-two tensors"),
    };
    ensure!(
        row.checked_add(rows).is_some_and(|end| end <= height)
            && column.checked_add(columns).is_some_and(|end| end <= width),
        "Tile outside tensor bounds"
    );
    let baseline = reference.map(super::Checkpoint::open).transpose()?;
    let baseline_tensors = baseline
        .as_ref()
        .map(super::Checkpoint::tensors)
        .transpose()?;
    let reference_tensor = baseline_tensors
        .as_ref()
        .map(|tensors| tensors.get(name).context("Reference tensor missing"))
        .transpose()?;
    if let Some(reference_tensor) = reference_tensor {
        ensure!(
            reference_tensor.shape() == tensor.shape()
                && reference_tensor.dtype() == tensor.dtype(),
            "Reference tensor shape or dtype mismatch"
        );
    }
    let element_width = tensor.dtype().bitsize() / 8;
    ensure!(
        element_width > 0 && tensor.dtype().bitsize() % 8 == 0,
        "Unsupported packed dtype"
    );
    let mut source = checkpoint.reader(tensor)?;
    let mut reference_source = reference_tensor
        .map(|tensor| baseline.as_ref().unwrap().reader(tensor))
        .transpose()?;
    let mut values = Vec::new();
    let mut raw_hex = Vec::new();
    let mut reference_values = Vec::new();
    let mut changed = Vec::new();
    let mut changed_count = 0usize;
    let mut nonfinite = 0usize;
    let mut hasher = Sha256::new();
    let mut previous_flat = 0usize;
    for y in row..row + rows {
        let flat = y
            .checked_mul(width)
            .and_then(|value| value.checked_add(column))
            .context("Tile offset overflow")?;
        let skip = (flat - previous_flat)
            .checked_mul(element_width)
            .context("Tile byte offset overflow")?;
        use std::io::{Seek, SeekFrom};
        source.seek(SeekFrom::Current(i64::try_from(skip)?))?;
        let mut bytes = vec![0u8; columns * element_width];
        source.read_exact(&mut bytes)?;
        hasher.update(&bytes);
        let mut ref_bytes = vec![0u8; bytes.len()];
        if let Some(source) = &mut reference_source {
            source.seek(SeekFrom::Current(i64::try_from(skip)?))?;
            source.read_exact(&mut ref_bytes)?;
        }
        let mut row_values = Vec::new();
        let mut row_hex = Vec::new();
        let mut row_refs = Vec::new();
        let mut row_changed = Vec::new();
        for (index, bytes) in bytes.chunks_exact(element_width).enumerate() {
            let number = super::number(bytes, tensor.dtype())?;
            if !number.is_finite() {
                nonfinite += 1;
            }
            row_values.push(json_number(number));
            row_hex.push(
                bytes
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<String>(),
            );
            if reference_source.is_some() {
                let reference_bytes =
                    &ref_bytes[index * element_width..(index + 1) * element_width];
                row_refs.push(json_number(super::number(reference_bytes, tensor.dtype())?));
                let different = bytes != reference_bytes;
                changed_count += usize::from(different);
                row_changed.push(different);
            }
        }
        values.push(row_values);
        raw_hex.push(row_hex);
        reference_values.push(row_refs);
        changed.push(row_changed);
        previous_flat = flat + columns;
    }
    Ok(
        json!({"tensor":name,"shape":tensor.shape(),"dtype":format!("{:?}",tensor.dtype()),"matrix_shape":[height,width],
        "row":row,"column":column,"rows":rows,"columns":columns,"values":values,"raw_le_hex":raw_hex,
        "reference":reference,"reference_values":if reference.is_some(){Some(reference_values)}else{None},
        "changed":if reference.is_some(){Some(changed)}else{None},"changed_cells":if reference.is_some(){Some(changed_count)}else{None},
        "nonfinite":nonfinite,"tile_sha256":format!("{:x}",hasher.finalize()),
        "coverage":"Bounded row-major tile only. Raw little-endian bytes preserve exact parameter representation; displayed integer values may lose float64 precision."}),
    )
}

fn json_number(value: f64) -> Value {
    if value.is_finite() {
        json!(value)
    } else {
        json!(value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tiles_follow_matrix_strides_and_preserve_fault_bytes() {
        let directory = tempfile::tempdir().unwrap();
        let reference = directory.path().join("reference.safetensors");
        let candidate = directory.path().join("candidate.safetensors");
        super::super::fixture(&reference, false).unwrap();
        super::super::fixture(&candidate, true).unwrap();
        let tile=tile(&candidate,Some(&reference),&json!({"tensor":"model.layers.2.mlp.down_proj.weight","row":7,"column":10,"rows":2,"columns":3})).unwrap();
        assert_eq!(tile["changed_cells"], 1);
        assert_eq!(tile["values"][0][1], "NaN");
        assert_eq!(tile["raw_le_hex"][0][1], "0000c07f");
        let expected=super::super::tensor_slice(&json!({"path":reference,"tensor":"model.layers.2.mlp.down_proj.weight","offset":8*64+10,"count":3})).unwrap();
        assert_eq!(tile["values"][1], expected["values"]);
    }
    #[test]
    fn tiles_reject_invalid_sizes_and_offsets() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("reference.safetensors");
        super::super::fixture(&path, false).unwrap();
        let mut args = json!({"tensor":"model.layers.0.self_attn.q_proj.weight","rows":0});
        assert!(tile(&path, None, &args).is_err());
        args["rows"] = json!(1);
        args["row"] = json!(-1);
        assert!(tile(&path, None, &args).is_err());
        args["row"] = json!(64);
        assert!(tile(&path, None, &args).is_err());
        args["row"] = json!(0);
        args["columns"] = json!(65);
        assert!(tile(&path, None, &args).is_err());
    }
    #[test]
    fn incomplete_topology_is_reported_without_invented_edges() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.safetensors");
        super::super::fixture(&path, false).unwrap();
        fs::write(directory.path().join("config.json"),r#"{"model_type":"qwen2","hidden_act":"silu","hidden_size":64,"intermediate_size":64,"num_attention_heads":4,"num_key_value_heads":2,"num_hidden_layers":4,"vocab_size":32}"#).unwrap();
        let report = graph(directory.path()).unwrap();
        assert_eq!(report["tensors"].as_array().unwrap().len(), 12);
        assert!(report["topology"].is_null());
        assert!(report["topology_error"]
            .as_str()
            .unwrap()
            .contains("embed_tokens"));
    }

    fn tiny_qwen(path: &Path) {
        let mut shapes = BTreeMap::from([
            ("model.embed_tokens.weight", vec![8, 4]),
            ("model.norm.weight", vec![4]),
            ("model.layers.0.input_layernorm.weight", vec![4]),
            ("model.layers.0.post_attention_layernorm.weight", vec![4]),
            ("model.layers.0.self_attn.q_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.k_proj.weight", vec![2, 4]),
            ("model.layers.0.self_attn.v_proj.weight", vec![2, 4]),
            ("model.layers.0.self_attn.o_proj.weight", vec![4, 4]),
            ("model.layers.0.mlp.gate_proj.weight", vec![8, 4]),
            ("model.layers.0.mlp.up_proj.weight", vec![8, 4]),
            ("model.layers.0.mlp.down_proj.weight", vec![4, 8]),
        ]);
        shapes.insert("model.layers.0.self_attn.q_proj.bias", vec![4]);
        let mut header = serde_json::Map::new();
        let mut bytes = Vec::new();
        for (name, shape) in shapes {
            let start = bytes.len();
            bytes.resize(start + shape.iter().product::<usize>() * 4, 0);
            header.insert(
                name.into(),
                json!({"dtype":"F32","shape":shape,"data_offsets":[start,bytes.len()]}),
            );
        }
        let mut serialized = serde_json::to_vec(&header).unwrap();
        while !serialized.len().is_multiple_of(8) {
            serialized.push(b' ');
        }
        let mut file = (serialized.len() as u64).to_le_bytes().to_vec();
        file.extend(serialized);
        file.extend(bytes);
        fs::write(path.join("model.safetensors"), file).unwrap();
        fs::write(path.join("config.json"),r#"{"model_type":"qwen2","hidden_act":"silu","hidden_size":4,"intermediate_size":8,"num_attention_heads":2,"num_key_value_heads":1,"num_hidden_layers":1,"vocab_size":8,"tie_word_embeddings":true}"#).unwrap();
    }

    #[test]
    fn qwen_adapter_checks_head_dimensions_and_tied_output() {
        let directory = tempfile::tempdir().unwrap();
        tiny_qwen(directory.path());
        let report = graph(directory.path()).unwrap();
        assert!(report["topology_error"].is_null());
        assert_eq!(report["topology"]["head_dimension"], 2);
        assert_eq!(
            report["topology"]["output_head"],
            "model.embed_tokens.weight"
        );
        let nodes = report["topology"]["layers"][0]["nodes"].as_array().unwrap();
        let query = nodes.iter().find(|node| node["id"] == "q").unwrap();
        assert_eq!(query["parameters"].as_array().unwrap().len(), 2);
        let mut config: Value =
            serde_json::from_slice(&fs::read(directory.path().join("config.json")).unwrap())
                .unwrap();
        config["num_key_value_heads"] = json!(2);
        fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let invalid = graph(directory.path()).unwrap();
        assert!(invalid["topology"].is_null());
        assert!(invalid["topology_error"]
            .as_str()
            .unwrap()
            .contains("Shape mismatch"));
    }

    #[test]
    fn tile_api_only_resolves_registered_numeric_models_and_compatible_references() {
        let directory = tempfile::tempdir().unwrap();
        tiny_qwen(directory.path());
        let models = vec![directory.path().to_path_buf()];
        let valid=url::Url::parse("http://localhost/api/tensor-tile?model=0&tensor=model.layers.0.self_attn.q_proj.weight&rows=2&columns=2&reference=0").unwrap();
        assert_eq!(http_request(&valid, &models).unwrap()["changed_cells"], 0);
        for query in [
            "model=1",
            "model=/etc/passwd",
            "path=/etc/passwd",
            "model=0&reference=4",
            "model=0&rows=1.5",
        ] {
            let url=url::Url::parse(&format!("http://localhost/api/tensor-tile?tensor=model.layers.0.self_attn.q_proj.weight&{query}")).unwrap();
            assert!(http_request(&url, &models).is_err());
        }
        let other = tempfile::tempdir().unwrap();
        super::super::fixture(&other.path().join("model.safetensors"), false).unwrap();
        assert!(tile(
            directory.path(),
            Some(other.path()),
            &json!({"tensor":"model.layers.0.self_attn.q_proj.weight","rows":1,"columns":1})
        )
        .is_err());
    }
}
