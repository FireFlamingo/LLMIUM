//! Instrumented Candle 0.11.1 Qwen2 forward path.
//! Derived from huggingface/candle, commit 59f8e903ca80a32ea7f8514b39748fc979e69e82,
//! candle-transformers/src/models/qwen2.rs, under the MIT license.
//! See third_party/CANDLE-LICENSE-MIT. LLMIUM changes add read-only activation
//! observations without changing forward operations or KV-cache behavior.

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Activation, VarBuilder};
use candle_transformers::models::qwen2::Config;
use candle_transformers::models::with_tracing::{linear, linear_no_bias, Linear, RmsNorm};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::sync::Arc;

#[derive(Default)]
pub struct Trace {
    pub records: Vec<Value>,
}

impl Trace {
    fn observe(&mut self, module: &str, operation: &str, xs: &Tensor) -> Result<()> {
        let values = xs.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let mut hash = Sha256::new();
        let mut nonfinite = 0usize;
        let mut first_nonfinite = Vec::new();
        let mut minimum = f64::INFINITY;
        let mut maximum = f64::NEG_INFINITY;
        let mut sum = 0f64;
        let mut squares = 0f64;
        for (index, &value) in values.iter().enumerate() {
            hash.update(value.to_le_bytes());
            if !value.is_finite() {
                nonfinite += 1;
                if first_nonfinite.len() < 8 {
                    let mut remainder = index;
                    let mut coordinates = vec![0; xs.rank()];
                    for (coordinate, &dimension) in coordinates.iter_mut().zip(xs.dims()).rev() {
                        *coordinate = remainder % dimension;
                        remainder /= dimension;
                    }
                    first_nonfinite.push(json!({"flat_index":index,"coordinates":coordinates,"value":value.to_string()}));
                }
            } else {
                let value = f64::from(value);
                minimum = minimum.min(value);
                maximum = maximum.max(value);
                sum += value;
                squares += value * value;
            }
        }
        let finite = values.len() - nonfinite;
        self.records.push(json!({"sequence":self.records.len(),"module":module,"operation":operation,
            "shape":xs.dims(),"dtype":format!("{:?}",xs.dtype()),"elements":values.len(),"nonfinite":nonfinite,
            "first_nonfinite":first_nonfinite,"f32_sha256":format!("{:x}",hash.finalize()),
            "finite_statistics":if finite>0 {Some(json!({"count":finite,"min":minimum,"max":maximum,"mean":sum/finite as f64,"rms":(squares/finite as f64).sqrt()}))}else{None},
            "first_values":values.iter().take(8).map(|&value|if value.is_finite(){json!(value)}else{json!(value.to_string())}).collect::<Vec<_>>()}));
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_sz, hidden_sz, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }
}

impl MLP {
    fn forward(&self, xs: &Tensor, trace: &mut Trace, prefix: &str) -> Result<Tensor> {
        let gate = xs.apply(&self.gate_proj)?;
        trace.observe(&format!("{prefix}.gate_proj"), "linear", &gate)?;
        let lhs = gate.apply(&self.act_fn)?;
        trace.observe(prefix, "gate_activation", &lhs)?;
        let rhs = xs.apply(&self.up_proj)?;
        trace.observe(&format!("{prefix}.up_proj"), "linear", &rhs)?;
        let gated = (lhs * rhs)?;
        trace.observe(prefix, "gated_product", &gated)?;
        let output = gated.apply(&self.down_proj)?;
        trace.observe(&format!("{prefix}.down_proj"), "linear", &output)?;
        Ok(output)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = hidden_sz / num_heads;
        let q_proj = linear(hidden_sz, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_sz, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_sz, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_sz, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size: hidden_sz,
            rotary_emb,
            kv_cache: None,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        trace: &mut Trace,
        prefix: &str,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs)?;
        trace.observe(&format!("{prefix}.q_proj"), "linear", &query_states)?;
        let key_states = self.k_proj.forward(xs)?;
        trace.observe(&format!("{prefix}.k_proj"), "linear", &key_states)?;
        let value_states = self.v_proj.forward(xs)?;
        trace.observe(&format!("{prefix}.v_proj"), "linear", &value_states)?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) =
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_states, &key_states, seqlen_offset)?;
        trace.observe(
            &format!("{prefix}.q_proj"),
            "rotary_embedding",
            &query_states,
        )?;
        trace.observe(&format!("{prefix}.k_proj"), "rotary_embedding", &key_states)?;

        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));

        let key_states =
            candle_transformers::utils::repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
        let value_states = candle_transformers::utils::repeat_kv(value_states, self.num_kv_groups)?
            .contiguous()?;

        let attn_output = {
            let scale = 1f64 / f64::sqrt(self.head_dim as f64);
            let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;
            trace.observe(prefix, "attention_scores_before_mask", &attn_weights)?;

            let attn_weights = match attention_mask {
                None => attn_weights,
                Some(mask) => attn_weights.broadcast_add(mask)?,
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            // Intentional negative-infinity mask entries are not activation faults.
            // Observe the post-softmax probabilities rather than masked scores.
            trace.observe(prefix, "attention_probabilities", &attn_weights)?;
            attn_weights.matmul(&value_states)?
        };
        trace.observe(prefix, "attention_context", &attn_output)?;
        let output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.hidden_size))?
            .apply(&self.o_proj)?;
        trace.observe(&format!("{prefix}.o_proj"), "linear", &output)?;
        Ok(output)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"))?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        trace: &mut Trace,
        prefix: &str,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        trace.observe(&format!("{prefix}.input_layernorm"), "rms_norm", &xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            seqlen_offset,
            trace,
            &format!("{prefix}.self_attn"),
        )?;
        let xs = (xs + residual)?;
        trace.observe(prefix, "attention_residual", &xs)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?;
        trace.observe(
            &format!("{prefix}.post_attention_layernorm"),
            "rms_norm",
            &xs,
        )?;
        let xs = self.mlp.forward(&xs, trace, &format!("{prefix}.mlp"))?;
        let output = (residual + xs)?;
        trace.observe(prefix, "block_output", &output)?;
        Ok(output)
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    sliding_window: usize,
    device: Device,
    dtype: DType,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            sliding_window: cfg.sliding_window,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn prepare_causal_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        // Sliding window mask?
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + self.sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), self.dtype, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(self.dtype)
    }

    fn prepare_attention_mask(&self, attn_mask: &Tensor) -> Result<Tensor> {
        let (b_sz, sql_len) = attn_mask.dims2()?;
        let mut mask: Vec<Tensor> = vec![];
        for b in 0..b_sz {
            mask.push(attn_mask.i((b, ..))?.expand((1, 1, sql_len, sql_len))?);
        }
        let mask = Tensor::cat(&mask, 0)?;
        let on_true = mask.zeros_like()?.to_dtype(self.dtype)?;
        let on_false = Tensor::new(f32::NEG_INFINITY, &self.device)?
            .broadcast_as(mask.shape())?
            .to_dtype(self.dtype)?;
        mask.where_cond(&on_true, &on_false)
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        attn_mask: Option<&Tensor>,
        trace: &mut Trace,
    ) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask: Option<Tensor> = match attn_mask {
            Some(mask) => Some(self.prepare_attention_mask(mask)?),
            None => {
                if seq_len <= 1 {
                    None
                } else {
                    Some(self.prepare_causal_attention_mask(b_size, seq_len, seqlen_offset)?)
                }
            }
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        trace.observe("model.embed_tokens", "embedding", &xs)?;
        for (index, layer) in self.layers.iter_mut().enumerate() {
            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                seqlen_offset,
                trace,
                &format!("model.layers.{index}"),
            )?
        }
        let output = xs.apply(&self.norm)?;
        trace.observe("model.norm", "rms_norm", &output)?;
        Ok(output)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelForCausalLM {
    base_model: Model,
    lm_head: Linear,
}

impl ModelForCausalLM {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let base_model = Model::new(cfg, vb.clone())?;
        let lm_head = if vb.contains_tensor("lm_head.weight") {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        } else {
            Linear::from_weights(base_model.embed_tokens.embeddings().clone(), None)
        };
        Ok(Self {
            base_model,
            lm_head,
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        trace: &mut Trace,
    ) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;
        let output = self
            .base_model
            .forward(input_ids, seqlen_offset, None, trace)?
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.lm_head)?;
        trace.observe("lm_head", "linear", &output)?;
        Ok(output)
    }

    pub fn clear_kv_cache(&mut self) {
        self.base_model.clear_kv_cache()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn fixture(dtype: DType, fault: bool, tied: bool) -> (Config, VarBuilder<'static>) {
        let config: Config = serde_json::from_value(json!({
            "vocab_size":8,"hidden_size":4,"intermediate_size":8,"num_hidden_layers":2,
            "num_attention_heads":2,"num_key_value_heads":1,"max_position_embeddings":32,
            "sliding_window":32,"max_window_layers":2,"tie_word_embeddings":tied,
            "rope_theta":10000.0,"rms_norm_eps":0.000001,"use_sliding_window":false,"hidden_act":"silu"
        })).unwrap();
        let mut weights = HashMap::new();
        let mut add = |name: String, shape: Vec<usize>| {
            let mut values = (0..shape.iter().product())
                .map(|index| {
                    if name.contains("norm") {
                        1.0
                    } else {
                        ((index + name.len()) as f32).sin() * 0.1
                    }
                })
                .collect::<Vec<_>>();
            if fault && name == "model.layers.0.self_attn.q_proj.weight" {
                values[0] = f32::NAN;
            }
            weights.insert(
                name,
                Tensor::from_vec(values, shape, &Device::Cpu)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap(),
            );
        };
        add("model.embed_tokens.weight".into(), vec![8, 4]);
        add("model.norm.weight".into(), vec![4]);
        if !tied {
            add("lm_head.weight".into(), vec![8, 4]);
        }
        for layer in 0..2 {
            for (suffix, shape) in [
                ("input_layernorm.weight", vec![4]),
                ("post_attention_layernorm.weight", vec![4]),
                ("self_attn.q_proj.weight", vec![4, 4]),
                ("self_attn.k_proj.weight", vec![2, 4]),
                ("self_attn.v_proj.weight", vec![2, 4]),
                ("self_attn.o_proj.weight", vec![4, 4]),
                ("self_attn.q_proj.bias", vec![4]),
                ("self_attn.k_proj.bias", vec![2]),
                ("self_attn.v_proj.bias", vec![2]),
                ("mlp.gate_proj.weight", vec![8, 4]),
                ("mlp.up_proj.weight", vec![8, 4]),
                ("mlp.down_proj.weight", vec![4, 8]),
            ] {
                add(format!("model.layers.{layer}.{suffix}"), shape);
            }
        }
        (
            config,
            VarBuilder::from_tensors(weights, dtype, &Device::Cpu),
        )
    }

    #[test]
    fn traced_forward_matches_upstream_prefill_decode_and_cache_reset() {
        for dtype in [DType::F32, DType::BF16] {
            for tied in [true, false] {
                let (config, builder) = fixture(dtype, false, tied);
                let mut upstream = candle_transformers::models::qwen2::ModelForCausalLM::new(
                    &config,
                    builder.clone(),
                )
                .unwrap();
                let mut traced = ModelForCausalLM::new(&config, builder).unwrap();
                for request in [vec![1u32, 2, 3], vec![4u32, 5], vec![1u32, 2, 3]] {
                    upstream.clear_kv_cache();
                    traced.clear_kv_cache();
                    for (tokens, offset) in [(request.clone(), 0), (vec![6u32], request.len())] {
                        let input = Tensor::new(tokens.as_slice(), &Device::Cpu)
                            .unwrap()
                            .unsqueeze(0)
                            .unwrap();
                        let expected = upstream
                            .forward(&input, offset)
                            .unwrap()
                            .flatten_all()
                            .unwrap()
                            .to_dtype(DType::F32)
                            .unwrap()
                            .to_vec1::<f32>()
                            .unwrap();
                        let mut trace = Trace::default();
                        let actual = traced
                            .forward(&input, offset, &mut trace)
                            .unwrap()
                            .flatten_all()
                            .unwrap()
                            .to_dtype(DType::F32)
                            .unwrap()
                            .to_vec1::<f32>()
                            .unwrap();
                        assert_eq!(actual, expected, "{dtype:?}, tied {tied}, offset {offset}");
                        assert_eq!(trace.records.len(), 39);
                        assert!(trace.records.iter().all(|record| record["nonfinite"] == 0));
                    }
                }
            }
        }
    }

    #[test]
    fn first_fault_is_query_projection_while_parallel_key_and_value_stay_finite() {
        let (config, builder) = fixture(DType::F32, true, true);
        let mut model = ModelForCausalLM::new(&config, builder).unwrap();
        let input = Tensor::new(&[[1u32, 2, 3]], &Device::Cpu).unwrap();
        let mut trace = Trace::default();
        let output = model.forward(&input, 0, &mut trace).unwrap();
        let first = trace
            .records
            .iter()
            .find(|record| record["nonfinite"].as_u64().unwrap() > 0)
            .unwrap();
        assert_eq!(first["module"], "model.layers.0.self_attn.q_proj");
        assert_eq!(first["operation"], "linear");
        assert_eq!(first["nonfinite"], 3);
        assert_eq!(first["first_nonfinite"][1]["coordinates"], json!([0, 1, 0]));
        for suffix in ["k_proj", "v_proj"] {
            assert_eq!(
                trace
                    .records
                    .iter()
                    .find(
                        |record| record["module"] == format!("model.layers.0.self_attn.{suffix}")
                            && record["operation"] == "linear"
                    )
                    .unwrap()["nonfinite"],
                0
            );
        }
        assert!(output
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|value| value.is_nan()));
    }
}
