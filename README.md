# LLMIUM

A Rust workbench for open-weight checkpoint assessment, exact reference comparison, verified tensor restoration, a hierarchical network and weight explorer and a read-only MCP interface.

## Build and run

Rust stable and a C linker are required. On this workstation the toolchain is available in Kali WSL.

```bash
cd /mnt/d/Projects/LLMIUM
~/.cargo/bin/cargo build --release
./target/release/llmium demo --output runs/rust-demo
./target/release/llmium serve --root runs
```

Open http://127.0.0.1:8765 and select `rust-demo/demo.json`. Inspect layer 2 and the affected tensor, then switch to After repair. The Rust demo seeds one NaN in a synthetic module, performs actual linear forward computations and checks exact output recovery.

## Assess and restore

```bash
./target/release/llmium scan models/model --output runs/assessment.json
./target/release/llmium scan models/candidate --reference models/trusted --output runs/comparison.json
./target/release/llmium tensor-slice models/candidate --tensor model.layers.2.mlp.down_proj.weight --offset 0 --count 16
./target/release/llmium repair models/candidate --reference models/trusted --tensor model.layers.2.mlp.down_proj.weight --output models/restored
```

Inputs are a safetensors file or directory of shards. The safetensors crate validates structure. Headers are validated through read-only memory mapping. Every supported element is scanned using bounded buffered reads and streaming hashes. Reports contain per-tensor SHA-256, numerical health statistics, exact changed-element counts and the first changed indices. Integer statistical summaries use float64. Element differences compare raw bytes.

NaNs and infinities receive high numerical-health findings. All-zero tensors receive informational findings because zero biases are common. Reference differences receive review findings. These findings describe measured properties. Security vulnerability classification, CWE mapping and CVSS scoring require demonstrated security impact.

Repair requires a new output directory and a compatible trusted reference. It restores selected tensors, rescans and publishes the new directory after verification. JSON configuration and a reference hash manifest are copied. Copy any additional tokenizer or deployment assets required by your runtime. Keep files immutable during scans and repair. Quantized packed formats, GGUF and unsupported float formats need future adapters.

## Agent interface

Run `llmium mcp` as a stdio MCP server. It exposes `assess_checkpoint` with a required `path` and optional `reference`, plus `tensor_slice` for bounded reads of actual tensor values. The slice tool requires `path` and `tensor`, and accepts `offset` and `count` up to 4096 elements. `checkpoint_graph` returns the complete tensor inventory and validated Qwen2 module topology. `tensor_tile` accepts a matrix `row`, `column`, `rows` and `columns`, with each tile dimension between 1 and 64, and an optional `reference` checkpoint. It includes exact raw bytes and changed-cell flags. `activation_trace` reads filtered, paged observations from saved runtime reports without loading weights. It accepts `path`, an optional comparison `stage` (default candidate), `evaluation`, `step`, `module_prefix`, `nonfinite_only`, `offset` and `count` (1–128, default 32). Report files are limited to 64 MiB. The server uses newline-delimited JSON-RPC and MCP protocol version 2024-11-05. Paths use the server's filesystem. An agent host can launch the WSL binary with a configuration like this.

```json
{
  "mcpServers": {
    "llmium": {
      "command": "wsl.exe",
      "args": ["-d", "kali-linux", "--", "/mnt/d/Projects/LLMIUM/target/release/llmium", "mcp"]
    }
  }
}
```

## Explore the network and actual weights

Register the checkpoints that the read-only viewer may inspect. IDs follow registration order. Existing report viewing works without registered checkpoints.

```bash
./target/release/llmium serve --root runs --checkpoint models/Qwen2.5-1.5B-Instruct --checkpoint runs/qwen-rust-buffered/candidate --checkpoint runs/qwen-rust-buffered/repaired
```

Choose registered checkpoint 1 and click Inspect checkpoint. The Qwen2 adapter validates tensor shapes and shows all decoder blocks, attention projections, residual paths, gated MLP modules, normalization, embedding and the tied output head. Select a module to inspect its parameters. Set Compare to checkpoint 0 to highlight exact byte changes. In layer 0's query projection, row 0 and column 0 show the seeded NaN. Select a cell to inspect its flat index and raw little-endian bytes.

Matrix heatmaps and connection diagrams read actual bounded tiles. Row, column, next-page and flat-index controls address other weights throughout the checkpoint. Vector and scalar tensors are also readable. Higher-rank tensors remain in the inventory and can be inspected with flattened `tensor-slice` reads. Keep checkpoint files immutable while browsing.

The native commands provide the same data.

```bash
./target/release/llmium graph models/Qwen2.5-1.5B-Instruct --output runs/qwen-atlas.json
./target/release/llmium tensor-tile runs/qwen-rust-buffered/candidate --reference models/Qwen2.5-1.5B-Instruct --tensor model.layers.0.self_attn.q_proj.weight --row 0 --column 0 --rows 16 --columns 16
```

Exported atlas JSON works as an offline metadata report. Actual tile reads require a registered live checkpoint. Qwen2 topology is architecture-derived and shape-validated. It represents module dataflow. Connection diagrams represent selected parameter coordinates; they do not establish the semantic role or causal activation of individual neurons. Other model families receive a complete tensor inventory without an invented topology.

## Full model runtime evaluation

The optional Candle adapter runs Qwen2 and Qwen2.5 checkpoints locally on CPU. It uses the checkpoint files and tokenizer assets already on disk. Build it with Rust.

```bash
~/.cargo/bin/cargo build --release --features inference
RAYON_NUM_THREADS=2 ./target/release/llmium evaluate models/Qwen2.5-1.5B-Instruct --assets models/Qwen2.5-1.5B-Instruct --prompt 'The capital of France is' --max-new-tokens 4 --output runs/runtime.json
```

The controlled workflow loads the reference, candidate and repaired checkpoints sequentially. It clears the KV cache between prompts, uses deterministic greedy decoding, records token IDs and hashes every output logit vector. Nonfinite logits stop decoding and are recorded as a numerical failure. BF16 is the default. F32 requires substantially more memory.

```bash
RAYON_NUM_THREADS=2 ./target/release/llmium runtime-compare --reference models/Qwen2.5-1.5B-Instruct --candidate runs/qwen-rust-buffered/candidate --repaired runs/qwen-rust-buffered/repaired --assets models/Qwen2.5-1.5B-Instruct --prompt 'The capital of France is' --prompt 'Two plus two equals' --prompt 'The capital of France is' --max-new-tokens 4 --output runs/qwen-runtime/comparison.json
```

Use a new output filename for each run. The report binds observations to checkpoint, configuration and tokenizer hashes. Verified recovery requires matching checkpoint hashes, identical generated tokens and logit hashes, and finite reference and repaired logits. Open runtime reports in the same local viewer. This adapter evaluates raw prompt continuations. Apply a model's chat template explicitly when evaluating chat behavior.

## Trace numerical fault propagation

Add `--trace` to `evaluate` or `runtime-compare` to record ordered intermediate outputs from an instrumented Qwen2 forward path. Each observation includes its module and operation, shape, dtype, nonfinite count and coordinates, finite-value statistics, a row-major F32 hash and eight leading values. Attention scores are observed before masking, and probabilities after softmax, so intentional mask infinities are excluded from numerical-fault counts.

```bash
RAYON_NUM_THREADS=2 ./target/release/llmium runtime-compare --reference models/Qwen2.5-1.5B-Instruct --candidate runs/qwen-rust-buffered/candidate --repaired runs/qwen-rust-buffered/repaired --assets models/Qwen2.5-1.5B-Instruct --prompt 'The capital of France is' --prompt 'Two plus two equals' --prompt 'The capital of France is' --max-new-tokens 2 --trace --output runs/qwen-activation/comparison.json
```

Tracing supports up to 256 input tokens per prompt and 16 generated steps. It adds observation overhead and retains summaries rather than complete activation tensors. Comparison reports identify the first changed recorded output and test exact recovery of every recorded observation. Runtime settings, tokenizer and configuration hashes must match across comparison stages for verified recovery.

Inspect a bounded page natively, including in builds without the inference feature. Omit `--stage` for single evaluation reports. Pagination offsets apply after filtering.

```bash
./target/release/llmium trace-slice runs/qwen-activation/comparison.json --stage candidate --evaluation 0 --step 0 --module-prefix model.layers.0.self_attn --nonfinite-only --count 8
```

Open Recorded activation trace within a runtime card to filter and page through observations. Select a row for coordinates and hashes. Show execution on checkpoint graph overlays recorded finite and nonfinite outputs on the registered checkpoint's modules. The overlay checks the recorded canonical path and configuration hash; the viewer does not rehash the weights. Preserve immutable checkpoint files and use the recorded file hashes for provenance.

The instrumented forward path is derived from Candle 0.11.1 and tested against its unmodified Qwen2 implementation for F32/BF16, tied/untied heads, prefill, decoding and cache resets. See [third_party/CANDLE-LICENSE-MIT](third_party/CANDLE-LICENSE-MIT) and [THIRD_PARTY.md](THIRD_PARTY.md). An observed numerical boundary alone does not establish a unique causal weight or explain neuron semantics.

## Verification

```bash
~/.cargo/bin/cargo test
~/.cargo/bin/cargo clippy --all-targets -- -D warnings
```

Include the runtime adapter when checking an inference build.

```bash
~/.cargo/bin/cargo test --release --features inference
~/.cargo/bin/cargo clippy --release --all-targets --features inference -- -D warnings
```

The implementation includes a synthetic fault experiment, malformed-container rejection, repair overwrite checks, tile bounds and raw-byte comparisons, and topology compatibility checks. The assessment layer map groups tensor names; the atlas additionally validates Qwen2 module dataflow. General semantic vulnerability discovery, causal neuron interpretation and multi-tenant isolation assessment remain research work. See [RESEARCH.md](RESEARCH.md) for positioning and evaluation requirements.

## Real checkpoint experiment

```bash
./target/release/llmium experiment models/Qwen2.5-1.5B-Instruct --tensor model.layers.0.self_attn.q_proj.weight --output runs/qwen-rust-buffered
```

This creates a checkpoint copy with one explicitly seeded NaN, preserves adjacent JSON model assets, compares every supported tensor element, restores the selected tensor and checks actual matrix forward outputs. It measures the selected linear module. Full language-model generation requires a separate runtime evaluation. The viewer labels this experiment as a seeded numerical fault.



## Measured Qwen assessment

The local Rust experiment inspected 1,543,714,304 stored parameters across 338 tensors from Qwen2.5-1.5B-Instruct. The downloaded file matched the publisher SHA-256 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`.

The candidate contains an explicitly seeded NaN at flattened index 0 in the BF16 matrix `model.layers.0.self_attn.q_proj.weight` with shape 1536 by 1536. The full comparison reported one nonfinite value and one changed element. Two findings describe the same seeded fault, numerical health and reference drift. This experiment does not establish a naturally occurring security vulnerability.

The completed experiment restored zero reference differences and zero implemented-check findings. The repaired checkpoint SHA-256 matches the publisher file exactly. A forward computation through the actual selected Qwen matrix produced eight nonfinite outputs with the fault, zero after restoration, and exact output recovery. The reproducible evidence snapshot is in `examples/qwen-demo.json`.

The full Qwen model was also evaluated with the Rust Candle CPU adapter using BF16 and greedy decoding. Two distinct raw prompts were supplied, with the first repeated after the second, and a four-token generation limit. The complete reference, candidate and repaired comparison took about eleven minutes on this workstation.

| Measurement | Reference | Seeded candidate | Repaired |
| --- | --- | --- | --- |
| Prompt evaluations | 3 | 3 | 3 |
| Nonfinite logits across recorded steps | 0 | 455,808 | 0 |
| Decoding outcome | Four tokens per prompt | Stopped at first step for every prompt | Four tokens per prompt |
| Exact checkpoint and runtime recovery | Baseline | Fault observed | Verified |

Every repaired logit hash and generated token sequence matched the reference, and the checkpoint file hash matched exactly. The repeated prompt produced the same tokens and logit hashes within each healthy stage. The continuation for `The capital of France is` was ` Paris. The capital` in both healthy checkpoints. Results cover these prompts and this runtime. The saved evidence is `examples/qwen-runtime.json`. See [DEMO.md](DEMO.md) for a presentation sequence. General semantic vulnerability discovery remains research work.

The instrumented comparison also completed on the real model, with two generated tokens per healthy prompt and the same three prompt evaluations. It recorded 507 intermediate outputs per forward step, totaling 7,605 observations. All 3,042 repaired observations matched the reference exactly. The 12 healthy logit hashes also matched the corresponding first two steps of the earlier uninstrumented run. The saved trace evidence is `examples/qwen-activation.json`.

For all three candidate prompts, the first changed output and first nonfinite output were the first layer's query projection, observation 2. In the five-token France prompt, the measured path was:

| Recorded output | Nonfinite elements |
| --- | ---: |
| Token embedding and input normalization | 0 |
| Query projection | 5 of 7,680 |
| Parallel key and value projections | 0 |
| Query after RoPE | 10 of 7,680 |
| Attention scores and probabilities, each | 25 of 300 |
| Attention context | 640 of 7,680 |
| Attention output projection | 7,680 of 7,680 |
| Final logits | 151,936 of 151,936 |

The controlled parameter comparison identifies one changed weight, while these counts describe its observed activation propagation. The corresponding reference and repaired outputs were finite. This establishes the recorded numerical behavior for this seeded experiment.
