# LLMIUM

A Rust workbench for open-weight checkpoint assessment, exact reference comparison, verified tensor restoration, a local layer explorer and a read-only MCP interface.

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
./target/release/llmium repair models/candidate --reference models/trusted --tensor model.layers.2.mlp.down_proj.weight --output models/restored
```

Inputs are a safetensors file or directory of shards. The safetensors crate validates structure. Every supported element is inspected using read-only memory mapping. Reports contain per-tensor SHA-256, numerical health statistics, exact changed-element counts and the first changed indices. Integer statistical summaries use float64. Element differences compare raw bytes.

NaNs and infinities receive high numerical-health findings. All-zero tensors receive informational findings because zero biases are common. Reference differences receive review findings. These findings describe measured properties. Security vulnerability classification, CWE mapping and CVSS scoring require demonstrated security impact.

Repair requires a new output directory and a compatible trusted reference. It restores selected tensors, rescans and publishes the new directory after verification. JSON configuration and a reference hash manifest are copied. Copy any additional tokenizer or deployment assets required by your runtime. Keep files immutable during scans and repair. Quantized packed formats, GGUF and unsupported float formats need future adapters.

## Agent interface

Run `llmium mcp` as a stdio MCP server. It exposes the read-only `assess_checkpoint` tool with a required `path` and optional `reference`. It uses newline-delimited JSON-RPC and MCP protocol version 2024-11-05. Paths use the server's filesystem. An agent host can launch the WSL binary with a configuration like this.

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

## Verification

```bash
~/.cargo/bin/cargo test
~/.cargo/bin/cargo clippy --all-targets -- -D warnings
```

The implementation includes a synthetic fault experiment, malformed-container rejection and repair overwrite checks. The layer map groups tensor names. General semantic vulnerability discovery, complete neuron connectivity and multi-tenant isolation assessment remain research work. See [RESEARCH.md](RESEARCH.md) for positioning and evaluation requirements.

## Real checkpoint experiment

```bash
./target/release/llmium experiment models/Qwen2.5-1.5B-Instruct --tensor model.layers.0.self_attn.q_proj.weight --output runs/qwen-rust
```

This creates a checkpoint copy with one explicitly seeded NaN, compares every supported tensor element, restores the selected tensor and checks actual matrix forward outputs. It measures the selected linear module. Full language-model generation requires a separate runtime evaluation. The viewer labels this experiment as a seeded numerical fault.
