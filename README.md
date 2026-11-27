# LLMIUM

An open-weight checkpoint assessment workbench with full tensor scans, exact reference comparisons, verified tensor restoration, a local layer explorer, and a read-only MCP interface.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\llmium demo --output runs/demo
.\.venv\Scripts\llmium serve --root runs
```

Open http://127.0.0.1:8765. Select `demo/demo.json`, inspect the affected layer and tensor, then switch to After repair. The demo seeds a NaN in one element of a synthetic module and measures actual forward-pass corruption and exact recovery. Each demo run needs a fresh output directory.

## Assess your checkpoint

```powershell
llmium scan models/model --output runs/assessment.json
llmium scan models/candidate --reference models/trusted --output runs/comparison.json
llmium repair models/candidate --reference models/trusted --tensor model.layers.2.mlp.down_proj.weight --output models/restored
```

Inputs can be a safetensors file or directory of shards. The scanner validates offsets, lengths, shapes and supported dtypes, measures every element, and emits per-tensor SHA-256, numerical statistics, exact changed-element counts and initial changed indices. Memory use is bounded by a tensor chunk. Model execution and custom Python code are unnecessary for inspection. Quantized packed formats and GGUF need future adapters. Integer numerical summaries use float64, while change detection compares exact element bytes.

NaNs and infinities receive a high numerical-health finding. All-zero tensors receive informational findings because zero biases are common. Differences from a supplied reference receive review findings. These findings carry measured evidence. Their severity describes assessment priority. Establishing a security vulnerability, a CVSS vector, information leakage or exploitability requires further evidence from the affected runtime.

Repair writes a new directory, restores only named tensors from the reference, and rescans to verify the selected bytes. Supply a trusted compatible reference. The CLI copies JSON configuration and an audit manifest. Copy tokenizer vocabulary and other required assets when preparing a standalone inference deployment. Keep source files stable while scanning or repairing. The viewer reads generated reports and has no model mutation endpoint.

## Connect an agent

Configure an MCP stdio server using an absolute executable path.

```json
{
  "mcpServers": {
    "llmium": {
      "command": "D:\\Projects\\LLMIUM\\.venv\\Scripts\\python.exe",
      "args": ["-m", "llmium.cli", "mcp"],
      "cwd": "D:\\Projects\\LLMIUM"
    }
  }
}
```

The tools `inspect_checkpoint` and `assess_checkpoint` return structured local evidence. The server implements newline-delimited MCP JSON-RPC stdio with protocol version 2024-11-05. Agent hosts must have access to the local checkpoint paths.

## Verify

```powershell
python -m unittest discover -s tests -v
```

See [RESEARCH.md](RESEARCH.md) for the threat model, related work, evaluation plan and demonstration narrative. The current prototype proves numerical localization and reference recovery. General semantic vulnerability discovery, automatic behavioral attribution, complete neuron connectivity and multi-tenant isolation assessment remain research work. Layer groups are inferred from tensor names.
