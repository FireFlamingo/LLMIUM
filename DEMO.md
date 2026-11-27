# LLMIUM demonstration

## Opening statement

LLMIUM is a Rust workbench that connects checkpoint inspection, exact change localization, runtime evidence and verified restoration. This demonstration uses a deliberately seeded numerical fault in the real Qwen2.5 1.5B Instruct checkpoint. The evidence identifies the changed parameter and measures its effects.

## Prepare the viewer

From Kali WSL, use the existing reports for a quick presentation. The reports contain measured results and can be inspected without loading the model.

```bash
cd /mnt/d/Projects/LLMIUM
./target/release/llmium serve --root runs
```

Open [the local viewer](http://127.0.0.1:8765). If the generated run directories are unavailable, serve the committed evidence snapshots.

```bash
./target/release/llmium serve --root examples --port 8766
```

## Demonstration sequence

1. Open `qwen-rust-buffered/demo.json` from runs or `qwen-demo.json` from examples. Show the seeded-fault label and coverage of 1,543,714,304 stored parameters across 338 tensors.
2. Choose Before repair. Show one nonfinite value and one changed element.
3. Filter for `model.layers.0.self_attn.q_proj.weight`. Select the tensor row and show BF16, shape 1536 by 1536, and flattened changed index 0.
4. Explain that the numerical-health and reference-drift findings describe the same parameter change. The experiment injects one NaN into a checkpoint copy.
5. Show the actual selected-matrix computation with eight nonfinite outputs. Choose After repair and show zero reference differences and exact matrix-output recovery.
6. When a runtime comparison report is available, open it separately. Compare reference, candidate and repaired prompt results. Expand token and logit evidence. Read the measured recovery status and finite-logit counts before claiming a full-model result.
7. Explain that the MCP interface lets an agent request an assessment or read a bounded tensor slice. The trusted reference supplies the restoration bytes.

## Reproduce the experiment

Use fresh output paths. Assessment reads every supported parameter. Model copies, hashing and CPU inference can take several minutes, particularly across the Windows-mounted filesystem.

```bash
~/.cargo/bin/cargo build --release --features inference
./target/release/llmium experiment models/Qwen2.5-1.5B-Instruct --tensor model.layers.0.self_attn.q_proj.weight --output runs/presentation-experiment
RAYON_NUM_THREADS=2 ./target/release/llmium runtime-compare --reference models/Qwen2.5-1.5B-Instruct --candidate runs/presentation-experiment/candidate --repaired runs/presentation-experiment/repaired --assets models/Qwen2.5-1.5B-Instruct --prompt 'The capital of France is' --prompt 'Two plus two equals' --prompt 'The capital of France is' --max-new-tokens 4 --output runs/presentation-runtime.json
```

## Questions to anticipate

| Question | Supported answer |
| --- | --- |
| Was this flaw discovered in the publisher's checkpoint? | The numerical fault was explicitly seeded into a local copy. The publisher file passed the implemented numerical checks. |
| Does a changed weight prove an attack? | Legitimate training also changes weights. Intent and security impact need separate evidence. |
| Can weights read another user's active conversation? | The serving runtime owns request contexts and caches. Cross-user access requires a demonstrated isolation failure in that runtime. |
| Is weight editing new? | ROME, MEMIT and integrity-verification research establish prior art. LLMIUM's current contribution is the integrated evidence workflow. |
| Is restoring one tensor always safe? | Restoration needs a compatible trusted reference and behavioral validation. Replacing fine-tuned weights can remove legitimate behavior. |
| Does the layer map show every neuron and connection? | It groups tensors from checkpoint names. Complete computation graphs and neuron-level interpretation remain future work. |
| Does zero findings mean secure? | It means the implemented checks found no numerical or reference-difference findings for that input. |

Research references and evaluation limits are recorded in [RESEARCH.md](RESEARCH.md).
