# LLMIUM demonstration

## Opening statement

LLMIUM is a Rust workbench that connects checkpoint inspection, exact change localization, runtime evidence and verified restoration. This demonstration uses a deliberately seeded numerical fault in the real Qwen2.5 1.5B Instruct checkpoint. The evidence identifies the changed parameter and measures its effects.

## Prepare the viewer

From Kali WSL, use the existing reports for a quick presentation. The reports contain measured results and can be inspected without loading the model.

```bash
cd /mnt/d/Projects/LLMIUM
./target/release/llmium serve --root runs --checkpoint models/Qwen2.5-1.5B-Instruct --checkpoint runs/qwen-rust-buffered/candidate --checkpoint runs/qwen-rust-buffered/repaired
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
6. Open `qwen-runtime/comparison.json` from runs or `qwen-runtime.json` from examples. Show verified recovery and 455,808 candidate nonfinite logits across three prompt evaluations. The clean and repaired continuation for `The capital of France is` is ` Paris. The capital`. The candidate stops on nonfinite logits. Expand token and logit evidence to show exact restored hashes. The first prompt was repeated after a different request and produced the same healthy outputs.
7. Explain that the MCP interface lets an agent request an assessment or read a bounded tensor slice. The trusted reference supplies the restoration bytes.
8. Choose registered checkpoint 1 and click Inspect checkpoint. Select layer 0, then its Q projection module. Set Compare to checkpoint 0. Show the orange weight at row 0, column 0, its raw bytes and its connection line. Select checkpoint 2 and repeat the comparison to show zero changed cells in the restored tile. Use flat-index and page controls to inspect other parameter coordinates.

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
| Is weight editing new? | ROME, MEMIT, Weight-Space Programming, Keel and integrity-verification research establish prior art. LLMIUM's current contribution is the integrated evidence workflow. |
| Is restoring one tensor always safe? | Restoration needs a compatible trusted reference and behavioral validation. Replacing fine-tuned weights can remove legitimate behavior. |
| What does the network explorer show? | It inventories every tensor, validates Qwen2 module dataflow and reads actual paged weight connections. Semantic neuron roles and causal activations need further analysis. |
| Does zero findings mean secure? | It means the implemented checks found no numerical or reference-difference findings for that input. |

Research references and evaluation limits are recorded in [RESEARCH.md](RESEARCH.md).
