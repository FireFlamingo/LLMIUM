#import "theme.typ": *
#show: setup.with("LLMIUM demonstration guide","Demonstration guide")
#cover("PRESENTER'S GUIDE","Inspect. Edit.\nReproduce. Restore.",[A practical walkthrough for presenting LLMIUM, selecting a single weight connection and demonstrating a measured, verifiable recovery.])
#pagebreak()
#label("Preparation")
= Start the local workspace
Use the existing real-model evidence for a short presentation. Use the synthetic fixture for an interactive edit that completes quickly. Full-model copies, verification and CPU generation take substantially longer.
== Build and launch
Run from Kali WSL. Python is not required.
```bash
cd /mnt/d/Projects/LLMIUM
~/.cargo/bin/cargo build --release --features inference
./target/release/llmium serve --root runs --allow-edits \
  --checkpoint models/Qwen2.5-1.5B-Instruct \
  --checkpoint runs/qwen-rust-buffered/candidate \
  --checkpoint runs/qwen-rust-buffered/repaired \
  --checkpoint runs/qwen-finite/candidate \
  --checkpoint runs/qwen-finite/repaired \
  --checkpoint runs/rust-demo/reference.safetensors
```
Open #link("http://127.0.0.1:8765")[the local workbench]. Keep this server running during the demonstration. Check that the sidebar says *Rust engine connected*.
== Before presenting
- Confirm the Qwen checkpoint and saved reports are present in the stated directories.
- Confirm the overview shows the measured 1.54B parameters and 338 tensors.
- Use *Inspect sample fixture* for the quick interactive section. It requires the registered fixture checkpoint in the launch command.
- Keep enough disk space for two copies of the selected checkpoint. For this Qwen model the weight file alone is approximately 3.09 GB per copy.
- Keep model files immutable while scanning or running inference. Edits create new directories under `runs/lab/`.
#note("If the fixture is missing",[Create it once with a fresh directory: `llmium demo --output runs/rust-demo`. Existing output directories are protected from accidental overwrite.])
#pagebreak()
#label("Interactive demonstration")
= Select one connection
Explain: “A displayed thread is one stored parameter coordinate in the selected tensor tile. We can inspect and edit that exact value.”
+ Select *Weight explorer*, choose the registered synthetic reference checkpoint, and click *Inspect checkpoint*. Alternatively, click *Inspect sample fixture* on the overview.
+ Select *Layer 2*. In the tensor menu choose `model.layers.2.mlp.down_proj.weight`.
+ Set *Tile size* to 8. Enter flat index `459` and click *Go to index*. The tile starts at row 7, column 11.
+ In *Connections in the selected matrix tile*, hover over the connection from column 11 to row 7. The hover readout shows the row, column, flat index and value.
+ Click the connection to lock the selection. Keyboard users can focus a connection and press Enter or Space. Clicking the corresponding weight cell selects the same parameter.
+ In *Selected connection*, confirm the tensor name and flat index `459`. Inspect the current dtype and raw little-endian bytes.
== Make the edit and rerun
+ Enter `100` as the new stored value.
+ Choose *Selected parameter probe · quick*.
+ Click *Create edit & rerun demonstration*. The server creates a new checkpoint copy and writes exactly this parameter.
+ Follow the progress stages: copying and editing, assessment, restoration and verification, then measured outputs.
+ When complete, click *Open experiment evidence*.
#note("What this fast demonstration proves",[It measures the selected matrix using deterministic synthetic inputs. A single matrix weight changes one output row for each of eight inputs. This is a parameter-level experiment, not a claim about complete language-model quality.])
#pagebreak()
#label("Read the result")
= Show the evidence chain
== Before repair
Show the deliberately seeded experiment label, the tensor and the flat index. The comparison should identify one changed element. Because `100` is finite, the nonfinite count stays zero.
In *Experiment evidence*, inspect `forward_pass`. Show the number of changed probe outputs and the maximum absolute delta. The reference, candidate and repaired hashes describe the actual recorded probe arrays.
== After repair
Switch *Demo phase* to *After repair*. Show zero reference differences for the restored checkpoint. Explain that the tool restores the selected tensor from the supplied trusted baseline.
Show `exact_forward_recovery: true`. This checks every recorded probe output bit against the reference.
== Inspect the edited bytes
Click *Inspect edited checkpoint* in the completed experiment panel. Select the affected tensor and compare its tile with the original registered reference. The changed cell is highlighted. Click it to display its exact coordinate and raw representation.
The original checkpoint remains unchanged. The job directory contains the candidate, repaired checkpoint, change assessment, manifests and demonstration JSON.
#card("Suggested presenter line",[“The edit is deliberate. The assessment localizes the changed parameter, the controlled probe measures its effect, and the trusted reference supplies the restoration. We verify the recorded recovery rather than infer it from a green status alone.”])
== Keep the distinction clear
The seeded edit is not an automatically discovered vulnerability. Byte differences also arise from legitimate fine-tuning. A finite value may affect behavior while passing nonfinite checks. Security impact needs its own reproducible experiment.
#pagebreak()
#label("Real language-model evidence")
= Follow a fault through Qwen
Use *Evidence & repair* to open the preserved reports. These recorded results can be presented without rerunning the model.
== Generation comparison
Open `qwen-runtime/comparison.json`. It evaluates two distinct prompts and repeats the first after the second, using four-token greedy decoding on CPU.
- The seeded candidate contains one NaN at index 0 of the first query-projection matrix.
- Each candidate prompt stops at its first step with 151,936 nonfinite logits. The three prompt evaluations total 455,808 nonfinite logits.
- Reference and repaired runs have finite logits, identical generated tokens and identical recorded logit hashes.
- The repaired weight-file hash matches the published checkpoint file exactly.
== Intermediate execution
Open `qwen-activation/comparison.json`, expand the first candidate's *Recorded activation trace*, and select *Only changed outputs*.
The first changed and nonfinite output is `model.layers.0.self_attn.q_proj`, observation 2. The parallel key and value projections remain finite at that boundary. Select a row to inspect counts, hashes and coordinates.
Click *Show execution on checkpoint graph*. Green marks finite outputs; orange marks nonfinite outputs; purple marks finite outputs changed from the aligned reference. Endpoint buttons expose embedding, final normalization and output-head observations.
#note("Trace scope",[The saved run records 507 observations per forward step. All 3,042 repaired observations match the reference exactly. The viewer matches canonical checkpoint path and configuration hash; it does not rehash weight files while browsing. Preserve the recorded checkpoints.])
#pagebreak()
#label("Full rerun and troubleshooting")
= Run a new generation experiment
To measure an edit through the complete Qwen model, select a Qwen parameter connection or vector cell, enter a finite value and choose *Full Qwen generation · CPU*.
Use the same prompt and token limit for the reference, candidate and repaired stages. A 16-token limit allows the descriptive repetition metrics to detect sufficiently long, low-diversity outputs. The result depends on the actual parameter and prompt; do not promise repetition before measuring it.
The worker loads stages sequentially, clears request caches, records token IDs and logit hashes, and can record intermediate outputs. Only one interactive experiment runs at a time.
== Understand finite differences
At the initial prompt, matching input tokens permit a controlled comparison of the recorded boundaries. Once generated tokens differ, later inputs differ too. The viewer reports this context divergence; downstream differences alone no longer localize a parameter.
== Common issues
#table(columns:(1fr,2fr),inset:8pt,stroke:0.4pt+rgb("d9dfe5"),fill:(x,y)=>if y==0{pale}else{none},
  [*Symptom*],[*Action*],
  [Editing disabled],[Restart with `--allow-edits`. Select an actual registered checkpoint and a parameter.],
  [Value unchanged],[Choose a value that encodes differently in the storage dtype. BF16 rounds many decimal inputs to the same bytes.],
  [Probe rejected],[Use a vector or matrix with at most 20 million parameters, or choose supported Qwen generation.],
  [Generation unavailable],[Build with the inference feature. Supply a Qwen2/2.5 checkpoint with configuration and tokenizer assets.],
  [Progress seems slow],[Hashing, copying and full scans read gigabytes. CPU inference takes minutes. Check the existing job before starting another.],
  [Overlay unavailable],[Register the exact recorded checkpoint path and preserve its configuration and weight files.],
)
#v(6pt)
#card("Closing line",[“LLMIUM connects an exact parameter change to measured execution and verified restoration. The current product is an integrity and regression workbench. General semantic discovery and serving isolation are the next research problems.”])
