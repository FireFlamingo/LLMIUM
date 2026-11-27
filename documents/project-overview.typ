#import "theme.typ": *
#show: setup.with("LLMIUM project overview","Project overview")
#cover("PROJECT BRIEF","A workbench for\nopen-weight evidence.",[Introduction, problem statement, solution, architecture, demonstrated results and the research goal behind LLMIUM.])
#pagebreak()
#label("Introduction / problem statement")
= Model files need an evidence workflow
Open-weight models expose their parameters, but billions of numerical values are difficult to inspect, compare and interpret. A deployment can load a corrupted or unintentionally changed checkpoint and still appear structurally valid. Finite parameter changes can also alter outputs while passing basic numerical checks.
Developers and researchers need to answer a concrete sequence of questions: What changed? Where is it stored? Which measured outputs changed? Can the effect be reproduced? Does restoration recover the observed behavior?
== The problem
- Container validation checks file structure, not the complete behavior of a language model.
- A large tensor difference can be hard to connect to a particular parameter coordinate or execution boundary.
- Generated text alone gives weak evidence about the cause of a regression.
- Restoring bytes without comparing execution leaves the practical recovery unverified.
- Agent workflows need bounded, structured access to weight and runtime evidence rather than enormous unfiltered dumps.
== Our solution
LLMIUM combines checkpoint assessment, exact parameter exploration, controlled editing, local inference, recorded intermediate execution, verified restoration and a read-only MCP interface in one Rust workbench.
#note("The objective",[Make model changes inspectable, their effects reproducible and their restoration verifiable. Build a foundation for weight-level security research while keeping measured findings separate from unproven vulnerability claims.])
#pagebreak()
#label("Solution / architecture")
= From bytes to a controlled experiment
#grid(columns:(1fr,1fr),gutter:14pt,
  card("Inspect",[Validate safetensors headers and shard compatibility. Scan every supported element. Record hashes, finite-value statistics and exact reference differences.]),
  card("Explore",[Inventory every tensor, validate Qwen2 module topology, and inspect bounded matrix or vector tiles with exact raw bytes and coordinates.]),
  card("Reproduce",[Select an individual parameter, create an edited copy, and compare deterministic probes or full Qwen generation under controlled settings.]),
  card("Restore",[Copy the selected tensor from a compatible trusted reference into a new checkpoint. Rescan and compare file hashes, tokens, logits and recorded activations.]))
== Rust engine
The checkpoint engine uses safetensors validation, read-only memory mapping, bounded reads and SHA-256 hashing. F32, F16 and BF16 parameter edits verify the encoded storage value and preserve the source checkpoint.
The optional Candle adapter runs Qwen2/2.5 on CPU with deterministic greedy decoding. Instrumented execution records intermediate module outputs, numerical statistics and F32 hashes. The traced forward path is tested against Candle's standard implementation.
== Local workbench and agent access
The browser workbench provides a hierarchical atlas, actual weight tiles, selectable connection lines, experiment progress and recorded runtime comparisons. The six MCP tools expose numerical assessment, bounded parameter reads, topology, tiles, activation observations and descriptive behavior metrics.
The local server checks loopback Host and same-origin requests. Editing requires `--allow-edits`; experiment POST requests require same-origin Origin and JSON. The interface accepts registered checkpoint IDs and writes fresh experiment directories beneath the report root.
#pagebreak()
#label("Demonstrated results")
= A real checkpoint, measured recovery
The preserved Qwen2.5-1.5B-Instruct experiment covers *1,543,714,304 stored parameters across 338 tensors*. The downloaded checkpoint file matches the publisher SHA-256 recorded with the evidence.
== Seeded numerical fault
A controlled candidate contains one BF16 NaN at index 0 in `model.layers.0.self_attn.q_proj.weight`. Full comparison localizes one changed and nonfinite element.
Three prompt evaluations of the complete model each stop at the first step with 151,936 nonfinite logits. The reference and repaired runs remain finite. Repaired generated token sequences, logit hashes and the weight-file hash match the reference exactly.
== Execution localization
The instrumented experiment records 507 module outputs per forward step. The first changed and first nonfinite candidate boundary is the first query projection on all three prompts. At that point the parallel key and value projections remain finite.
All *3,042 repaired activation observations* match the reference exactly. Twelve healthy logit hashes also match the corresponding prefixes of the independent standard-runtime experiment.
== Finite editing
The browser-verified fixture experiment changes parameter 459 in a 64 by 64 matrix to 100. Exactly eight of 512 deterministic probe outputs change, all remain finite, and restoration recovers every reference output bit. `examples/connection-edit.json` preserves the measured comparison.

The workbench supports explicit finite single-parameter edits and compares their effects. The existing finite candidate changes one final-normalization weight from 3.0 to 2,998,272.0; its full numerical scan reports exactly one changed parameter and zero nonfinite values. The selected tensor has been restored and the repaired file hash verified. Full generation evidence for any new finite edit must be read from its completed run.
#note("Evidence scope",[These results establish controlled observations and exact recorded recovery for the supplied checkpoints, inputs and runtime. They do not demonstrate a naturally occurring flaw, universal model equivalence or access to other users' contexts.])
#pagebreak()
#label("Research position / goals")
= What we are trying to achieve
The near-term product goal is a dependable reverse-engineering workflow: precise coordinates, reproducible edits, measured effects, evidence retention and restoration. The long-term research goal is to improve localization of behavioral defects and evaluate whether exploitable failures can be found and repaired at weight level.
== Current contribution
LLMIUM integrates existing methods into a local workflow that joins parameter-level evidence with controlled runtime observations. A changed weight is inspectable; a demonstrated effect is measurable; a repaired checkpoint is checked against the supplied reference.
== Open research questions
- How can a behavioral failure be localized when no trusted reference exists?
- Which inputs reliably distinguish harmful changes from legitimate fine-tuning?
- Can candidate changes be ranked with calibrated false-positive and missed-case rates?
- Which interventions establish causal roles rather than correlated activation differences?
- What specific serving-runtime failures could expose data across requests or tenants?
== Novelty and prior work
Weight editing, causal localization and integrity verification already have substantial prior art. ROME and MEMIT study model editing. VerIDeep studies integrity verification with sensitive inputs. Weight-Space Programming and Keel are direct prior art for agent-guided diagnosis and editing of looping behavior.
Algorithmic novelty, a new vulnerability class and general autonomous repair have not been established by the current seeded demonstrations.
== Boundaries that matter
Shared immutable weights do not themselves provide access to another user's live conversation. User context, caches, tools and network permissions belong to the serving runtime. Cross-user leakage requires a concrete, reproduced isolation failure there.
Descriptive repetition and token concentration can also occur in legitimate tasks. They prioritize review and do not prove an infinite reasoning loop or malicious intent.
#pagebreak()
#label("Evaluation / next steps")
= A defensible path forward
== Evaluate beyond seeded examples
Test clean baselines, deliberate changes, legitimate fine-tunes and dtype conversions separately. Use held-out prompts and measure false positives, missed cases, elapsed time, peak memory and localization quality. Compare with established integrity and model-editing methods before claiming research novelty.
== Broaden model and runtime support
The present runtime adapter supports Qwen2 and Qwen2.5. Other safetensors models receive tensor inventories without an invented architecture graph. Quantized packed formats, GGUF and additional architecture adapters need further implementation and validation.
== Improve diagnosis and repair
Extend controlled interventions, trigger evaluation and reference-free defect localization. Exact restoration from a supplied compatible checkpoint is the implemented repair method. Learning an original semantic repair is separate research work.
== Sources and reproducibility
The repository includes `README.md`, `DEMO.md`, `RESEARCH.md`, `OBJECTIVE_AUDIT.md`, test code and evidence snapshots in `examples/`. Keep the run's checkpoint, configuration and tokenizer hashes with its results.
#set text(size:8.5pt)
- #link("https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct")[Qwen2.5-1.5B-Instruct model card]
- #link("https://github.com/safetensors/safetensors")[Safetensors container implementation]
- #link("https://arxiv.org/abs/2202.05262")[ROME: Locating and Editing Factual Associations in GPT]
- #link("https://arxiv.org/abs/2210.07229")[MEMIT: Mass-Editing Memory in a Transformer]
- #link("https://arxiv.org/abs/1808.03277")[VerIDeep: Verifying Integrity of Deep Neural Networks through Sensitive-Sample Fingerprinting]
- #link("https://rikiyat.github.io/blog/weight-space-programming/")[Sparks of Weight-Space Programming]
- #link("https://arxiv.org/abs/2611.25257")[Keel preprint]
#v(8pt)
#note("Success criterion",[Every presentation claim should be traceable to measured output, source code, tests or a preserved report. The goal is to make a model failure reviewable and a recovery reproducible.])
