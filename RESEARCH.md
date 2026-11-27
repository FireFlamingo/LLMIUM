# LLMIUM research and demonstration

LLMIUM joins numerical checkpoint inspection, exact reference comparison, behavioral evidence and verified restoration in a local workbench. Novelty remains an empirical research question.

## Threat model and evidence

The operator controls the checkpoint and supplies a trusted reference. Numerical faults can damage inference. Parameter changes can also alter behavior while preserving ordinary statistics. The serving runtime controls users, requests, caches, tools and network access. Cross-user leakage needs a demonstrated path through that runtime and its isolation boundary. Shared immutable weights alone do not demonstrate access to another user's live context.

Implemented findings measure numerical health and exact element differences. Legitimate fine-tuning also changes weights. Malicious intent, triggers and exploitability require further experiments. Report severity describes assessment priority. Assign CWE and CVSS only after establishing an actual security weakness.

## Prior work

The November 2, 2026 author article *Sparks of Weight-Space Programming* appears to match the motivating example. It reports a frontier agent diagnosing reasoning loops in Qwen3.5-0.8B and editing weights without gradient training. The November 18 Keel preprint reports agentic weight editing for looping vision-language models. These are direct prior art for agent-guided diagnosis and repair. Their reported benchmark gains have not been independently reproduced in LLMIUM. [Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/), [Keel](https://arxiv.org/abs/2611.25257)

VerIDeep studies integrity verification with sensitive input samples. It establishes prior work connecting parameter changes to observable outputs. Its results do not establish performance for this implementation or for billion-parameter language models. [VerIDeep](https://arxiv.org/abs/1808.03277)

ROME uses causal interventions to localize factual recall and edits feed-forward weights to change factual associations. MEMIT extends direct model editing to many associations. These establish substantial prior art for locating and changing model behavior through weights. LLMIUM cannot claim that weight-level inspection or editing is a new research direction. [ROME](https://arxiv.org/abs/2202.05262), [MEMIT](https://arxiv.org/abs/2210.07229)

Safetensors defines names, shapes, dtypes and explicit offsets. It provides the container used by this implementation. Numerical and behavioral assessment extends beyond container validation. [Safetensors](https://github.com/safetensors/safetensors)

The real-checkpoint evaluation uses the published Qwen2.5 1.5B Instruct model. Record file digests and the downloaded revision with the evidence. [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)

## Current evidence and limits

The prototype checks every supported element for finite values, compares compatible tensors byte-for-byte, localizes seeded numerical faults, restores selected tensors into a new directory, and verifies the result. The synthetic linear experiment measures actual fault propagation and exact output recovery. Real-checkpoint claims must be assessed from their generated reports.

The completed Qwen2.5 1.5B Instruct experiment covers 338 tensors and 1,543,714,304 stored parameters. A single seeded NaN was localized to index 0 in the first layer's query projection. Full-model BF16 CPU evaluation used two distinct prompts and repeated the first, with a four-token greedy limit. Each candidate evaluation stopped with 151,936 nonfinite logits on the first step. Reference and repaired evaluations had finite logits, identical token sequences and identical per-step logit hashes. The repaired file hash matched the publisher checkpoint. Evidence is preserved in `examples/qwen-demo.json` and `examples/qwen-runtime.json`.

General semantic vulnerability discovery, automatic trigger identification, causal interpretation of individual neurons and tenant isolation assessment remain research work. The checkpoint atlas inventories every stored tensor and validates a Qwen2 architecture-derived module graph. Its connection diagrams read actual bounded matrix tiles with exact bytes and optional reference comparison. They describe selected parameter coordinates and declared architecture dataflow. Statistical classification and cryptographic provenance verification require additional implementation and evaluation.

## Evaluation plan

Measure clean baselines, seeded faults, legitimate fine-tunes and dtype conversions separately. Report coverage, elapsed time, memory, false positives and missed cases. Calibrate statistical thresholds on independent data before testing. Validate behavioral claims using held-out inputs. A clean numerical scan covers the implemented checks.

The defensible product contribution is an integrated, reproducible workflow from checkpoint evidence to localized change, runtime observations and verified restoration. A research contribution would need an independently evaluated localization method, a new vulnerability class or measured gains over existing integrity and model-editing methods. The current seeded experiment establishes none of those novelty claims.

For runtime evaluation, retain per-step logit hashes and generated token IDs for clean, changed and restored checkpoints with identical assets and settings. Repeat a prompt after another request to test cache reset. Compare finite-logit counts before interpreting generated text. Exact output recovery on a few prompts establishes those observations only. It does not establish universal behavioral equivalence for different runtimes, hardware or inputs.

## Five-minute demo

1. Run `llmium demo --output runs/demo2` using a fresh directory.
2. Start `llmium serve --root runs` and select the demo report.
3. State that the synthetic experiment contains a seeded numerical fault.
4. Select layer 2 and `model.layers.2.mlp.down_proj.weight`. Show element 459 and the nonfinite count.
5. Show the eight nonfinite forward outputs.
6. Switch to After repair and show zero differences and exact output recovery.
7. Open the real Qwen report when available and describe its measured coverage.

The Rust implementation validates synthetic and real-checkpoint matrix recovery. The optional runtime adapter records full language-model comparisons separately. Check the generated evidence before making claims. [DEMO.md](DEMO.md) provides the presentation sequence and reproducible commands.

