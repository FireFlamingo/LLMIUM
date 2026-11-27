# LLMIUM research and demonstration

LLMIUM joins numerical checkpoint inspection, exact reference comparison, behavioral evidence and verified restoration in a local workbench. Novelty remains an empirical research question.

## Threat model and evidence

The operator controls the checkpoint and supplies a trusted reference. Numerical faults can damage inference. Parameter changes can also alter behavior while preserving ordinary statistics. The serving runtime controls users, requests, caches, tools and network access. Cross-user leakage needs a demonstrated path through that runtime and its isolation boundary. Shared immutable weights alone do not demonstrate access to another user's live context.

Implemented findings measure numerical health and exact element differences. Legitimate fine-tuning also changes weights. Malicious intent, triggers and exploitability require further experiments. Report severity describes assessment priority. Assign CWE and CVSS only after establishing an actual security weakness.

## Prior work

VerIDeep studies integrity verification with sensitive input samples. It establishes prior work connecting parameter changes to observable outputs. Its results do not establish performance for this implementation or for billion-parameter language models. [VerIDeep](https://arxiv.org/abs/1808.03277)

Safetensors defines names, shapes, dtypes and explicit offsets. It provides the container used by this implementation. Numerical and behavioral assessment extends beyond container validation. [Safetensors](https://github.com/safetensors/safetensors)

The real-checkpoint evaluation uses the published Qwen2.5 1.5B Instruct model. Record file digests and the downloaded revision with the evidence. [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)

## Current evidence and limits

The prototype checks every supported element for finite values, compares compatible tensors byte-for-byte, localizes seeded numerical faults, restores selected tensors into a new directory, and verifies the result. The synthetic linear experiment measures actual fault propagation and exact output recovery. Real-checkpoint claims must be assessed from their generated reports.

General semantic vulnerability discovery, automatic trigger identification, full neuron connectivity and tenant isolation assessment remain research work. The layer map groups tensor names. Statistical classification and cryptographic provenance verification require additional implementation and evaluation.

## Evaluation plan

Measure clean baselines, seeded faults, legitimate fine-tunes and dtype conversions separately. Report coverage, elapsed time, memory, false positives and missed cases. Calibrate statistical thresholds on independent data before testing. Validate behavioral claims using held-out inputs. A clean numerical scan covers the implemented checks.

## Five-minute demo

1. Run `llmium demo --output runs/demo2` using a fresh directory.
2. Start `llmium serve --root runs` and select the demo report.
3. State that the synthetic experiment contains a seeded numerical fault.
4. Select layer 2 and `model.layers.2.mlp.down_proj.weight`. Show element 459 and the nonfinite count.
5. Show the eight nonfinite forward outputs.
6. Switch to After repair and show zero differences and exact output recovery.
7. Open the real Qwen report when available and describe its measured coverage.

The Rust implementation currently validates synthetic forward recovery. A full language-model runtime experiment remains pending. Check generated evidence before making claims.

