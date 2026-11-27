# LLMIUM: Defensive Open-Weight Checkpoint Assessment Tool
## Research & Verification Plan

---

## 1. Executive Summary & Mission Scope

Open-weight Large Language Models (LLMs) are routinely downloaded from public hubs (e.g., Hugging Face, ModelScope) and deployed directly into enterprise inference pipelines. While standard container security checks software dependencies, model weights themselves exist in an evaluation gray zone:
- File serialization security verifies *how* weights are stored, not *what* weights compute.
- Serving runtime security isolates *execution engines*, not *behavioral fidelity*.
- Behavioral evaluations (e.g., lm-eval-harness) measure task benchmark performance, but are compute-intensive, non-deterministic across hardware, and insensitive to targeted sub-layer anomalies.

**LLMIUM** is conceived as a defensive, pre-flight assessment utility for open-weight model checkpoints. It provides deterministic, reproducible health, integrity, and anomaly screening for model weights before they enter GPU memory pools and production inference endpoints.

---

## 2. Taxonomy of Failures: Strict Boundary Separation

A primary risk in model security tooling is conflating file corruption, behavioral backdoors, and engine security flaws into a single vague "vulnerability" category. LLMIUM strictly distinguishes between three orthogonal threat and fault categories:

| Dimension | Checkpoint Corruption | Malicious Behavior (Poisoning/Backdoors) | Serving Isolation & Memory Bugs |
| :--- | :--- | :--- | :--- |
| **Root Cause** | Accidental transport, hardware, or precision failure (network truncation, silent bit rot, underflow/overflow, faulty quantization scales). | Deliberate adversarial parameter tuning (Trojan triggers, targeted safety-guardrail suppression, unauthorized fine-tuned steering deltas). | Runtime infrastructure vulnerabilities in inference engines (vLLM, Ollama, TensorRT-LLM) or GPU hardware drivers. |
| **Manifestation** | Output collapse (repetitive tokens, NaNs, gibberish), localized layer divergence, broken tensor dimensions, unaligned offsets. | Preserved standard benchmark performance alongside dormant conditional behaviors activated only by specific prompt triggers. | Cross-tenant KV-cache exposure, uninitialized GPU memory leaks, CUDA out-of-bounds reads, process escape via deserializer bugs. |
| **Inspection Boundary** | **Detectable statically and via local numerical checks**: Layer-wise statistics, parity checks, IEEE 754 compliance, canonical digest matching. | **Partially detectable via differential probing**: Static heuristics can flag abnormal weight deltas vs. base models; reliable trigger detection requires behavioral testing. | **Out of scope for static checkpoint assessment**: Belongs to container sandboxing, memory sanitizers (ASan/MSan), and runtime kernel verification. |
| **Remediation** | Automated redownload, slice re-fetching, or local canonical reference reconstruction. | Model rejection, provenance blacklisting, or retraining from trusted base weights. | Host patching, driver upgrades, and runtime memory fence enforcement. |

---

## 3. Limits of Static Weight Inspection

Static weight inspection involves parsing and analyzing raw parameter arrays on disk without executing a full dynamic forward pass over an input corpus. To maintain technical defensibility, LLMIUM explicitly acknowledges the mathematical and operational limits of purely static analysis.

### What Static Inspection Can Definitively Detect:
1. **Structural and Numerical Inconsistencies:** Broken tensor shapes, header-payload byte offset mismatches, non-contiguous stride anomalies, and dangling tensor names.
2. **IEEE 754 Anomalies:** Presence of `NaN`, `+Inf`, `-Inf`, unnormalized denormals, or unexpected zero-density in weight matrices.
3. **Severe Distributional Shifts:** Anomalous weight kurtosis, extreme outlier clipping, broken scale factors in quantized models (e.g., zeroed scales in FP8/INT4 metadata), and spectral norm deviations against architectural standard distributions.
4. **Cryptographic & Shard Discrepancies:** Shard hash divergence against published manifest trees and missing layer blocks.

### Fundamental Blindspots of Static Inspection:
1. **Dynamic Activation Inaccessibility:** A static tensor matrix $W$ gives no direct signal regarding the dynamic distribution of activations $h = \sigma(W \cdot x + b)$ without the activation input distribution $p(x)$. Low-magnitude parameter changes in high-curvature directions can produce dramatic behavioral shifts, while high-magnitude parameter modifications in dormant subspaces may yield negligible changes.
2. **Subtle LoRA / Targeted Delta Concealment:** A low-rank adaptation ($A \times B$ where $r \le 16$) designed to suppress alignment or introduce a backdoor can have an $L_2$ and Frobenius norm shift indistinguishable from benign fine-tuning or quantization noise.
3. **Trigger-Action Opacity:** Pure static inspection cannot deduce the semantic input string (trigger) that routes inference through an adversarial sub-circuit without input-space sampling or symbolic execution.

---

## 4. Prior Work & Positioning

LLMIUM builds on lessons learned from academic verification and industry file safety standards.

### 4.1 VerIDeep (arXiv:1808.03277)
- **Concept:** *VerIDeep: Verifying Integrity of Deep Neural Networks through Efficient and Robust Fingerprinting.* Focused on formal and empirical verification of DNN integrity using selected input queries (fingerprints) sensitive to model parameter modifications.
- **Strengths:** Established that verifying DNN integrity does not require proving properties over the infinite input domain; targeted boundary-point fingerprinting can detect unauthorized fine-tuning or model modification.
- **Scalability Bottlenecks to Modern LLMs:** VerIDeep targeted classification CNNs (e.g., MNIST, CIFAR-10) with millions of parameters and fixed categorical output spaces. Modern LLMs have billions of parameters, autoregressive token generation, diverse prompt spaces, and stochastic decoding.
- **LLMIUM Adaptation:** Adapts the concept of integrity verification into *hierarchical pre-flight checks*: rather than requiring exhaustive dynamic fingerprinting upfront, LLMIUM uses layered screening (Layer 0: header/format; Layer 1: numerical statistics; Layer 2: sparse reference differential check).

### 4.2 Safetensors (Hugging Face)
- **Concept:** Designed specifically to replace Python's native `pickle` format (e.g., PyTorch `.pt` / `.bin` files) to eradicate arbitrary code execution during deserialization.
- **Mechanisms:** Zero-copy memory mapping (`mmap`), strict header size limits, single JSON header parsing, and validated byte offsets.
- **Safety Boundary vs. LLMIUM:**
  - `safetensors` guarantees **storage and deserialization safety** (preventing `os.system()` payload exploits inside model files).
  - `safetensors` does **not** guarantee **semantic validity or tensor sanity**. A `.safetensors` file can be completely filled with `NaN`s, intentionally perturbed weights, or corrupted layer allocations while remaining 100% valid according to the Safetensors specification.
- **LLMIUM Positioning:** LLMIUM treats `safetensors` as the baseline transport format and performs the verification that `safetensors` intentionally omits: tensor value integrity, architectural distribution validation, and differential verification.

---

## 5. Evaluation Controls & Experimental Rigor

To prevent subjective heuristics, LLMIUM's validation benchmarks enforce strict evaluation controls:

```
[Canonical Reference Checkpoint]
          │
          ├── (Control 0) Unmodified Base Model (Negative Control / Baseline)
          ├── (Control 1) Benign Quantization (AWQ, GPTQ, FP8) -> Measure false-positive tolerance
          ├── (Control 2) Benign Fine-Tuning Delta (LoRA on Alpaca/GSM8K) -> Measure drift baseline
          │
          ├── [Experimental Mutations]
          │     ├── Perturbation A: Random single-bit IEEE 754 mantissa flip
          │     ├── Perturbation B: Targeted zeroing of attention projection matrices (e.g. o_proj)
          │     ├── Perturbation C: Extreme value scaling / NaN injection
          │     └── Perturbation D: Low-rank rank-1 orthogonal perturbation
          │
          └── [LLMIUM Multi-Tier Assessor]
                │
                └── Output: Anomaly Score, Confidence Interval, Diagnostic Attribution
```

### Key Controls & Statistical Discipline:
1. **Negative Controls (Pristine Checkpoints):** Runs on official releases from Hugging Face / Mistral / Qwen repositories must yield zero anomaly false positives across all layers.
2. **Acceptable Drift Calibration:** Known transformations (INT8 quantization, bfloat16-to-fp16 cast, legitimate instruction fine-tuning) must not trigger corruption alarms. Anomaly score thresholds must be calibrated against these expected statistical variances.
3. **Statistical Sensitivity (FDR Control):** For high-dimensional parameter matrices, multiple hypothesis testing corrections (Benjamini-Hochberg) are applied across layer-wise statistical comparisons to maintain a False Discovery Rate (FDR) $\le 0.01$.

---

## 6. Novelty Claims Requiring Empirical Evidence

To prevent ungrounded claims, any assertion of novelty must be tethered to specific empirical criteria:

| Stated Claim | Required Empirical Evidence | Risk / Failure Mode |
| :--- | :--- | :--- |
| **Claim 1: Sub-Second Tensor Sanity Screening** | Benchmark demonstrating full header, byte-offset, and IEEE 754 NaN/Inf scanning of a 7B parameter model in $\le 1.5$ seconds using parallel memory-mapped block sampling without loading weights to GPU. | I/O bottlenecks on non-NVMe storage; high false negatives if sampling rate is too sparse. |
| **Claim 2: Statistical Separation of Benign Compression vs. Perturbation** | Distributional metric (e.g., standardized Wasserstein distance or kurtosis drift) that reliably separates benign 4-bit/8-bit quantization noise from catastrophic numerical corruption across all transformer blocks. | High sensitivity to model architecture (e.g., SwiGLU vs. GELU layer variance differences). |
| **Claim 3: Deterministic Delta Isolation & Local Restoration** | Ability to pinpoint an isolated corrupted tensor shard and re-hydrate only the corrupted slice from a signed canonical reference, restoring 100% baseline output fidelity without full redownload. | Shard dependency cascading; network overhead if remote range-requests are unsupported by model hub. |

---

## 7. Five-Minute Honest Demo Plan

This demonstration illustrates LLMIUM's core capability using safe, benign numerical faults and transparent reference restoration on an open-weight base model (e.g., `HuggingFaceTB/SmolLM-135M` or `Qwen/Qwen2.5-0.5B`).

### Demo Setup
- **Model:** Open-weight small model (`Qwen2.5-0.5B` or `SmolLM-135M`) serialized in `.safetensors`.
- **Constraint:** Strictly zero attack code, zero exploits, zero malware payloads. All operations operate on pure numerical arrays.
- **Environment:** Single workstation, CPU/RAM only (no GPU required for verification).

### Minute-by-Minute Script

```
00:00 - 01:00  [Baseline Verification]
               - Run: `llmium verify --model ./models/Qwen2.5-0.5B`
               - Output: Inspects 24 layers, validates safetensors headers, checks IEEE 754 compliance.
               - Result: Status PASS (All tensors healthy, 0 NaNs, expected Kurtosis range).
               - Quick inference check: Model generates coherent text for "The color of the sky is".

01:00 - 02:00  [Injection of Benign Numerical Fault]
               - Run: `python scripts/inject_fault.py --model ./models/Qwen2.5-0.5B --layer 12 --type nan_zero`
               - Action: Sets a 64x64 sub-block in `model.layers.12.self_attn.o_proj.weight` to NaN/0.0.
               - Run inference: Prompting the model outputs repetitive gibberish or NaN token crash.

02:00 - 03:00  [LLMIUM Pre-Flight Diagnostics]
               - Run: `llmium scan ./models/Qwen2.5-0.5B`
               - Detection: In < 400ms, LLMIUM flags:
                 * Tensor: `model.layers.12.self_attn.o_proj.weight`
                 * Anomaly: NaN detection + Extreme sparsity outlier (Z-score > 18.4)
                 * Recommendation: QUARANTINE / CORRUPTED_SLICE

03:00 - 04:00  [Reference Restoration]
               - Run: `llmium restore --corrupted ./models/Qwen2.5-0.5B --reference ./reference/Qwen2.5-0.5B`
               - Action: Computes SHA256 difference per tensor slice, extracts canonical bytes for layer 12
                 from reference, and surgically writes back only the altered byte range.
               - Output: "Slice repaired: model.layers.12.self_attn.o_proj.weight (Checksum matches canonical)".

04:00 - 05:00  [Post-Restoration Validation]
               - Run: `llmium verify ./models/Qwen2.5-0.5B` -> Status PASS.
               - Re-run test inference: Output returns to original coherent text.
               - Conclusion: Demonstrates closed-loop detection and recovery without redownloading 1GB+ model.
```

---

## 8. Summary of Principles

1. **Defensive Rigor:** Focus strictly on model validation, health assessment, and pipeline resilience.
2. **Transparent Boundaries:** Acknowledge what static inspection can solve (corruption, basic numerical faults, known delta diffs) and where it must defer to behavioral and runtime testing.
3. **Reproducibility:** All metrics must be deterministic, mathematically defined, and verifiable on commodity developer hardware.
