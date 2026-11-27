"""Reproducible benign numerical fault, localization, and restoration experiment."""
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from .core import Checkpoint, repair, scan


def run_demo(output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("Demo requires an empty output directory")
    rng = np.random.default_rng(42)
    tensors = {f"model.layers.{layer}.{part}.weight": rng.normal(0, .03, (64, 64)).astype(np.float32)
               for layer in range(4) for part in ["self_attn.q_proj", "self_attn.v_proj", "mlp.down_proj"]}
    reference = root / "reference.safetensors"
    candidate = root / "candidate.safetensors"
    save_file(tensors, reference)
    target = "model.layers.2.mlp.down_proj.weight"
    tensors[target] = tensors[target].copy()
    tensors[target][7, 11] = np.nan
    save_file(tensors, candidate)
    before = scan(candidate, reference)
    repair(candidate, reference, root / "repaired", [target])
    after = scan(root / "repaired", reference)
    # An actual forward computation shows NaN propagation and exact recovery.
    inputs = rng.normal(size=(8, 64)).astype(np.float32)
    outputs = {}
    arrays = {}
    for label, path in [("reference", reference), ("candidate", candidate), ("repaired", root / "repaired")]:
        cp = Checkpoint(path)
        weight = cp.array(target).reshape(64, 64)
        arrays[label] = inputs @ weight.T
        outputs[label] = {"nonfinite_outputs": int((~np.isfinite(arrays[label])).sum()),
                          "output_shape": list(arrays[label].shape)}
    evidence = {"experiment": "Seeded numerical fault in a synthetic linear module",
                "synthetic": True, "seed": 42, "target": target, "flat_index": 459,
                "before": before, "after": after, "forward_pass": outputs,
                "exact_forward_recovery": bool(np.array_equal(arrays["reference"], arrays["repaired"])),
                "limits": "Synthetic module experiment. Full language-model generation is a separate evaluation."}
    (root / "demo.json").write_text(json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8")
    return evidence
