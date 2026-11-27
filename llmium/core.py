"""Bounded-memory tensor inspection. Checkpoints are parsed as data only."""
import hashlib
import json
import math
import shutil
import struct
import tempfile
from pathlib import Path

import numpy as np

DTYPES = {"F64": "<f8", "F32": "<f4", "F16": "<f2", "BF16": "<u2",
          "I64": "<i8", "I32": "<i4", "I16": "<i2", "I8": "i1", "U8": "u1", "BOOL": "?"}
CHUNK = 262144


def digest(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


class Checkpoint:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.tensors = {}
        files = sorted(self.path.glob("*.safetensors")) if self.path.is_dir() else [self.path]
        if not files:
            raise ValueError("No safetensors files found")
        self.files = files
        for file in files:
            self._header(file)

    def _header(self, file):
        if file.suffix != ".safetensors":
            raise ValueError("Only safetensors checkpoints are supported")
        with file.open("rb") as source:
            length_bytes = source.read(8)
            if len(length_bytes) != 8:
                raise ValueError("Truncated safetensors header")
            length = struct.unpack("<Q", length_bytes)[0]
            if length > 16 * 1024 * 1024 or length > file.stat().st_size - 8:
                raise ValueError("Invalid or oversized safetensors header")
            def unique(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError("Duplicate JSON key")
                    result[key] = value
                return result
            header = json.loads(source.read(length), object_pairs_hook=unique)
        spans = []
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if name in self.tensors:
                raise ValueError(f"Duplicate tensor {name}")
            dtype = entry["dtype"]
            if dtype not in DTYPES:
                raise ValueError(f"Unsupported dtype {dtype}")
            shape = entry["shape"]
            if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
                raise ValueError("Invalid tensor shape")
            start, end = entry["data_offsets"]
            size = math.prod(shape)
            if type(start) is not int or type(end) is not int or start < 0 or end < start:
                raise ValueError("Invalid tensor offsets")
            if end - start != size * np.dtype(DTYPES[dtype]).itemsize or end + 8 + length > file.stat().st_size:
                raise ValueError("Tensor size or bounds mismatch")
            self.tensors[name] = {"file": file, "offset": 8 + length + start, "bytes": end - start,
                                  "shape": shape, "dtype": dtype, "size": size}
            spans.append((start, end))
        cursor = 0
        for start, end in sorted(spans):
            if start != cursor:
                raise ValueError("Overlapping tensors or gaps in tensor data")
            cursor = end
        if cursor + 8 + length != file.stat().st_size:
            raise ValueError("Trailing checkpoint data")

    def array(self, name):
        tensor = self.tensors[name]
        if tensor["size"] == 0:
            return np.empty(0, dtype=DTYPES[tensor["dtype"]])
        return np.memmap(tensor["file"], dtype=DTYPES[tensor["dtype"]], mode="r",
                         offset=tensor["offset"], shape=(tensor["size"],))

    def values(self, name, block):
        if self.tensors[name]["dtype"] == "BF16":
            return (block.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        return block.astype(np.float64)


def scan(path, reference=None):
    checkpoint = Checkpoint(path)
    baseline = Checkpoint(reference) if reference else None
    rows, findings = [], []
    for name, tensor in checkpoint.tensors.items():
        array = checkpoint.array(name)
        hasher = hashlib.sha256()
        count, zeros, nonfinite, square, total, maximum = 0, 0, 0, 0.0, 0.0, 0.0
        changed, first_changes, delta_square = 0, [], 0.0
        compatible = baseline is not None and name in baseline.tensors
        if compatible:
            other = baseline.tensors[name]
            compatible = other["shape"] == tensor["shape"] and other["dtype"] == tensor["dtype"]
        refarray = baseline.array(name) if compatible else None
        for start in range(0, tensor["size"], CHUNK):
            raw = array[start:start + CHUNK]
            hasher.update(raw.tobytes())
            values = checkpoint.values(name, raw)
            valid = np.isfinite(values)
            finite = values[valid]
            nonfinite += int((~valid).sum())
            zeros += int((finite == 0).sum())
            count += finite.size
            total += float(finite.sum())
            square += float(np.dot(finite, finite))
            if finite.size:
                maximum = max(maximum, float(np.abs(finite).max()))
            if compatible:
                other_raw = refarray[start:start + raw.size]
                difference = raw.view(np.dtype(f"V{raw.dtype.itemsize}")) != other_raw.view(np.dtype(f"V{raw.dtype.itemsize}"))
                indices = np.flatnonzero(difference)
                changed += indices.size
                for index in indices[:max(0, 8 - len(first_changes))]:
                    first_changes.append(int(start + index))
                delta = values - baseline.values(name, other_raw)
                finite_delta = delta[np.isfinite(delta)]
                delta_square += float(np.dot(finite_delta, finite_delta))
        row = {"name": name, "shape": tensor["shape"], "dtype": tensor["dtype"],
               "parameters": tensor["size"], "bytes": tensor["bytes"], "sha256": hasher.hexdigest(),
               "finite": count, "nonfinite": nonfinite, "zeros": zeros,
               "mean": total / count if count else None,
               "rms": math.sqrt(square / count) if count else None, "max_abs": maximum,
               "changed_elements": int(changed) if compatible else None,
               "first_changed_indices": first_changes,
               "finite_delta_l2": math.sqrt(delta_square) if compatible else None}
        rows.append(row)
        if nonfinite:
            findings.append({"severity": "high", "rule": "NONFINITE", "tensor": name,
                             "evidence": f"{nonfinite} NaN or infinity values", "confidence": "measured"})
        if tensor["size"] and zeros == tensor["size"]:
            findings.append({"severity": "info", "rule": "ALL_ZERO", "tensor": name,
                             "evidence": "Every parameter is zero. Some bias tensors are intentionally zero.", "confidence": "measured"})
        if compatible and changed:
            findings.append({"severity": "review", "rule": "REFERENCE_DRIFT", "tensor": name,
                             "evidence": f"{changed} elements differ byte-for-byte from supplied reference",
                             "confidence": "measured"})
        if baseline and not compatible:
            findings.append({"severity": "review", "rule": "SCHEMA_DRIFT", "tensor": name,
                             "evidence": "Tensor missing from reference or shape/dtype differs", "confidence": "measured"})
    if baseline:
        for name in baseline.tensors.keys() - checkpoint.tensors.keys():
            findings.append({"severity": "review", "rule": "MISSING_TENSOR", "tensor": name,
                             "evidence": "Reference tensor absent from candidate", "confidence": "measured"})
    return {"schema_version": 1, "checkpoint": str(checkpoint.path),
            "reference": str(baseline.path) if baseline else None,
            "coverage": "All supported tensor elements scanned", "tensors": rows, "findings": findings,
            "parameters": sum(row["parameters"] for row in rows),
            "files": [{"name": file.name, "sha256": digest(file), "bytes": file.stat().st_size} for file in checkpoint.files],
            "interpretation": "Numerical health and reference differences require behavioral validation. Reference trust is supplied by the operator."}


def repair(path, reference, output, names):
    """Restore selected tensor bytes into a new checkpoint directory atomically."""
    candidate, baseline = Checkpoint(path), Checkpoint(reference)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Repair output must be a new directory")
    names = list(dict.fromkeys(names))
    if not names:
        raise ValueError("Select at least one tensor")
    for name in names:
        if name not in candidate.tensors or name not in baseline.tensors:
            raise ValueError(f"Unknown tensor {name}")
        a, b = candidate.tensors[name], baseline.tensors[name]
        if (a["shape"], a["dtype"]) != (b["shape"], b["dtype"]):
            raise ValueError("Repair requires matching tensor shape and dtype")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".llmium-", dir=output.parent))
    try:
        for file in candidate.files:
            shutil.copyfile(file, staging / file.name)
        root = candidate.path if candidate.path.is_dir() else candidate.path.parent
        for file in root.iterdir():
            if file.is_file() and file.suffix == ".json":
                shutil.copyfile(file, staging / file.name)
        for name in names:
            a, b = candidate.tensors[name], baseline.tensors[name]
            with b["file"].open("rb") as source, (staging / a["file"].name).open("r+b") as target:
                source.seek(b["offset"])
                target.seek(a["offset"])
                remaining = b["bytes"]
                while remaining:
                    block = source.read(min(1048576, remaining))
                    if not block:
                        raise ValueError("Reference changed during repair")
                    target.write(block)
                    remaining -= len(block)
        verified = scan(staging, reference)
        restored = {row["name"]: row for row in verified["tensors"]}
        if any(restored[name]["changed_elements"] != 0 for name in names):
            raise ValueError("Repair verification failed")
        (staging / "llmium-repair.json").write_text(json.dumps({"restored": names, "reference": str(baseline.path),
            "reference_files": [{"name": f.name, "sha256": digest(f)} for f in baseline.files]}, indent=2), encoding="utf-8")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return {"output": str(output), "restored": names, "verified": True}
