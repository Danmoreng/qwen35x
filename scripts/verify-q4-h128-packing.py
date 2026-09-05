"""Verify lossless canonical-to-CPU-X8 packing, including merged projections.

Requires NumPy. This is a correctness check, not a benchmark. Both inputs
should first pass the production artifact reader's checksum validation.
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def read_directory(path):
    tensors = {}
    with path.open("rb") as stream:
        header = stream.read(96)
        if header[:8] != b"Q35H128\0" or struct.unpack_from("<I", header, 8)[0] != 1:
            raise ValueError(f"Unsupported artifact: {path}")
        count, offset = struct.unpack_from("<QQ", header, 24)
        stream.seek(offset)
        for _ in range(count):
            name_size, rank, encoding, transform, group, reserved, seed, elements, offset, size, checksum = (
                struct.unpack("<6I5Q", stream.read(64))
            )
            name = stream.read(name_size).decode("utf-8")
            shape = struct.unpack(f"<{rank}Q", stream.read(rank * 8))
            tensors[name] = dict(shape=shape, encoding=encoding, transform=transform,
                                 group=group, seed=seed, offset=offset, size=size)
    return tensors


def source_names(name):
    for suffix, parts in (
        ("mlp.gate_up_proj.weight", ("mlp.gate_proj.weight", "mlp.up_proj.weight")),
        ("linear_attn.in_proj_all.weight", tuple(f"linear_attn.in_proj_{p}.weight" for p in ("qkv", "z", "b", "a"))),
        ("self_attn.qkv_proj.weight", tuple(f"self_attn.{p}_proj.weight" for p in ("q", "k", "v"))),
    ):
        if name.endswith(suffix):
            return [name[:-len(suffix)] + part for part in parts]
    return [name]


def payload(stream, info):
    stream.seek(info["offset"])
    data = stream.read(info["size"])
    if len(data) != info["size"]:
        raise ValueError("Truncated payload")
    return data


def verify(canonical_path, packed_path):
    original = read_directory(canonical_path)
    packed = read_directory(packed_path)
    consumed = set()
    with canonical_path.open("rb") as old, packed_path.open("rb") as new:
        for name, info in packed.items():
            names = source_names(name)
            expected = []
            rows = 0
            for source in names:
                source_info = original[source]
                if source in consumed:
                    raise ValueError(f"Source consumed twice: {source}")
                consumed.add(source)
                data = payload(old, source_info)
                if info["encoding"] == 0:
                    if info["shape"] != source_info["shape"] or source_info["encoding"] != 0:
                        raise ValueError(f"F32 metadata mismatch: {source}")
                    expected.append(data)
                    continue
                if info["encoding"] not in (3, 4) or source_info["encoding"] != info["encoding"] - 2:
                    raise ValueError(f"Encoding mismatch: {source}")
                if any(info[key] != source_info[key] for key in ("transform", "group", "seed")):
                    raise ValueError(f"Quantization metadata changed: {source}")
                r, c = source_info["shape"]
                if r % 8 or c != info["shape"][1]:
                    raise ValueError(f"Invalid source tile shape: {source}")
                rows += r
                blocks = np.frombuffer(data, dtype=np.uint8).reshape(r // 8, 8, c // 32, 18)
                scales = blocks[..., :2].transpose(0, 2, 1, 3).reshape(r // 8, c // 32, 16)
                quants = blocks[..., 2:].reshape(r // 8, 8, c // 32, 2, 8)
                quants = quants.transpose(0, 2, 3, 1, 4).reshape(r // 8, c // 32, 128)
                expected.append(np.concatenate((scales, quants), axis=2).tobytes())
            if info["encoding"] != 0 and rows != info["shape"][0]:
                raise ValueError(f"Merged row count mismatch: {name}")
            if payload(new, info) != b"".join(expected):
                raise ValueError(f"Payload mismatch: {name}")
    if consumed != set(original):
        raise ValueError(f"Missing source tensors: {set(original) - consumed}")
    return dict(canonical_tensors=len(original), packed_tensors=len(packed),
                result="All FP16 scales, Q4 nibbles and F32 tensors are byte-exact.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("canonical", type=Path)
    parser.add_argument("packed", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.canonical, args.packed), indent=2))
