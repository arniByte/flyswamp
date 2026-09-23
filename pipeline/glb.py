"""Minimal glTF 2.0 binary writer (positions, optional normals, indices, node tree)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class GlbWriter:
    def __init__(self, materials: dict[str, tuple]):
        self.buf = bytearray()
        self.views, self.accessors, self.meshes, self.nodes = [], [], [], []
        self.material_names = list(materials)
        self.materials = [
            {"name": k, "pbrMetallicRoughness": {"baseColorFactor": list(c), "metallicFactor": 0.0, "roughnessFactor": 0.6}}
            for k, c in materials.items()
        ]

    def _view(self, data: bytes, target: int) -> int:
        while len(self.buf) % 4:
            self.buf.append(0)
        self.views.append({"buffer": 0, "byteOffset": len(self.buf), "byteLength": len(data), "target": target})
        self.buf.extend(data)
        return len(self.views) - 1

    def _accessor(self, arr: np.ndarray, kind: str, target: int, minmax: bool = False) -> int:
        comp = {np.dtype(np.float32): 5126, np.dtype(np.uint32): 5125, np.dtype(np.uint16): 5123}[arr.dtype]
        acc = {"bufferView": self._view(np.ascontiguousarray(arr).tobytes(), target), "componentType": comp, "count": len(arr), "type": kind}
        if minmax:
            acc["min"] = arr.min(axis=0).tolist()
            acc["max"] = arr.max(axis=0).tolist()
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def mesh(self, name: str, v: np.ndarray, f: np.ndarray, material: str, normals: np.ndarray | None = None) -> int:
        v = np.asarray(v, np.float32)
        attrs = {"POSITION": self._accessor(v, "VEC3", 34962, minmax=True)}
        if normals is not None:
            attrs["NORMAL"] = self._accessor(np.asarray(normals, np.float32), "VEC3", 34962)
        idx = np.asarray(f).reshape(-1)
        idx = idx.astype(np.uint16 if len(v) < 65536 else np.uint32)
        prim = {"attributes": attrs, "indices": self._accessor(idx, "SCALAR", 34963), "material": self.material_names.index(material)}
        self.meshes.append({"name": name, "primitives": [prim]})
        return len(self.meshes) - 1

    def node(self, name: str, mesh: int | None = None, translation=None, rotation_xyzw=None) -> int:
        n = {"name": name}
        if mesh is not None:
            n["mesh"] = mesh
        if translation is not None:
            n["translation"] = list(map(float, translation))
        if rotation_xyzw is not None:
            n["rotation"] = list(map(float, rotation_xyzw))
        self.nodes.append(n)
        return len(self.nodes) - 1

    def parent(self, parent: int, child: int) -> None:
        self.nodes[parent].setdefault("children", []).append(child)

    def write(self, path: Path, roots: list[int]) -> int:
        gltf = {
            "asset": {"version": "2.0", "generator": "flyswamp pipeline"},
            "scene": 0,
            "scenes": [{"nodes": roots}],
            "nodes": self.nodes, "meshes": self.meshes, "materials": self.materials,
            "accessors": self.accessors, "bufferViews": self.views, "buffers": [{"byteLength": len(self.buf)}],
        }
        js = json.dumps(gltf, separators=(",", ":")).encode()
        js += b" " * (-len(js) % 4)
        while len(self.buf) % 4:
            self.buf.append(0)
        out = bytearray()
        out += np.array([0x46546C67, 2, 12 + 8 + len(js) + 8 + len(self.buf)], np.uint32).tobytes()
        out += np.array([len(js), 0x4E4F534A], np.uint32).tobytes() + js
        out += np.array([len(self.buf), 0x004E4942], np.uint32).tobytes() + bytes(self.buf)
        path.write_bytes(bytes(out))
        return len(out)
