"""Mesh decimation shared by the pipeline scripts."""
from __future__ import annotations

import fast_simplification
import numpy as np
import trimesh


def decimate(v: np.ndarray, f: np.ndarray, tris: int, max_passes: int = 8) -> trimesh.Trimesh:
    """Quadric decimation towards `tris` triangles. One fast_simplification pass often stops
    short of the target on large meshes, so repeat until the target is met or progress stalls."""
    m = trimesh.Trimesh(v, f, process=True)
    for _ in range(max_passes):
        n = len(m.faces)
        if n <= tris:
            break
        vv, ff = fast_simplification.simplify(m.vertices, m.faces, target_reduction=1 - tris / n)
        m = trimesh.Trimesh(vv, ff, process=True)
        if len(m.faces) > 0.97 * n:
            break
    return m
