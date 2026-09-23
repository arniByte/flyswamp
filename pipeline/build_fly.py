"""Convert the flybody fruit fly model (TuragaLab, Apache-2.0) into a web-ready rig.

MuJoCo loads the MJCF so that mesh re-centering and geom/body frames come out exactly as in
the simulator. We decimate each visual mesh and write:
  web/public/data/fly.glb       node per body (local rest transform), mesh nodes per visual geom
  web/public/data/fly_rig.json  hinge joints: body, axis, range (radians), in MuJoCo order

Frames stay in MuJoCo convention (x forward, y left, z up). Units: millimetres.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from common import CACHE, WEB_DATA
from glb import GlbWriter
from meshutil import decimate

FLYBODY_REPO = "https://github.com/TuragaLab/flybody.git"
ASSETS = CACHE / "flybody" / "flybody" / "fruitfly" / "assets"
CM_TO_MM = 10.0

# Colour classes by MJCF rgba; the renderer swaps these for its own materials.
MATERIALS = {
    "cuticle": (0.674, 0.35, 0.143),
    "black": (0.0, 0.0, 0.0),
    "eye": (0.8, 0.028, 0.002),
    "ocelli": (0.128, 0.049, 0.015),
    "pale": (0.799, 0.61, 0.386),
    "vein": (0.202, 0.078, 0.026),
    "membrane": (0.539, 0.686, 0.8),
}
# Triangle budgets for the heaviest meshes; everything else keeps at most DEFAULT_TRIS.
BUDGET = {"head_red": 9000, "thorax": 5000, "thorax_black": 3500, "head": 3000, "head_black": 3000, "head_ocelli": 800}
DEFAULT_TRIS = 900


def ensure_flybody() -> None:
    if (ASSETS / "fruitfly.xml").exists():
        return
    dest = CACHE / "flybody"
    subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", FLYBODY_REPO, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", "flybody/fruitfly/assets"], check=True)


def material_of(rgba: np.ndarray) -> str:
    return min(MATERIALS, key=lambda k: np.sum((np.array(MATERIALS[k]) - rgba[:3]) ** 2))


def simplify(v: np.ndarray, f: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray]:
    mesh = decimate(v, f, target)  # also merges MuJoCo's per-face duplicated vertices
    return np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.uint32)


def main() -> None:
    ensure_flybody()
    m = mujoco.MjModel.from_xml_path(str(ASSETS / "fruitfly.xml"))
    name = lambda obj, i: mujoco.mj_id2name(m, obj, i)

    bodies = []
    for b in range(1, m.nbody):
        bodies.append({
            "name": name(mujoco.mjtObj.mjOBJ_BODY, b),
            "parent": name(mujoco.mjtObj.mjOBJ_BODY, m.body_parentid[b]) if m.body_parentid[b] > 0 else None,
            "pos": (m.body_pos[b] * CM_TO_MM).tolist(),
            "quat": m.body_quat[b].tolist(),  # w, x, y, z
        })

    geoms, total = [], 0
    for g in range(m.ngeom):
        if m.geom_group[g] != 1 or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mi = m.geom_dataid[g]
        va, vn = m.mesh_vertadr[mi], m.mesh_vertnum[mi]
        fa, fn = m.mesh_faceadr[mi], m.mesh_facenum[mi]
        gname = name(mujoco.mjtObj.mjOBJ_GEOM, g)
        rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
        v, f = simplify(m.mesh_vert[va:va + vn] * CM_TO_MM, m.mesh_face[fa:fa + fn], BUDGET.get(gname, DEFAULT_TRIS))
        total += len(f)
        geoms.append({
            "name": gname,
            "body": name(mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]),
            "pos": (m.geom_pos[g] * CM_TO_MM).tolist(),
            "quat": m.geom_quat[g].tolist(),
            "material": material_of(rgba),
            "v": v,
            "f": f,
        })

    joints = []
    for j in range(m.njnt):
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        joints.append({
            "name": name(mujoco.mjtObj.mjOBJ_JOINT, j),
            "body": name(mujoco.mjtObj.mjOBJ_BODY, m.jnt_bodyid[j]),
            "axis": m.jnt_axis[j].tolist(),
            "range": m.jnt_range[j].tolist() if m.jnt_limited[j] else None,
        })

    WEB_DATA.mkdir(parents=True, exist_ok=True)
    write_glb(WEB_DATA / "fly.glb", bodies, geoms)
    (WEB_DATA / "fly_rig.json").write_text(json.dumps({
        "source": "flybody (TuragaLab), Apache-2.0, https://github.com/TuragaLab/flybody",
        "frame": "MuJoCo: x forward, y left, z up; millimetres",
        "joints": joints,
    }, indent=1))
    print(f"{len(bodies)} bodies, {len(geoms)} meshes, {total} triangles, {len(joints)} hinge joints", file=sys.stderr)


def write_glb(path: Path, bodies: list, geoms: list) -> None:
    """Body hierarchy as nodes, one mesh node per visual geom under its body."""
    mats = {k: (*c, 0.4 if k == "membrane" else 1.0) for k, c in MATERIALS.items()}
    w = GlbWriter(mats)
    body_node = {}
    for b in bodies:
        qw, qx, qy, qz = b["quat"]
        body_node[b["name"]] = w.node(b["name"], translation=b["pos"], rotation_xyzw=[qx, qy, qz, qw])
    for b in bodies:
        if b["parent"]:
            w.parent(body_node[b["parent"]], body_node[b["name"]])
    for g in geoms:
        normals = np.array(trimesh.Trimesh(g["v"], g["f"], process=False).vertex_normals, np.float32)
        normals[np.linalg.norm(normals, axis=1) < 1e-6] = (0, 0, 1)  # zero normals become NaN in shaders
        mesh = w.mesh(g["name"], g["v"], g["f"], g["material"], normals=normals)
        qw, qx, qy, qz = g["quat"]
        w.parent(body_node[g["body"]], w.node(g["name"], mesh=mesh, translation=g["pos"], rotation_xyzw=[qx, qy, qz, qw]))
    size = w.write(path, [body_node[b["name"]] for b in bodies if not b["parent"]])
    print(f"wrote {path} ({size / 1e6:.2f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
