"""Anatomy for the x-ray view: real MaleCNS geometry placed inside the flybody model.

Outputs (web/public/data/):
  cns.glb        brain/VNC shells, neuropils, antennal lobe glomeruli, mushroom body compartments
  somas.bin      every annotated soma in the CNS (int16 xyz + region byte)
  neurons.bin    simplified skeletons of all circuit neurons (see circuit.json) as parent-linked nodes
  anatomy.json   layout of the binary files, the CNS→fly transform and mesh categories

The CNS is placed with a single rigid transform at true scale (millimetres). MaleCNS axes:
x = towards the fly's left, y = ventral, z = posterior. Fly (MuJoCo) axes: x forward, y left, z up.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh

from common import ANNOTATIONS, CACHE, GCS, WEB_DATA, fetch, flat
from glb import GlbWriter
from meshutil import decimate

ROIS = f"{GCS}/rois"
SWC = f"{GCS}/v1.0/segmentation/skeletons-malecns/skeletons-swc"
NM_PER_VOXEL = 8.0

# CNS (x left, y ventral, z posterior) → fly (x forward, y left, z up)
AXES = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], float)
BRAIN_CENTER_NM = np.array([384256.0, 237825.0, 247552.0])  # centre of the brain shell bounding box
HEAD_ANCHOR_MM = np.array([0.745, 0.0, -0.035])             # where that centre sits in the fly's head
Q = 20000.0                                                 # int16 quantization: units per mm (0.05 µm)

NEUROPILS = ["AL(L)", "AL(R)", "CA(L)", "CA(R)", "PED(L)", "PED(R)", "LH(L)", "LH(R)", "EB", "FB", "PB", "NO",
             "ME(L)", "ME(R)", "LO(L)", "LO(R)", "LOP(L)", "LOP(R)", "GNG"]
MB_COMPARTMENTS = ["a1", "a2", "a3", "a'1", "a'2", "a'3", "b1", "b2", "b'1", "b'2", "g1", "g2", "g3", "g4", "g5"]

# Skeleton simplification per group: resampling step (µm) and minimum twig length (µm).
SIMPLIFY = {"ORN": (10, 10), "PN": (9, 8), "KC": (12, 8), "MBON": (7, 6), "PAM": (8, 8), "PPL1": (7, 6), "APL": (6, 6), "DN": (8, 8)}

REGION = {"cb": 0, "ol": 1, "visual": 1, "vnc": 2, "descending": 3, "ascending": 3, "sensory": 4}


def to_fly(p_nm: np.ndarray) -> np.ndarray:
    return (np.asarray(p_nm, float) - BRAIN_CENTER_NM) / 1e6 @ AXES.T + HEAD_ANCHOR_MM


def quantize(p_mm: np.ndarray) -> np.ndarray:
    q = np.round(p_mm * Q)
    assert np.abs(q).max() < 32767, "position outside int16 range"
    return q.astype(np.int16)


# ---------- meshes ----------

def read_labels(layer: str, scale: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Read a neuroglancer precomputed ROI segmentation at a coarse scale.
    Returns the label volume (x, y, z), its voxel offset and the voxel size in nm."""
    import tensorstore as ts

    cache = CACHE / "malecns" / "rois" / f"{layer}-{scale}nm.npy"
    info = json.loads(fetch(f"{ROIS}/{layer}/info", CACHE / "malecns" / "rois" / layer / "info.json").read_text())
    sc = next(x for x in info["scales"] if x["resolution"][0] == scale)
    offset = np.array(sc.get("voxel_offset", [0, 0, 0]), float)
    if cache.exists():
        return np.load(cache), offset, float(scale)
    store = ts.open({"driver": "neuroglancer_precomputed", "kvstore": f"{ROIS}/{layer}/", "scale_index": info["scales"].index(sc)}).result()
    vol = np.asarray(store.read().result())[..., 0]
    small = vol.astype(np.uint16) if vol.max() < 65535 else vol
    np.save(cache, small)
    return small, offset, float(scale)


def segment_ids(layer: str) -> dict[str, str]:
    info = json.loads(fetch(f"{ROIS}/{layer}/segment_properties/info", CACHE / "malecns" / "rois" / layer / "props.json").read_text())["inline"]
    return dict(zip(info["properties"][0]["values"], info["ids"]))


def label_mesh(vol: np.ndarray, offset: np.ndarray, res: float, label: int, tris: int, sigma: float = 0.8):
    """Smooth surface of one label: blurred mask → marching cubes → quadric decimation → fly frame."""
    from scipy import ndimage
    from skimage.measure import marching_cubes

    idx = np.argwhere(vol == label)
    lo = np.maximum(idx.min(0) - 3, 0)
    hi = np.minimum(idx.max(0) + 4, vol.shape)
    mask = (vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] == label).astype(np.float32)
    mask = np.pad(mask, 2)
    field = ndimage.gaussian_filter(mask, sigma)
    v, f, _, _ = marching_cubes(field, 0.5)
    v_nm = (v - 2 + lo + offset + 0.5) * res
    m = decimate(v_nm, f[:, ::-1], tris)  # flipped winding: normals point outward
    return to_fly(m.vertices).astype(np.float32), np.asarray(m.faces, np.uint32)


def build_meshes() -> list[dict]:
    w = GlbWriter({"shell": (0.5, 0.8, 1.0, 1.0), "neuropil": (0.6, 0.7, 1.0, 1.0), "glomerulus": (1.0, 0.8, 0.4, 1.0), "compartment": (1.0, 0.5, 0.7, 1.0)})
    items, nodes = [], []

    def add(name: str, kind: str, vol, label: int, tris: int, sigma: float = 0.8):
        vv, ff = label_mesh(*vol, label, tris, sigma)
        items.append({"name": name, "kind": kind, "tris": len(ff)})
        nodes.append(w.node(name, mesh=w.mesh(name, vv, ff, kind)))

    add("brain", "shell", read_labels("brain-shell-v2.2", 1024), 1, 24000, 1.5)
    add("vnc", "shell", read_labels("vnc-shell-v2", 2048), 1, 12000, 1.0)
    roi_vol = read_labels("fullbrain-roi-v4", 2048)
    roi = segment_ids("fullbrain-roi-v4")
    for name in NEUROPILS:
        add(name, "neuropil", roi_vol, int(roi[name]), 1500 if name[:2] in ("ME", "LO") else 2500)
    sub_vol = read_labels("malecns-subcompartments-v3", 1024)
    present = set(np.unique(sub_vol[0]).tolist())
    for label, seg in segment_ids("malecns-subcompartments-v3").items():
        if int(seg) not in present:
            continue
        if label.startswith("AL-"):
            add(label[3:], "glomerulus", sub_vol, int(seg), 300, 0.7)
        elif label.split("(")[0] in MB_COMPARTMENTS:
            add(label, "compartment", sub_vol, int(seg), 800, 0.8)
    size = w.write(WEB_DATA / "cns.glb", nodes)
    print(f"cns.glb: {len(items)} meshes, {sum(i['tris'] for i in items)} triangles, {size / 1e6:.2f} MB", file=sys.stderr)
    return items


# ---------- somas ----------

def region_code(superclass: str) -> int:
    s = superclass or ""
    for k, v in REGION.items():
        if s.startswith(k) or k in s:
            return v
    return 5


def build_somas(ann: pd.DataFrame) -> tuple[bytes, int]:
    s = ann[ann.somaLocation.notna()]
    p = to_fly(np.stack(s.somaLocation.values).astype(float) * NM_PER_VOXEL)
    q = quantize(p)
    reg = np.array([region_code(x) for x in s.superclass.fillna("")], np.uint8)
    return q.tobytes() + reg.tobytes(), len(s)


# ---------- skeletons ----------

def read_swc(body: int) -> np.ndarray | None:
    path = CACHE / "malecns" / "swc" / f"{body}.swc"
    try:
        fetch(f"{SWC}/{body}.swc", path)
    except Exception as e:  # noqa: BLE001 - a few bodies may lack skeletons
        print(f"no skeleton for {body}: {e}", file=sys.stderr)
        return None
    rows = [ln.split() for ln in path.read_text().splitlines() if ln and not ln.startswith("#")]
    return np.array(rows, float) if rows else None


def simplify_skeleton(swc: np.ndarray, step_um: float, twig_um: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep root, branch points and leaves; resample paths every step_um; drop short twigs.
    Returns node positions (nm), parent index per node (-1 for roots) and path distance from the root (µm)."""
    ids = swc[:, 0].astype(int)
    pos = swc[:, 2:5] * NM_PER_VOXEL
    parent = swc[:, 6].astype(int)
    index = {i: k for k, i in enumerate(ids)}
    par = np.array([index.get(p, -1) for p in parent])
    n = len(ids)
    children = [[] for _ in range(n)]
    for k in range(n):
        if par[k] >= 0:
            children[par[k]].append(k)
    roots = [k for k in range(n) if par[k] < 0]

    # distance from root and subtree extent for twig pruning
    dist = np.zeros(n)
    order = []
    stack = list(roots)
    while stack:
        k = stack.pop()
        order.append(k)
        for c in children[k]:
            dist[c] = dist[k] + np.linalg.norm(pos[c] - pos[k]) / 1000
            stack.append(c)
    reach = dist.copy()  # furthest path distance below each node
    for k in reversed(order):
        if par[k] >= 0:
            reach[par[k]] = max(reach[par[k]], reach[k])

    out_pos, out_dist, out_par = [], [], []

    def emit(k, parent_out):
        out_pos.append(pos[k])
        out_dist.append(dist[k])
        out_par.append(parent_out)
        return len(out_pos) - 1

    stack = [(r, emit(r, -1)) for r in roots]
    while stack:
        k, ok = stack.pop()
        for c in children[k]:
            if reach[c] - dist[k] < twig_um and len(children[k]) > 1:
                continue  # short side twig
            # walk down the unbranched path from c, emitting every step_um
            last_emit_d, cur, prev_out = dist[k], c, ok
            while True:
                kids = [x for x in children[cur] if not (reach[x] - dist[cur] < twig_um and len(children[cur]) > 1)]
                if len(kids) != 1 or dist[cur] - last_emit_d >= step_um:
                    o = emit(cur, prev_out)
                    prev_out, last_emit_d = o, dist[cur]
                if len(kids) != 1:
                    stack.append((cur, prev_out))
                    break
                cur = kids[0]
    return np.array(out_pos), np.array(out_par, int), np.array(out_dist)


def build_neurons(ann: pd.DataFrame, circuit: dict) -> tuple[bytes, dict]:
    bodies = circuit["neurons"]["bodyId"]
    group_of = np.zeros(len(bodies), int)
    names = [g["name"] for g in circuit["groups"]]
    for gi, g in enumerate(circuit["groups"]):
        group_of[g["offset"]:g["offset"] + g["count"]] = gi

    with ThreadPoolExecutor(32) as ex:
        swcs = list(ex.map(read_swc, bodies))

    soma_nm = ann.set_index("bodyId").somaLocation
    pos_all, nid_all, dist_all, par_all, soma_all = [], [], [], [], []
    ranges, base = [], 0
    for i, (body, swc) in enumerate(zip(bodies, swcs)):
        step, twig = SIMPLIFY[names[group_of[i]]]
        if swc is None or len(swc) < 2:
            ranges.append([base, 0])
            soma_all.append(BRAIN_CENTER_NM)
            continue
        p, par, d = simplify_skeleton(swc, step, twig)
        pos_all.append(p)
        nid_all.append(np.full(len(p), i, np.uint16))
        dist_all.append(np.round(255 * d / max(d.max(), 1e-6)).astype(np.uint8))
        par_all.append(np.where(par >= 0, par + base, -1))
        ranges.append([base, len(p)])
        s = soma_nm.get(body)
        soma_all.append(np.asarray(s, float) * NM_PER_VOXEL if isinstance(s, (list, np.ndarray)) else p[0])
        base += len(p)

    pos = quantize(to_fly(np.concatenate(pos_all)))
    nid = np.concatenate(nid_all)
    dist = np.concatenate(dist_all)
    parent = np.concatenate(par_all).astype(np.int32)
    soma = quantize(to_fly(np.array(soma_all)))

    parts = [("positions", pos.tobytes()), ("nid", nid.tobytes()), ("dist", dist.tobytes()),
             ("parent", parent.tobytes()), ("soma", soma.tobytes())]
    blob, layout = bytearray(), {}
    for name, data in parts:
        while len(blob) % 4:
            blob.append(0)
        layout[name] = {"offset": len(blob), "bytes": len(data)}
        blob.extend(data)
    layout.update({"vertices": int(len(pos)), "neurons": len(bodies), "ranges": ranges})
    print(f"neurons.bin: {len(bodies)} neurons, {len(pos)} vertices, {len(blob) / 1e6:.2f} MB", file=sys.stderr)
    return bytes(blob), layout


def main() -> None:
    WEB_DATA.mkdir(parents=True, exist_ok=True)
    ann = pd.read_feather(flat(ANNOTATIONS))
    circuit = json.loads((WEB_DATA / "circuit.json").read_text())

    meshes = build_meshes()
    soma_bytes, n_soma = build_somas(ann)
    (WEB_DATA / "somas.bin").write_bytes(soma_bytes)
    neuron_bytes, layout = build_neurons(ann, circuit)
    (WEB_DATA / "neurons.bin").write_bytes(neuron_bytes)

    (WEB_DATA / "anatomy.json").write_text(json.dumps({
        "frame": "fly (MuJoCo): x forward, y left, z up; millimetres; int16 positions are mm * scale",
        "scale": Q,
        "transform": {"axes": AXES.tolist(), "brain_center_nm": BRAIN_CENTER_NM.tolist(), "head_anchor_mm": HEAD_ANCHOR_MM.tolist()},
        "meshes": meshes,
        "somas": {"count": n_soma, "regions": ["central brain", "optic lobe", "VNC", "descending/ascending", "sensory", "other"]},
        "neurons": layout,
    }, separators=(",", ":")))
    print(f"somas.bin: {n_soma} somas", file=sys.stderr)


if __name__ == "__main__":
    main()
