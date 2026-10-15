"""
renderer.py — NeuroDash 3D Volume Renderer
==========================================
Converts .npy MRI volumes (4-channel: FLAIR, T1, T1ce, T2) into
interactive Plotly 3D figures streamed back to the browser.

Modality channel order in .npy: [FLAIR=0, T1=1, T1ce=2, T2=3]
Segmentation .npy:              scalar label map (0=bg, 1=necrotic, 2=edema, 3/4=enhancing)
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import zoom

logger = logging.getLogger("renderer")

# ─── Constants ────────────────────────────────────────────────────────────────

# Channel indices inside the 4-channel .npy volume
CHANNEL = {"FLAIR": 0, "T1": 1, "T1ce": 2, "T2": 3}

# Plotly colorscale for each modality — transparent at 0, vivid at 1
# rgba(r,g,b,alpha) so low-intensity brain tissue fades out, pathology pops
COLORSCALES: dict[str, list] = {
    "FLAIR": [
        [0.00, "rgba(0,0,0,0)"],
        [0.10, "rgba(80,20,0,0)"],
        [0.35, "rgba(210,90,0,0.55)"],
        [0.65, "rgba(255,175,30,0.85)"],
        [1.00, "rgba(255,245,160,1.0)"],
    ],
    "T1": [
        [0.00, "rgba(0,0,0,0)"],
        [0.10, "rgba(0,15,60,0)"],
        [0.40, "rgba(10,100,230,0.55)"],
        [0.70, "rgba(60,190,255,0.85)"],
        [1.00, "rgba(220,245,255,1.0)"],
    ],
    "T1ce": [
        [0.00, "rgba(0,0,0,0)"],
        [0.15, "rgba(70,0,0,0)"],
        [0.45, "rgba(230,40,40,0.55)"],
        [0.75, "rgba(255,110,90,0.85)"],
        [1.00, "rgba(255,220,215,1.0)"],
    ],
    "T2": [
        [0.00, "rgba(0,0,0,0)"],
        [0.10, "rgba(0,50,0,0)"],
        [0.40, "rgba(10,180,70,0.55)"],
        [0.70, "rgba(60,235,110,0.85)"],
        [1.00, "rgba(210,255,225,1.0)"],
    ],
}
# COLORSCALES: dict[str, list] = {
#     "FLAIR": [
#         [0.00, "rgba(0,0,0,0)"],
#         [0.15, "rgba(60,10,0,0)"],
#         [0.40, "rgba(180,60,0,0.35)"],
#         [0.70, "rgba(240,140,20,0.65)"],
#         [1.00, "rgba(255,230,120,0.90)"],
#     ],
#     "T1": [
#         [0.00, "rgba(0,0,0,0)"],
#         [0.15, "rgba(0,10,50,0)"],
#         [0.45, "rgba(20,80,200,0.35)"],
#         [0.75, "rgba(80,170,255,0.65)"],
#         [1.00, "rgba(200,235,255,0.90)"],
#     ],
#     "T1ce": [
#         [0.00, "rgba(0,0,0,0)"],
#         [0.20, "rgba(60,0,0,0)"],
#         [0.50, "rgba(200,30,30,0.40)"],
#         [0.80, "rgba(255,90,90,0.70)"],
#         [1.00, "rgba(255,210,210,0.90)"],
#     ],
#     "T2": [
#         [0.00, "rgba(0,0,0,0)"],
#         [0.15, "rgba(0,40,0,0)"],
#         [0.45, "rgba(20,160,60,0.35)"],
#         [0.75, "rgba(80,220,100,0.65)"],
#         [1.00, "rgba(200,255,215,0.90)"],
#     ],
# }

# Segmentation label colors (BraTS convention)
SEG_COLORS: dict[int, str] = {
    1: "rgba(255,60,60,0.85)",    # necrotic core — red
    2: "rgba(60,220,255,0.70)",   # peritumoral edema — cyan
    3: "rgba(255,230,0,0.90)",    # enhancing tumor — yellow
    4: "rgba(255,160,0,0.85)",    # enhancing (alt label) — orange
}

# Preprocess target — store data at this resolution in the session
_DS_TARGET = 64

# Render-time targets — further downsample at render time based on complexity.
# ALL4 at 64³ = 4 × 262K points = ~1M points → ~20 MB JSON → browser hangs.
# ALL4 at 32³ = 4 ×  33K points = ~131K points → ~2.5 MB JSON → fast.
_RENDER_TARGETS: dict[int, int] = {
    1: 56,   # single modality  → 56³ ≈ 175K pts/trace
    2: 44,   # two modalities   → 44³ ≈  85K pts/trace
    3: 38,   # three modalities → 38³ ≈  55K pts/trace
    4: 32,   # four modalities  → 32³ ≈  33K pts/trace  (8× less than 64³!)
}

# Surface count: fewer levels = less WebGL mesh work; still looks good at 5-10
_SURFACE_COUNTS: dict[int, int] = {
    1: 10,   # single: good quality
    2:  7,   # pair: balanced
    3:  6,   # triple
    4:  5,   # ALL4: fastest possible
}

# Dark scene background (matches NeuroDash UI)
_SCENE_BG = "rgb(6,6,18)"
_PAPER_BG = "rgb(6,6,18)"

# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class SessionVolume:
    """Preprocessed, downsampled volumes for a render session."""
    data_ds: dict[str, np.ndarray]   # {modality: (H,W,D) float32 in [0,1]}
    seg_ds: Optional[np.ndarray]     # (H,W,D) int labels or None
    shape_ds: tuple[int, int, int]
    grid: tuple[np.ndarray, np.ndarray, np.ndarray]   # mgrid coords
    has_seg: bool

    def free(self) -> None:
        """Explicitly release numpy arrays for GC."""
        self.data_ds.clear()
        self.seg_ds = None
        del self.grid
        gc.collect()


# Canonical casing for every view type (used to normalise user-supplied strings)
_CANONICAL_VIEWS = [
    "FLAIR", "T1", "T1ce", "T2",
    "T1_FLAIR", "T1_T1ce", "T1_T2", "ALL4",
    "FLAIR_SEG", "T1_SEG", "T1ce_SEG", "T2_SEG", "ALL5",
]
_CANON_MAP: dict[str, str] = {v.upper(): v for v in _CANONICAL_VIEWS}


def _canon(vt: str) -> str:
    """Normalise a view type string to its canonical casing.
    e.g. 'T1CE' → 'T1ce',  't1_t1ce' → 'T1_T1ce'
    Falls back to the input unchanged if not recognised.
    """
    return _CANON_MAP.get(vt.upper(), vt)


# ─── Shape Normalisation ──────────────────────────────────────────────────────

def normalize_shape(arr: np.ndarray) -> np.ndarray:
    """
    Return array in (4, H, W, D) float32.

    Accepts:
      (4, H, W, D)  — channels-first (standard)
      (H, W, D, 4)  — channels-last
      (H, W, D)     — single-channel (treated as one modality, replicated ×4)
    """
    arr = arr.astype(np.float32)
    if arr.ndim == 4:
        if arr.shape[0] == 4:
            return arr
        if arr.shape[-1] == 4:
            return np.transpose(arr, (3, 0, 1, 2))
        # Ambiguous — take the first 4 channels along the smallest axis
        axis = int(np.argmin(arr.shape))
        if arr.shape[axis] >= 4:
            slices = [slice(None)] * 4
            slices[axis] = slice(0, 4)
            arr = arr[tuple(slices)]
            # move channel axis to front
            return np.moveaxis(arr, axis, 0)
        raise ValueError(
            f"Cannot interpret shape {arr.shape} as a 4-channel volume."
        )
    if arr.ndim == 3:
        logger.warning("Single-channel .npy detected; replicating to 4 channels.")
        return np.stack([arr] * 4, axis=0)
    raise ValueError(f"Unsupported ndim={arr.ndim}, shape={arr.shape}")


# ─── Preprocessing ────────────────────────────────────────────────────────────

def _downsample(vol: np.ndarray, target: int = _DS_TARGET) -> np.ndarray:
    """Trilinear downsample 3-D volume to fit inside target³."""
    max_dim = max(vol.shape)
    if max_dim <= target:
        return vol.astype(np.float32)
    factor = target / max_dim
    return zoom(vol.astype(np.float32), factor, order=1)


def _percentile_norm(vol: np.ndarray) -> np.ndarray:
    """Clip to [2nd, 98th] percentile then scale to [0, 1]."""
    brain_mask = vol > 0
    if brain_mask.sum() < 10:
        return np.zeros_like(vol)
    lo, hi = np.percentile(vol[brain_mask], [2, 98])
    if hi - lo < 1e-6:
        return np.zeros_like(vol)
    return np.clip((vol - lo) / (hi - lo), 0, 1).astype(np.float32)


def preprocess_session(
    raw_data: np.ndarray,           # (4, H, W, D) or convertible
    raw_seg: Optional[np.ndarray],  # (H, W, D) int labels, or None
) -> SessionVolume:
    """
    Downsample + normalise everything once per session upload.
    All subsequent render calls operate on the pre-processed data.
    """
    # Ensure (4, H, W, D)
    data4 = normalize_shape(raw_data)

    # Downsample each modality independently (they might differ in max value)
    data_ds: dict[str, np.ndarray] = {}
    for name, ch in CHANNEL.items():
        ds = _downsample(data4[ch], _DS_TARGET)
        data_ds[name] = _percentile_norm(ds)

    # Ensure consistent spatial size (take FLAIR as reference)
    H, W, D = data_ds["FLAIR"].shape

    # Segmentation
    seg_ds: Optional[np.ndarray] = None
    if raw_seg is not None:
        seg_raw = raw_seg.squeeze().astype(np.float32)
        seg_ds = _downsample(seg_raw, _DS_TARGET)
        # Re-snap to integer labels after interpolation
        seg_ds = np.round(seg_ds).astype(np.int8)
        # Align spatial shape to data_ds
        if seg_ds.shape != (H, W, D):
            factors = (H / seg_ds.shape[0], W / seg_ds.shape[1], D / seg_ds.shape[2])
            seg_ds = zoom(seg_ds, factors, order=0).astype(np.int8)

    # Pre-compute coordinate grid once (reused by every render call)
    gx, gy, gz = np.mgrid[0:H, 0:W, 0:D]

    return SessionVolume(
        data_ds=data_ds,
        seg_ds=seg_ds,
        shape_ds=(H, W, D),
        grid=(gx, gy, gz),
        has_seg=(seg_ds is not None),
    )


# ─── Trace Builders ───────────────────────────────────────────────────────────

def _volume_trace(
    vol: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
    colorscale: list,
    name: str,
    surface_count: int = 18,
    opacity: float = 0.12,
    isomin: float = 0.08,
) -> go.Volume:
    """Single semi-transparent volume trace for one modality."""
    return go.Volume(
        x=gx.flatten(),
        y=gy.flatten(),
        z=gz.flatten(),
        value=vol.flatten(),
        isomin=isomin,
        isomax=1.0,
        opacity=opacity,
        surface_count=surface_count,
        colorscale=colorscale,
        showscale=False,
        name=name,
        caps=dict(x_show=False, y_show=False, z_show=False),
        lighting=dict(
            ambient=0.6,
            diffuse=0.8,
            specular=0.3,
            roughness=0.5,
            fresnel=0.2,
        ),
        lightposition=dict(x=100, y=200, z=0),
    )


def _seg_traces(
    seg: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
) -> list[go.Isosurface]:
    """One Isosurface trace per unique non-background segmentation label."""
    traces = []
    labels = np.unique(seg[seg > 0])
    for label in labels:
        color = SEG_COLORS.get(int(label), "rgba(200,200,200,0.75)")
        label_names = {1: "Necrotic Core", 2: "Edema", 3: "Enhancing Tumor", 4: "Enhancing (alt)"}
        traces.append(
            go.Isosurface(
                x=gx.flatten(),
                y=gy.flatten(),
                z=gz.flatten(),
                value=seg.astype(np.float32).flatten(),
                isomin=float(label) - 0.45,
                isomax=float(label) + 0.45,
                surface=dict(count=1, fill=1.0),
                colorscale=[[0, color], [1, color]],
                showscale=False,
                name=label_names.get(int(label), f"Seg {int(label)}"),
                opacity=0.75,
                caps=dict(x_show=False, y_show=False, z_show=False),
                lighting=dict(ambient=0.7, diffuse=0.9, specular=0.5, roughness=0.4),
                flatshading=False,
            )
        )
    return traces

def _orientation_labels(H: int, W: int, D: int) -> go.Scatter3d:
    """
    Anatomical orientation labels at volume extremes.
    BraTS standard: X=L→R, Y=P→A, Z=I→S
    """
    cx, cy, cz = H / 2, W / 2, D / 2
    m = 3   # small inset from edge so labels don't clip
    return go.Scatter3d(
        x=[m,   H-m, cx,  cx,  cx,  cx ],
        y=[cy,  cy,  m,   W-m, cy,  cy ],
        z=[cz,  cz,  cz,  cz,  m,   D-m],
        mode="text",
        text=["L", "R", "P", "A", "I", "S"],
        textfont=dict(
            color=["#ff7070","#ff7070",   # L/R — red axis
                   "#70ff70","#70ff70",   # P/A — green axis
                   "#7070ff","#7070ff"],  # I/S — blue axis
            size=13,
            family="JetBrains Mono, monospace",
        ),
        showlegend=False,
        hoverinfo="skip",
        name="orientation",
    )

# ─── Layout ───────────────────────────────────────────────────────────────────

def _make_layout(title: str, H: int, W: int, D: int) -> go.Layout:
    axis_cfg = dict(
        showgrid=False,
        showticklabels=False,
        showline=False,
        zeroline=False,
        showspikes=False,
        title="",
        backgroundcolor=_SCENE_BG,
    )
    return go.Layout(
        title=dict(
            text=title,
            font=dict(color="rgba(200,210,230,0.8)", size=13, family="JetBrains Mono, monospace"),
            x=0.5,
            y=0.97,
        ),
        scene=dict(
            bgcolor=_SCENE_BG,
            xaxis={**axis_cfg},
            yaxis={**axis_cfg},
            zaxis={**axis_cfg},
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=1.2),
                center=dict(x=0, y=0, z=-0.1),
            ),
            dragmode="orbit",
        ),
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PAPER_BG,
        margin=dict(l=0, r=0, t=28, b=0),
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(0,0,0,0.55)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1,
            font=dict(color="rgba(200,210,230,0.85)", size=10, family="monospace"),
            xanchor="left",
            yanchor="top",
        ),
        uirevision="neurodash",   # keeps camera between re-renders of same view
    )


# ─── View Rendering ───────────────────────────────────────────────────────────

# opacity tuning per number of overlapping volumes
_OPACITY_TABLE = {1: 0.14, 2: 0.11, 3: 0.09, 4: 0.08, 5: 0.07}


def _opacity_for(n_volumes: int) -> float:
    return _OPACITY_TABLE.get(n_volumes, 0.07)


def render_view(sv: SessionVolume, view_type: str) -> str:
    """
    Compute a Plotly 3D figure JSON string for the given view_type.

    view_type options:
      "FLAIR" | "T1" | "T1ce" | "T2"          — single modality
      "FLAIR_SEG" | "T1_SEG" | ...             — modality + segmentation
      "T1_FLAIR" | "T1_T1ce" | "T1_T2"        — T1 paired with one other
      "ALL4"                                    — all 4 modalities overlaid
      "ALL5"                                    — all 4 + segmentation
    """
    gx, gy, gz = sv.grid
    H, W, D = sv.shape_ds
    vt = _canon(view_type)   # preserves T1ce casing; maps T1CE → T1ce etc.

    # ── Determine which modalities + whether seg is included ──────────────
    include_seg = vt.endswith("_SEG") or vt == "ALL5"
    if include_seg and not sv.has_seg:
        # gracefully fall back to no-seg version
        vt = vt.replace("_SEG", "").replace("ALL5", "ALL4")
        include_seg = False

    if vt in CHANNEL:                        # e.g. "FLAIR", "T1", "T1ce", "T2"
        modalities = [vt]
    elif vt.endswith("_SEG"):                # e.g. "FLAIR_SEG" → modality = "FLAIR"
        modalities = [vt[:-4]]
    elif vt.startswith("T1_") and vt not in ("T1_SEG",):
        # e.g. "T1_FLAIR", "T1_T1CE", "T1_T2"
        other = vt[3:]                       # strip leading "T1_"
        modalities = ["T1", other]
    elif vt == "ALL4":
        modalities = ["FLAIR", "T1", "T1ce", "T2"]
    elif vt == "ALL5":
        modalities = ["FLAIR", "T1", "T1ce", "T2"]
    else:
        raise ValueError(f"Unknown view_type: {view_type!r}")

    # Validate all requested modalities exist
    for m in modalities:
        if m not in sv.data_ds:
            raise ValueError(f"Modality {m!r} not found in session data.")

    n_mods  = len(modalities)
    op      = _opacity_for(n_mods)
    if include_seg:
         op = min(op * 1.8, 0.22)  # boost opacity if seg is included, to help it stand out
    rt      = _RENDER_TARGETS.get(n_mods, 32)       # render-time target size
    sc      = _SURFACE_COUNTS.get(n_mods, 5)         # surface count
    # Tighter isomin for multi-mod cuts ~30% more near-zero voxels
    isomin_val = 0.08 if n_mods == 1 else 0.15

    # ── Render-time downsample (never upsamples, only reduces if needed) ──
    stored_max = max(sv.shape_ds)
    if stored_max > rt:
        factor = rt / stored_max
        data_render = {
            mod: zoom(sv.data_ds[mod], factor, order=1) for mod in modalities
        }
        H2, W2, D2 = next(iter(data_render.values())).shape
        rg = np.mgrid[0:H2, 0:W2, 0:D2]
        gx2, gy2, gz2 = rg[0], rg[1], rg[2]
        if include_seg and sv.seg_ds is not None:
            f2 = (H2 / sv.shape_ds[0], W2 / sv.shape_ds[1], D2 / sv.shape_ds[2])
            seg_render: Optional[np.ndarray] = (
                zoom(sv.seg_ds.astype(np.float32), f2, order=0)
                .round().astype(np.int8)
            )
        else:
            seg_render = sv.seg_ds
    else:
        data_render = {mod: sv.data_ds[mod] for mod in modalities}
        gx2, gy2, gz2 = sv.grid
        H2, W2, D2 = sv.shape_ds
        seg_render = sv.seg_ds

    # ── Build traces ──────────────────────────────────────────────────────
    traces: list = []
    for mod in modalities:
        traces.append(
            _volume_trace(
                vol=data_render[mod],
                gx=gx2, gy=gy2, gz=gz2,
                colorscale=COLORSCALES[mod],
                name=mod,
                surface_count=sc,
                opacity=op,
                isomin=isomin_val,
            )
        )

    if include_seg and seg_render is not None:
        traces.extend(_seg_traces(seg_render, gx2, gy2, gz2))

    # ── Layout ────────────────────────────────────────────────────────────
    title_map = {
        "ALL4":     "All 4 Modalities",
        "ALL5":     "All 4 + Segmentation",
        "T1_FLAIR": "T1 ⊕ FLAIR",
        "T1_T1ce":  "T1 ⊕ T1ce",
        "T1_T2":    "T1 ⊕ T2",
    }
    if vt in title_map:
        title = title_map[vt]
    elif include_seg:
        title = f"{modalities[0]} + Seg"
    else:
        title = " ⊕ ".join(modalities)

    layout = _make_layout(title, H, W, D)
    traces.append(_orientation_labels(H2, W2, D2))
    fig = go.Figure(data=traces, layout=layout)
    return fig.to_json()


# ─── All-Views Manifest ───────────────────────────────────────────────────────

def all_view_types(has_seg: bool) -> list[str]:
    """
    Ordered list of all view types to stream.
    Single-modality views first, then seg overlays, then combined.
    """
    views = ["FLAIR", "T1", "T1ce", "T2", "ALL4"]
    if has_seg:
        views += ["FLAIR_SEG", "T1_SEG", "T1ce_SEG", "T2_SEG", "ALL5"]
    return views