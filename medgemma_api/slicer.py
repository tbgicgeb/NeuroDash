"""
slicer.py — NeuroDash MedGemma Slice Extractor
===============================================
Extracts diagnostically rich 2-D PNG slices from 4-channel MRI .npy volumes
to feed into a multimodal LLM (MedGemma) for report generation.

Slice strategy
──────────────
1.  Find tumor centroid from seg mask (or volume centre as fallback).
2.  For each of the 3 anatomical planes (Axial, Coronal, Sagittal):
      - Extract the centre slice  + one offset slice (Δ=5 voxels) per plane.
3.  For every (plane × position) combination, produce:
      - A 4-modality mosaic  (FLAIR | T1 | T1ce | T2 in a 2×2 grid).
      - A T1ce + segmentation overlay  (colour-coded tumour regions).
4.  Return base-64 PNG strings + a structured text manifest for the prompt.

Output: ~8 PNG images   (3 planes × mosaic + 3 planes × seg + 2 extra T1ce)
        enough context for MedGemma without hitting token limits.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("slicer")

# ─── Colour maps ──────────────────────────────────────────────────────────────

# Segmentation label → RGBA
SEG_COLOURS: dict[int, tuple[int, int, int, int]] = {
    1: (255,  60,  60, 200),   # necrotic core      — red
    2: ( 60, 220, 255, 170),   # peritumoral edema  — cyan
    3: (255, 230,   0, 210),   # enhancing tumour   — yellow
    4: (255, 160,   0, 200),   # enhancing (alt)    — orange
}

SEG_LABEL_NAMES = {
    1: "Necrotic",
    2: "Edema",
    3: "Enhancing",
    4: "Enhancing",
}

# ─── Config ───────────────────────────────────────────────────────────────────

SLICE_OFFSET   = 5      # voxels away from centroid for the ±offset slices
MOSAIC_SIZE    = 256    # pixels per modality tile in mosaic
SEG_OVERLAY_SZ = 320    # pixels for seg-overlay image
FONT_SIZE      = 11


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class SliceImage:
    """One extracted image ready for the LLM."""
    label: str            # short human-readable label, e.g. "Axial Z=64 Mosaic"
    description: str      # longer description for the prompt
    b64_png: str          # base64-encoded PNG


@dataclass
class SlicePackage:
    """Everything the report generator needs."""
    images: list[SliceImage]
    meta: dict                # stats: centroid, tumor volume, etc.
    prompt_context: str       # pre-built natural-language description of images


# ─── Normalisation helpers ────────────────────────────────────────────────────

def _norm_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Percentile-clip + scale to [0, 255] uint8."""
    brain = arr > 0
    if brain.sum() < 10:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(arr[brain], [1, 99])
    if hi - lo < 1e-6:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = np.clip((arr - lo) / (hi - lo), 0, 1) * 255
    return out.astype(np.uint8)


def _apply_window(arr: np.ndarray, wl: float, ww: float) -> np.ndarray:
    """CT-style windowing (useful for T1/T1ce contrast enhancement)."""
    lo, hi = wl - ww / 2, wl + ww / 2
    return np.clip((arr - lo) / (hi - lo), 0, 1)


# ─── Shape normalisation ─────────────────────────────────────────────────────

def _to_channels_first(raw: np.ndarray) -> np.ndarray:
    """Return (4, H, W, D) float32 regardless of input layout."""
    arr = raw.astype(np.float32)
    if arr.ndim == 4:
        if arr.shape[0] == 4:
            return arr
        if arr.shape[-1] == 4:
            return np.transpose(arr, (3, 0, 1, 2))
        # best-guess: move smallest axis to front
        ax = int(np.argmin(arr.shape))
        return np.moveaxis(arr[:, :, :, :4] if ax == 3 else arr, ax, 0)
    if arr.ndim == 3:
        return np.stack([arr] * 4, axis=0)
    raise ValueError(f"Unsupported array shape: {arr.shape}")


# ─── Tumour localisation ──────────────────────────────────────────────────────

def _tumour_centroid(seg: Optional[np.ndarray], shape: tuple) -> tuple[int, int, int]:
    """Return (z_axial, y_coronal, x_sagittal) of tumour centroid.
    Falls back to volume centre if no seg or empty seg."""
    H, W, D = shape
    if seg is not None:
        mask = seg > 0
        if mask.sum() > 0:
            coords = np.argwhere(mask)
            cz, cy, cx = coords.mean(axis=0).astype(int)
            return int(cz), int(cy), int(cx)
    return H // 2, W // 2, D // 2


def _tumour_stats(seg: Optional[np.ndarray], voxel_vol_mm3: float = 1.0) -> dict:
    """Return volume + per-label stats."""
    if seg is None:
        return {"has_seg": False}
    stats: dict = {"has_seg": True, "labels": {}}
    for lbl in [1, 2, 3, 4]:
        n = int((seg == lbl).sum())
        if n > 0:
            stats["labels"][lbl] = {
                "name": SEG_LABEL_NAMES[lbl],
                "voxels": n,
                "approx_mm3": round(n * voxel_vol_mm3, 1),
            }
    stats["total_tumour_voxels"] = int((seg > 0).sum())
    return stats


# ─── Slice extraction ─────────────────────────────────────────────────────────

def _extract_slice(vol: np.ndarray, plane: str, idx: int) -> np.ndarray:
    """Extract a 2-D slice from a 3-D volume. Returns (H, W) array."""
    if plane == "axial":     # z-cut → (H, W)
        return vol[idx, :, :]
    if plane == "coronal":   # y-cut → (H, D)
        return vol[:, idx, :]
    if plane == "sagittal":  # x-cut → (W, D)
        return vol[:, :, idx]
    raise ValueError(f"Unknown plane: {plane}")


def _clamp_idx(idx: int, size: int, margin: int = 2) -> int:
    return max(margin, min(size - 1 - margin, idx))


# ─── Image builders ───────────────────────────────────────────────────────────

_MODALITY_ORDER = ["FLAIR", "T1", "T1ce", "T2"]
_MODALITY_COLORMAPS = {
    "FLAIR": "hot",
    "T1":    "bone",
    "T1ce":  "magma",
    "T2":    "viridis",
}

def _grey_to_rgb(arr2d: np.ndarray) -> np.ndarray:
    """Convert uint8 greyscale (H,W) to RGB (H,W,3) using a simple hot ramp."""
    r = arr2d.astype(np.float32) / 255.0
    # warm grey ramp — looks good for MRI
    rgb = np.stack([r, r * 0.92, r * 0.82], axis=-1)
    return (rgb * 255).astype(np.uint8)


def _modality_tile(slice2d: np.ndarray, modality: str, tile_px: int) -> Image.Image:
    """One normalised modality tile as a PIL Image."""
    u8 = _norm_to_uint8(slice2d)
    rgb = _grey_to_rgb(u8)
    img = Image.fromarray(rgb, mode="RGB").resize((tile_px, tile_px), Image.LANCZOS)

    # Label in corner
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 60, 15], fill=(0, 0, 0, 180))
    try:
        draw.text((3, 2), modality, fill=(200, 230, 255), font=ImageFont.load_default())
    except Exception:
        draw.text((3, 2), modality, fill=(200, 230, 255))
    return img


def _build_mosaic(
    channels: dict[str, np.ndarray],   # {modality: 3-D volume}
    plane: str,
    idx: int,
    tile_px: int = MOSAIC_SIZE,
) -> Image.Image:
    """2×2 mosaic of all 4 modalities at the given slice."""
    tiles = []
    for mod in _MODALITY_ORDER:
        sl = _extract_slice(channels[mod], plane, idx)
        tiles.append(_modality_tile(sl, mod, tile_px))

    # 2×2 grid
    grid = Image.new("RGB", (tile_px * 2, tile_px * 2), color=(10, 10, 20))
    grid.paste(tiles[0], (0,       0))
    grid.paste(tiles[1], (tile_px, 0))
    grid.paste(tiles[2], (0,       tile_px))
    grid.paste(tiles[3], (tile_px, tile_px))
    return grid


def _build_seg_overlay(
    t1ce_vol: np.ndarray,
    seg_vol: Optional[np.ndarray],
    plane: str,
    idx: int,
    out_px: int = SEG_OVERLAY_SZ,
) -> Image.Image:
    """T1ce slice with coloured segmentation overlay."""
    sl_t1ce = _extract_slice(t1ce_vol, plane, idx)
    u8 = _norm_to_uint8(sl_t1ce)
    rgb = _grey_to_rgb(u8)
    base = Image.fromarray(rgb).resize((out_px, out_px), Image.LANCZOS).convert("RGBA")

    if seg_vol is not None:
        sl_seg = _extract_slice(seg_vol.astype(np.float32), plane, idx)
        # resize seg to match
        seg_small = np.array(
            Image.fromarray(sl_seg.astype(np.float32)).resize((out_px, out_px), Image.NEAREST)
        )
        overlay = Image.new("RGBA", (out_px, out_px), (0, 0, 0, 0))
        opix = overlay.load()
        for lbl, colour in SEG_COLOURS.items():
            ys, xs = np.where(np.abs(seg_small - lbl) < 0.5)
            for y, x in zip(ys, xs):
                opix[x, y] = colour
        base = Image.alpha_composite(base, overlay)

    img_rgb = base.convert("RGB")

    # Legend
    draw = ImageDraw.Draw(img_rgb)
    y0 = 5
    for lbl, colour in SEG_COLOURS.items():
        if seg_vol is not None and int(lbl) in np.unique(seg_vol):
            draw.rectangle([5, y0, 16, y0 + 10], fill=colour[:3])
            try:
                draw.text((20, y0), SEG_LABEL_NAMES[lbl], fill=(240, 240, 240))
            except Exception:
                draw.text((20, y0), SEG_LABEL_NAMES[lbl], fill=(240, 240, 240))
            y0 += 14

    return img_rgb


def _pil_to_b64(img: Image.Image) -> str:
    """PIL image → base-64 PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─── Public API ───────────────────────────────────────────────────────────────

def extract_slices(
    raw_data: np.ndarray,
    raw_seg: Optional[np.ndarray] = None,
    n_offset_slices: int = 1,
) -> SlicePackage:
    """
    Main entry-point.

    Parameters
    ----------
    raw_data  : 4-channel MRI volume — (4,H,W,D) or (H,W,D,4)
    raw_seg   : segmentation label map (H,W,D) int, optional
    n_offset_slices : number of ±offset slices around centroid (1 = centre + 1 offset each way)

    Returns
    -------
    SlicePackage with images, metadata, and a ready-to-use prompt context string.
    """
    logger.info("extract_slices: normalising data …")
    data4 = _to_channels_first(raw_data)          # (4, H, W, D)
    H, W, D = data4.shape[1], data4.shape[2], data4.shape[3]

    channels: dict[str, np.ndarray] = {
        "FLAIR": data4[0], "T1": data4[1], "T1ce": data4[2], "T2": data4[3],
    }

    seg: Optional[np.ndarray] = None
    if raw_seg is not None:
        seg = raw_seg.squeeze()
        # snap floats → int labels
        if seg.dtype.kind == "f":
            seg = np.round(seg).astype(np.int8)
        if seg.shape != (H, W, D):
            from scipy.ndimage import zoom as _zoom
            f = (H / seg.shape[0], W / seg.shape[1], D / seg.shape[2])
            seg = np.round(_zoom(seg.astype(np.float32), f, order=0)).astype(np.int8)

    # Tumour centroid
    cz, cy, cx = _tumour_centroid(seg, (H, W, D))
    anatomy = _estimate_anatomy(cz, cy, cx, H, W, D)
    logger.info(f"Tumour centroid: z={cz}, y={cy}, x={cx}")
    logger.info(f"Anatomical location: {anatomy['summary']}")

    # Stats
    meta = {
        "volume_shape": (H, W, D),
        "centroid": {"axial": int(cz), "coronal": int(cy), "sagittal": int(cx)},
        **_tumour_stats(seg),
    }

    images: list[SliceImage] = []

    # ── Planes × positions ────────────────────────────────────────────────────
    planes = [
        ("axial",    cz, H, "Axial (Z)"),
        ("coronal",  cy, W, "Coronal (Y)"),
        ("sagittal", cx, D, "Sagittal (X)"),
    ]

    for plane, centre, size, plane_name in planes:
        positions = [centre]
        for k in range(1, n_offset_slices + 1):
            for sign in [1, -1]:
                pos = _clamp_idx(centre + sign * SLICE_OFFSET * k, size)
                if pos != centre and pos not in positions:
                    positions.append(pos)

        for pos in positions:
            pos_label = f"{plane_name} idx={pos}"

            # Mosaic
            try:
                mosaic = _build_mosaic(channels, plane, pos)
                images.append(SliceImage(
                    label=f"{pos_label} | 4-Modality Mosaic",
                    description=(
                        f"{plane_name} slice at index {pos}. "
                        "2×2 grid: top-left=FLAIR, top-right=T1, "
                        "bottom-left=T1ce, bottom-right=T2."
                    ),
                    b64_png=_pil_to_b64(mosaic),
                ))
            except Exception as e:
                logger.warning(f"Mosaic failed for {pos_label}: {e}")

            # Seg overlay (T1ce base)
            try:
                seg_img = _build_seg_overlay(channels["T1ce"], seg, plane, pos)
                images.append(SliceImage(
                    label=f"{pos_label} | T1ce + Seg Overlay",
                    description=(
                        f"{plane_name} slice at index {pos}. "
                        "T1ce with colour-coded tumour segmentation: "
                        "red=necrotic core, cyan=peritumoral edema, yellow=enhancing tumour."
                    ),
                    b64_png=_pil_to_b64(seg_img),
                ))
            except Exception as e:
                logger.warning(f"Seg overlay failed for {pos_label}: {e}")

    logger.info(f"Extracted {len(images)} slice images.")

    # ── Prompt context ─────────────────────────────────────────────────────────
    lines = [
        f"MRI volume dimensions: {H}×{W}×{D} voxels (H×W×D).",
        f"Modalities: FLAIR, T1, T1ce (gadolinium contrast), T2.",
        f"Tumour centroid voxel: Axial={cz}, Coronal={cy}, Sagittal={cx}.",
        f"Estimated anatomical location: {anatomy['summary']} "
        f"(normalised coords: x={anatomy['normalised']['x']}, "
        f"y={anatomy['normalised']['y']}, z={anatomy['normalised']['z']}).",
        f"Hemisphere: {anatomy['hemisphere']}. "
        f"Vertical position: {anatomy['vertical']}. "
        f"Anteroposterior: {anatomy['anteroposterior']}.",
    ]
    if meta.get("has_seg"):
        total_v = meta.get("total_tumour_voxels", 0)
        lines.append(f"Total tumour voxels: {total_v}.")
        for lbl, info in meta.get("labels", {}).items():
            lines.append(
                f"  - {info['name']} (label {lbl}): {info['voxels']} voxels "
                f"(≈{info['approx_mm3']} mm³)."
            )
    else:
        lines.append("No segmentation mask provided; describe findings from images alone.")

    lines.append("")
    lines.append("The following images are provided (in order):")
    for i, img in enumerate(images, 1):
        lines.append(f"  [{i}] {img.label}: {img.description}")

    prompt_context = "\n".join(lines)

    return SlicePackage(images=images, meta=meta, prompt_context=prompt_context)


#extra
def _estimate_anatomy(cz: int, cy: int, cx: int, H: int, W: int, D: int) -> dict:
    """
    Estimate anatomical location from normalised centroid coordinates.
    Assumes BraTS-standard orientation: X=L→R, Y=P→A, Z=I→S
    """
    # Normalise to [0,1]
    nz = cz / H   # inferior(0) → superior(1)
    ny = cy / W   # posterior(0) → anterior(1)
    nx = cx / D   # left(0) → right(1)

    hemisphere = "right" if nx > 0.5 else "left"

    # Superior/inferior thirds
    if nz > 0.66:
        vert = "superior"
    elif nz > 0.33:
        vert = "middle"
    else:
        vert = "inferior"

    # Anterior/posterior thirds
    if ny > 0.60:
        ap = "anterior"
    elif ny > 0.35:
        ap = "mid"
    else:
        ap = "posterior"

    # Rough lobe estimate from combined position
    if ap == "anterior" and vert in ("superior", "middle"):
        lobe = "frontal lobe"
    elif ap == "anterior" and vert == "inferior":
        lobe = "orbitofrontal / temporal pole region"
    elif ap == "mid" and vert == "superior":
        lobe = "parietal lobe"
    elif ap == "mid" and vert == "middle":
        lobe = "parietal / temporal lobe"
    elif ap == "mid" and vert == "inferior":
        lobe = "temporal lobe"
    elif ap == "posterior" and vert == "superior":
        lobe = "occipital / posterior parietal lobe"
    elif ap == "posterior":
        lobe = "occipital lobe"
    else:
        lobe = "frontoparietal region"

    return {
        "hemisphere": hemisphere,
        "lobe": lobe,
        "vertical": vert,
        "anteroposterior": ap,
        "normalised": {"x": round(nx, 2), "y": round(ny, 2), "z": round(nz, 2)},
        "summary": f"{vert} {hemisphere} {lobe}",
    }
#extra
