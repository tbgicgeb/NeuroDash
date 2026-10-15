"""
report_generator.py — NeuroDash MedGemma Report Generator
==========================================================
Sends slice images + context to MedGemma (local) and streams back a
structured radiology report.

Supported backends
──────────────────
  "ollama"        — MedGemma via Ollama REST API  (recommended for local use)
                    Install: https://ollama.com   then: ollama pull medgemma3:4b
  "transformers"  — HuggingFace transformers (needs CUDA GPU)
                    Model: google/medgemma-4b-it
  "mock"          — Returns deterministic fake output for UI testing

Set via MEDGEMMA_BACKEND env var (default: "ollama").
Set MEDGEMMA_MODEL  to override the model name.
Set OLLAMA_BASE_URL to override the Ollama server URL (default: http://localhost:11434).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import AsyncGenerator, Optional

import httpx

from slicer import SlicePackage

logger = logging.getLogger("report_generator")

# ─── Config ───────────────────────────────────────────────────────────────────

BACKEND        = os.getenv("MEDGEMMA_BACKEND", "ollama").lower()
OLLAMA_URL     = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME     = os.getenv("MEDGEMMA_MODEL", "medgemma:4b")
HF_MODEL_ID    = os.getenv("HF_MODEL_ID", "google/medgemma-4b-it")
STREAM_TIMEOUT = 300   # seconds

# ─── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert neuroradiologist reporting on brain tumour MRI.

You will receive:
- Multi-planar MRI slices (axial, coronal, sagittal) as 2×2 mosaics (FLAIR | T1 | T1ce | T2)
- T1ce slices with colour-coded segmentation overlays (red=necrotic, cyan=edema, yellow=enhancing)
- Volumetric stats and an estimated anatomical location from the centroid coordinates

CRITICAL RULES:
- Use the provided anatomical location estimate and CONFIRM or REFINE it based on the images.
- NEVER use placeholder text like "(Specify location)" — always give a concrete answer.
- Report EXACT observations from the images: signal intensity, margins, extent of edema.
- Quantify size from the voxel counts provided (state them in mm³ and estimate cm dimensions).
- If you cannot determine something from the images, state "not clearly delineated on available sequences" — never leave a blank.

Generate a structured report with these exact ## Markdown section headings:

## Clinical Summary
## Tumour Characterisation
## Modality-Specific Findings
## Segmentation Analysis
## Differential Diagnosis
## Impression"""


# ─── Report structure ─────────────────────────────────────────────────────────

REPORT_SECTIONS = [
    "clinical_summary",
    "tumour_characterisation",
    "modality_findings",
    "segmentation_analysis",
    "differential_diagnosis",
    "impression",
]

SECTION_TITLES = {
    "clinical_summary":       "Clinical Summary",
    "tumour_characterisation": "Tumour Characterisation",
    "modality_findings":      "Modality-Specific Findings",
    "segmentation_analysis":  "Segmentation Analysis",
    "differential_diagnosis": "Differential Diagnosis",
    "impression":             "Impression",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

# REMOVE the old _build_user_content function entirely, replace with these two:

def _build_ollama_messages(pkg: SlicePackage, max_images: int = 10) -> list[dict]:
    """Ollama format: images are a separate base64 list, content is plain string."""
    images_b64 = [img.b64_png for img in pkg.images[:max_images]]
    text = (
        f"{pkg.prompt_context}\n\n"
        "Analyse the images and generate a structured radiology report "
        "using ## Markdown headings for each section."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},          # must be plain string
        {"role": "user",   "content": text, "images": images_b64},
    ]


async def _stream_ollama(pkg: SlicePackage) -> AsyncGenerator[str, None]:
    payload = {
        "model":    MODEL_NAME,
        "messages": _build_ollama_messages(pkg),
        "stream":   True,
        "options":  {"temperature": 0.2, "top_p": 0.9, "num_predict": 2048},
    }
    url = f"{OLLAMA_URL}/api/chat"
    logger.info(f"Calling Ollama at {url} with model={MODEL_NAME}")

    async with httpx.AsyncClient(timeout=STREAM_TIMEOUT) as client:
        async with client.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise RuntimeError(f"Ollama error {resp.status_code}: {body.decode()[:400]}")
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue


# ─── Ollama backend ───────────────────────────────────────────────────────────

# async def _stream_ollama(pkg: SlicePackage) -> AsyncGenerator[str, None]:
#     """Stream report text from Ollama /api/chat endpoint."""
#     messages = [
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {"role": "user",   "content": _build_user_content(pkg)},
#     ]

#     payload = {
#         "model":    MODEL_NAME,
#         "messages": messages,
#         "stream":   True,
#         "options": {
#             "temperature": 0.2,
#             "top_p":       0.9,
#             "num_predict": 2048,
#         },
#     }

#     url = f"{OLLAMA_URL}/api/chat"
#     logger.info(f"Calling Ollama at {url} with model={MODEL_NAME}")

#     async with httpx.AsyncClient(timeout=STREAM_TIMEOUT) as client:
#         async with client.stream("POST", url, json=payload) as resp:
#             if resp.status_code != 200:
#                 body = await resp.aread()
#                 raise RuntimeError(
#                     f"Ollama error {resp.status_code}: {body.decode()[:400]}"
#                 )
#             async for line in resp.aiter_lines():
#                 if not line.strip():
#                     continue
#                 try:
#                     chunk = json.loads(line)
#                     token = chunk.get("message", {}).get("content", "")
#                     if token:
#                         yield token
#                     if chunk.get("done"):
#                         break
#                 except json.JSONDecodeError:
#                     continue


# ─── HuggingFace Transformers backend ────────────────────────────────────────

async def _stream_transformers(pkg: SlicePackage) -> AsyncGenerator[str, None]:
    """
    Stream report from a local HuggingFace transformers model.
    Uses a thread pool to avoid blocking the event loop.
    Requires: pip install transformers torch accelerate Pillow
    """
    import asyncio
    import base64
    import io
    from concurrent.futures import ThreadPoolExecutor

    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor, AutoModelForImageTextToText, TextIteratorStreamer
        import threading
    except ImportError as e:
        raise RuntimeError(
            f"HuggingFace backend requires: pip install transformers torch accelerate. Error: {e}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading {HF_MODEL_ID} on {device} …")

    processor = AutoProcessor.from_pretrained(HF_MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        HF_MODEL_ID,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)

    # Build PIL images list (cap at 4 for memory)
    pil_images = []
    for img_obj in pkg.images[:4]:
        raw = base64.b64decode(img_obj.b64_png)
        pil_images.append(Image.open(io.BytesIO(raw)).convert("RGB"))

    # Compose messages in processor format
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": (
                [{"type": "image"} for _ in pil_images]
                + [{"type": "text", "text": (
                    f"{pkg.prompt_context}\n\n"
                    "Please generate a structured radiology report using ## Markdown headings."
                )}]
            ),
        },
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    streamer = TextIteratorStreamer(
        processor.tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    gen_kwargs = {
        **inputs,
        "max_new_tokens": 2048,
        "temperature":    0.2,
        "do_sample":      True,
        "streamer":       streamer,
    }

    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=1)

    def _gen():
        model.generate(**gen_kwargs)

    future = loop.run_in_executor(executor, _gen)

    # Yield tokens as they come from the streamer
    for token in streamer:
        yield token
        await asyncio.sleep(0)   # give event loop a tick

    await future


# ─── Mock backend (for testing / UI dev) ─────────────────────────────────────

_MOCK_REPORT = """## Clinical Summary

Multi-modal MRI examination of the brain demonstrates a heterogeneous intra-axial mass
lesion in the right frontal lobe with associated perilesional oedema. The lesion exhibits
irregular peripheral enhancement on T1ce sequences, central necrosis, and extensive
T2/FLAIR signal abnormality, consistent with a high-grade glioma (WHO Grade IV).

## Tumour Characterisation

- **Location**: Right frontal lobe, involving the posterior frontal gyrus. Approaches but
  does not cross the midline. No involvement of eloquent motor cortex identified on
  current sequences.
- **Morphology**: Irregular, lobulated mass measuring approximately 4.2 × 3.8 × 3.5 cm.
  Heterogeneous internal signal. Poorly defined infiltrative margins.
- **Enhancement**: Thick, irregular rim of avid enhancement on T1ce. Central non-enhancing
  necrotic core. No leptomeningeal spread identified.

## Modality-Specific Findings

- **FLAIR**: Extensive hyperintense signal surrounding the lesion, indicating vasogenic
  and possibly infiltrative oedema extending into the frontal and parietal white matter.
- **T1**: Hypointense central core consistent with necrosis/liquefaction. Peripheral
  isointense to slightly hypointense tumour rim.
- **T1ce**: Florid irregular rim enhancement confirming blood-brain barrier disruption.
  Enhancing nodule in the posterior aspect suggests maximally proliferative tumour region.
- **T2**: Hyperintense perilesional zone larger than FLAIR abnormality, suggesting
  tumour infiltration beyond the enhancing rim.

## Segmentation Analysis

| Region | Volume (voxels) | Clinical Significance |
|--------|----------------|----------------------|
| Necrotic Core | 1,842 | Central liquefactive necrosis |
| Peritumoral Oedema | 8,305 | Vasogenic + infiltrative component |
| Enhancing Tumour | 3,217 | Active proliferating tumour |

Total tumour burden (all regions): ~13,364 voxels.

## Differential Diagnosis

1. **Glioblastoma (GBM)** — Most likely. Ring-enhancing necrotic mass with extensive
   oedema in a supratentorial location. High-grade features throughout.
2. **High-grade anaplastic astrocytoma** — Cannot be excluded. Tissue sampling required
   for definitive grading (IDH status, MGMT methylation).
3. **Brain metastasis** — Less likely given infiltrative margins and lack of discrete
   nodularity, but solitary metastasis should be considered without known primary.

## Impression

Heterogeneous right frontal mass with necrosis, rim enhancement, and extensive perilesional
oedema is highly suspicious for **glioblastoma (WHO Grade IV)**. Urgent neurosurgical review
and multidisciplinary team discussion recommended. Stereotactic biopsy or maximal safe
resection advised. Recommend MR spectroscopy, perfusion imaging, and full staging workup."""


async def _stream_mock(pkg: SlicePackage) -> AsyncGenerator[str, None]:
    """Yield the mock report word-by-word to simulate streaming."""
    import asyncio
    words = _MOCK_REPORT.split(" ")
    chunk_size = 4
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size]) + " "
        yield chunk
        await asyncio.sleep(0.04)


# ─── Public API ───────────────────────────────────────────────────────────────

async def stream_report(pkg: SlicePackage) -> AsyncGenerator[str, None]:
    """
    Dispatch to the configured backend and stream report tokens.

    Yields raw text tokens/chunks as they arrive.
    """
    logger.info(f"Generating report with backend={BACKEND}")

    if BACKEND == "ollama":
        async for token in _stream_ollama(pkg):
            yield token

    elif BACKEND == "transformers":
        async for token in _stream_transformers(pkg):
            yield token

    elif BACKEND == "mock":
        async for token in _stream_mock(pkg):
            yield token

    else:
        raise ValueError(
            f"Unknown MEDGEMMA_BACKEND={BACKEND!r}. "
            "Valid options: 'ollama', 'transformers', 'mock'."
        )


async def check_backend_health() -> dict:
    """Probe the configured backend and return a status dict."""
    if BACKEND == "ollama":
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{OLLAMA_URL}/api/tags")
                models = [m["name"] for m in r.json().get("models", [])]
                has_model = any(MODEL_NAME in m for m in models)
                return {
                    "backend": "ollama",
                    "url": OLLAMA_URL,
                    "reachable": True,
                    "model": MODEL_NAME,
                    "model_available": has_model,
                    "available_models": models,
                }
        except Exception as e:
            return {
                "backend": "ollama",
                "url": OLLAMA_URL,
                "reachable": False,
                "error": str(e),
                "hint": f"Run: ollama pull {MODEL_NAME}",
            }

    if BACKEND == "transformers":
        try:
            import torch
            return {
                "backend": "transformers",
                "model": HF_MODEL_ID,
                "cuda_available": torch.cuda.is_available(),
                "device": "cuda" if torch.cuda.is_available() else "cpu (slow)",
            }
        except ImportError:
            return {
                "backend": "transformers",
                "error": "torch not installed",
                "hint": "pip install torch transformers accelerate",
            }

    if BACKEND == "mock":
        return {"backend": "mock", "status": "always ready (testing mode)"}

    return {"backend": BACKEND, "error": "unknown backend"}
