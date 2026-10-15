# NeuroDash — MedGemma Report API

FastAPI service that accepts multi-modal MRI `.npy` uploads, extracts
diagnostically rich 2-D slices from all modalities + segmentation masks,
and streams a structured radiology report from a locally-running MedGemma model.

---

## Architecture

```
medgemma_api/
├── main.py              FastAPI app + all endpoints
├── slicer.py            MRI slice extraction (finds tumour centroid, builds mosaics + seg overlays)
├── report_generator.py  MedGemma integration (Ollama / HuggingFace / Mock backends)
├── requirements.txt
└── start.sh             Launch script
```

---

## Setup

### 1. Install Python dependencies

```bash
cd medgemma_api
pip install -r requirements.txt
```

### 2. Choose a backend

#### Option A — Ollama (recommended, easiest)

```bash
# Install Ollama: https://ollama.com/download
# Pull MedGemma:
ollama pull medgemma3:4b          # 4B parameters (~3 GB)
# or the larger model:
ollama pull medgemma3:27b         # 27B parameters (~15 GB)

# Run
MEDGEMMA_BACKEND=ollama uvicorn main:app --port 8001 --reload
```

#### Option B — HuggingFace Transformers (needs CUDA GPU)

```bash
pip install torch transformers accelerate sentencepiece

MEDGEMMA_BACKEND=transformers \
HF_MODEL_ID=google/medgemma-4b-it \
uvicorn main:app --port 8001 --reload
```

> **Note:** MedGemma on HuggingFace requires accepting the model licence at
> https://huggingface.co/google/medgemma-4b-it

#### Option C — Mock (UI development / testing)

```bash
MEDGEMMA_BACKEND=mock uvicorn main:app --port 8001 --reload
```

Returns a realistic deterministic report without needing any model running.

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Liveness + backend probe |
| `GET`  | `/api/backend/status` | Detailed backend health |
| `POST` | `/api/report/generate` | Upload `.npy` files → `report_id` |
| `GET`  | `/api/report/{rid}/stream` | SSE: stream report tokens |
| `GET`  | `/api/report/{rid}` | Get completed report JSON |
| `GET`  | `/api/report/{rid}/slices` | Get extracted slice images (base64 PNGs) |
| `DELETE` | `/api/report/{rid}` | Free memory |

### Upload files

```bash
curl -X POST http://localhost:8001/api/report/generate \
  -F "data_file=@/path/to/data.npy" \
  -F "seg_file=@/path/to/seg.npy"
```

Returns:
```json
{
  "report_id": "abc123...",
  "stream_url": "/api/report/abc123.../stream",
  "n_images": 12,
  "meta": { "centroid": {...}, "labels": {...}, ... }
}
```

### Stream the report

```bash
curl -N http://localhost:8001/api/report/abc123.../stream
```

SSE events:
```
data: {"type": "token",    "text": "## Clinical Summary\n"}
data: {"type": "progress", "section": "clinical_summary", "title": "Clinical Summary"}
data: {"type": "token",    "text": "Multi-modal MRI..."}
...
data: {"type": "done",     "report_md": "...", "meta": {...}}
```

---

## Slice extraction strategy

The slicer finds the tumour centroid from the segmentation mask (or uses the
volume centre as a fallback), then extracts slices in all 3 anatomical planes:

- **Axial** (Z-axis) — standard brain tumour view
- **Coronal** (Y-axis) — for oedema extent assessment
- **Sagittal** (X-axis) — for midline involvement

For each plane × position, two image types are produced:

1. **4-Modality Mosaic** — 2×2 grid of FLAIR / T1 / T1ce / T2
2. **T1ce + Seg Overlay** — T1ce with colour-coded tumour regions
   - 🔴 Red = Necrotic core (label 1)
   - 🔵 Cyan = Peritumoral edema (label 2)
   - 🟡 Yellow = Enhancing tumour (label 3/4)

Typically 10–14 images are sent to MedGemma per scan.

---

## Frontend integration

The `src/reportApi.ts` client provides:

```ts
// 1. Upload + extract slices
const { report_id, meta } = await generateReport(dataFile, segFile);

// 2. Stream report into UI
const stop = streamReport(report_id, {
  onToken:    (t)       => setReportMd(prev => prev + t),
  onProgress: (sec, title) => setCurrentSection(title),
  onDone:     (md, meta)   => setStatus('done'),
  onError:    (msg)        => setError(msg),
});

// 3. Cleanup
stop();  // aborts stream
```

Set the API URL in your `.env`:
```
VITE_REPORT_API_URL=http://localhost:8001
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEDGEMMA_BACKEND` | `ollama` | `ollama` / `transformers` / `mock` |
| `MEDGEMMA_MODEL` | `medgemma3:4b` | Ollama model tag |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `HF_MODEL_ID` | `google/medgemma-4b-it` | HuggingFace model ID |
