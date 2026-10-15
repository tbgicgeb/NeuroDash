/**
 * reportApi.ts — NeuroDash MedGemma Report API Client
 * =====================================================
 * Thin TypeScript wrapper around the MedGemma Report API.
 *
 * Usage
 * ─────
 *   const { reportId, meta } = await generateReport(dataFile, segFile);
 *   streamReport(reportId, {
 *     onToken:    (t)   => setReport(prev => prev + t),
 *     onProgress: (sec) => setCurrentSection(sec),
 *     onDone:     (md)  => setReportMd(md),
 *     onError:    (err) => setError(err),
 *   });
 */

const REPORT_API_BASE =
  (import.meta as any).env?.VITE_REPORT_API_URL ?? "http://localhost:8001";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface ReportMeta {
  volume_shape: [number, number, number];
  centroid: { axial: number; coronal: number; sagittal: number };
  has_seg: boolean;
  total_tumour_voxels?: number;
  labels?: Record<
    string,
    { name: string; voxels: number; approx_mm3: number }
  >;
}

export interface GenerateReportResponse {
  report_id: string;
  stream_url: string;
  n_images: number;
  meta: ReportMeta;
}

export interface ReportResponse {
  report_id: string;
  status: "pending" | "streaming" | "done" | "error";
  report_md: string;
  meta: ReportMeta;
  error?: string;
}

export interface SliceImage {
  label: string;
  description: string;
  b64_png: string;
}

export interface StreamCallbacks {
  onToken?: (token: string) => void;
  onProgress?: (section: string, title: string) => void;
  onDone?: (reportMd: string, meta: ReportMeta) => void;
  onError?: (message: string) => void;
}

// ─── Generate (upload + slice extraction) ─────────────────────────────────────

/**
 * Upload .npy files to the report API.
 * Returns a report_id and slice metadata.
 * Does NOT start LLM generation yet — call streamReport() next.
 */
export async function generateReport(
  dataFile: File,
  segFile?: File | null
): Promise<GenerateReportResponse> {
  const form = new FormData();
  form.append("data_file", dataFile);
  if (segFile) form.append("seg_file", segFile);

  const res = await fetch(`${REPORT_API_BASE}/api/report/generate`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Report API error ${res.status}: ${body}`);
  }

  return res.json();
}

// ─── Stream report tokens ─────────────────────────────────────────────────────

/**
 * Open an SSE connection to the report stream.
 * Returns a cleanup function — call it to abort the stream.
 */
export function streamReport(
  reportId: string,
  callbacks: StreamCallbacks
): () => void {
  const url = `${REPORT_API_BASE}/api/report/${reportId}/stream`;
  const source = new EventSource(url);

  source.onmessage = (e) => {
    try {
      const msg = JSON.parse(e.data) as {
        type: string;
        text?: string;
        section?: string;
        title?: string;
        report_md?: string;
        meta?: ReportMeta;
        message?: string;
      };

      switch (msg.type) {
        case "token":
          callbacks.onToken?.(msg.text ?? "");
          break;
        case "progress":
          callbacks.onProgress?.(msg.section ?? "", msg.title ?? "");
          break;
        case "done":
          callbacks.onDone?.(msg.report_md ?? "", msg.meta!);
          source.close();
          break;
        case "error":
          callbacks.onError?.(msg.message ?? "Unknown error");
          source.close();
          break;
      }
    } catch {
      // ignore malformed events
    }
  };

  source.onerror = () => {
    callbacks.onError?.("SSE connection error — is the report API running?");
    source.close();
  };

  return () => source.close();
}

// ─── Fetch completed report ────────────────────────────────────────────────────

export async function fetchReport(reportId: string): Promise<ReportResponse> {
  const res = await fetch(`${REPORT_API_BASE}/api/report/${reportId}`);
  if (!res.ok) throw new Error(`Fetch report error ${res.status}`);
  return res.json();
}

// ─── Fetch slice images ────────────────────────────────────────────────────────

export async function fetchSlices(
  reportId: string
): Promise<SliceImage[]> {
  const res = await fetch(`${REPORT_API_BASE}/api/report/${reportId}/slices`);
  if (!res.ok) throw new Error(`Fetch slices error ${res.status}`);
  const data = await res.json();
  return data.images as SliceImage[];
}

// ─── Delete report ─────────────────────────────────────────────────────────────

export async function deleteReport(reportId: string): Promise<void> {
  await fetch(`${REPORT_API_BASE}/api/report/${reportId}`, {
    method: "DELETE",
  });
}

// ─── Backend health ────────────────────────────────────────────────────────────

export async function checkBackendStatus(): Promise<Record<string, unknown>> {
  const res = await fetch(`${REPORT_API_BASE}/api/backend/status`);
  if (!res.ok) throw new Error("Backend unreachable");
  return res.json();
}
