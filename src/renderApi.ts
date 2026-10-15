/**
 * renderApi.ts — typed client for the NeuroDash Render API
 * ──────────────────────────────────────────────────────────
 * Exports:
 *   createSession()        → sessionId + metadata
 *   fetchSingleRender()    → single Plotly figure JSON
 *   streamAllRenders()     → callback-based SSE consumer
 *   deleteSession()        → cleanup
 */

export const API_BASE =
  (import.meta.env.VITE_RENDER_API_URL as string) || "http://localhost:8000";

// ── Types ────────────────────────────────────────────────────────────────────

export interface SessionMeta {
  session_id: string;
  shape_downsampled: [number, number, number];
  modalities: string[];
  has_segmentation: boolean;
  available_views: string[];
}

export interface RenderResult {
  view: string;
  figure: string;   // JSON string — parse with JSON.parse() then pass to Plotly
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function checkOk(res: Response, label: string): void {
  if (!res.ok) throw new Error(`${label}: HTTP ${res.status}`);
}

// ── API calls ────────────────────────────────────────────────────────────────

/**
 * Upload .npy files and create a render session.
 * @param dataFile  Multi-channel MRI .npy (FLAIR, T1, T1ce, T2)
 * @param segFile   Optional segmentation label map .npy
 */
export async function createSession(
  dataFile: File,
  segFile?: File | null
): Promise<SessionMeta> {
  const form = new FormData();
  form.append("data_file", dataFile);
  if (segFile) form.append("seg_file", segFile);

  const res = await fetch(`${API_BASE}/api/session`, {
    method: "POST",
    body: form,
  });
  checkOk(res, "createSession");
  return res.json();
}

/**
 * Request a single view render on demand.
 * view_type examples: "T1", "ALL4", "T1_FLAIR", "T2_SEG"
 */
export async function fetchSingleRender(
  sessionId: string,
  viewType: string
): Promise<RenderResult> {
  const res = await fetch(
    `${API_BASE}/api/render/${sessionId}/single/${encodeURIComponent(viewType)}`
  );
  checkOk(res, `fetchSingleRender(${viewType})`);
  return res.json();
}

export interface StreamEvent {
  type?: "render" | "done" | "error";
  done?: boolean;
  error?: string;
  view?: string;
  figure?: string;
  total?: number;
  message?: string;
}

/**
 * Subscribe to the SSE stream that delivers all renders in parallel.
 *
 * @param sessionId   Session returned by createSession()
 * @param onRender    Called each time a view finishes rendering
 * @param onDone      Called when all views are complete
 * @param onError     Called if a render fails
 * @returns           AbortController — call .abort() to cancel early
 */
export function streamAllRenders(
  sessionId: string,
  onRender: (result: RenderResult) => void,
  onDone: (total: number) => void,
  onError?: (view: string, message: string) => void
): AbortController {
  const controller = new AbortController();
  const url = `${API_BASE}/api/render/${sessionId}/stream`;

  // Use fetch-based SSE so we respect AbortController
  (async () => {
    try {
      const res = await fetch(url, { signal: controller.signal });
      checkOk(res, "streamAllRenders");

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() ?? "";   // keep incomplete tail

        for (const chunk of lines) {
          const line = chunk.replace(/^data: /, "").trim();
          if (!line) continue;
          try {
            const evt = JSON.parse(line) as StreamEvent;
            if (evt.done) {
              onDone(evt.total ?? 0);
            } else if (evt.error) {
              onError?.(evt.error!, evt.message ?? "");
            } else if (evt.view && evt.figure) {
              onRender({ view: evt.view, figure: evt.figure });
            }
          } catch {
            // malformed chunk — skip
          }
        }
      }
    } catch (err: unknown) {
      if ((err as Error).name !== "AbortError") {
        console.error("SSE stream error:", err);
      }
    }
  })();

  return controller;
}

/** Delete a session and free server-side memory. */
export async function deleteSession(sessionId: string): Promise<void> {
  await fetch(`${API_BASE}/api/session/${sessionId}`, { method: "DELETE" });
}
