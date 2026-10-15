/**
 * Viewer3D.tsx — Interactive 3D MRI Volume Viewer
 * ─────────────────────────────────────────────────
 * Renders Plotly 3D volume figures returned by the Render API.
 * Displays a tab-bar of all available views; renders stream in as they arrive.
 *
 * Install dependencies first:
 *   npm install react-plotly.js plotly.js-dist-min
 *   npm install -D @types/react-plotly.js
 *
 * In vite.config.ts add the alias so plotly.js-dist-min is used instead of the
 * full ~3 MB bundle:
 *
 *   resolve: {
 *     alias: { "plotly.js": "plotly.js-dist-min" }
 *   }
 */

import React, { useCallback, useEffect, useRef, useState } from "react";
import Plot from "react-plotly.js";
import { Loader2, RefreshCw, LayoutGrid, BrainCircuit, AlertCircle } from "lucide-react";
import { streamAllRenders, fetchSingleRender, type RenderResult } from "./renderApi";

// ── Types ─────────────────────────────────────────────────────────────────────

interface Viewer3DProps {
  sessionId: string | null;
  availableViews: string[];
  hasSeg: boolean;
  scanName: string;
}

type RenderMap = Record<string, string>;   // viewType → plotly JSON string

// ── Human-readable labels & groups ───────────────────────────────────────────

const VIEW_LABELS: Record<string, string> = {
  FLAIR:    "FLAIR",
  T1:       "T1",
  T1ce:     "T1ce",
  T2:       "T2",
  T1_FLAIR: "T1 + FLAIR",
  T1_T1ce:  "T1 + T1ce",
  T1_T2:    "T1 + T2",
  ALL4:     "All 4",
  FLAIR_SEG: "FLAIR + Seg",
  T1_SEG:   "T1 + Seg",
  T1ce_SEG: "T1ce + Seg",
  T2_SEG:   "T2 + Seg",
  ALL5:     "All + Seg",
};

const GROUP_ORDER = [
  ["FLAIR", "T1", "T1ce", "T2"],           // 4 separate modalities
  ["FLAIR_SEG", "T1_SEG", "T1ce_SEG", "T2_SEG"],  // each modality + seg
  ["ALL4", "ALL5"],                          // all combined
];

const MODALITY_ACCENT: Record<string, string> = {
  FLAIR:    "#e07820",
  T1:       "#3fa0f0",
  T1ce:     "#e04040",
  T2:       "#30c060",
  T1_FLAIR: "#b06030",
  T1_T1ce:  "#a04040",
  T1_T2:    "#508040",
  ALL4:     "#8060e0",
  FLAIR_SEG: "#e07820",
  T1_SEG:   "#3fa0f0",
  T1ce_SEG: "#e04040",
  T2_SEG:   "#30c060",
  ALL5:     "#b050c0",
};

// ── Component ─────────────────────────────────────────────────────────────────

export default function Viewer3D({
  sessionId,
  availableViews,
  hasSeg,
  scanName,
}: Viewer3DProps) {
  const [renders, setRenders] = useState<RenderMap>({});
  const [activeView, setActiveView] = useState<string>("T1");
  const [streamDone, setStreamDone] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Cache of already-parsed Plotly figures — avoids re-parsing same JSON on tab switch
  const parsedCache = useRef<Record<string, { data: Plotly.Data[]; layout: Partial<Plotly.Layout> }>>({});
  const parseRafRef = useRef<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // ── Start streaming when sessionId changes ────────────────────────────────

  useEffect(() => {
    if (!sessionId) return;

    // Cancel any previous stream
    abortRef.current?.abort();
    setRenders({});
    setStreamDone(false);
    setError(null);

    // Start with the lightest single-modality view for fast first render
    const preferred = ["T1", "FLAIR", "T2", "T1ce", "ALL4"];
    const first = preferred.find((v) => availableViews.includes(v)) ?? availableViews[0];
    setActiveView(first ?? "T1");
    // Clear the parsed cache for the new session
    parsedCache.current = {};

    const controller = streamAllRenders(
      sessionId,
      (result: RenderResult) => {
        setRenders((prev) => ({ ...prev, [result.view]: result.figure }));
      },
      (total) => {
        setStreamDone(true);
        console.info(`[Viewer3D] All ${total} renders complete.`);
      },
      (view, message) => {
        console.error(`[Viewer3D] Render error for ${view}: ${message}`);
      }
    );

    abortRef.current = controller;
    return () => controller.abort();
  }, [sessionId]);

  // ── On-demand single render (click on a tab not yet loaded) ───────────────

  const handleTabClick = useCallback(
    async (view: string) => {
      setActiveView(view);
      if (renders[view] || !sessionId) return;

      try {
        const result = await fetchSingleRender(sessionId, view);
        setRenders((prev) => ({ ...prev, [view]: result.figure }));
      } catch (err) {
        setError(`Failed to render ${view}: ${(err as Error).message}`);
      }
    },
    [renders, sessionId]
  );

  // ── Parse Plotly figure JSON — deferred off the main thread ─────────────
  // useMemo would block the UI thread for 200-500ms on heavy JSON (ALL4 = ~20MB).
  // Instead we defer with requestAnimationFrame so React can finish painting first.

  const [plotData, setPlotData] = useState<{
    data: Plotly.Data[];
    layout: Partial<Plotly.Layout>;
  } | null>(null);

  useEffect(() => {
    const raw = renders[activeView];

    // If we already parsed this view, reuse it immediately (no freeze)
    if (parsedCache.current[activeView]) {
      setPlotData(parsedCache.current[activeView]);
      return;
    }

    if (!raw) {
      setPlotData(null);
      return;
    }

    // Show loading skeleton while we defer the parse
    setPlotData(null);

    // Cancel any in-flight parse for a previous view
    if (parseRafRef.current !== null) cancelAnimationFrame(parseRafRef.current);

    // Defer to the next animation frame — React paints the skeleton first
    parseRafRef.current = requestAnimationFrame(() => {
      // Use setTimeout(0) as a second deferral for very large JSON
      setTimeout(() => {
        try {
          const parsed = JSON.parse(raw) as {
            data: Plotly.Data[];
            layout: Partial<Plotly.Layout>;
          };
          parsedCache.current[activeView] = parsed;   // cache for instant re-use
          setPlotData(parsed);
        } catch {
          setPlotData(null);
        }
        parseRafRef.current = null;
      }, 0);
    });

    return () => {
      if (parseRafRef.current !== null) cancelAnimationFrame(parseRafRef.current);
    };
  }, [renders, activeView]);

  // ── Render ────────────────────────────────────────────────────────────────

  if (!sessionId) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-gray-500 gap-3">
        <BrainCircuit className="w-12 h-12 opacity-20" />
        <p className="text-sm text-gray-400">Load a .npy scan to begin 3D visualization</p>
      </div>
    );
  }

  const grouped = GROUP_ORDER
    .map((group) => group.filter((v) => availableViews.includes(v)))
    .filter((g) => g.length > 0);

  const isLoading = (view: string) => !renders[view];

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-[#06060e]">
      {/* ── Tab bar ────────────────────────────────────────────────────────── */}
      <div className="flex items-center gap-0.5 px-2 py-1.5 border-b border-white/5 overflow-x-auto shrink-0 scrollbar-none">
        {grouped.map((group, gi) => (
          <React.Fragment key={gi}>
            {gi > 0 && <div className="w-px h-4 bg-white/10 mx-1 shrink-0" />}
            {group.map((view) => {
              const active = view === activeView;
              const loaded = !!renders[view];
              const accent = MODALITY_ACCENT[view] ?? "#8888aa";
              return (
                <button
                  key={view}
                  onClick={() => handleTabClick(view)}
                  className={`
                    relative shrink-0 px-2.5 py-1 rounded text-[10px] font-bold
                    transition-all duration-150 flex items-center gap-1.5
                    ${active
                      ? "bg-white/10 text-white ring-1 ring-white/15"
                      : "text-gray-500 hover:text-gray-300 hover:bg-white/5"}
                  `}
                  style={active ? { borderBottom: `2px solid ${accent}` } : {}}
                >
                  {!loaded && (
                    <span className="w-1.5 h-1.5 rounded-full bg-amber-500/70 animate-pulse" />
                  )}
                  {VIEW_LABELS[view] ?? view}
                </button>
              );
            })}
          </React.Fragment>
        ))}

        {/* stream progress pill */}
        <div className="ml-auto shrink-0">
          {!streamDone ? (
            <div className="flex items-center gap-1.5 text-[10px] text-amber-400/80 font-mono px-2">
              <Loader2 className="w-3 h-3 animate-spin" />
              {Object.keys(renders).length}/{availableViews.length}
            </div>
          ) : (
            <div className="text-[10px] text-emerald-500/70 font-mono px-2">
              ✓ {availableViews.length} views
            </div>
          )}
        </div>
      </div>

      {/* ── 3D Viewer ──────────────────────────────────────────────────────── */}
      <div className="flex-1 relative overflow-hidden">
        {error && (
          <div className="absolute top-2 right-2 z-20 bg-red-900/80 text-red-200 text-xs
                          px-3 py-2 rounded flex items-center gap-2 max-w-xs">
            <AlertCircle className="w-3.5 h-3.5 shrink-0" />
            {error}
          </div>
        )}

        {plotData ? (
          <Plot
            data={plotData.data}
            layout={{
              ...plotData.layout,
              autosize: true,
              width: undefined,
              height: undefined,
              // Keep camera position when switching between views
              uirevision: "neurodash-camera",
            }}
            config={{
              displayModeBar: true,
              displaylogo: false,
              // Render at device pixel ratio 1 — halves GPU memory vs default 2
              plotGlPixelRatio: 1,
              modeBarButtonsToRemove: [
                "sendDataToCloud",
                "select2d",
                "lasso2d",
                "resetCameraLastSave3d",
              ],
              responsive: true,
            }}
            style={{ width: "100%", height: "100%" }}
            useResizeHandler
          />
        ) : (
          /* Loading skeleton for active view */
          <div className="flex-1 flex flex-col items-center justify-center gap-4 h-full">
            <div className="relative">
              <div className="w-24 h-24 rounded-full border-2 border-blue-500/20 animate-pulse" />
              <div className="absolute inset-3 rounded-full border-2 border-blue-400/15 animate-pulse" style={{ animationDelay: "150ms" }} />
              <BrainCircuit className="absolute inset-0 m-auto w-10 h-10 text-blue-500/50" />
            </div>
            <div className="text-center">
              <p className="text-xs text-gray-400 font-mono tracking-wider">
                RENDERING {activeView.replace(/_/g, " + ")}
              </p>
              <p className="text-[10px] text-gray-600 mt-1">{scanName}</p>
            </div>
          </div>
        )}

        {/* Mini legend overlay */}
        {plotData && (
          <div className="absolute bottom-3 left-3 text-[9px] font-mono text-gray-500 space-y-0.5 select-none pointer-events-none">
            <div>scroll — zoom  |  drag — orbit  |  shift+drag — pan</div>
          </div>
        )}
      </div>
    </div>
  );
}