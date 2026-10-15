/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */
import Viewer3D from './Viewer3D';
import { createSession, deleteSession, type SessionMeta } from './renderApi';
import {
  generateReport,
  streamReport,
  fetchSlices,
  deleteReport,
  type ReportMeta,
  type SliceImage,
} from './reportApi';


import React, { useState, useEffect, useRef } from 'react';
import {
  ChevronLeft, ChevronRight, CheckCircle2,
  Brain, FileImage, Layers, Loader2, Activity, Play,
  ZoomIn, ZoomOut, Rotate3d, Sparkles, ChevronDown,
  FolderOpen, FolderArchive, Database, Image, Folder,
  File, ChevronRight as FolderArrow
} from 'lucide-react';

const ALL_MODALITIES = ['T1', 'T1c', 'T2', 'FLAIR'];

// File tree node type
interface FileNode {
  name: string;
  type: 'folder' | 'file';
  children?: FileNode[];
}

// Build a tree structure from flat file list
function buildFileTree(files: File[]): FileNode[] {
  const root: FileNode[] = [];
  const folderMap: Record<string, FileNode> = {};

  files.forEach(file => {
    const parts = ((file as any).webkitRelativePath as string || file.name).split('/');
    // parts[0] = root folder, parts[1..n-1] = subfolders, parts[n] = filename
    let current = root;
    for (let i = 0; i < parts.length; i++) {
      const part = parts[i];
      const isLast = i === parts.length - 1;
      const pathKey = parts.slice(0, i + 1).join('/');

      if (isLast) {
        current.push({ name: part, type: 'file' });
      } else {
        if (!folderMap[pathKey]) {
          const node: FileNode = { name: part, type: 'folder', children: [] };
          folderMap[pathKey] = node;
          current.push(node);
        }
        current = folderMap[pathKey].children!;
      }
    }
  });

  return root;
}

// Recursive tree renderer
function FileTreeNode({ node, depth = 0, segBaseName = '', hasSegTick = false }: {
  node: FileNode;
  depth?: number;
  key?: number;
  segBaseName?: string;
  hasSegTick?: boolean;
}) {
  const [open, setOpen] = useState(depth <= 1);
  const indent = depth * 12;

  // Show tick on .npy / .nii / .nii.gz files that belong to the current patient
  // segBaseName is the patient ID (e.g. "BraTS2021_00003") — tick any file whose name contains it
  const isRelevantFileType = node.name.endsWith('.npy') || node.name.endsWith('.nii') || node.name.endsWith('.nii.gz') || node.name.endsWith('.mha') || node.name.endsWith('.mhd');
  const isSegFile = hasSegTick && node.type === 'file' && isRelevantFileType
    && (segBaseName === '' || node.name.includes(segBaseName));

  if (node.type === 'folder') {
    return (
      <div>
        <button
          onClick={() => setOpen(o => !o)}
          className="w-full flex items-center gap-1.5 py-0.5 px-2 hover:bg-gray-100 rounded text-left text-xs text-gray-700"
          style={{ paddingLeft: `${8 + indent}px` }}
        >
          <span className={`transition-transform ${open ? 'rotate-90' : ''}`}>
            <FolderArrow className="w-3 h-3 text-gray-400 shrink-0" />
          </span>
          <Folder className="w-3.5 h-3.5 text-amber-500 shrink-0" />
          <span className="truncate font-medium">{node.name}</span>
        </button>
        {open && node.children && (
          <div>
            {node.children.map((child, i) => (
              <FileTreeNode key={i} node={child} depth={depth + 1} segBaseName={segBaseName} hasSegTick={hasSegTick} />
            ))}
          </div>
        )}
      </div>
    );
  }

  return (
    <div
      className={`flex items-center gap-1.5 py-0.5 px-2 hover:bg-gray-100 rounded text-xs cursor-default ${isSegFile ? 'text-emerald-700 bg-emerald-50/60' : 'text-gray-600'}`}
      style={{ paddingLeft: `${8 + indent}px` }}
    >
      <File className={`w-3.5 h-3.5 shrink-0 ${isSegFile ? 'text-emerald-500' : 'text-blue-400'}`} />
      <span className="truncate flex-1">{node.name}</span>
      {isSegFile && (
        <CheckCircle2 className="w-3 h-3 text-emerald-500 shrink-0 ml-auto" />
      )}
    </div>
  );
}

export default function App() {
  // --- Folder & Data Navigation State ---
  const dataFileInputRef = useRef<HTMLInputElement>(null);
  const segFileInputRef = useRef<HTMLInputElement>(null);

  const [isDataLoaded, setIsDataLoaded] = useState(false);
  const [dataFolderName, setDataFolderName] = useState('');
  const [scanFiles, setScanFiles] = useState<File[]>([]);

  const [isSegLoaded, setIsSegLoaded] = useState(false);
  const [segFolderName, setSegFolderName] = useState('');
  const [segFiles, setSegFiles] = useState<File[]>([]);

  const [currentIndex, setCurrentIndex] = useState(1);
  const [totalScans, setTotalScans] = useState(0);

  // Track which scans have been processed by AI
  const [predictions, setPredictions] = useState<Record<number, boolean>>({});
  const hasPrediction = predictions[currentIndex] || false;

  //use state
  const [sessionId, setSessionId]       = useState<string | null>(null);
  const [sessionMeta, setSessionMeta]   = useState<SessionMeta | null>(null);
  const [renderReady, setRenderReady]   = useState(false);

  // UI States
  const [activeModalities, setActiveModalities] = useState<string[]>(['T1', 'T1c']);
  const [isProcessing, setIsProcessing] = useState(false);
  const [aiLoading, setAiLoading] = useState(false);
  const [aiExplanation, setAiExplanation] = useState<string | null>(null);
  const [expandedSection, setExpandedSection] = useState<'original' | 'segmentation' | null>('original');

  // ── MedGemma report state ──────────────────────────────────────────────────
  const [reportId, setReportId]           = useState<string | null>(null);
  const [reportMd, setReportMd]           = useState<string>('');
  const [reportMeta, setReportMeta]       = useState<ReportMeta | null>(null);
  const [reportStatus, setReportStatus]   = useState<'idle'|'uploading'|'streaming'|'done'|'error'>('idle');
  const [reportError, setReportError]     = useState<string | null>(null);
  const [reportSection, setReportSection] = useState<string>('');
  const [sliceImages, setSliceImages]     = useState<SliceImage[]>([]);
  const [showSlices, setShowSlices]       = useState(false);
  const _stopStream                       = useRef<(() => void) | null>(null);

  // Path input state (editable)
  const [datasetPath, setDatasetPath] = useState('/path/to/dataset');

  // File trees
  const [dataFileTree, setDataFileTree] = useState<FileNode[]>([]);
  const [segFileTree, setSegFileTree] = useState<FileNode[]>([]);

  // --- Dynamic Scan Name & Ground Truth Matching Logic ---
  const currentScanFile = scanFiles[currentIndex - 1];
  const currentScanName = currentScanFile
    ? currentScanFile.name
    : 'No Slide Loaded';

  // Extract patient ID: strip extension(s) then remove known suffixes like _image, _scan, _t1, etc.
  // e.g. "BraTS2021_00003_image.npy" -> "BraTS2021_00003"
  //      "Patient_001_T1.nii.gz"    -> "Patient_001"
  const stripExtensions = (name: string) => name.replace(/\.nii\.gz$|\.nii$|\.npy$|\.mha$|\.mhd$|\.dcm$/i, '');
  const extractPatientId = (name: string): string => {
    const noExt = stripExtensions(name);
    // Remove common suffixes: _image, _scan, _volume, _t1, _t1c, _t2, _flair, _data
    return noExt.replace(/_?(image|scan|volume|t1c|t1|t2|flair|data|input)$/i, '');
  };

  const currentBaseName = currentScanFile ? stripExtensions(currentScanFile.name) : '';
  const currentPatientId = currentScanFile ? extractPatientId(currentScanFile.name) : '';

  // Match seg files: first try exact base name, then by patient ID prefix
  const hasGroundTruthForCurrentScan = isSegLoaded && segFiles.some(file => {
    const segBase = stripExtensions(file.name);
    const segPatient = extractPatientId(file.name);
    return segBase === currentBaseName
      || segPatient === currentPatientId
      || file.name.includes(currentPatientId)
      || (currentPatientId.length > 3 && segBase.startsWith(currentPatientId));
  });

  // For tree tick: use patient ID so both _image and _mask files highlight together
  const segTickId = currentPatientId || currentBaseName;

  // --- Real File Loading Logic ---
  const handleDataFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const files: File[] = Array.from(e.target.files ?? new FileList());
      if (files.length === 0) return;

      const folderName = (files[0] as any).webkitRelativePath?.split('/')[0] || 'Dataset';
      setDataFolderName(folderName);
      const npyFiles = files.filter(f => f.name.endsWith('.npy'));
      setScanFiles(npyFiles);
      setTotalScans(npyFiles.length);
      setCurrentIndex(1);
      setIsDataLoaded(true);
      setPredictions({});
      setDatasetPath(folderName);
      setDataFileTree(buildFileTree(files));

      // Upload first .npy to the Render API
      loadSessionForIndex(1, npyFiles);
      // const firstNpy = files.find(f => f.name.endsWith('.npy'));
      // if (firstNpy) {
      //   setRenderReady(false);
      //   if (sessionId) deleteSession(sessionId).catch(() => {});
      //   const pid = extractPatientId(firstNpy.name);
      //   const matchingSeg = segFiles.find(f => f.name.includes(pid)) ?? null;
      //   createSession(firstNpy, matchingSeg)
      //     .then(meta => { setSessionId(meta.session_id); setSessionMeta(meta); setRenderReady(true); })
      //     .catch(err => console.error('Render API:', err));
      //   // createSession(firstNpy, null)
      //   //   .then(meta => { setSessionId(meta.session_id); setSessionMeta(meta); setRenderReady(true); })
      //   //   .catch(err => console.error('Render API:', err));
      // }
    };

  const handleSegFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files: File[] = Array.from(e.target.files ?? new FileList());
    if (files.length === 0) return;

    const folderName = (files[0] as any).webkitRelativePath?.split('/')[0] || 'Masks';
    setSegFolderName(folderName);
    setSegFiles(files);
    setIsSegLoaded(true);
    setSegFileTree(buildFileTree(files));

    const currentFile = scanFiles[currentIndex - 1];
    if (currentFile?.name.endsWith('.npy')) {
      const pid = extractPatientId(currentFile.name);
      const matchingSeg = files.find(f => f.name.includes(pid)) ?? null;
      if (matchingSeg) {
        setRenderReady(false);
        if (sessionId) deleteSession(sessionId).catch(() => {});
        createSession(currentFile, matchingSeg)
          .then(meta => { setSessionId(meta.session_id); setSessionMeta(meta); setRenderReady(true); })
          .catch(err => console.error('Render API:', err));
      }
    }
  };

  const loadSessionForIndex = (index: number, fileList: File[]) => {
    const file = fileList[index - 1];
    if (!file?.name.endsWith('.npy')) return;

    setRenderReady(false);
    if (sessionId) deleteSession(sessionId).catch(() => {});

    const pid = extractPatientId(file.name);
    const matchingSeg = segFiles.find(f => f.name.includes(pid)) ?? null;

    createSession(file, matchingSeg)
      .then(meta => {
        setSessionId(meta.session_id);
        setSessionMeta(meta);
        setRenderReady(true);
      })
      .catch(err => console.error('Render API:', err));
  };

  const handlePrev = () => {
    if (currentIndex > 1) {
      const newIndex = currentIndex - 1;
      setCurrentIndex(newIndex);
      loadSessionForIndex(newIndex, scanFiles);
    }
  };

  const handleNext = () => {
    if (currentIndex < totalScans) {
      const newIndex = currentIndex + 1;
      setCurrentIndex(newIndex);
      loadSessionForIndex(newIndex, scanFiles);
    }
  };

  const toggleModality = (mod: string) => {
    setActiveModalities(prev =>
      prev.includes(mod) ? prev.filter(m => m !== mod) : [...prev, mod]
    );
  };

  const handleProcess = () => {
    setIsProcessing(true);
    setTimeout(() => {
      setPredictions(prev => ({ ...prev, [currentIndex]: true }));
      setIsProcessing(false);
      setExpandedSection('segmentation');
    }, 2000);
  };

  const handleGenerateExplanation = async () => {
    // Get the current scan file (data) + matching seg file
    const dataFile = scanFiles[currentIndex - 1];
    if (!dataFile) {
      setReportError('No scan file loaded.');
      return;
    }

    //Extra
    const segFile = segFiles.find(f => {
      const segBase = f.name.replace(/\.nii\.gz$|\.nii$|\.npy$|\.mha$|\.mhd$/i, '');
      return segBase.includes(currentPatientId) || currentPatientId.includes(segBase.split('_')[0]);
    }) ?? null;

    if (!segFile) {
      setReportStatus('error');
      setReportError('No segmentation file found for this patient. Load a matching seg .npy file to generate a report.');
      return;
    }
    //Extra

    // Find the best-matching seg file for the current scan
    // const segFile = segFiles.find(f => {
    //   const segBase = f.name.replace(/\.nii\.gz$|\.nii$|\.npy$|\.mha$|\.mhd$/i, '');
    //   return segBase.includes(currentPatientId) || currentPatientId.includes(segBase.split('_')[0]);
    // }) ?? null;

    // Cancel any running stream
    _stopStream.current?.();
    _stopStream.current = null;

    // Reset state
    setReportStatus('uploading');
    setReportMd('');
    setReportError(null);
    setReportSection('');
    setAiLoading(true);
    setAiExplanation(null);

    try {
      // 1. Upload files + extract slices
      const { report_id, meta, n_images } = await generateReport(dataFile, segFile);
      setReportId(report_id);
      setReportMeta(meta);
      setReportStatus('streaming');

      // 2. Start SSE stream
      const stop = streamReport(report_id, {
        onToken: (t) => {
          setReportMd(prev => prev + t);
        },
        onProgress: (section, title) => {
          setReportSection(title);
        },
        onDone: async (md, finalMeta) => {
          setReportMd(md);
          setReportMeta(finalMeta);
          setReportStatus('done');
          setAiLoading(false);
          setReportSection('');
          // Also fetch the slice preview images
          try {
            const imgs = await fetchSlices(report_id);
            setSliceImages(imgs);
          } catch { /* non-critical */ }
        },
        onError: (msg) => {
          setReportError(msg);
          setReportStatus('error');
          setAiLoading(false);
        },
      });
      _stopStream.current = stop;

    } catch (err: any) {
      setReportError(err.message ?? 'Failed to contact report API.');
      setReportStatus('error');
      setAiLoading(false);
    }
  };

  // --- Dynamic Resizing Handlers ---
  const [rightSidebarWidth, setRightSidebarWidth] = useState(220);
  const [isDraggingSidebar, setIsDraggingSidebar] = useState(false);

  const [aiHeight, setAiHeight] = useState(220);
  const [isDraggingAi, setIsDraggingAi] = useState(false);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (isDraggingSidebar) {
        const newWidth = document.body.clientWidth - e.clientX;
        if (newWidth > 160 && newWidth < 500) {
          setRightSidebarWidth(newWidth);
        }
      }
      if (isDraggingAi) {
        const newHeight = window.innerHeight - e.clientY;
        if (newHeight > 120 && newHeight < 600) {
          setAiHeight(newHeight);
        }
      }
    };

    const handleMouseUp = () => {
      setIsDraggingSidebar(false);
      setIsDraggingAi(false);
    };

    if (isDraggingSidebar || isDraggingAi) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDraggingSidebar, isDraggingAi]);

  // Combined file tree for right panel
  const combinedTree: FileNode[] = [
    ...(dataFileTree.length > 0 ? dataFileTree : []),
    ...(segFileTree.length > 0 ? segFileTree : []),
  ];

  return (
    <div className="flex h-screen w-full bg-gray-100 text-gray-900 font-sans overflow-hidden">

      {/* Hidden File Inputs for Directory Selection */}
      <input
        type="file"
        ref={dataFileInputRef}
        onChange={handleDataFileChange}
        className="hidden"
        {...({ webkitdirectory: "true", directory: "true" } as any)}
      />
      <input
        type="file"
        ref={segFileInputRef}
        onChange={handleSegFileChange}
        className="hidden"
        {...({ webkitdirectory: "true", directory: "true" } as any)}
      />

      {/* ── Left Narrow Icon Sidebar ── */}
      <aside className="w-11 bg-slate-900 border-r border-slate-800 flex flex-col items-center shrink-0 z-10 py-2 gap-1">
        {/* Logo mark */}
        <div className="w-8 h-8 flex items-center justify-center mb-2">
          <Brain className="w-5 h-5 text-blue-500" />
        </div>

        <div className="w-full h-px bg-slate-800 mb-1" />

        {/* Load Data */}
        <button
          title={isDataLoaded ? `Data: ${dataFolderName}` : 'Load Data Folder'}
          onClick={() => dataFileInputRef.current?.click()}
          className={`w-8 h-8 flex items-center justify-center rounded-md transition-colors ${
            isDataLoaded ? 'bg-blue-600 text-white' : 'text-slate-400 hover:bg-slate-700 hover:text-white'
          }`}
        >
          <Database className="w-4 h-4" />
        </button>

        {/* Load Segmentation — functional */}
        <button
          title={isSegLoaded ? `Seg: ${segFolderName}` : 'Load Segmentation Folder'}
          onClick={() => segFileInputRef.current?.click()}
          disabled={!isDataLoaded}
          className={`w-8 h-8 flex items-center justify-center rounded-md transition-colors disabled:opacity-30 disabled:cursor-not-allowed ${
            isSegLoaded
              ? (hasGroundTruthForCurrentScan ? 'bg-emerald-600 text-white' : 'bg-amber-600 text-white')
              : 'text-slate-400 hover:bg-slate-700 hover:text-white'
          }`}
        >
          <FolderArchive className="w-4 h-4" />
        </button>

        {/* Image viewer icon */}
        <button
          title="Image View"
          className="w-8 h-8 flex items-center justify-center rounded-md text-slate-400 hover:bg-slate-700 hover:text-white transition-colors"
        >
          <Image className="w-4 h-4" />
        </button>

        {/* Layers icon */}
        <button
          title="Layers"
          className="w-8 h-8 flex items-center justify-center rounded-md text-slate-400 hover:bg-slate-700 hover:text-white transition-colors"
        >
          <Layers className="w-4 h-4" />
        </button>

        {/* Spacer pushes arrows to bottom */}
        <div className="flex-1" />

        <div className="w-full h-px bg-slate-800 mb-1" />

        {/* Navigation arrows — left first, then right */}
        <button
          onClick={handlePrev}
          disabled={!isDataLoaded || currentIndex === 1}
          title="Previous Scan"
          className="w-8 h-8 flex items-center justify-center rounded-md text-slate-400 hover:bg-slate-700 hover:text-white disabled:opacity-25 disabled:cursor-not-allowed transition-colors"
        >
          <ChevronLeft className="w-4 h-4" />
        </button>

        <button
          onClick={handleNext}
          disabled={!isDataLoaded || currentIndex === totalScans}
          title="Next Scan"
          className="w-8 h-8 flex items-center justify-center rounded-md text-slate-400 hover:bg-slate-700 hover:text-white disabled:opacity-25 disabled:cursor-not-allowed transition-colors mb-2"
        >
          <ChevronRight className="w-4 h-4" />
        </button>
      </aside>

      {/* ── Main Content Area ── */}
      <div className="flex flex-col flex-1 min-w-0 bg-gray-50 relative">

        {/* ── Top Bar (compact, matches screenshot) ── */}
        <header className="h-10 bg-white border-b border-gray-200 flex items-center px-3 gap-3 shadow-sm z-10 shrink-0">
          {/* VIEWING label + scan name */}
          <div className="flex items-center gap-2 shrink-0">
            <span className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">VIEWING</span>
            <span className="text-xs font-semibold text-gray-700 max-w-[160px] truncate">
              {isDataLoaded ? currentScanName : 'No Slide Loaded'}
            </span>
          </div>

          <div className="h-4 w-px bg-gray-200 shrink-0" />

          {/* PATH input — center */}
          <div className="flex items-center gap-1.5 flex-1 min-w-0">
            <span className="text-[10px] font-bold text-gray-400 uppercase tracking-widest shrink-0">PATH</span>
            <input
              type="text"
              value={datasetPath}
              onChange={e => setDatasetPath(e.target.value)}
              className="flex-1 min-w-0 text-xs text-gray-600 bg-gray-50 border border-gray-200 rounded px-2 py-0.5 focus:outline-none focus:ring-1 focus:ring-blue-400 font-mono"
              placeholder="/path/to/dataset"
            />
          </div>

          {/* Segmentation Load Button — shown when data is loaded */}
          {isDataLoaded && (
            <>
              <div className="h-4 w-px bg-gray-200 shrink-0" />
              <button
                onClick={() => segFileInputRef.current?.click()}
                title={isSegLoaded ? `Seg loaded: ${segFolderName}` : 'Load Segmentation Folder'}
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-[10px] font-bold border transition-all shrink-0 ${
                  isSegLoaded
                    ? hasGroundTruthForCurrentScan
                      ? 'bg-emerald-50 border-emerald-300 text-emerald-700 hover:bg-emerald-100'
                      : 'bg-amber-50 border-amber-300 text-amber-700 hover:bg-amber-100'
                    : 'bg-gray-50 border-gray-200 text-gray-500 hover:bg-gray-100'
                }`}
              >
                <FolderArchive className="w-3.5 h-3.5" />
                {isSegLoaded
                  ? hasGroundTruthForCurrentScan
                    ? <><CheckCircle2 className="w-3 h-3" /><span>SEG LOADED</span></>
                    : <span>SEG MISSING</span>
                  : <span>LOAD SEG</span>}
              </button>
            </>
          )}

          {/* Modality toggles */}
          <div className="flex items-center gap-1 shrink-0">
            {ALL_MODALITIES.map(mod => {
              const isActive = activeModalities.includes(mod);
              return (
                <button
                  key={mod}
                  onClick={() => toggleModality(mod)}
                  disabled={!isDataLoaded}
                  className={`px-2.5 py-0.5 rounded text-[10px] font-bold transition-all disabled:opacity-40 disabled:cursor-not-allowed ${
                    isActive
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-500 border border-gray-200 hover:bg-gray-200'
                  }`}
                >
                  {mod}
                </button>
              );
            })}
          </div>

          <div className="h-4 w-px bg-gray-200 shrink-0" />

          {/* Icon buttons + START PROCESSING */}
          <div className="flex items-center gap-1.5 shrink-0">
            {/* Zoom icon */}
            <button
              title="Zoom"
              className="w-7 h-7 flex items-center justify-center rounded text-gray-500 hover:bg-gray-100 transition-colors"
            >
              <ZoomIn className="w-4 h-4" />
            </button>
            {/* Rotate icon */}
            <button
              title="Rotate"
              className="w-7 h-7 flex items-center justify-center rounded text-gray-500 hover:bg-gray-100 transition-colors"
            >
              <Rotate3d className="w-4 h-4" />
            </button>

            <button
              onClick={handleProcess}
              disabled={!isDataLoaded || isProcessing}
              className="flex items-center gap-1.5 bg-emerald-600 hover:bg-emerald-700 text-white px-3 py-1 rounded text-[11px] font-bold shadow transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isProcessing ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Play className="w-3 h-3 fill-current" />
              )}
              START PROCESSING
            </button>
          </div>
        </header>

        {/* ── Workspace ── */}
        <main className="flex-1 flex overflow-hidden">

          {/* Dashboard / Vis Area */}
          <div className="flex-1 relative flex items-center justify-center bg-[#f0f0f0] overflow-hidden flex-col group" style={{ backgroundImage: 'radial-gradient(circle at 1.5px 1.5px, #d1d5db 1px, transparent 0)', backgroundSize: '24px 24px' }}>

            {isDataLoaded ? (
              <>
                {/* 3D Tools — floating left */}
                <div className="absolute left-3 top-[42px] flex flex-col gap-1.5 z-20">
                  <button className="p-2 bg-white/80 hover:bg-white text-gray-500 rounded-lg shadow border border-gray-200 transition-colors"><ZoomIn className="w-4 h-4"/></button>
                  <button className="p-2 bg-white/80 hover:bg-white text-gray-500 rounded-lg shadow border border-gray-200 transition-colors"><ZoomOut className="w-4 h-4"/></button>
                  <button className="p-2 bg-white/80 hover:bg-white text-gray-500 rounded-lg shadow border border-gray-200 transition-colors"><Rotate3d className="w-4 h-4"/></button>
                </div>

                {/* Scan info — top left */}
                <div className="absolute left-14 top-[42px] bg-white/80 backdrop-blur rounded-lg px-3 py-1.5 border border-gray-200 shadow z-20">
                  <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Current Scan</p>
                  <p className="text-xs font-semibold text-gray-800 truncate max-w-[200px]">{currentScanName}</p>
                  <p className="text-[10px] text-gray-400 font-mono">{currentIndex} / {totalScans}</p>
                </div>

                {/* Simulated 3D Volume Viewer */}
                {/* Real 3D Volume Viewer */}
                <div className="relative w-full flex-1 flex overflow-hidden">
                  {renderReady && sessionId && sessionMeta ? (
                    <Viewer3D
                      sessionId={sessionId}
                      availableViews={sessionMeta.available_views}
                      hasSeg={sessionMeta.has_segmentation}
                      scanName={currentScanName}
                    />
                  ) : isDataLoaded ? (
                    <div className="flex-1 flex flex-col items-center justify-center gap-4 h-full">
                      <div className="w-16 h-16 rounded-full border-2 border-blue-500/30 animate-spin border-t-blue-500" />
                      <p className="text-xs text-gray-400 font-mono tracking-wider">UPLOADING TO RENDER API…</p>
                    </div>
                  ) : (
                    <div className="text-gray-500 flex flex-col items-center justify-center z-10 bg-white/70 p-10 rounded-2xl border border-gray-200 backdrop-blur-sm shadow">
                      <FileImage className="w-14 h-14 opacity-25 mb-5 text-gray-400" />
                      <h3 className="text-base font-semibold text-gray-600 mb-1">Please load an image or folder to begin</h3>
                      <p className="text-xs text-gray-400 text-center max-w-xs mt-1">Use the left sidebar icons to load your data and segmentation folders.</p>
                    </div>
                  )}
                </div>

                {/* <div className="relative w-full flex-1 flex items-center justify-center">
                  {activeModalities.length > 0 ? (
                    <div className="relative w-80 h-80 flex items-center justify-center">
                      <div className="absolute inset-0 border-2 border-blue-400/20 rounded-full animate-pulse z-0" />
                      <div className="absolute inset-4 border-2 border-blue-300/20 rounded-full z-0" />
                      <div className="z-10 w-64 h-72 bg-gray-800 rounded-xl shadow-2xl relative overflow-hidden ring-1 ring-white/10 flex flex-col items-center justify-center">
                        <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent"></div>
                        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 flex flex-col items-center z-10 w-full px-4">
                          <Brain className="w-16 h-16 text-blue-500 opacity-80 mb-4 drop-shadow-lg" />
                          <div className="text-white font-mono text-[10px] tracking-[0.2em] opacity-80 text-center uppercase">
                            <p className="text-blue-300 font-bold mb-2 truncate max-w-[200px]">3D: {currentScanName}</p>
                            <p>Modalities: {activeModalities.join(' + ')}</p>
                          </div>
                          {(hasGroundTruthForCurrentScan || hasPrediction) && (
                            <div className="mt-4 flex items-center justify-center gap-2 text-emerald-400 font-bold px-3 py-1 rounded bg-black/20 text-xs text-center border border-emerald-500/20 backdrop-blur-md">
                              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                              SEGMENTATION LOADED
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="text-gray-400 font-medium flex flex-col items-center gap-3 z-10">
                      <Layers className="w-12 h-12 opacity-40" />
                      <p className="text-sm">Select a modality to begin visualization.</p>
                    </div>
                  )}
                </div> */}

                {activeModalities.length > 0 && (
                  <div className="mb-6 flex space-x-6 z-10">
                    <div className="text-center">
                      <div className="text-[10px] text-gray-400 font-bold tracking-widest">RENDER MODE</div>
                      <div className="text-xl font-light text-gray-800">Full 3D</div>
                    </div>
                    <div className="text-center">
                      <div className="text-[10px] text-gray-400 font-bold tracking-widest">ROTATION</div>
                      <div className="text-xl font-light text-gray-800">45°</div>
                    </div>
                    <div className="text-center">
                      <div className="text-[10px] text-gray-400 font-bold tracking-widest">ZOOM</div>
                      <div className="text-xl font-light text-gray-800">1.4x</div>
                    </div>
                  </div>
                )}
              </>
            ) : (
              // Empty State
              <div className="text-gray-500 flex flex-col items-center justify-center z-10 bg-white/70 p-10 rounded-2xl border border-gray-200 backdrop-blur-sm shadow">
                <FileImage className="w-14 h-14 opacity-25 mb-5 text-gray-400" />
                <h3 className="text-base font-semibold text-gray-600 mb-1">Please load an image or folder to begin</h3>
                <p className="text-xs text-gray-400 text-center max-w-xs mt-1">
                  Use the left sidebar icons to load your data and segmentation folders.
                </p>
              </div>
            )}

            {/* Status bar */}
            <div className="absolute bottom-3 right-3 bg-white/80 backdrop-blur px-2.5 py-1 rounded text-[10px] text-gray-400 font-mono flex items-center gap-1.5 border border-gray-200 shadow">
              <Activity className="w-3 h-3 text-emerald-500" />
              API: Connected
            </div>
          </div>

          {/* Sidebar Vertical Resizer Handle */}
          <div
            className="w-1 bg-gray-200 hover:bg-blue-400 active:bg-blue-500 cursor-col-resize transition-all shrink-0 z-20"
            onMouseDown={() => setIsDraggingSidebar(true)}
          />

          {/* ── Right Sidebar — File Browser ── */}
          <aside
            style={{ width: rightSidebarWidth }}
            className="bg-white border-l border-gray-200 flex flex-col shrink-0 z-10 h-full"
          >
            {/* Header with patient nav arrows */}
            <div className="border-b border-gray-200 bg-gray-50 shrink-0">
              {/* Top row: Files label + folder icon */}
              <div className="h-8 px-3 flex items-center justify-between">
                <span className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Files</span>
                <FolderOpen className="w-3.5 h-3.5 text-gray-400" />
              </div>
              {/* Patient navigation row */}
              <div className="px-2 pb-2 flex items-center gap-1">
                <button
                  onClick={handlePrev}
                  disabled={!isDataLoaded || currentIndex === 1}
                  title="Previous Patient"
                  className="w-6 h-6 flex items-center justify-center rounded bg-white border border-gray-200 text-gray-500 hover:bg-gray-100 disabled:opacity-30 disabled:cursor-not-allowed transition-colors shrink-0"
                >
                  <ChevronLeft className="w-3.5 h-3.5" />
                </button>
                <div className="flex-1 min-w-0 text-center">
                  <div className="text-[10px] font-semibold text-gray-700 truncate leading-tight">
                    {isDataLoaded ? currentScanName : 'No file loaded'}
                  </div>
                  <div className="text-[9px] text-gray-400 font-mono">
                    {isDataLoaded ? `${currentIndex} / ${totalScans}` : '—'}
                  </div>
                </div>
                <button
                  onClick={handleNext}
                  disabled={!isDataLoaded || currentIndex === totalScans}
                  title="Next Patient"
                  className="w-6 h-6 flex items-center justify-center rounded bg-white border border-gray-200 text-gray-500 hover:bg-gray-100 disabled:opacity-30 disabled:cursor-not-allowed transition-colors shrink-0"
                >
                  <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            {/* File Tree */}
            <div className="flex-1 overflow-y-auto py-1 text-xs">
              {combinedTree.length === 0 ? (
                <div className="px-3 py-4 text-center text-gray-400 text-[11px]">
                  No files loaded
                </div>
              ) : (
                combinedTree.map((node, i) => (
                  <FileTreeNode
                    key={i}
                    node={node}
                    depth={0}
                    segBaseName={segTickId}
                    hasSegTick={isSegLoaded || hasPrediction}
                  />
                ))
              )}
            </div>

            {/* Horizontal Resizer for AI box */}
            <div
              className="h-1 w-full bg-gray-200 hover:bg-blue-400 cursor-row-resize transition-all shrink-0"
              onMouseDown={() => setIsDraggingAi(true)}
            />

            {/* MedGemma Report Section */}
            <div
              style={{ height: aiHeight }}
              className="bg-white flex flex-col shrink-0 min-h-[160px] border-t border-gray-100"
            >
              {/* Header row */}
              <div className="flex items-center justify-between px-3 pt-2.5 pb-1.5 shrink-0">
                <div className="flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-violet-500" />
                  <label className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">
                    MedGemma Report
                  </label>
                  {reportStatus === 'streaming' && (
                    <span className="text-[9px] text-violet-500 font-medium animate-pulse">
                      {reportSection ? `▸ ${reportSection}` : 'Generating…'}
                    </span>
                  )}
                  {reportStatus === 'done' && (
                    <CheckCircle2 className="w-3 h-3 text-emerald-500" />
                  )}
                </div>
                {/* Slice preview toggle */}
                {sliceImages.length > 0 && (
                  <button
                    onClick={() => setShowSlices(v => !v)}
                    className="text-[9px] text-blue-500 hover:text-blue-700 font-medium"
                  >
                    {showSlices ? 'Hide slices' : `Show ${sliceImages.length} slices`}
                  </button>
                )}
              </div>

              {/* Report content area */}
              <div className="flex-1 overflow-y-auto px-3 pb-2 min-h-0">

                {/* Slice preview strip */}
                {showSlices && sliceImages.length > 0 && (
                  <div className="flex gap-1.5 overflow-x-auto pb-1.5 mb-2">
                    {sliceImages
                      .filter(img => img.label.includes('Mosaic'))
                      .slice(0, 6)
                      .map((img, i) => (
                        <div key={i} className="shrink-0">
                          <img
                            src={`data:image/png;base64,${img.b64_png}`}
                            alt={img.label}
                            title={img.description}
                            className="w-20 h-20 rounded object-cover border border-gray-200"
                          />
                          <div className="text-[8px] text-gray-400 truncate w-20 mt-0.5">{img.label.split('|')[0].trim()}</div>
                        </div>
                      ))}
                  </div>
                )}

                {/* Idle state */}
                {reportStatus === 'idle' && (
                  <div className="flex-1 flex items-center justify-center text-[11px] text-gray-400 text-center py-4">
                    Load a scan and click<br/>
                    <strong className="mx-1 text-gray-500">Generate Report</strong>
                    to run MedGemma.
                  </div>
                )}

                {/* Uploading */}
                {reportStatus === 'uploading' && (
                  <div className="flex flex-col items-center justify-center gap-2 py-4 text-xs text-blue-500">
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Extracting slices from all modalities…
                  </div>
                )}

                {/* Streaming / Done — render markdown inline */}
                {(reportStatus === 'streaming' || reportStatus === 'done') && reportMd && (
                  <div className="text-[11px] text-gray-800 leading-relaxed space-y-1 font-mono whitespace-pre-wrap">
                    {reportMd.split('\n').map((line, i) => {
                      if (line.startsWith('## '))
                        return <div key={i} className="font-bold text-[11px] text-violet-700 mt-2 border-b border-violet-100 pb-0.5">{line.replace('## ', '')}</div>;
                      if (line.startsWith('**') && line.endsWith('**'))
                        return <div key={i} className="font-semibold text-gray-700">{line.replace(/\*\*/g, '')}</div>;
                      if (line.startsWith('- '))
                        return <div key={i} className="pl-3 text-gray-600">• {line.slice(2)}</div>;
                      if (line.startsWith('| '))
                        return <div key={i} className="text-[10px] font-mono text-gray-500 truncate">{line}</div>;
                      return <div key={i} className={line === '' ? 'h-1' : ''}>{line}</div>;
                    })}
                    {reportStatus === 'streaming' && (
                      <span className="inline-block w-1.5 h-3 bg-violet-400 animate-pulse ml-0.5" />
                    )}
                  </div>
                )}

                {/* Error */}
                {reportStatus === 'error' && reportError && (
                  <div className="bg-red-50 border border-red-200 rounded p-2 text-[10px] text-red-700">
                    <strong>Error:</strong> {reportError}
                    <div className="text-red-500 mt-1">Is MedGemma running? Check MEDGEMMA_BACKEND env var.</div>
                  </div>
                )}

                {/* Tumour stats badge strip */}
                {reportMeta?.has_seg && reportMeta.labels && (
                  <div className="flex gap-1.5 flex-wrap mt-2">
                    {Object.values(reportMeta.labels).map((lbl: any, i) => (
                      <span key={i} className="text-[9px] px-1.5 py-0.5 bg-gray-100 rounded text-gray-600 font-mono">
                        {lbl.name}: {lbl.voxels.toLocaleString()} vox
                      </span>
                    ))}
                  </div>
                )}
              </div>

              {/* Generate button */}
              <div className="px-3 pb-3 shrink-0">
                <button
                  onClick={handleGenerateExplanation}
                  disabled={!isDataLoaded || !hasGroundTruthForCurrentScan ||  reportStatus === 'uploading' || reportStatus === 'streaming'}
                  title={!hasGroundTruthForCurrentScan ? 'Load a segmentation file for this patient first' : ''}
                  className="w-full py-2 bg-violet-600 text-white text-[10px] font-bold rounded uppercase tracking-wider hover:bg-violet-700 transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex justify-center items-center gap-1.5"
                >
                  {(reportStatus === 'uploading' || reportStatus === 'streaming')
                    ? <><Loader2 className="w-3.5 h-3.5 animate-spin" /> {reportStatus === 'uploading' ? 'Extracting…' : 'Generating…'}</>
                    : <><Brain className="w-3.5 h-3.5" /> Generate Report</>
                  }
                </button>
              </div>
            </div>
          </aside>
        </main>
      </div>
    </div>
  );
}






















// /**
//  * @license
//  * SPDX-License-Identifier: Apache-2.0
//  */
// import Viewer3D from './Viewer3D';
// import { createSession, deleteSession, type SessionMeta } from './renderApi';


// import React, { useState, useEffect, useRef } from 'react';
// import {
//   ChevronLeft, ChevronRight, CheckCircle2,
//   Brain, FileImage, Layers, Loader2, Activity, Play,
//   ZoomIn, ZoomOut, Rotate3d, Sparkles, ChevronDown,
//   FolderOpen, FolderArchive, Database, Image, Folder,
//   File, ChevronRight as FolderArrow
// } from 'lucide-react';

// const ALL_MODALITIES = ['T1', 'T1c', 'T2', 'FLAIR'];

// // File tree node type
// interface FileNode {
//   name: string;
//   type: 'folder' | 'file';
//   children?: FileNode[];
// }

// // Build a tree structure from flat file list
// function buildFileTree(files: File[]): FileNode[] {
//   const root: FileNode[] = [];
//   const folderMap: Record<string, FileNode> = {};

//   files.forEach(file => {
//     const parts = ((file as any).webkitRelativePath as string || file.name).split('/');
//     // parts[0] = root folder, parts[1..n-1] = subfolders, parts[n] = filename
//     let current = root;
//     for (let i = 0; i < parts.length; i++) {
//       const part = parts[i];
//       const isLast = i === parts.length - 1;
//       const pathKey = parts.slice(0, i + 1).join('/');

//       if (isLast) {
//         current.push({ name: part, type: 'file' });
//       } else {
//         if (!folderMap[pathKey]) {
//           const node: FileNode = { name: part, type: 'folder', children: [] };
//           folderMap[pathKey] = node;
//           current.push(node);
//         }
//         current = folderMap[pathKey].children!;
//       }
//     }
//   });

//   return root;
// }

// // Recursive tree renderer
// function FileTreeNode({ node, depth = 0, segBaseName = '', hasSegTick = false }: {
//   node: FileNode;
//   depth?: number;
//   key?: number;
//   segBaseName?: string;
//   hasSegTick?: boolean;
// }) {
//   const [open, setOpen] = useState(depth <= 1);
//   const indent = depth * 12;

//   // Show tick on .npy / .nii / .nii.gz files that belong to the current patient
//   // segBaseName is the patient ID (e.g. "BraTS2021_00003") — tick any file whose name contains it
//   const isRelevantFileType = node.name.endsWith('.npy') || node.name.endsWith('.nii') || node.name.endsWith('.nii.gz') || node.name.endsWith('.mha') || node.name.endsWith('.mhd');
//   const isSegFile = hasSegTick && node.type === 'file' && isRelevantFileType
//     && (segBaseName === '' || node.name.includes(segBaseName));

//   if (node.type === 'folder') {
//     return (
//       <div>
//         <button
//           onClick={() => setOpen(o => !o)}
//           className="w-full flex items-center gap-1.5 py-0.5 px-2 hover:bg-gray-100 rounded text-left text-xs text-gray-700"
//           style={{ paddingLeft: `${8 + indent}px` }}
//         >
//           <span className={`transition-transform ${open ? 'rotate-90' : ''}`}>
//             <FolderArrow className="w-3 h-3 text-gray-400 shrink-0" />
//           </span>
//           <Folder className="w-3.5 h-3.5 text-amber-500 shrink-0" />
//           <span className="truncate font-medium">{node.name}</span>
//         </button>
//         {open && node.children && (
//           <div>
//             {node.children.map((child, i) => (
//               <FileTreeNode key={i} node={child} depth={depth + 1} segBaseName={segBaseName} hasSegTick={hasSegTick} />
//             ))}
//           </div>
//         )}
//       </div>
//     );
//   }

//   return (
//     <div
//       className={`flex items-center gap-1.5 py-0.5 px-2 hover:bg-gray-100 rounded text-xs cursor-default ${isSegFile ? 'text-emerald-700 bg-emerald-50/60' : 'text-gray-600'}`}
//       style={{ paddingLeft: `${8 + indent}px` }}
//     >
//       <File className={`w-3.5 h-3.5 shrink-0 ${isSegFile ? 'text-emerald-500' : 'text-blue-400'}`} />
//       <span className="truncate flex-1">{node.name}</span>
//       {isSegFile && (
//         <CheckCircle2 className="w-3 h-3 text-emerald-500 shrink-0 ml-auto" />
//       )}
//     </div>
//   );
// }

// export default function App() {
//   // --- Folder & Data Navigation State ---
//   const dataFileInputRef = useRef<HTMLInputElement>(null);
//   const segFileInputRef = useRef<HTMLInputElement>(null);

//   const [isDataLoaded, setIsDataLoaded] = useState(false);
//   const [dataFolderName, setDataFolderName] = useState('');
//   const [scanFiles, setScanFiles] = useState<File[]>([]);

//   const [isSegLoaded, setIsSegLoaded] = useState(false);
//   const [segFolderName, setSegFolderName] = useState('');
//   const [segFiles, setSegFiles] = useState<File[]>([]);

//   const [currentIndex, setCurrentIndex] = useState(1);
//   const [totalScans, setTotalScans] = useState(0);

//   // Track which scans have been processed by AI
//   const [predictions, setPredictions] = useState<Record<number, boolean>>({});
//   const hasPrediction = predictions[currentIndex] || false;

//   //use state
//   const [sessionId, setSessionId]       = useState<string | null>(null);
//   const [sessionMeta, setSessionMeta]   = useState<SessionMeta | null>(null);
//   const [renderReady, setRenderReady]   = useState(false);

//   // UI States
//   const [activeModalities, setActiveModalities] = useState<string[]>(['T1', 'T1c']);
//   const [isProcessing, setIsProcessing] = useState(false);
//   const [aiLoading, setAiLoading] = useState(false);
//   const [aiExplanation, setAiExplanation] = useState<string | null>(null);
//   const [expandedSection, setExpandedSection] = useState<'original' | 'segmentation' | null>('original');

//   // Path input state (editable)
//   const [datasetPath, setDatasetPath] = useState('/path/to/dataset');

//   // File trees
//   const [dataFileTree, setDataFileTree] = useState<FileNode[]>([]);
//   const [segFileTree, setSegFileTree] = useState<FileNode[]>([]);

//   // --- Dynamic Scan Name & Ground Truth Matching Logic ---
//   const currentScanFile = scanFiles[currentIndex - 1];
//   const currentScanName = currentScanFile
//     ? currentScanFile.name
//     : 'No Slide Loaded';

//   // Extract patient ID: strip extension(s) then remove known suffixes like _image, _scan, _t1, etc.
//   // e.g. "BraTS2021_00003_image.npy" -> "BraTS2021_00003"
//   //      "Patient_001_T1.nii.gz"    -> "Patient_001"
//   const stripExtensions = (name: string) => name.replace(/\.nii\.gz$|\.nii$|\.npy$|\.mha$|\.mhd$|\.dcm$/i, '');
//   const extractPatientId = (name: string): string => {
//     const noExt = stripExtensions(name);
//     // Remove common suffixes: _image, _scan, _volume, _t1, _t1c, _t2, _flair, _data
//     return noExt.replace(/_?(image|scan|volume|t1c|t1|t2|flair|data|input)$/i, '');
//   };

//   const currentBaseName = currentScanFile ? stripExtensions(currentScanFile.name) : '';
//   const currentPatientId = currentScanFile ? extractPatientId(currentScanFile.name) : '';

//   // Match seg files: first try exact base name, then by patient ID prefix
//   const hasGroundTruthForCurrentScan = isSegLoaded && segFiles.some(file => {
//     const segBase = stripExtensions(file.name);
//     const segPatient = extractPatientId(file.name);
//     return segBase === currentBaseName
//       || segPatient === currentPatientId
//       || file.name.includes(currentPatientId)
//       || (currentPatientId.length > 3 && segBase.startsWith(currentPatientId));
//   });

//   // For tree tick: use patient ID so both _image and _mask files highlight together
//   const segTickId = currentPatientId || currentBaseName;

//   // --- Real File Loading Logic ---
//   const handleDataFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
//       const files: File[] = Array.from(e.target.files ?? new FileList());
//       if (files.length === 0) return;

//       const folderName = (files[0] as any).webkitRelativePath?.split('/')[0] || 'Dataset';
//       setDataFolderName(folderName);
//       const npyFiles = files.filter(f => f.name.endsWith('.npy'));
//       setScanFiles(npyFiles);
//       setTotalScans(npyFiles.length);
//       setCurrentIndex(1);
//       setIsDataLoaded(true);
//       setPredictions({});
//       setDatasetPath(folderName);
//       setDataFileTree(buildFileTree(files));

//       // Upload first .npy to the Render API
//       loadSessionForIndex(1, npyFiles);
//       // const firstNpy = files.find(f => f.name.endsWith('.npy'));
//       // if (firstNpy) {
//       //   setRenderReady(false);
//       //   if (sessionId) deleteSession(sessionId).catch(() => {});
//       //   const pid = extractPatientId(firstNpy.name);
//       //   const matchingSeg = segFiles.find(f => f.name.includes(pid)) ?? null;
//       //   createSession(firstNpy, matchingSeg)
//       //     .then(meta => { setSessionId(meta.session_id); setSessionMeta(meta); setRenderReady(true); })
//       //     .catch(err => console.error('Render API:', err));
//       //   // createSession(firstNpy, null)
//       //   //   .then(meta => { setSessionId(meta.session_id); setSessionMeta(meta); setRenderReady(true); })
//       //   //   .catch(err => console.error('Render API:', err));
//       // }
//     };

//   const handleSegFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
//     const files: File[] = Array.from(e.target.files ?? new FileList());
//     if (files.length === 0) return;

//     const folderName = (files[0] as any).webkitRelativePath?.split('/')[0] || 'Masks';
//     setSegFolderName(folderName);
//     setSegFiles(files);
//     setIsSegLoaded(true);
//     setSegFileTree(buildFileTree(files));

//     const currentFile = scanFiles[currentIndex - 1];
//     if (currentFile?.name.endsWith('.npy')) {
//       const pid = extractPatientId(currentFile.name);
//       const matchingSeg = files.find(f => f.name.includes(pid)) ?? null;
//       if (matchingSeg) {
//         setRenderReady(false);
//         if (sessionId) deleteSession(sessionId).catch(() => {});
//         createSession(currentFile, matchingSeg)
//           .then(meta => { setSessionId(meta.session_id); setSessionMeta(meta); setRenderReady(true); })
//           .catch(err => console.error('Render API:', err));
//       }
//     }
//   };

//   const loadSessionForIndex = (index: number, fileList: File[]) => {
//     const file = fileList[index - 1];
//     if (!file?.name.endsWith('.npy')) return;

//     setRenderReady(false);
//     if (sessionId) deleteSession(sessionId).catch(() => {});

//     const pid = extractPatientId(file.name);
//     const matchingSeg = segFiles.find(f => f.name.includes(pid)) ?? null;

//     createSession(file, matchingSeg)
//       .then(meta => {
//         setSessionId(meta.session_id);
//         setSessionMeta(meta);
//         setRenderReady(true);
//       })
//       .catch(err => console.error('Render API:', err));
//   };

//   const handlePrev = () => {
//     if (currentIndex > 1) {
//       const newIndex = currentIndex - 1;
//       setCurrentIndex(newIndex);
//       loadSessionForIndex(newIndex, scanFiles);
//     }
//   };

//   const handleNext = () => {
//     if (currentIndex < totalScans) {
//       const newIndex = currentIndex + 1;
//       setCurrentIndex(newIndex);
//       loadSessionForIndex(newIndex, scanFiles);
//     }
//   };

//   const toggleModality = (mod: string) => {
//     setActiveModalities(prev =>
//       prev.includes(mod) ? prev.filter(m => m !== mod) : [...prev, mod]
//     );
//   };

//   const handleProcess = () => {
//     setIsProcessing(true);
//     setTimeout(() => {
//       setPredictions(prev => ({ ...prev, [currentIndex]: true }));
//       setIsProcessing(false);
//       setExpandedSection('segmentation');
//     }, 2000);
//   };

//   const handleGenerateExplanation = () => {
//     setAiLoading(true);
//     setAiExplanation(null);
//     setTimeout(() => {
//       setAiExplanation("The segmentation mask identifies a hyperintense region primarily visible in T2 and FLAIR modalities, suggesting vasogenic edema. The central necrotic core is evident in T1c. The model exhibits high confidence in the tumor boundaries with no significant artifacts detected in the current slices.");
//       setAiLoading(false);
//     }, 1500);
//   };

//   // --- Dynamic Resizing Handlers ---
//   const [rightSidebarWidth, setRightSidebarWidth] = useState(220);
//   const [isDraggingSidebar, setIsDraggingSidebar] = useState(false);

//   const [aiHeight, setAiHeight] = useState(220);
//   const [isDraggingAi, setIsDraggingAi] = useState(false);

//   useEffect(() => {
//     const handleMouseMove = (e: MouseEvent) => {
//       if (isDraggingSidebar) {
//         const newWidth = document.body.clientWidth - e.clientX;
//         if (newWidth > 160 && newWidth < 500) {
//           setRightSidebarWidth(newWidth);
//         }
//       }
//       if (isDraggingAi) {
//         const newHeight = window.innerHeight - e.clientY;
//         if (newHeight > 120 && newHeight < 600) {
//           setAiHeight(newHeight);
//         }
//       }
//     };

//     const handleMouseUp = () => {
//       setIsDraggingSidebar(false);
//       setIsDraggingAi(false);
//     };

//     if (isDraggingSidebar || isDraggingAi) {
//       document.addEventListener('mousemove', handleMouseMove);
//       document.addEventListener('mouseup', handleMouseUp);
//     }
//     return () => {
//       document.removeEventListener('mousemove', handleMouseMove);
//       document.removeEventListener('mouseup', handleMouseUp);
//     };
//   }, [isDraggingSidebar, isDraggingAi]);

//   // Combined file tree for right panel
//   const combinedTree: FileNode[] = [
//     ...(dataFileTree.length > 0 ? dataFileTree : []),
//     ...(segFileTree.length > 0 ? segFileTree : []),
//   ];

//   return (
//     <div className="flex h-screen w-full bg-gray-100 text-gray-900 font-sans overflow-hidden">

//       {/* Hidden File Inputs for Directory Selection */}
//       <input
//         type="file"
//         ref={dataFileInputRef}
//         onChange={handleDataFileChange}
//         className="hidden"
//         {...({ webkitdirectory: "true", directory: "true" } as any)}
//       />
//       <input
//         type="file"
//         ref={segFileInputRef}
//         onChange={handleSegFileChange}
//         className="hidden"
//         {...({ webkitdirectory: "true", directory: "true" } as any)}
//       />

//       {/* ── Left Narrow Icon Sidebar ── */}
//       <aside className="w-11 bg-slate-900 border-r border-slate-800 flex flex-col items-center shrink-0 z-10 py-2 gap-1">
//         {/* Logo mark */}
//         <div className="w-8 h-8 flex items-center justify-center mb-2">
//           <Brain className="w-5 h-5 text-blue-500" />
//         </div>

//         <div className="w-full h-px bg-slate-800 mb-1" />

//         {/* Load Data */}
//         <button
//           title={isDataLoaded ? `Data: ${dataFolderName}` : 'Load Data Folder'}
//           onClick={() => dataFileInputRef.current?.click()}
//           className={`w-8 h-8 flex items-center justify-center rounded-md transition-colors ${
//             isDataLoaded ? 'bg-blue-600 text-white' : 'text-slate-400 hover:bg-slate-700 hover:text-white'
//           }`}
//         >
//           <Database className="w-4 h-4" />
//         </button>

//         {/* Load Segmentation — functional */}
//         <button
//           title={isSegLoaded ? `Seg: ${segFolderName}` : 'Load Segmentation Folder'}
//           onClick={() => segFileInputRef.current?.click()}
//           disabled={!isDataLoaded}
//           className={`w-8 h-8 flex items-center justify-center rounded-md transition-colors disabled:opacity-30 disabled:cursor-not-allowed ${
//             isSegLoaded
//               ? (hasGroundTruthForCurrentScan ? 'bg-emerald-600 text-white' : 'bg-amber-600 text-white')
//               : 'text-slate-400 hover:bg-slate-700 hover:text-white'
//           }`}
//         >
//           <FolderArchive className="w-4 h-4" />
//         </button>

//         {/* Image viewer icon */}
//         <button
//           title="Image View"
//           className="w-8 h-8 flex items-center justify-center rounded-md text-slate-400 hover:bg-slate-700 hover:text-white transition-colors"
//         >
//           <Image className="w-4 h-4" />
//         </button>

//         {/* Layers icon */}
//         <button
//           title="Layers"
//           className="w-8 h-8 flex items-center justify-center rounded-md text-slate-400 hover:bg-slate-700 hover:text-white transition-colors"
//         >
//           <Layers className="w-4 h-4" />
//         </button>

//         {/* Spacer pushes arrows to bottom */}
//         <div className="flex-1" />

//         <div className="w-full h-px bg-slate-800 mb-1" />

//         {/* Navigation arrows — left first, then right */}
//         <button
//           onClick={handlePrev}
//           disabled={!isDataLoaded || currentIndex === 1}
//           title="Previous Scan"
//           className="w-8 h-8 flex items-center justify-center rounded-md text-slate-400 hover:bg-slate-700 hover:text-white disabled:opacity-25 disabled:cursor-not-allowed transition-colors"
//         >
//           <ChevronLeft className="w-4 h-4" />
//         </button>

//         <button
//           onClick={handleNext}
//           disabled={!isDataLoaded || currentIndex === totalScans}
//           title="Next Scan"
//           className="w-8 h-8 flex items-center justify-center rounded-md text-slate-400 hover:bg-slate-700 hover:text-white disabled:opacity-25 disabled:cursor-not-allowed transition-colors mb-2"
//         >
//           <ChevronRight className="w-4 h-4" />
//         </button>
//       </aside>

//       {/* ── Main Content Area ── */}
//       <div className="flex flex-col flex-1 min-w-0 bg-gray-50 relative">

//         {/* ── Top Bar (compact, matches screenshot) ── */}
//         <header className="h-10 bg-white border-b border-gray-200 flex items-center px-3 gap-3 shadow-sm z-10 shrink-0">
//           {/* VIEWING label + scan name */}
//           <div className="flex items-center gap-2 shrink-0">
//             <span className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">VIEWING</span>
//             <span className="text-xs font-semibold text-gray-700 max-w-[160px] truncate">
//               {isDataLoaded ? currentScanName : 'No Slide Loaded'}
//             </span>
//           </div>

//           <div className="h-4 w-px bg-gray-200 shrink-0" />

//           {/* PATH input — center */}
//           <div className="flex items-center gap-1.5 flex-1 min-w-0">
//             <span className="text-[10px] font-bold text-gray-400 uppercase tracking-widest shrink-0">PATH</span>
//             <input
//               type="text"
//               value={datasetPath}
//               onChange={e => setDatasetPath(e.target.value)}
//               className="flex-1 min-w-0 text-xs text-gray-600 bg-gray-50 border border-gray-200 rounded px-2 py-0.5 focus:outline-none focus:ring-1 focus:ring-blue-400 font-mono"
//               placeholder="/path/to/dataset"
//             />
//           </div>

//           {/* Segmentation Load Button — shown when data is loaded */}
//           {isDataLoaded && (
//             <>
//               <div className="h-4 w-px bg-gray-200 shrink-0" />
//               <button
//                 onClick={() => segFileInputRef.current?.click()}
//                 title={isSegLoaded ? `Seg loaded: ${segFolderName}` : 'Load Segmentation Folder'}
//                 className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-[10px] font-bold border transition-all shrink-0 ${
//                   isSegLoaded
//                     ? hasGroundTruthForCurrentScan
//                       ? 'bg-emerald-50 border-emerald-300 text-emerald-700 hover:bg-emerald-100'
//                       : 'bg-amber-50 border-amber-300 text-amber-700 hover:bg-amber-100'
//                     : 'bg-gray-50 border-gray-200 text-gray-500 hover:bg-gray-100'
//                 }`}
//               >
//                 <FolderArchive className="w-3.5 h-3.5" />
//                 {isSegLoaded
//                   ? hasGroundTruthForCurrentScan
//                     ? <><CheckCircle2 className="w-3 h-3" /><span>SEG LOADED</span></>
//                     : <span>SEG MISSING</span>
//                   : <span>LOAD SEG</span>}
//               </button>
//             </>
//           )}

//           {/* Modality toggles */}
//           <div className="flex items-center gap-1 shrink-0">
//             {ALL_MODALITIES.map(mod => {
//               const isActive = activeModalities.includes(mod);
//               return (
//                 <button
//                   key={mod}
//                   onClick={() => toggleModality(mod)}
//                   disabled={!isDataLoaded}
//                   className={`px-2.5 py-0.5 rounded text-[10px] font-bold transition-all disabled:opacity-40 disabled:cursor-not-allowed ${
//                     isActive
//                       ? 'bg-blue-600 text-white'
//                       : 'bg-gray-100 text-gray-500 border border-gray-200 hover:bg-gray-200'
//                   }`}
//                 >
//                   {mod}
//                 </button>
//               );
//             })}
//           </div>

//           <div className="h-4 w-px bg-gray-200 shrink-0" />

//           {/* Icon buttons + START PROCESSING */}
//           <div className="flex items-center gap-1.5 shrink-0">
//             {/* Zoom icon */}
//             <button
//               title="Zoom"
//               className="w-7 h-7 flex items-center justify-center rounded text-gray-500 hover:bg-gray-100 transition-colors"
//             >
//               <ZoomIn className="w-4 h-4" />
//             </button>
//             {/* Rotate icon */}
//             <button
//               title="Rotate"
//               className="w-7 h-7 flex items-center justify-center rounded text-gray-500 hover:bg-gray-100 transition-colors"
//             >
//               <Rotate3d className="w-4 h-4" />
//             </button>

//             <button
//               onClick={handleProcess}
//               disabled={!isDataLoaded || isProcessing}
//               className="flex items-center gap-1.5 bg-emerald-600 hover:bg-emerald-700 text-white px-3 py-1 rounded text-[11px] font-bold shadow transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
//             >
//               {isProcessing ? (
//                 <Loader2 className="w-3.5 h-3.5 animate-spin" />
//               ) : (
//                 <Play className="w-3 h-3 fill-current" />
//               )}
//               START PROCESSING
//             </button>
//           </div>
//         </header>

//         {/* ── Workspace ── */}
//         <main className="flex-1 flex overflow-hidden">

//           {/* Dashboard / Vis Area */}
//           <div className="flex-1 relative flex items-center justify-center bg-[#f0f0f0] overflow-hidden flex-col group" style={{ backgroundImage: 'radial-gradient(circle at 1.5px 1.5px, #d1d5db 1px, transparent 0)', backgroundSize: '24px 24px' }}>

//             {isDataLoaded ? (
//               <>
//                 {/* 3D Tools — floating left */}
//                 <div className="absolute left-3 top-[42px] flex flex-col gap-1.5 z-20">
//                   <button className="p-2 bg-white/80 hover:bg-white text-gray-500 rounded-lg shadow border border-gray-200 transition-colors"><ZoomIn className="w-4 h-4"/></button>
//                   <button className="p-2 bg-white/80 hover:bg-white text-gray-500 rounded-lg shadow border border-gray-200 transition-colors"><ZoomOut className="w-4 h-4"/></button>
//                   <button className="p-2 bg-white/80 hover:bg-white text-gray-500 rounded-lg shadow border border-gray-200 transition-colors"><Rotate3d className="w-4 h-4"/></button>
//                 </div>

//                 {/* Scan info — top left */}
//                 <div className="absolute left-14 top-[42px] bg-white/80 backdrop-blur rounded-lg px-3 py-1.5 border border-gray-200 shadow z-20">
//                   <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Current Scan</p>
//                   <p className="text-xs font-semibold text-gray-800 truncate max-w-[200px]">{currentScanName}</p>
//                   <p className="text-[10px] text-gray-400 font-mono">{currentIndex} / {totalScans}</p>
//                 </div>

//                 {/* Simulated 3D Volume Viewer */}
//                 {/* Real 3D Volume Viewer */}
//                 <div className="relative w-full flex-1 flex overflow-hidden">
//                   {renderReady && sessionId && sessionMeta ? (
//                     <Viewer3D
//                       sessionId={sessionId}
//                       availableViews={sessionMeta.available_views}
//                       hasSeg={sessionMeta.has_segmentation}
//                       scanName={currentScanName}
//                     />
//                   ) : isDataLoaded ? (
//                     <div className="flex-1 flex flex-col items-center justify-center gap-4 h-full">
//                       <div className="w-16 h-16 rounded-full border-2 border-blue-500/30 animate-spin border-t-blue-500" />
//                       <p className="text-xs text-gray-400 font-mono tracking-wider">UPLOADING TO RENDER API…</p>
//                     </div>
//                   ) : (
//                     <div className="text-gray-500 flex flex-col items-center justify-center z-10 bg-white/70 p-10 rounded-2xl border border-gray-200 backdrop-blur-sm shadow">
//                       <FileImage className="w-14 h-14 opacity-25 mb-5 text-gray-400" />
//                       <h3 className="text-base font-semibold text-gray-600 mb-1">Please load an image or folder to begin</h3>
//                       <p className="text-xs text-gray-400 text-center max-w-xs mt-1">Use the left sidebar icons to load your data and segmentation folders.</p>
//                     </div>
//                   )}
//                 </div>

//                 {/* <div className="relative w-full flex-1 flex items-center justify-center">
//                   {activeModalities.length > 0 ? (
//                     <div className="relative w-80 h-80 flex items-center justify-center">
//                       <div className="absolute inset-0 border-2 border-blue-400/20 rounded-full animate-pulse z-0" />
//                       <div className="absolute inset-4 border-2 border-blue-300/20 rounded-full z-0" />
//                       <div className="z-10 w-64 h-72 bg-gray-800 rounded-xl shadow-2xl relative overflow-hidden ring-1 ring-white/10 flex flex-col items-center justify-center">
//                         <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent"></div>
//                         <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 flex flex-col items-center z-10 w-full px-4">
//                           <Brain className="w-16 h-16 text-blue-500 opacity-80 mb-4 drop-shadow-lg" />
//                           <div className="text-white font-mono text-[10px] tracking-[0.2em] opacity-80 text-center uppercase">
//                             <p className="text-blue-300 font-bold mb-2 truncate max-w-[200px]">3D: {currentScanName}</p>
//                             <p>Modalities: {activeModalities.join(' + ')}</p>
//                           </div>
//                           {(hasGroundTruthForCurrentScan || hasPrediction) && (
//                             <div className="mt-4 flex items-center justify-center gap-2 text-emerald-400 font-bold px-3 py-1 rounded bg-black/20 text-xs text-center border border-emerald-500/20 backdrop-blur-md">
//                               <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
//                               SEGMENTATION LOADED
//                             </div>
//                           )}
//                         </div>
//                       </div>
//                     </div>
//                   ) : (
//                     <div className="text-gray-400 font-medium flex flex-col items-center gap-3 z-10">
//                       <Layers className="w-12 h-12 opacity-40" />
//                       <p className="text-sm">Select a modality to begin visualization.</p>
//                     </div>
//                   )}
//                 </div> */}

//                 {activeModalities.length > 0 && (
//                   <div className="mb-6 flex space-x-6 z-10">
//                     <div className="text-center">
//                       <div className="text-[10px] text-gray-400 font-bold tracking-widest">RENDER MODE</div>
//                       <div className="text-xl font-light text-gray-800">Full 3D</div>
//                     </div>
//                     <div className="text-center">
//                       <div className="text-[10px] text-gray-400 font-bold tracking-widest">ROTATION</div>
//                       <div className="text-xl font-light text-gray-800">45°</div>
//                     </div>
//                     <div className="text-center">
//                       <div className="text-[10px] text-gray-400 font-bold tracking-widest">ZOOM</div>
//                       <div className="text-xl font-light text-gray-800">1.4x</div>
//                     </div>
//                   </div>
//                 )}
//               </>
//             ) : (
//               // Empty State
//               <div className="text-gray-500 flex flex-col items-center justify-center z-10 bg-white/70 p-10 rounded-2xl border border-gray-200 backdrop-blur-sm shadow">
//                 <FileImage className="w-14 h-14 opacity-25 mb-5 text-gray-400" />
//                 <h3 className="text-base font-semibold text-gray-600 mb-1">Please load an image or folder to begin</h3>
//                 <p className="text-xs text-gray-400 text-center max-w-xs mt-1">
//                   Use the left sidebar icons to load your data and segmentation folders.
//                 </p>
//               </div>
//             )}

//             {/* Status bar */}
//             <div className="absolute bottom-3 right-3 bg-white/80 backdrop-blur px-2.5 py-1 rounded text-[10px] text-gray-400 font-mono flex items-center gap-1.5 border border-gray-200 shadow">
//               <Activity className="w-3 h-3 text-emerald-500" />
//               API: Connected
//             </div>
//           </div>

//           {/* Sidebar Vertical Resizer Handle */}
//           <div
//             className="w-1 bg-gray-200 hover:bg-blue-400 active:bg-blue-500 cursor-col-resize transition-all shrink-0 z-20"
//             onMouseDown={() => setIsDraggingSidebar(true)}
//           />

//           {/* ── Right Sidebar — File Browser ── */}
//           <aside
//             style={{ width: rightSidebarWidth }}
//             className="bg-white border-l border-gray-200 flex flex-col shrink-0 z-10 h-full"
//           >
//             {/* Header with patient nav arrows */}
//             <div className="border-b border-gray-200 bg-gray-50 shrink-0">
//               {/* Top row: Files label + folder icon */}
//               <div className="h-8 px-3 flex items-center justify-between">
//                 <span className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Files</span>
//                 <FolderOpen className="w-3.5 h-3.5 text-gray-400" />
//               </div>
//               {/* Patient navigation row */}
//               <div className="px-2 pb-2 flex items-center gap-1">
//                 <button
//                   onClick={handlePrev}
//                   disabled={!isDataLoaded || currentIndex === 1}
//                   title="Previous Patient"
//                   className="w-6 h-6 flex items-center justify-center rounded bg-white border border-gray-200 text-gray-500 hover:bg-gray-100 disabled:opacity-30 disabled:cursor-not-allowed transition-colors shrink-0"
//                 >
//                   <ChevronLeft className="w-3.5 h-3.5" />
//                 </button>
//                 <div className="flex-1 min-w-0 text-center">
//                   <div className="text-[10px] font-semibold text-gray-700 truncate leading-tight">
//                     {isDataLoaded ? currentScanName : 'No file loaded'}
//                   </div>
//                   <div className="text-[9px] text-gray-400 font-mono">
//                     {isDataLoaded ? `${currentIndex} / ${totalScans}` : '—'}
//                   </div>
//                 </div>
//                 <button
//                   onClick={handleNext}
//                   disabled={!isDataLoaded || currentIndex === totalScans}
//                   title="Next Patient"
//                   className="w-6 h-6 flex items-center justify-center rounded bg-white border border-gray-200 text-gray-500 hover:bg-gray-100 disabled:opacity-30 disabled:cursor-not-allowed transition-colors shrink-0"
//                 >
//                   <ChevronRight className="w-3.5 h-3.5" />
//                 </button>
//               </div>
//             </div>

//             {/* File Tree */}
//             <div className="flex-1 overflow-y-auto py-1 text-xs">
//               {combinedTree.length === 0 ? (
//                 <div className="px-3 py-4 text-center text-gray-400 text-[11px]">
//                   No files loaded
//                 </div>
//               ) : (
//                 combinedTree.map((node, i) => (
//                   <FileTreeNode
//                     key={i}
//                     node={node}
//                     depth={0}
//                     segBaseName={segTickId}
//                     hasSegTick={isSegLoaded || hasPrediction}
//                   />
//                 ))
//               )}
//             </div>

//             {/* Horizontal Resizer for AI box */}
//             <div
//               className="h-1 w-full bg-gray-200 hover:bg-blue-400 cursor-row-resize transition-all shrink-0"
//               onMouseDown={() => setIsDraggingAi(true)}
//             />

//             {/* AI Explanation Section */}
//             <div
//               style={{ height: aiHeight }}
//               className="bg-white flex flex-col shrink-0 min-h-[120px] border-t border-gray-100 p-3"
//             >
//               <label className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-2 block shrink-0">AI Explanation</label>

//               <div className="bg-blue-50/50 rounded-lg p-3 border border-blue-100 mb-3 flex-1 overflow-y-auto flex flex-col">
//                 {!aiExplanation && !aiLoading ? (
//                   <div className="flex-1 flex items-center justify-center text-[11px] text-gray-400 text-center py-3">
//                     Select modalities + segmentation<br/>to generate explanation.
//                   </div>
//                 ) : aiLoading ? (
//                   <div className="flex-1 flex flex-col items-center justify-center gap-2 text-xs text-blue-500 py-3">
//                     <Loader2 className="w-5 h-5 animate-spin" />
//                     Synthesizing...
//                   </div>
//                 ) : (
//                   <div className="flex items-start gap-2">
//                     <div className="w-5 h-5 bg-blue-600 rounded flex items-center justify-center shrink-0 mt-0.5">
//                       <Sparkles className="w-3 h-3 text-white" />
//                     </div>
//                     <p className="text-[11px] text-blue-900 leading-relaxed">
//                       {aiExplanation}
//                     </p>
//                   </div>
//                 )}
//               </div>

//               <button
//                 onClick={handleGenerateExplanation}
//                 disabled={aiLoading || (!hasGroundTruthForCurrentScan && !hasPrediction) || activeModalities.length === 0}
//                 className="w-full shrink-0 py-2 bg-gray-100 text-gray-600 text-[10px] font-bold rounded uppercase tracking-wider hover:bg-gray-200 transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex justify-center items-center gap-1.5"
//               >
//                 <Brain className="w-3.5 h-3.5 text-gray-500" />
//                 Generate Explanation
//               </button>
//             </div>
//           </aside>
//         </main>
//       </div>
//     </div>
//   );
// }
