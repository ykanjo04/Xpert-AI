/**
 * ARMode — immersive AR visualization connected to real pipeline data.
 *
 * Props:
 *   onClose       – callback to return to chat
 *   imageData     – base64 data URL of the original uploaded image
 *   pipelineData  – { prediction, score, images: { enhanced, mask, gradcam }, gradcam_regions, quality }
 */

import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  ArrowLeft,
  Activity,
  Settings,
  Layers,
  Eye,
  RefreshCw,
  Camera,
} from 'lucide-react';
import { arStyles } from '../styles/customStyles';

const ARMode = ({ onClose, imageData, pipelineData = {} }) => {
  const [layers, setLayers] = useState({ heatmap: true, skeleton: true, organs: true, vessels: false });
  const [opacity, setOpacity] = useState(80);
  const [activePanel, setActivePanel] = useState('reference'); // reference | enhanced | mask | gradcam
    const unityFrameRef = useRef(null);

  const gradcamUrl = pipelineData?.images?.gradcam || null;

  const resolvedGradcamUrl = useMemo(() => {
    if (!gradcamUrl) return null;

    if (/^https?:\/\//i.test(gradcamUrl)) return gradcamUrl;

    const BACKEND_ORIGIN = "http://localhost:8000";
    return `${BACKEND_ORIGIN}${gradcamUrl}`;
  }, [gradcamUrl]);

  useEffect(() => {
    if (!resolvedGradcamUrl) return;

    unityFrameRef.current?.contentWindow?.postMessage(
      { type: "XPERT_SET_HEATMAP_URL", url: resolvedGradcamUrl },
      "*"
    );
  }, [resolvedGradcamUrl]);

  useEffect(() => {
    if (!pipelineData?.gradcam_regions) return;
  
    unityFrameRef.current?.contentWindow?.postMessage(
      {
        type: "XPERT_SET_HEATMAP_REGIONS",
        regions: pipelineData.gradcam_regions
      },
      "*"
    );
  }, [pipelineData]);

  useEffect(() => {
    unityFrameRef.current?.contentWindow?.postMessage(
      { type: "XPERT_SET_OVERLAY", opacity, enabled: !!layers.heatmap },
      "*"
    );
  }, [opacity, layers.heatmap]);

  const toggleLayer = (layer) => setLayers((prev) => ({ ...prev, [layer]: !prev[layer] }));

  const score = pipelineData.score ?? 0;
  const prediction = pipelineData.prediction || 'unknown';
  const scorePercent = (score * 100).toFixed(1);
  const qualityLabel = pipelineData.quality?.label || '—';
  const regions = pipelineData.gradcam_regions || [];

  const panelImages = {
    reference: imageData,
    enhanced: pipelineData.images?.enhanced,
    mask: pipelineData.images?.mask,
    gradcam: pipelineData.images?.gradcam,
  };

  return (
    <div className="fixed inset-0 z-[200] bg-[#03060a] text-white font-sans overflow-hidden flex flex-col">
      <style>{arStyles}</style>

      {/* Background FX */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-cyan-900/20 rounded-full blur-[100px] ar-pulse"></div>
        <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.03)_1px,transparent_1px)] bg-[size:100px_100px]"></div>
      </div>

      {/* Header */}
      <header className="h-16 border-b border-white/10 bg-[#060d17]/80 backdrop-blur-md flex items-center justify-between px-6 z-20 relative">
        <div className="flex items-center gap-4">
          <button onClick={onClose} className="flex items-center gap-2 text-white/60 hover:text-white transition-colors">
            <ArrowLeft size={20} />
            <span className="text-sm font-medium">Back to Analysis</span>
          </button>
          <div className="h-6 w-px bg-white/10 mx-2"></div>
          <div className="flex items-center gap-2">
            <Activity size={18} className="text-cyan-400" />
            <span className="font-bold tracking-wide">AR Visualization Mode</span>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className="px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-300 text-xs font-mono tracking-wider flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></div>
            LIVE RENDER
          </div>
          <button className="p-2 rounded-lg hover:bg-white/10 transition-colors">
            <Settings size={20} />
          </button>
        </div>
      </header>

      {/* Main */}
      <div className="flex-1 relative flex overflow-hidden z-10">
        {/* LEFT PANEL */}
        <div className="w-80 p-6 flex flex-col gap-6 pointer-events-auto overflow-y-auto">
          {/* Image selector */}
          <div className="ar-glass-panel p-4 rounded-2xl">
            <h3 className="text-xs font-bold text-white/50 uppercase tracking-widest mb-3">Scan Views</h3>
            <div className="grid grid-cols-2 gap-2 mb-3">
              {(['reference', 'enhanced', 'mask', 'gradcam']).map((key) => (
                <button
                  key={key}
                  onClick={() => setActivePanel(key)}
                  className={`px-3 py-2 rounded-lg text-[10px] font-bold uppercase tracking-wider border transition-all ${
                    activePanel === key
                      ? 'bg-cyan-500/20 border-cyan-500/40 text-cyan-300'
                      : 'border-white/10 text-white/40 hover:bg-white/5'
                  }`}
                >
                  {key}
                </button>
              ))}
            </div>
            <div className="aspect-[3/4] bg-black/50 rounded-xl overflow-hidden border border-white/10 relative flex items-center justify-center">
              {panelImages[activePanel] ? (
                <img src={panelImages[activePanel]} alt={activePanel} className="w-full h-full object-contain" />
              ) : (
                <div className="text-center text-white/20 font-mono text-xs">NO DATA</div>
              )}
            </div>
          </div>

          {/* Diagnostics — REAL DATA */}
          <div className="ar-glass-panel p-5 rounded-2xl flex-1">
            <h3 className="text-xs font-bold text-white/50 uppercase tracking-widest mb-4">Diagnostic Context</h3>
            <div className="space-y-4">
              <div className={`p-3 rounded-xl border ${prediction === 'pneumonia' ? 'bg-red-500/10 border-red-500/20' : 'bg-green-500/10 border-green-500/20'}`}>
                <h4 className={`font-bold text-sm mb-1 ${prediction === 'pneumonia' ? 'text-red-300' : 'text-green-300'}`}>
                  {prediction === 'pneumonia' ? 'Anomaly Detected' : 'No Anomaly Detected'}
                </h4>
                <p className="text-xs text-white/70 leading-relaxed">
                  {prediction === 'pneumonia'
                    ? `AI detected features consistent with pneumonia (score: ${scorePercent}%).`
                    : `Scan classified as normal (score: ${scorePercent}%).`}
                </p>
              </div>

              <div className="flex justify-between items-center text-sm border-b border-white/5 pb-2">
                <span className="text-white/60">Confidence</span>
                <span className="font-mono text-cyan-300">{scorePercent}%</span>
              </div>
              <div className="flex justify-between items-center text-sm border-b border-white/5 pb-2">
                <span className="text-white/60">Enhanced</span>
                <span className="font-mono text-green-300">Yes</span>
              </div>
              <div className="flex justify-between items-center text-sm border-b border-white/5 pb-2">
                <span className="text-white/60">Grad-CAM Regions</span>
                <span className="font-mono text-white">{regions.length}</span>
              </div>
              {regions.length > 0 && (
                <div className="space-y-2">
                  <p className="text-[10px] text-white/40 uppercase tracking-widest">Top Regions</p>
                  {regions.slice(0, 3).map((r, idx) => (
                    <div key={idx} className="flex justify-between items-center text-xs bg-white/5 rounded-lg px-3 py-2 border border-white/5">
                      <span className="text-white/60">Region {idx + 1} ({r.w}×{r.h})</span>
                      <span className="font-mono text-red-300">{(r.intensity * 100).toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* CENTER: AR VIEWPORT */}
        <div className="flex-1 relative flex items-center justify-center">
          <div className="relative w-[820px] h-[620px] ar-glass-panel rounded-2xl overflow-hidden border border-white/10">
            <iframe
              ref={unityFrameRef}
              title="Xpert Unity Viewer"
              src="/unity/index.html"
              className="w-full h-full"
              style={{ border: 0 }}
              allow="fullscreen"
            />
          </div>
        
          <div className="absolute bottom-8 px-6 py-3 ar-glass-panel rounded-full flex items-center gap-6">
            <div className="flex items-center gap-2 border-r border-white/10 pr-6">
              <Layers size={16} className="text-cyan-400" />
              <span className="text-sm font-medium">
                Layer: {layers.skeleton ? 'Skeletal' : 'Soft Tissue'}
              </span>
            </div>
        
            <div className="flex items-center gap-2">
              <Activity
                size={16}
                className={prediction === 'pneumonia' ? 'text-red-400' : 'text-green-400'}
              />
              <span className="text-sm font-medium">
                Prediction: {prediction.toUpperCase()} ({scorePercent}%)
              </span>
            </div>
          </div>
        </div>

        {/* RIGHT PANEL: Controls */}
        <div className="w-72 p-6 flex flex-col gap-6 pointer-events-auto">
          <div className="ar-glass-panel p-5 rounded-2xl">
            <h3 className="text-xs font-bold text-white/50 uppercase tracking-widest mb-4 flex items-center gap-2">
              <Layers size={14} /> Visibility Layers
            </h3>
            <div className="space-y-3">
              {[
                { id: 'heatmap', label: 'AI Heatmap', color: 'bg-red-500' },
                { id: 'skeleton', label: 'Skeletal Structure', color: 'bg-white' },
                { id: 'organs', label: 'Vital Organs', color: 'bg-blue-500' },
                { id: 'vessels', label: 'Blood Vessels', color: 'bg-purple-500' },
              ].map((layer) => (
                <button
                  key={layer.id}
                  onClick={() => toggleLayer(layer.id)}
                  className={`w-full flex items-center justify-between p-3 rounded-xl border transition-all ${
                    layers[layer.id]
                      ? 'bg-white/10 border-cyan-500/50 shadow-[0_0_10px_rgba(34,211,238,0.1)]'
                      : 'bg-transparent border-white/5 hover:bg-white/5'
                  }`}
                >
                  <span className="text-sm font-medium text-white/90">{layer.label}</span>
                  <div className={`w-2 h-2 rounded-full ${layers[layer.id] ? layer.color : 'bg-white/10'}`}></div>
                </button>
              ))}
            </div>
          </div>

          <div className="ar-glass-panel p-5 rounded-2xl">
            <h3 className="text-xs font-bold text-white/50 uppercase tracking-widest mb-4 flex items-center gap-2">
              <Eye size={14} /> Overlay Opacity
            </h3>
            <input
              type="range"
              min="0"
              max="100"
              value={opacity}
              onChange={(e) => setOpacity(Number(e.target.value))}
              className="w-full h-2 bg-white/10 rounded-lg appearance-none cursor-pointer accent-cyan-400"
            />
            <div className="flex justify-between mt-2 text-xs text-white/40 font-mono">
              <span>0%</span>
              <span>{opacity}%</span>
              <span>100%</span>
            </div>
          </div>

          <div className="mt-auto grid grid-cols-2 gap-3">
            <button className="ar-glass-panel ar-control-btn p-4 rounded-xl flex flex-col items-center justify-center gap-2 text-white/70 hover:text-white">
              <RefreshCw size={20} />
              <span className="text-xs">Reset View</span>
            </button>
            <button className="ar-glass-panel ar-control-btn p-4 rounded-xl flex flex-col items-center justify-center gap-2 text-white/70 hover:text-white">
              <Camera size={20} />
              <span className="text-xs">Snapshot</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ARMode;
