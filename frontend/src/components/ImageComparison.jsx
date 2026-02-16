/**
 * ImageComparison — full-screen lightbox to compare pipeline output images.
 *
 * Supports:
 *   • Tab switching between Original / Enhanced / Mask / Grad-CAM
 *   • Overlay mode with adjustable opacity
 */

import React, { useState } from 'react';
import { X, Layers, Eye } from 'lucide-react';

const TABS = [
  { key: 'original', label: 'Original',   color: 'white' },
  { key: 'enhanced', label: 'Enhanced',   color: 'cyan' },
  { key: 'mask',     label: 'Mask',       color: 'purple' },
  { key: 'gradcam',  label: 'Grad-CAM',   color: 'red' },
];

const ImageComparison = ({ images, originalImage, onClose }) => {
  const [activeTab, setActiveTab] = useState('enhanced');
  const [overlayOn, setOverlayOn] = useState(false);
  const [overlayOpacity, setOverlayOpacity] = useState(50);

  const srcMap = {
    original: originalImage,
    enhanced: images?.enhanced,
    mask:     images?.mask,
    gradcam:  images?.gradcam,
  };

  return (
    <div className="fixed inset-0 z-[100] bg-black/95 backdrop-blur-md flex flex-col text-white font-sans">
      {/* Header */}
      <header className="h-16 border-b border-white/10 flex items-center justify-between px-6 bg-[#060d17]/80 backdrop-blur-md z-20 flex-shrink-0">
        <div className="flex items-center gap-4">
          <Layers size={18} className="text-cyan-400" />
          <span className="font-bold tracking-wide">Image Comparison</span>
        </div>
        <button onClick={onClose} className="p-2 rounded-lg hover:bg-white/10 transition-colors">
          <X size={22} />
        </button>
      </header>

      {/* Tabs */}
      <div className="flex items-center justify-center gap-2 px-6 py-3 border-b border-white/5 bg-white/[0.02] flex-shrink-0">
        {TABS.map((tab) => {
          const src = srcMap[tab.key];
          const available = !!src;
          return (
            <button
              key={tab.key}
              disabled={!available}
              onClick={() => setActiveTab(tab.key)}
              className={`px-5 py-2 rounded-full text-sm font-bold uppercase tracking-wider transition-all border
                ${activeTab === tab.key
                  ? `bg-${tab.color === 'white' ? 'white' : tab.color}-500/20 border-${tab.color === 'white' ? 'white' : tab.color}-500/50 text-${tab.color === 'white' ? 'white' : tab.color}-300`
                  : available
                    ? 'border-white/10 text-white/50 hover:bg-white/5 hover:text-white'
                    : 'border-white/5 text-white/20 cursor-not-allowed'}
              `}
            >
              {tab.label}
            </button>
          );
        })}

        <div className="w-px h-6 bg-white/10 mx-2"></div>

        {/* Overlay toggle */}
        <button
          onClick={() => setOverlayOn(!overlayOn)}
          className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-bold border transition-all
            ${overlayOn
              ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-300'
              : 'border-white/10 text-white/50 hover:bg-white/5'}
          `}
        >
          <Eye size={14} />
          Overlay
        </button>
      </div>

      {/* Image display */}
      <div className="flex-1 flex items-center justify-center p-8 relative overflow-hidden">
        {overlayOn ? (
          /* Overlay mode: base image + gradcam overlay */
          <div className="relative max-w-2xl max-h-[70vh]">
            <img
              src={srcMap.enhanced || srcMap.original}
              alt="Base"
              className="max-w-full max-h-[70vh] object-contain rounded-2xl border border-white/10"
            />
            {srcMap.gradcam && (
              <img
                src={srcMap.gradcam}
                alt="Overlay"
                className="absolute inset-0 w-full h-full object-contain rounded-2xl mix-blend-screen"
                style={{ opacity: overlayOpacity / 100 }}
              />
            )}
          </div>
        ) : (
          /* Single image mode */
          <div className="max-w-2xl max-h-[70vh]">
            {srcMap[activeTab] ? (
              <img
                src={srcMap[activeTab]}
                alt={activeTab}
                className="max-w-full max-h-[70vh] object-contain rounded-2xl border border-white/10"
              />
            ) : (
              <div className="w-96 h-96 rounded-2xl border border-white/10 bg-white/5 flex items-center justify-center">
                <p className="text-white/30 text-sm">Image not available</p>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Overlay opacity slider */}
      {overlayOn && (
        <div className="px-8 pb-6 flex items-center justify-center gap-4 flex-shrink-0">
          <span className="text-xs text-white/40 font-mono w-8">0%</span>
          <input
            type="range"
            min="0"
            max="100"
            value={overlayOpacity}
            onChange={(e) => setOverlayOpacity(Number(e.target.value))}
            className="w-64 h-2 bg-white/10 rounded-lg appearance-none cursor-pointer accent-cyan-400"
          />
          <span className="text-xs text-white/40 font-mono w-12">{overlayOpacity}%</span>
        </div>
      )}
    </div>
  );
};

export default ImageComparison;
