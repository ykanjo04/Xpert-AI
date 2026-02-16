/**
 * Shared glass-morphism UI primitives used across Xpert.
 */

import React, { useRef } from 'react';
import {
  X,
  UploadCloud,
  Wand2,
  Send,
  Stethoscope,
} from 'lucide-react';

/* ── GlassCard ─────────────────────────────────────────────────────────── */

export const GlassCard = ({ children, className = '', onClick, hoverEffect = true }) => (
  <div
    onClick={onClick}
    className={`
      relative overflow-hidden rounded-3xl
      bg-white/5 backdrop-blur-md
      shadow-[inset_0_1px_0_0_rgba(255,255,255,0.1),0_0_0_1px_rgba(255,255,255,0.05),0_8px_20px_-5px_rgba(0,0,0,0.5)]
      transition-all duration-500 ease-out
      border-t border-white/10
      z-10
      ${hoverEffect
        ? 'hover:-translate-y-1 hover:shadow-[inset_0_1px_0_0_rgba(255,255,255,0.2),0_0_0_1px_rgba(34,211,238,0.4),0_10px_40px_-10px_rgba(34,211,238,0.3)] hover:bg-cyan-900/10 cursor-pointer group'
        : ''}
      ${className}
    `}
  >
    <div className="relative z-10">{children}</div>
  </div>
);

/* ── GlassButton ───────────────────────────────────────────────────────── */

export const GlassButton = ({ children, variant = 'primary', onClick, className = '' }) => {
  const base =
    'relative flex items-center justify-center gap-2 rounded-2xl px-6 py-3 font-bold tracking-wide transition-all duration-300 overflow-hidden text-sm uppercase shadow-lg';
  const variants = {
    primary:
      'bg-gradient-to-r from-cyan-500/30 to-blue-600/30 text-white shadow-[0_0_20px_-5px_rgba(34,211,238,0.5)] border border-cyan-400/50 hover:shadow-[0_0_30px_rgba(34,211,238,0.8)] hover:border-cyan-300 hover:from-cyan-500/50 hover:to-blue-600/50',
    secondary:
      'bg-white/5 text-white border border-white/10 hover:bg-white/10 hover:border-white/30',
    text: 'text-white/70 hover:text-cyan-300 hover:bg-white/5',
    icon: 'p-3 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 hover:border-cyan-400/50 text-white hover:text-cyan-300 shadow-none',
  };
  return (
    <button onClick={onClick} className={`${base} ${variants[variant]} ${className}`}>
      <span className="relative z-10 flex items-center gap-2 text-shadow-sm">{children}</span>
    </button>
  );
};

/* ── GlassInput ────────────────────────────────────────────────────────── */

export const GlassInput = ({ type, placeholder, icon: Icon, value, onChange, className = '' }) => (
  <div className={`relative group ${className}`}>
    <input
      type={type}
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      className="w-full rounded-2xl border border-white/10 bg-white/5 px-6 py-4 pl-14 text-lg text-white placeholder-white/30 outline-none focus:border-cyan-500/50 focus:ring-2 focus:ring-cyan-500/20 focus:bg-white/10 transition-all shadow-inner font-sans"
    />
    <div className="absolute left-5 top-1/2 -translate-y-1/2 text-white/40 group-focus-within:text-cyan-400 transition-colors">
      {Icon ? (
        <Icon size={24} />
      ) : type === 'email' ? (
        <span className="text-xl font-bold">@</span>
      ) : (
        <span className="text-xl font-bold">*</span>
      )}
    </div>
  </div>
);

/* ── GlassOtpInput ─────────────────────────────────────────────────────── */

export const GlassOtpInput = ({ length = 6, value, onChange }) => (
  <div className="relative flex justify-center gap-3 w-full mb-2">
    {Array.from({ length }).map((_, index) => (
      <div
        key={index}
        className={`w-12 h-16 rounded-xl border flex items-center justify-center text-3xl font-bold transition-all duration-300
          ${value.length === index ? 'border-cyan-400 bg-cyan-900/20 shadow-[0_0_15px_rgba(34,211,238,0.3)] scale-110' : 'border-white/10 bg-white/5'}
          ${value.length > index ? 'text-white border-white/40' : 'text-white/10'}`}
      >
        {value[index] || ''}
      </div>
    ))}
    <input
      type="text"
      maxLength={length}
      value={value}
      onChange={(e) => {
        if (/^\d*$/.test(e.target.value)) onChange(e.target.value);
      }}
      className="absolute inset-0 w-full h-full opacity-0 cursor-pointer text-transparent bg-transparent tracking-[1em]"
      autoFocus
    />
  </div>
);

/* ── GlassModal ────────────────────────────────────────────────────────── */

export const GlassModal = ({ isOpen, onClose, children }) => {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 animate-in fade-in duration-300">
      <div className="absolute inset-0 bg-[#060d17]/90 backdrop-blur-md" onClick={onClose}></div>
      <GlassCard
        className="relative z-10 w-full max-w-xl p-8 animate-in zoom-in-95 duration-300 !shadow-[inset_0_1px_0_0_rgba(255,255,255,0.2),0_0_0_1px_rgba(255,255,255,0.1),0_20px_60px_-10px_rgba(0,0,0,0.8)]"
        hoverEffect={false}
      >
        <button onClick={onClose} className="absolute right-6 top-6 text-white/50 hover:text-white transition-colors">
          <X size={24} />
        </button>
        {children}
      </GlassCard>
    </div>
  );
};

/* ── FloatingInput (chat bar) ──────────────────────────────────────────── */

export const FloatingInput = ({ value, onChange, onSend, placeholder, onFileUpload, disabled = false }) => {
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file && !disabled) {
      onFileUpload(file);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  return (
    <div className="relative w-full max-w-4xl mx-auto mb-4">
      <div className="absolute inset-0 bg-cyan-500/10 blur-3xl rounded-full transform translate-y-4 scale-y-75 scale-x-95 opacity-60"></div>

      <div
        className={`bg-white/10 backdrop-blur-2xl border border-white/20 rounded-full px-2 py-2 flex items-center shadow-[0_8px_32px_0_rgba(0,0,0,0.37)] transition-all focus-within:border-cyan-400/50 focus-within:bg-white/15 focus-within:shadow-[0_0_30px_rgba(34,211,238,0.2)] ${disabled ? 'opacity-60 pointer-events-none' : ''}`}
      >
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled}
          className="flex items-center gap-2 px-4 py-2 bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 border border-cyan-500/30 rounded-full transition-all mr-2 shadow-[0_0_15px_rgba(34,211,238,0.15)] hover:shadow-[0_0_20px_rgba(34,211,238,0.3)]"
        >
          <UploadCloud size={18} />
          <span className="text-sm font-semibold">Upload X-ray</span>
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*,.dcm"
          onChange={handleFileChange}
          className="hidden"
        />

        <input
          type="text"
          value={value}
          onChange={onChange}
          onKeyDown={(e) => e.key === 'Enter' && !disabled && onSend()}
          placeholder={placeholder}
          disabled={disabled}
          className="flex-1 bg-transparent border-none outline-none text-white px-2 placeholder-white/40 font-light text-base h-10"
        />

        <div className="flex items-center gap-3 pr-2 border-l border-white/10 pl-3">
          <button className="flex items-center gap-2 text-white/60 hover:text-cyan-300 transition-colors group">
            <Wand2 size={18} className="group-hover:text-cyan-300 text-white/50" />
            <span className="text-xs font-medium hidden sm:inline-block">Enhance</span>
          </button>

          <button
            onClick={onSend}
            disabled={disabled}
            className={`p-2.5 rounded-full transition-all duration-300 ${
              value.trim() && !disabled
                ? 'bg-cyan-500 text-black shadow-[0_0_20px_rgba(34,211,238,0.4)]'
                : 'bg-white/10 text-white/20'
            }`}
          >
            <Send size={18} fill={value.trim() ? 'currentColor' : 'none'} />
          </button>
        </div>
      </div>
    </div>
  );
};
