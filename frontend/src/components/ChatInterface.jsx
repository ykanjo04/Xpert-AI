/**
 * ChatInterface — main chat experience with:
 *   • Markdown rendering (react-markdown)
 *   • Pipeline progress indicator (animated stages)
 *   • Conversational AI (text → /api/chat)
 *   • Image analysis (file → /api/analyze)
 *   • Image comparison lightbox
 *   • AR visualization with real data
 *   • Session history sidebar
 */

import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  Brain,
  Stethoscope,
  ScanEye,
  ShieldCheck,
  Plus,
  MessageSquare,
  Settings,
  Menu,
  LogOut,
  User,
  Layers,
  BarChart3,
  Clock,
  Image as ImageIcon,
} from 'lucide-react';

import { FloatingInput } from './ui';
import ARMode from './ARMode';
import ImageComparison from './ImageComparison';
import { customStyles } from '../styles/customStyles';
import { analyzeImage, sendChatMessage } from '../services/api';
import { AuthService } from '../services/auth';
import logo from '../logo.png';

/* ── Pipeline stage config ────────────────────────────────────────────── */

const PIPELINE_STAGES = [
  { key: 'quality',  icon: '🔍', label: 'Assessing image quality …' },
  { key: 'enhance',  icon: '✨', label: 'Enhancing image …' },
  { key: 'segment',  icon: '🫁', label: 'Segmenting lung region …' },
  { key: 'classify', icon: '🧠', label: 'Running classification …' },
  { key: 'gradcam',  icon: '🔥', label: 'Generating Grad-CAM heatmap …' },
  { key: 'adaptive', icon: '💡', label: 'Running adaptive engine …' },
];

/* ── PipelineProgress sub-component ───────────────────────────────────── */

const PipelineProgress = ({ currentStage }) => {
  const idx = PIPELINE_STAGES.findIndex((s) => s.key === currentStage);
  return (
    <div className="space-y-2 py-2">
      {PIPELINE_STAGES.map((stage, i) => {
        const done = i < idx;
        const active = i === idx;
        return (
          <div key={stage.key} className={`flex items-center gap-3 transition-all duration-300 ${done ? 'text-green-300' : active ? 'text-cyan-300' : 'text-white/20'}`}>
            <span className="w-5 text-center">
              {done ? '✅' : active ? (
                <span className="inline-block w-3 h-3 rounded-full border-2 border-cyan-400 border-t-transparent animate-spin" />
              ) : '⬜'}
            </span>
            <span className="text-sm">{stage.icon} {stage.label}</span>
          </div>
        );
      })}
    </div>
  );
};

/* ── Markdown renderer ────────────────────────────────────────────────── */

const MarkdownBody = ({ children }) => (
  <ReactMarkdown
    remarkPlugins={[remarkGfm]}
    components={{
      p: ({ children: c }) => <p className="mb-2 last:mb-0 leading-relaxed">{c}</p>,
      strong: ({ children: c }) => <strong className="text-cyan-300 font-bold">{c}</strong>,
      em: ({ children: c }) => <em className="text-white/70 italic">{c}</em>,
      h1: ({ children: c }) => <h1 className="text-xl font-bold mb-2 text-white">{c}</h1>,
      h2: ({ children: c }) => <h2 className="text-lg font-bold mb-2 text-white">{c}</h2>,
      h3: ({ children: c }) => <h3 className="text-base font-bold mb-1 text-white">{c}</h3>,
      ul: ({ children: c }) => <ul className="list-disc pl-5 space-y-1 mb-2">{c}</ul>,
      ol: ({ children: c }) => <ol className="list-decimal pl-5 space-y-1 mb-2">{c}</ol>,
      li: ({ children: c }) => <li className="text-white/80 text-sm">{c}</li>,
      code: ({ children: c, className }) => {
        const isBlock = className?.includes('language-');
        return isBlock
          ? <pre className="bg-black/40 rounded-xl p-3 my-2 overflow-x-auto border border-white/10"><code className="text-xs text-cyan-200 font-mono">{c}</code></pre>
          : <code className="bg-white/10 px-1.5 py-0.5 rounded text-cyan-200 text-sm font-mono">{c}</code>;
      },
      blockquote: ({ children: c }) => <blockquote className="border-l-2 border-cyan-500/50 pl-3 my-2 text-white/60 italic">{c}</blockquote>,
    }}
  >
    {children}
  </ReactMarkdown>
);

/* ── ChatInterface ────────────────────────────────────────────────────── */

const ChatInterface = ({ role, onLogout, sessionId, onShowDashboard }) => {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [showAR, setShowAR] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [arData, setArData] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [pipelineStage, setPipelineStage] = useState(null);
  const [comparisonData, setComparisonData] = useState(null);
  const chatEndRef = useRef(null);

  const themeColor = role === 'student' ? 'cyan' : 'blue';

  // Auto-scroll
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Pipeline progress timer
  useEffect(() => {
    if (!isAnalyzing) { setPipelineStage(null); return; }
    const keys = PIPELINE_STAGES.map((s) => s.key);
    let idx = 0;
    setPipelineStage(keys[0]);
    const timer = setInterval(() => {
      idx++;
      if (idx < keys.length) setPipelineStage(keys[idx]);
    }, 4000);
    return () => clearInterval(timer);
  }, [isAnalyzing]);

  /* ── Send text message (conversational AI) ── */
  const handleSend = async () => {
    if (!inputText.trim() || isAnalyzing) return;
    const text = inputText;
    setInputText('');
    const userMsg = { id: Date.now(), sender: 'user', text };
    setMessages((prev) => [...prev, userMsg]);

    // Show typing indicator
    const aiId = Date.now() + 1;
    setMessages((prev) => [...prev, { id: aiId, sender: 'ai', text: '', isTyping: true }]);

    try {
      const data = await sendChatMessage(sessionId, text, role);
      setMessages((prev) =>
        prev.map((m) => (m.id === aiId ? { ...m, text: data.response || 'No response.', isTyping: false } : m)),
      );
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === aiId
            ? { ...m, text: `Unable to reach the AI service: ${err.message}. Try uploading an X-ray instead.`, isTyping: false }
            : m,
        ),
      );
    }
  };

  /* ── Upload image (pipeline analysis) ── */
  const handleFileUpload = async (file) => {
    const reader = new FileReader();
    reader.onload = async (e) => {
      const imageUrl = e.target?.result;
      const userMsg = { id: Date.now(), sender: 'user', text: `Uploaded X-ray: ${file.name}`, image: imageUrl, fileName: file.name };
      setMessages((prev) => [...prev, userMsg]);

      const processingId = Date.now() + 1;
      setMessages((prev) => [...prev, { id: processingId, sender: 'ai', text: '', isProcessing: true }]);
      setIsAnalyzing(true);

      try {
        const data = await analyzeImage(file, {
          studentId: 'S001',
          studentInput: inputText || 'normal',
          action: 'diagnose',
          result: 'pending',
          sessionId,
        });

        // Parse adaptive response
        let adaptive = data.adaptive || {};
        if (adaptive.message && typeof adaptive.message === 'string') {
          try {
            const cleaned = adaptive.message.replace(/```json\n?/g, '').replace(/```/g, '').trim();
            adaptive = JSON.parse(cleaned);
          } catch { /* keep original */ }
        }

        const summary = adaptive.summary || adaptive.message || '';
        const recommendation = adaptive.learning_recommendation || adaptive.explanation || '';
        const score = data.pneumonia_score ?? 0;
        const prediction = data.prediction || 'unknown';
        const pct = (score * 100).toFixed(1);

        const resultText = [
          `## 🔬 AI Analysis Complete\n`,
          `**Prediction:** ${prediction.toUpperCase()}`,
          `**Pneumonia Score:** ${pct}%`,
          data.needs_human_review ? `\n> ⚠️ **Uncertainty zone** — this score falls in the ambiguous range (0.45–0.55). Careful review recommended.\n` : '',
          summary ? `\n### 💡 Summary\n${summary}\n` : '',
          recommendation ? `\n### 📚 Recommendation\n${recommendation}` : '',
          `\n---\n_Upload another X-ray or ask me a question. Type "quiz me" for a knowledge check!_`,
        ].filter(Boolean).join('\n');

        setMessages((prev) =>
          prev.map((m) =>
            m.id === processingId
              ? {
                  ...m,
                  text: resultText,
                  isProcessing: false,
                  showARButton: true,
                  pipelineImages: data.images || {},
                  pipelineData: {
                    prediction,
                    score,
                    images: data.images || {},
                    gradcam_regions: data.gradcam_regions || [],
                    quality: data.quality || {},
                  },
                }
              : m,
          ),
        );
      } catch (err) {
        setMessages((prev) =>
          prev.map((m) => (m.id === processingId ? { ...m, text: `❌ Analysis failed: ${err.message}`, isProcessing: false } : m)),
        );
      } finally {
        setIsAnalyzing(false);
      }
    };
    reader.readAsDataURL(file);
  };

  /* ── AR visualization ── */
  const handleOpenAR = (msg) => {
    const idx = messages.findIndex((m) => m.id === msg.id);
    let imageData = null;
    for (let i = idx - 1; i >= 0; i--) {
      if (messages[i].image) { imageData = messages[i].image; break; }
    }
    if (!imageData) { alert('No X-ray image found. Please upload an image first.'); return; }
    setSelectedImage(imageData);
    setArData(msg.pipelineData || {});
    setShowAR(true);
  };

  /* ── Image comparison ── */
  const handleOpenComparison = (msg) => {
    const idx = messages.findIndex((m) => m.id === msg.id);
    let original = null;
    for (let i = idx - 1; i >= 0; i--) {
      if (messages[i].image) { original = messages[i].image; break; }
    }
    setComparisonData({ images: msg.pipelineImages, originalImage: original });
  };

  /* ── Render ── */
  if (showAR) {
    return <ARMode onClose={() => { setShowAR(false); setSelectedImage(null); setArData(null); }} imageData={selectedImage} pipelineData={arData} />;
  }

  if (comparisonData) {
    return <ImageComparison images={comparisonData.images} originalImage={comparisonData.originalImage} onClose={() => setComparisonData(null)} />;
  }

  return (
    <div className="flex h-screen w-full bg-[#03060a] text-white font-sans overflow-hidden relative">
      <style>{customStyles}</style>

      {/* Background */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,_var(--tw-gradient-stops))] from-blue-900/30 via-[#060d17] to-[#03060a]"></div>
        <div className="absolute top-[-10%] right-[-5%] w-[40vw] h-[40vw] bg-cyan-500/10 rounded-full blur-[120px] opacity-60"></div>
        <div className="absolute bottom-[-10%] left-[-5%] w-[30vw] h-[30vw] bg-blue-600/10 rounded-full blur-[100px] opacity-50"></div>
      </div>

      {/* ── LEFT SIDEBAR ── */}
      <aside className="w-80 hidden md:flex flex-col border-r border-cyan-500/20 bg-white/5 backdrop-blur-xl z-20 relative shadow-[5px_0_30px_rgba(0,0,0,0.3)]">
        <div className="p-6 border-b border-white/10 flex items-center gap-3 cursor-pointer group" onClick={onLogout}>
          <img src={logo} alt="Xpert" className="h-12 w-12 object-contain" />
          <span className="text-2xl font-bold tracking-wide text-white/90 group-hover:text-cyan-300 transition-colors">Xpert</span>
        </div>

        <div className="p-5 flex-1 overflow-y-auto glass-scroll">
          <button className="w-full py-4 rounded-2xl mb-4 flex items-center justify-center gap-2 font-bold transition-all bg-gradient-to-r from-cyan-500 to-blue-500 text-white shadow-[0_0_20px_rgba(34,211,238,0.4)] hover:shadow-[0_0_30px_rgba(34,211,238,0.6)] border border-white/20 transform hover:scale-[1.02]">
            <Plus size={20} strokeWidth={3} /> New Session +
          </button>

          {/* Doctor-only: reports button */}
          {role === 'doctor' && onShowDashboard && (
            <button
              onClick={onShowDashboard}
              className="w-full py-3 rounded-2xl mb-6 flex items-center justify-center gap-2 font-bold transition-all bg-gradient-to-r from-blue-500/20 to-blue-600/20 text-blue-300 border border-blue-500/30 hover:bg-blue-500/30"
            >
              <BarChart3 size={18} /> Student Reports
            </button>
          )}

          <div className="space-y-3">
            <p className="text-[10px] font-bold text-white/40 uppercase tracking-[0.2em] px-3 mb-1">Recent History</p>
            <div className="flex flex-col items-center justify-center py-8 text-center">
              <MessageSquare size={24} className="text-white/10 mb-3" />
              <p className="text-sm text-white/30">No sessions yet</p>
              <p className="text-[10px] text-white/20 mt-1">Upload an X-ray to start</p>
            </div>
          </div>
        </div>

        <div className="p-4 border-t border-white/10 bg-black/20 backdrop-blur-sm">
          <div className="flex items-center justify-between p-3 rounded-xl hover:bg-white/5 transition-colors cursor-pointer group border border-transparent hover:border-white/10">
            <div className="flex items-center gap-3">
              <div className={`h-10 w-10 rounded-full bg-${themeColor}-900/30 flex items-center justify-center text-${themeColor}-300 border border-${themeColor}-500/20`}>
                <User size={18} />
              </div>
              <div className="flex flex-col">
                <span className="text-sm font-semibold capitalize text-white/90">{role} Account</span>
                <span className="text-[10px] text-white/40 uppercase tracking-wider">
                  {sessionId ? `Session: ${sessionId}` : 'Verified ID'}
                </span>
              </div>
            </div>
            <button onClick={onLogout} className="text-white/30 hover:text-red-400 p-2 rounded-lg transition-all">
              <LogOut size={16} />
            </button>
          </div>
        </div>
      </aside>

      {/* ── MAIN CHAT AREA ── */}
      <main className="flex-1 flex flex-col relative z-10 h-full">
        <header className="flex flex-col z-30">
          <div className="h-20 border-b border-white/10 flex items-center justify-between px-8 bg-white/[0.05] backdrop-blur-xl shadow-sm">
            <div className="flex items-center gap-4">
              <span className="md:hidden p-2 hover:bg-white/10 rounded-lg cursor-pointer" onClick={onLogout}>
                <Menu className="text-white" />
              </span>
              <div className={`flex items-center gap-2 px-4 py-2 rounded-full border text-xs font-bold uppercase tracking-widest shadow-[0_0_15px_rgba(34,211,238,0.2)] ${
                role === 'student'
                  ? 'bg-gradient-to-r from-cyan-500/20 to-cyan-900/20 border-cyan-500/40 text-cyan-300'
                  : 'bg-gradient-to-r from-blue-500/20 to-blue-900/20 border-blue-500/40 text-blue-300'
              }`}>
                <Activity size={14} className="animate-pulse" />
                {role === 'student' ? 'Student Mode' : 'Diagnostic Mode'}
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button className="text-white/60 hover:text-cyan-300 p-2 rounded-lg hover:bg-white/5 transition-all">
                <Settings size={22} />
              </button>
            </div>
          </div>

          <div className="h-8 bg-white/[0.02] border-b border-white/5 backdrop-blur-md flex items-center justify-center">
            <span className="text-[10px] font-medium tracking-widest text-cyan-200/70 uppercase flex items-center gap-2">
              <ShieldCheck size={10} /> Educational use only. Not for medical diagnosis.
            </span>
          </div>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 md:p-10 space-y-6 glass-scroll flex flex-col items-center">
          {messages.length === 0 ? (
            <div className="flex-1 flex items-center justify-center w-full">
              <div className="relative w-full max-w-3xl p-12 rounded-[3rem] bg-white/5 backdrop-blur-2xl border border-white/10 shadow-[0_0_60px_-15px_rgba(34,211,238,0.1)] flex flex-col items-center text-center">
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[80%] h-[80%] bg-cyan-500/20 blur-[100px] rounded-full pointer-events-none"></div>

                <div className="relative z-10 mb-8">
                  <div className="w-32 h-32 rounded-2xl bg-gradient-to-br from-cyan-400 to-blue-600 p-[3px] shadow-[0_0_40px_rgba(34,211,238,0.6)]">
                    <div className="w-full h-full rounded-2xl bg-gradient-to-br from-cyan-900/80 to-blue-900/80 backdrop-blur-lg flex items-center justify-center overflow-hidden border border-cyan-300/20">
                      <div className="flex flex-col items-center justify-center">
                        <div className="relative">
                          <div className="w-12 h-12 rounded-full bg-gradient-to-b from-cyan-200 to-cyan-100 mx-auto mb-1"></div>
                          <div className="w-16 h-10 rounded-t-2xl bg-gradient-to-b from-blue-300 to-blue-200 mx-auto flex items-start justify-center pt-1">
                            <Stethoscope size={20} className="text-blue-600" />
                          </div>
                        </div>
                        <div className="mt-2 text-xs font-bold text-cyan-200 tracking-widest">XPERT</div>
                      </div>
                    </div>
                  </div>
                  <div className="absolute inset-0 rounded-2xl border-2 border-cyan-400/30 animate-pulse"></div>
                </div>

                <h1 className="relative z-10 text-4xl md:text-5xl font-bold text-white mb-6 tracking-tight glow-text-cyan">
                  Upload an X-ray <br /> to start learning
                </h1>
                <p className="relative z-10 text-lg text-white/60 max-w-xl leading-relaxed">
                  I can analyze scans, highlight anomalies, and answer your questions about medical imaging. Try typing a question or uploading a file!
                </p>
              </div>
            </div>
          ) : (
            <div className="max-w-5xl mx-auto w-full pb-4">
              {messages.map((msg) => (
                <div key={msg.id} className={`flex w-full ${msg.sender === 'user' ? 'justify-end' : 'justify-start'} mb-8 group`}>
                  <div className={`flex max-w-[85%] md:max-w-[70%] gap-4 ${msg.sender === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                    {/* Avatar */}
                    <div className={`w-10 h-10 rounded-full flex-shrink-0 flex items-center justify-center border shadow-lg backdrop-blur-md ${
                      msg.sender === 'user'
                        ? 'bg-white/10 border-white/20'
                        : `bg-${themeColor}-900/40 border-${themeColor}-500/30 text-${themeColor}-400`
                    }`}>
                      {msg.sender === 'user' ? <User size={16} /> : <Brain size={16} />}
                    </div>

                    <div className="flex flex-col gap-3">
                      {/* User image */}
                      {msg.image && (
                        <div className={`rounded-3xl overflow-hidden shadow-xl border ${msg.sender === 'user' ? `border-${themeColor}-500/20` : 'border-white/10'}`}>
                          <img src={msg.image} alt={msg.fileName} className="max-w-sm max-h-64 object-cover" />
                        </div>
                      )}

                      {/* Message bubble */}
                      <div className={`rounded-3xl p-6 shadow-xl backdrop-blur-xl border relative overflow-hidden ${
                        msg.sender === 'user'
                          ? `bg-${themeColor}-500/10 border-${themeColor}-500/20 text-white rounded-tr-sm`
                          : 'bg-white/5 border-white/10 text-white/90 rounded-tl-sm'
                      }`}>
                        <div className={`absolute inset-0 pointer-events-none opacity-20 bg-gradient-to-b ${msg.sender === 'user' ? `from-${themeColor}-400/20 to-transparent` : 'from-white/10 to-transparent'}`}></div>

                        {msg.isTyping ? (
                          <div className="flex items-center gap-2 relative z-10">
                            <div className="flex gap-1">
                              <div className="w-2 h-2 rounded-full bg-cyan-400 animate-bounce" style={{ animationDelay: '0ms' }}></div>
                              <div className="w-2 h-2 rounded-full bg-cyan-400 animate-bounce" style={{ animationDelay: '150ms' }}></div>
                              <div className="w-2 h-2 rounded-full bg-cyan-400 animate-bounce" style={{ animationDelay: '300ms' }}></div>
                            </div>
                            <span className="text-sm text-white/40">Thinking …</span>
                          </div>
                        ) : msg.isProcessing ? (
                          <div className="relative z-10">
                            <p className="text-sm text-white/60 mb-3">Analyzing your X-ray through the AI pipeline …</p>
                            <PipelineProgress currentStage={pipelineStage} />
                          </div>
                        ) : (
                          <div className="text-[15px] leading-relaxed font-light relative z-10">
                            <MarkdownBody>{msg.text}</MarkdownBody>
                          </div>
                        )}
                      </div>

                      {/* Pipeline images */}
                      {msg.pipelineImages && !msg.isProcessing && (
                        <div className="flex flex-wrap gap-3 mt-1">
                          {msg.pipelineImages.enhanced && (
                            <button onClick={() => handleOpenComparison(msg)} className="rounded-2xl overflow-hidden border border-cyan-500/20 bg-black/40 backdrop-blur-md hover:border-cyan-400/50 transition-all cursor-pointer group/img">
                              <p className="text-[10px] font-bold uppercase tracking-widest text-cyan-300/70 px-3 pt-2">Enhanced</p>
                              <img src={msg.pipelineImages.enhanced} alt="Enhanced" className="max-w-[180px] max-h-[180px] object-contain p-2 group-hover/img:scale-105 transition-transform" />
                            </button>
                          )}
                          {msg.pipelineImages.mask && (
                            <button onClick={() => handleOpenComparison(msg)} className="rounded-2xl overflow-hidden border border-purple-500/20 bg-black/40 backdrop-blur-md hover:border-purple-400/50 transition-all cursor-pointer group/img">
                              <p className="text-[10px] font-bold uppercase tracking-widest text-purple-300/70 px-3 pt-2">Segmentation</p>
                              <img src={msg.pipelineImages.mask} alt="Mask" className="max-w-[180px] max-h-[180px] object-contain p-2 group-hover/img:scale-105 transition-transform" />
                            </button>
                          )}
                          {msg.pipelineImages.gradcam && (
                            <button onClick={() => handleOpenComparison(msg)} className="rounded-2xl overflow-hidden border border-red-500/20 bg-black/40 backdrop-blur-md hover:border-red-400/50 transition-all cursor-pointer group/img">
                              <p className="text-[10px] font-bold uppercase tracking-widest text-red-300/70 px-3 pt-2">Grad-CAM</p>
                              <img src={msg.pipelineImages.gradcam} alt="Grad-CAM" className="max-w-[180px] max-h-[180px] object-contain p-2 group-hover/img:scale-105 transition-transform" />
                            </button>
                          )}
                        </div>
                      )}

                      {/* Action buttons */}
                      {msg.showARButton && msg.sender === 'ai' && !msg.isProcessing && (
                        <div className="flex flex-wrap gap-3">
                          <button
                            onClick={() => handleOpenAR(msg)}
                            className="flex items-center gap-2 px-6 py-3 rounded-2xl bg-gradient-to-r from-cyan-500/30 to-blue-600/30 text-white border border-cyan-400/50 hover:shadow-[0_0_30px_rgba(34,211,238,0.8)] hover:border-cyan-300 transition-all duration-300 font-bold shadow-[0_0_20px_-5px_rgba(34,211,238,0.5)] w-fit"
                          >
                            <ScanEye size={18} /> AR Visualization
                          </button>
                          <button
                            onClick={() => handleOpenComparison(msg)}
                            className="flex items-center gap-2 px-6 py-3 rounded-2xl bg-white/5 text-white/70 border border-white/10 hover:bg-white/10 hover:text-white transition-all duration-300 font-bold w-fit"
                          >
                            <Layers size={18} /> Compare Images
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
              <div ref={chatEndRef} className="h-4" />
            </div>
          )}
        </div>

        {/* Input bar */}
        <div className="p-6 md:p-8 pt-0">
          <FloatingInput
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onSend={handleSend}
            onFileUpload={handleFileUpload}
            disabled={isAnalyzing}
            placeholder={
              isAnalyzing
                ? 'Analyzing X-ray … please wait'
                : role === 'student'
                  ? 'Ask a question or type "quiz me" …'
                  : 'Enter clinical findings or request analysis …'
            }
          />
        </div>
      </main>
    </div>
  );
};

// We need this icon import inline since it's used in the header
const Activity = ({ size, className, ...props }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className} {...props}>
    <path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2" />
  </svg>
);

export default ChatInterface;
