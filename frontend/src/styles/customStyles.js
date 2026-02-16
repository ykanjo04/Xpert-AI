/**
 * Shared CSS-in-JS style strings injected via <style> tags.
 */

export const customStyles = `
  body, .text-ivory, p, h1, h2, h3, span, input, button, textarea {
    color: #ffffff !important;
  }

  @keyframes deepBreath {
    0%, 100% { opacity: 0.6; transform: translate(-50%, -50%) scale(1); }
    50% { opacity: 0.9; transform: translate(-50%, -50%) scale(1.02); }
  }
  .animate-deep-breath {
    animation: deepBreath 8s ease-in-out infinite;
  }

  @keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-20px); }
  }
  .animate-float {
    animation: float 10s ease-in-out infinite;
  }
  .animate-float-delayed {
    animation: float 12s ease-in-out infinite;
    animation-delay: 5s;
  }

  @keyframes shimmer {
    0%, 100% { text-shadow: 0 0 10px rgba(34,211,238,0.3); }
    50% { text-shadow: 0 0 20px rgba(34,211,238,0.6); }
  }
  .animate-shimmer {
    animation: shimmer 3s ease-in-out infinite;
  }

  .glass-scroll::-webkit-scrollbar { width: 6px; }
  .glass-scroll::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
  .glass-scroll::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 10px; }
  .glass-scroll::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.25); }
  
  .text-shadow-sm { text-shadow: 0 2px 8px rgba(0,0,0,0.8); }
  .text-shadow-lg { text-shadow: 0 4px 15px rgba(0,0,0,0.9); }
  
  .glow-text-cyan {
    text-shadow: 0 0 20px rgba(34,211,238,0.7), 0 0 40px rgba(34,211,238,0.4);
  }
  .glow-text-blue {
    text-shadow: 0 0 20px rgba(59,130,246,0.7), 0 0 40px rgba(59,130,246,0.4);
  }
`;

export const arStyles = `
  .ar-glass-panel {
    background: rgba(10, 25, 47, 0.7);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
  }
  .ar-control-btn {
    transition: all 0.3s ease;
  }
  .ar-control-btn:hover {
    background: rgba(34, 211, 238, 0.1);
    border-color: rgba(34, 211, 238, 0.4);
    box-shadow: 0 0 15px rgba(34, 211, 238, 0.2);
  }
  @keyframes pulse-glow {
    0%, 100% { opacity: 0.4; filter: blur(20px); }
    50% { opacity: 0.7; filter: blur(25px); }
  }
  .ar-pulse {
    animation: pulse-glow 4s ease-in-out infinite;
  }
`;
