import React, { useState } from 'react';
import { Brain, Stethoscope, ChevronRight } from 'lucide-react';
import { GlassCard, GlassButton, GlassInput, GlassOtpInput, GlassModal } from './ui';
import lungBg from '../lung-bg.png';
import logo from '../logo.png';

const LandingPage = ({ onLogin }) => {
  const [activeModal, setActiveModal] = useState(null);
  const [loginStep, setLoginStep] = useState('input');
  const [inputValue, setInputValue] = useState('');
  const [otpValue, setOtpValue] = useState('');

  const handleOpenModal = (type) => {
    setActiveModal(type);
    setLoginStep('input');
    setInputValue('');
    setOtpValue('');
  };

  const handleDoctorLogin = () => {
    if (inputValue.trim()) setLoginStep('otp');
    else alert('Please enter a valid Medical License ID.');
  };

  const handleDoctorVerify = () => {
    if (otpValue.length >= 1) onLogin('doctor');
    else alert('Please enter the verification code.');
  };

  const handleStudentVerify = () => {
    if (otpValue.length >= 1) onLogin('student');
    else alert('Please enter the verification code.');
  };

  return (
    <div className="relative font-sans">
      {/* Fixed BG effects */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute inset-0 bg-[#060d17]"></div>
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_center,_var(--tw-gradient-stops))] from-[#0a2a4a] via-[#060d17] to-[#060d17] opacity-90"></div>
        <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-blue-600/10 rounded-full blur-[120px] animate-pulse pointer-events-none"></div>
        <div
          className="absolute bottom-0 left-0 w-[600px] h-[600px] bg-cyan-600/10 rounded-full blur-[120px] animate-pulse pointer-events-none"
          style={{ animationDelay: '2s' }}
        ></div>
      </div>

      {/* Lung BG — dark, subtle, blends into the background */}
      <div className="absolute top-20 left-1/2 -translate-x-1/2 w-[900px] h-[600px] pointer-events-none z-[1] overflow-hidden">
        <img
          src={lungBg}
          alt=""
          className="w-full h-full object-cover opacity-20 mix-blend-lighten"
          style={{
            maskImage: 'radial-gradient(circle at center, black 15%, transparent 60%)',
            WebkitMaskImage: 'radial-gradient(circle at center, black 15%, transparent 60%)',
            filter: 'brightness(0.3) saturate(1.2)',
          }}
        />
      </div>

      {/* Navbar */}
      <nav className="fixed top-0 z-40 w-full border-b border-white/5 bg-[#060d17]/95 backdrop-blur-xl">
        <div className="mx-auto flex h-24 max-w-7xl items-center justify-between px-6">
          <div
            className="flex items-center gap-4 cursor-pointer group"
            onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
          >
            <img src={logo} alt="Xpert Logo" className="h-20 w-20 object-contain" />
            <span className="text-3xl font-extrabold tracking-tight text-white text-shadow-sm font-sans group-hover:text-cyan-300 transition-colors">
              Xpert
            </span>
          </div>
          <div className="flex items-center gap-4"></div>
        </div>
      </nav>

      {/* Hero */}
      <main className="relative z-10 pt-24 font-sans">
        <section className="relative flex min-h-[90vh] flex-col items-center justify-center px-6 py-20 text-center">
          <div className="relative z-20 w-full max-w-6xl flex flex-col items-center">
            <h1 className="mb-8 text-4xl font-extrabold tracking-tight md:text-6xl text-white text-shadow-lg font-sans animate-shimmer">
              How can Xpert <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-300 via-cyan-200 to-blue-400 drop-shadow-[0_0_15px_rgba(34,211,238,0.6)]">
                help you today?
              </span>
            </h1>
            <p className="mb-16 max-w-3xl text-2xl text-white/80 font-light leading-relaxed font-sans text-shadow-sm">
              Choose your path to get started with intelligent, AI-powered visualization.
            </p>

            {/* Role cards */}
            <div className="mt-8 flex flex-col gap-6 md:flex-row md:gap-10 w-full justify-center items-center">
              <GlassCard
                className="group flex-1 w-full max-w-[380px] p-8 flex flex-col items-center text-center"
                onClick={() => handleOpenModal('student')}
              >
                <div className="mb-6 flex h-20 w-20 items-center justify-center rounded-3xl bg-gradient-to-br from-cyan-900/40 to-cyan-500/10 text-cyan-300 ring-1 ring-cyan-500/30 shadow-[inset_0_0_20px_rgba(34,211,238,0.2)] group-hover:bg-cyan-500/20 group-hover:text-cyan-200 transition-all duration-500 group-hover:shadow-[0_0_30px_rgba(34,211,238,0.6)] mx-auto">
                  <Brain size={40} />
                </div>
                <h2 className="mb-3 text-2xl font-bold text-white text-shadow-lg font-sans">Student Assistance</h2>
                <p className="text-base text-white/70 leading-relaxed mb-6 font-sans text-shadow-sm">
                  Learn, visualize, and explore X-rays with AI guidance.
                </p>
                <div className="mt-auto flex justify-center items-center text-cyan-300 font-bold text-lg group-hover:gap-4 gap-2 transition-all font-sans">
                  Enter Portal <ChevronRight size={20} />
                </div>
              </GlassCard>

              <GlassCard
                className="group flex-1 w-full max-w-[380px] p-8 flex flex-col items-center text-center"
                onClick={() => handleOpenModal('doctor')}
              >
                <div className="mb-6 flex h-20 w-20 items-center justify-center rounded-3xl bg-gradient-to-br from-blue-900/40 to-blue-500/10 text-blue-300 ring-1 ring-blue-500/30 shadow-[inset_0_0_20px_rgba(59,130,246,0.2)] group-hover:bg-blue-500/20 group-hover:text-blue-200 transition-all duration-500 group-hover:shadow-[0_0_30px_rgba(59,130,246,0.6)] mx-auto">
                  <Stethoscope size={40} />
                </div>
                <h2 className="mb-3 text-2xl font-bold text-white text-shadow-lg font-sans">Doctor Assistance</h2>
                <p className="text-base text-white/70 leading-relaxed mb-6 font-sans text-shadow-sm">
                  Analyze, verify, and visualize medical cases with AI support.
                </p>
                <div className="mt-auto flex justify-center items-center text-blue-300 font-bold text-lg group-hover:gap-4 gap-2 transition-all font-sans">
                  Enter Portal <ChevronRight size={20} />
                </div>
              </GlassCard>
            </div>
          </div>
        </section>
      </main>

      {/* ── Student modal ── */}
      <GlassModal isOpen={activeModal === 'student'} onClose={() => setActiveModal(null)}>
        <div className="mb-6 flex items-center gap-4">
          <div className="rounded-2xl border border-white/10 bg-gradient-to-br from-white/10 to-transparent p-4 text-cyan-400 shadow-inner">
            <Brain size={32} />
          </div>
          <h2 className="text-3xl font-bold text-white">Student Portal</h2>
        </div>
        {loginStep === 'input' ? (
          <>
            <p className="mb-8 font-sans">
              Welcome to the learning center. Please enter your verified university email.
            </p>
            <GlassInput
              placeholder="University Email (e.g student@med.edu)"
              type="email"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
            />
            <GlassButton
              variant="primary"
              className="w-full mt-8 justify-center text-lg py-4"
              onClick={() => {
                if (inputValue) setLoginStep('otp');
                else alert('Please enter an email');
              }}
            >
              Send Verification Code
            </GlassButton>
          </>
        ) : (
          <>
            <p className="mb-8 font-sans">
              A one-time password has been sent to{' '}
              <span className="text-cyan-400 font-bold">{inputValue}</span>.
            </p>
            <GlassOtpInput length={6} value={otpValue} onChange={setOtpValue} />
            <GlassButton variant="primary" className="w-full mt-8 justify-center text-lg py-4" onClick={handleStudentVerify}>
              Verify &amp; Login
            </GlassButton>
            <button
              onClick={() => setLoginStep('input')}
              className="w-full text-center mt-4 text-sm text-white/50 hover:text-white transition-colors"
            >
              Change Email
            </button>
          </>
        )}
      </GlassModal>

      {/* ── Doctor modal ── */}
      <GlassModal isOpen={activeModal === 'doctor'} onClose={() => setActiveModal(null)}>
        <div className="mb-6 flex items-center gap-4">
          <div className="rounded-2xl border border-white/10 bg-gradient-to-br from-white/10 to-transparent p-4 text-blue-400 shadow-inner">
            <Stethoscope size={32} />
          </div>
          <h2 className="text-3xl font-bold text-white">Doctor Access</h2>
        </div>
        {loginStep === 'input' ? (
          <>
            <p className="mb-8 font-sans">
              Secure clinical environment. Please enter your Medical License ID to verify credentials.
            </p>
            <GlassInput
              placeholder="Medical License ID"
              type="text"
              icon={Stethoscope}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
            />
            <GlassButton
              variant="primary"
              className="w-full mt-8 justify-center text-lg py-4 !from-blue-600/30 !to-blue-800/30 !border-blue-500/50 hover:!shadow-blue-500/50"
              onClick={handleDoctorLogin}
            >
              Send Verification Code
            </GlassButton>
          </>
        ) : (
          <>
            <p className="mb-8 font-sans">
              A one-time password has been sent to verify your Medical License ID{' '}
              <span className="text-blue-400 font-bold">{inputValue}</span>.
            </p>
            <GlassOtpInput length={6} value={otpValue} onChange={setOtpValue} />
            <GlassButton
              variant="primary"
              className="w-full mt-8 justify-center text-lg py-4 !from-blue-600/30 !to-blue-800/30 !border-blue-500/50 hover:!shadow-blue-500/50"
              onClick={handleDoctorVerify}
            >
              Verify &amp; Login
            </GlassButton>
            <button
              onClick={() => setLoginStep('input')}
              className="w-full text-center mt-4 text-sm text-white/50 hover:text-white transition-colors"
            >
              Change License ID
            </button>
          </>
        )}
      </GlassModal>
    </div>
  );
};

export default LandingPage;
