/**
 * App.jsx — root component (routing shell only).
 *
 * All visual logic lives in dedicated components:
 *   • LandingPage     – role selection + OTP login
 *   • ChatInterface   – chat, upload, pipeline progress, markdown
 *   • DoctorDashboard – student reports view
 *   • ARMode          – AR visualization (invoked from ChatInterface)
 *   • ImageComparison – lightbox comparison (invoked from ChatInterface)
 */

import React, { useState, useEffect } from 'react';
import LandingPage from './components/LandingPage';
import ChatInterface from './components/ChatInterface';
import DoctorDashboard from './components/DoctorDashboard';
import { AuthService } from './services/auth';
import { createSession } from './services/api';
import { customStyles } from './styles/customStyles';

export default function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(AuthService.isAuthenticated());
  const [userRole, setUserRole] = useState(AuthService.getRole());
  const [sessionId, setSessionId] = useState(AuthService.getSessionId());
  const [showDashboard, setShowDashboard] = useState(false);

  // Inject global styles
  useEffect(() => {
    const tag = document.createElement('style');
    tag.textContent = customStyles;
    document.head.appendChild(tag);
    return () => document.head.removeChild(tag);
  }, []);

  // Login: create a backend session, persist locally
  const handleLogin = async (role) => {
    try {
      const data = await createSession(role);
      AuthService.login(role, data.session_id);
      setUserRole(role);
      setSessionId(data.session_id);
      setIsAuthenticated(true);
    } catch (err) {
      console.error('Session creation failed', err);
      // Fallback — work without a backend session
      AuthService.login(role, '');
      setUserRole(role);
      setSessionId('');
      setIsAuthenticated(true);
    }
  };

  const handleLogout = () => {
    AuthService.logout();
    setIsAuthenticated(false);
    setUserRole(null);
    setSessionId(null);
    setShowDashboard(false);
  };

  // Not logged in → landing page
  if (!isAuthenticated) {
    return <LandingPage onLogin={handleLogin} />;
  }

  // Doctor dashboard
  if (showDashboard && userRole === 'doctor') {
    return (
      <DoctorDashboard
        onBack={() => setShowDashboard(false)}
        onLogout={handleLogout}
      />
    );
  }

  // Chat interface
  return (
    <ChatInterface
      role={userRole}
      onLogout={handleLogout}
      sessionId={sessionId}
      onShowDashboard={userRole === 'doctor' ? () => setShowDashboard(true) : undefined}
    />
  );
}
