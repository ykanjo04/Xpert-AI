/**
 * AuthService — client-side session token and role management.
 *
 * Stores a random token, the user role, and the backend session ID
 * in localStorage so they survive page refreshes.
 */

export const AuthService = {
  isAuthenticated: () => !!localStorage.getItem('xpert_token'),
  getRole:         () => localStorage.getItem('xpert_role'),
  getSessionId:    () => localStorage.getItem('xpert_session'),

  login(role, sessionId) {
    const token = Math.random().toString(36).substring(2);
    localStorage.setItem('xpert_token', token);
    localStorage.setItem('xpert_role', role);
    localStorage.setItem('xpert_session', sessionId || '');
  },

  logout() {
    localStorage.removeItem('xpert_token');
    localStorage.removeItem('xpert_role');
    localStorage.removeItem('xpert_session');
  },
};
