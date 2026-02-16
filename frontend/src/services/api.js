/**
 * API service — all fetch calls to the FastAPI backend.
 */

const BASE = '';  // same-origin

async function _json(response) {
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.error || `Server error ${response.status}`);
  }
  return response.json();
}

/* ── Sessions ──────────────────────────────────────────────────────────── */

export async function createSession(role, studentId = 'anonymous') {
  const res = await fetch(`${BASE}/api/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ role, student_id: studentId }),
  });
  return _json(res);
}

export async function listSessions(role) {
  const url = role ? `${BASE}/api/sessions?role=${role}` : `${BASE}/api/sessions`;
  return _json(await fetch(url));
}

export async function getSession(sessionId) {
  return _json(await fetch(`${BASE}/api/sessions/${sessionId}`));
}

export async function getSessionReport(sessionId) {
  return _json(await fetch(`${BASE}/api/sessions/${sessionId}/report`));
}

export async function endSession(sessionId) {
  return _json(await fetch(`${BASE}/api/sessions/${sessionId}/end`, { method: 'POST' }));
}

export async function listStudentSessions() {
  return _json(await fetch(`${BASE}/api/doctor/students`));
}

/* ── Image analysis ────────────────────────────────────────────────────── */

export async function analyzeImage(file, { studentId, studentInput, action, result, sessionId } = {}) {
  const fd = new FormData();
  fd.append('file', file);
  fd.append('student_id', studentId || 'S001');
  fd.append('student_input', studentInput || 'normal');
  fd.append('action', action || 'diagnose');
  fd.append('result', result || 'pending');
  fd.append('session_id', sessionId || '');

  const res = await fetch(`${BASE}/api/analyze`, { method: 'POST', body: fd });
  return _json(res);
}

/* ── Conversational chat ───────────────────────────────────────────────── */

export async function sendChatMessage(sessionId, message, role = 'student') {
  const res = await fetch(`${BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, message, role }),
  });
  return _json(res);
}

/* ── Health ─────────────────────────────────────────────────────────────── */

export async function healthCheck() {
  return _json(await fetch(`${BASE}/api/health`));
}
