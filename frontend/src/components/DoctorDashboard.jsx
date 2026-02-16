/**
 * DoctorDashboard — shows student session reports for doctors.
 *
 * Features:
 *   • Session list with interaction counts
 *   • Per-session accuracy, competency gaps, recommendations
 *   • Interaction details (prediction, score, student input)
 */

import React, { useState, useEffect } from 'react';
import {
  ArrowLeft,
  User,
  Activity,
  BarChart3,
  AlertTriangle,
  BookOpen,
  FileText,
  Clock,
  LogOut,
  Brain,
} from 'lucide-react';
import { listStudentSessions, getSessionReport } from '../services/api';
import { customStyles } from '../styles/customStyles';
import logo from '../logo.png';

const DoctorDashboard = ({ onBack, onLogout }) => {
  const [sessions, setSessions] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [reportLoading, setReportLoading] = useState(false);

  // Fetch student sessions
  useEffect(() => {
    listStudentSessions()
      .then((data) => {
        setSessions(data.sessions || []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  // Fetch report when a session is selected
  useEffect(() => {
    if (!selectedId) { setReport(null); return; }
    setReportLoading(true);
    getSessionReport(selectedId)
      .then((r) => { setReport(r); setReportLoading(false); })
      .catch(() => setReportLoading(false));
  }, [selectedId]);

  return (
    <div className="flex h-screen w-full bg-[#03060a] text-white font-sans overflow-hidden relative">
      <style>{customStyles}</style>

      {/* Background */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,_var(--tw-gradient-stops))] from-blue-900/30 via-[#060d17] to-[#03060a]"></div>
        <div className="absolute top-[-10%] right-[-5%] w-[40vw] h-[40vw] bg-blue-500/10 rounded-full blur-[120px] opacity-60"></div>
      </div>

      {/* Sidebar */}
      <aside className="w-80 hidden md:flex flex-col border-r border-blue-500/20 bg-white/5 backdrop-blur-xl z-20 relative shadow-[5px_0_30px_rgba(0,0,0,0.3)]">
        <div className="p-6 border-b border-white/10 flex items-center gap-3">
          <img src={logo} alt="Xpert" className="h-12 w-12 object-contain" />
          <span className="text-2xl font-bold tracking-wide text-white/90">Reports</span>
        </div>

        <div className="p-5 flex-1 overflow-y-auto glass-scroll">
          <button
            onClick={onBack}
            className="w-full py-3 rounded-2xl mb-6 flex items-center justify-center gap-2 font-bold transition-all bg-gradient-to-r from-blue-500 to-blue-600 text-white shadow-[0_0_20px_rgba(59,130,246,0.4)] hover:shadow-[0_0_30px_rgba(59,130,246,0.6)] border border-white/20"
          >
            <ArrowLeft size={18} /> Back to Chat
          </button>

          <p className="text-[10px] font-bold text-white/40 uppercase tracking-[0.2em] px-3 mb-3">Student Sessions</p>

          {loading ? (
            <div className="flex items-center justify-center py-12">
              <div className="w-6 h-6 border-2 border-blue-400/30 border-t-blue-400 rounded-full animate-spin"></div>
            </div>
          ) : sessions.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 text-center">
              <FileText size={24} className="text-white/10 mb-3" />
              <p className="text-sm text-white/30">No student sessions yet</p>
            </div>
          ) : (
            <div className="space-y-2">
              {sessions.map((s) => (
                <button
                  key={s.id}
                  onClick={() => setSelectedId(s.id)}
                  className={`w-full text-left p-4 rounded-xl border transition-all ${
                    selectedId === s.id
                      ? 'bg-blue-500/10 border-blue-500/30'
                      : 'border-white/5 hover:bg-white/5'
                  }`}
                >
                  <div className="flex items-center gap-3 mb-1">
                    <div className="h-8 w-8 rounded-full bg-blue-900/30 flex items-center justify-center text-blue-300 border border-blue-500/20">
                      <User size={14} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-semibold truncate">{s.student_id || 'anonymous'}</p>
                      <p className="text-[10px] text-white/40">{s.interaction_count || 0} analyses</p>
                    </div>
                  </div>
                  <p className="text-[10px] text-white/30 mt-1">
                    <Clock size={10} className="inline mr-1" />
                    {new Date(s.created_at).toLocaleString()}
                  </p>
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="p-4 border-t border-white/10 bg-black/20">
          <button
            onClick={onLogout}
            className="flex items-center gap-2 text-white/30 hover:text-red-400 transition-colors px-3 py-2 w-full rounded-lg hover:bg-white/5"
          >
            <LogOut size={16} /> Sign Out
          </button>
        </div>
      </aside>

      {/* Main area */}
      <main className="flex-1 flex flex-col relative z-10 h-full overflow-y-auto glass-scroll">
        <header className="h-20 border-b border-white/10 flex items-center px-8 bg-white/[0.05] backdrop-blur-xl flex-shrink-0">
          <div className="flex items-center gap-2 px-4 py-2 rounded-full border text-xs font-bold uppercase tracking-widest bg-gradient-to-r from-blue-500/20 to-blue-900/20 border-blue-500/40 text-blue-300 shadow-[0_0_15px_rgba(59,130,246,0.2)]">
            <BarChart3 size={14} />
            Student Performance Reports
          </div>
        </header>

        <div className="flex-1 p-8 md:p-12">
          {!selectedId ? (
            <div className="flex-1 flex items-center justify-center h-full">
              <div className="text-center">
                <BarChart3 size={64} className="text-white/10 mx-auto mb-4" />
                <p className="text-xl text-white/30 font-bold">Select a student session</p>
                <p className="text-sm text-white/20 mt-2">Click on a session in the sidebar to view the report</p>
              </div>
            </div>
          ) : reportLoading ? (
            <div className="flex items-center justify-center h-64">
              <div className="w-8 h-8 border-2 border-blue-400/30 border-t-blue-400 rounded-full animate-spin"></div>
            </div>
          ) : report ? (
            <div className="max-w-4xl mx-auto space-y-8">
              {/* Summary cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="rounded-2xl bg-white/5 border border-white/10 p-6 backdrop-blur-md">
                  <div className="flex items-center gap-3 mb-4">
                    <Activity size={20} className="text-cyan-400" />
                    <span className="text-sm font-bold text-white/60 uppercase tracking-wider">Analyses</span>
                  </div>
                  <p className="text-4xl font-bold text-white">{report.total_analyses}</p>
                  <p className="text-xs text-white/40 mt-1">{report.message_count || 0} chat messages</p>
                </div>

                <div className="rounded-2xl bg-white/5 border border-white/10 p-6 backdrop-blur-md">
                  <div className="flex items-center gap-3 mb-4">
                    <BarChart3 size={20} className="text-green-400" />
                    <span className="text-sm font-bold text-white/60 uppercase tracking-wider">Accuracy</span>
                  </div>
                  <p className="text-4xl font-bold text-white">{report.accuracy_percent}%</p>
                  <p className="text-xs text-white/40 mt-1">Student vs AI agreement</p>
                </div>

                <div className="rounded-2xl bg-white/5 border border-white/10 p-6 backdrop-blur-md">
                  <div className="flex items-center gap-3 mb-4">
                    <Brain size={20} className="text-purple-400" />
                    <span className="text-sm font-bold text-white/60 uppercase tracking-wider">Avg Score</span>
                  </div>
                  <p className="text-4xl font-bold text-white">{(report.avg_pneumonia_score * 100).toFixed(1)}%</p>
                  <p className="text-xs text-white/40 mt-1">Mean pneumonia probability</p>
                </div>
              </div>

              {/* Competency gaps */}
              {report.competency_gaps.length > 0 && (
                <div className="rounded-2xl bg-white/5 border border-white/10 p-6 backdrop-blur-md">
                  <div className="flex items-center gap-3 mb-4">
                    <AlertTriangle size={20} className="text-yellow-400" />
                    <span className="text-sm font-bold text-white/60 uppercase tracking-wider">Competency Gaps</span>
                  </div>
                  <div className="space-y-3">
                    {report.competency_gaps.map((gap, i) => (
                      <div key={i} className="p-3 bg-yellow-500/5 border border-yellow-500/10 rounded-xl text-sm text-white/70">
                        {gap}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Recommendations */}
              {report.recommendations.length > 0 && (
                <div className="rounded-2xl bg-white/5 border border-white/10 p-6 backdrop-blur-md">
                  <div className="flex items-center gap-3 mb-4">
                    <BookOpen size={20} className="text-blue-400" />
                    <span className="text-sm font-bold text-white/60 uppercase tracking-wider">Learning Recommendations</span>
                  </div>
                  <div className="space-y-3">
                    {report.recommendations.map((rec, i) => (
                      <div key={i} className="p-3 bg-blue-500/5 border border-blue-500/10 rounded-xl text-sm text-white/70">
                        {rec}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Interaction timeline */}
              {report.interactions.length > 0 && (
                <div className="rounded-2xl bg-white/5 border border-white/10 p-6 backdrop-blur-md">
                  <div className="flex items-center gap-3 mb-4">
                    <Clock size={20} className="text-white/40" />
                    <span className="text-sm font-bold text-white/60 uppercase tracking-wider">Interaction Timeline</span>
                  </div>
                  <div className="space-y-4">
                    {report.interactions.map((ix, i) => (
                      <div key={i} className="flex gap-4 items-start border-l-2 border-white/10 pl-4 py-2 hover:border-cyan-500/50 transition-colors">
                        <div className="flex-1">
                          <div className="flex items-center gap-3 mb-1">
                            <span className={`text-xs font-bold px-2 py-0.5 rounded-full border ${
                              ix.prediction === 'pneumonia'
                                ? 'bg-red-500/10 border-red-500/20 text-red-300'
                                : 'bg-green-500/10 border-green-500/20 text-green-300'
                            }`}>
                              {ix.prediction?.toUpperCase()}
                            </span>
                            <span className="text-xs text-white/40 font-mono">
                              Score: {((ix.pneumonia_score || 0) * 100).toFixed(1)}%
                            </span>
                          </div>
                          <p className="text-sm text-white/60">
                            Student said: <span className="text-white/80 font-medium">{ix.student_input || '—'}</span>
                          </p>
                          {ix.competency_gap && (
                            <p className="text-xs text-yellow-300/60 mt-1">Gap: {ix.competency_gap}</p>
                          )}
                        </div>
                        <span className="text-[10px] text-white/30 whitespace-nowrap">
                          {new Date(ix.created_at).toLocaleTimeString()}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-center text-white/30">Report not found.</div>
          )}
        </div>
      </main>
    </div>
  );
};

export default DoctorDashboard;
