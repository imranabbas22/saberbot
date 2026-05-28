import { useState, useRef, useEffect, useCallback } from 'react'
import './index.css'

const SESSION_TIMEOUT = 180  // 3 minutes
const WARN_BEFORE = 30       // warn with 30s remaining
const API_URL = '/api/chat'

const getSessionId = () => {
  let sid = localStorage.getItem('saberbot_session')
  if (!sid) {
    sid = crypto.randomUUID?.() || Math.random().toString(36).substring(2) + Date.now().toString(36)
    localStorage.setItem('saberbot_session', sid)
  }
  return sid
}

// Check if legal terms were accepted
const hasAcceptedTerms = () => localStorage.getItem('saberbot_terms_accepted') === 'true'
const setTermsAccepted = () => localStorage.setItem('saberbot_terms_accepted', 'true')

function App() {
  const [step, setStep] = useState(hasAcceptedTerms() ? 'chat' : 'landing')
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [expandedSources, setExpandedSources] = useState({})
  const [remaining, setRemaining] = useState(5)
  const [sessionId] = useState(getSessionId)
  const [showFeedback, setShowFeedback] = useState(false)
  const [feedbackSent, setFeedbackSent] = useState(false)
  const [globalStats, setGlobalStats] = useState({ total_requests: 0, feedback_count: 0 })
  const [timer, setTimer] = useState(SESSION_TIMEOUT)
  const [showTimerWarn, setShowTimerWarn] = useState(false)
  const [showEula, setShowEula] = useState(false)
  const [eulaTab, setEulaTab] = useState('terms')
  const [agreed, setAgreed] = useState(false)
  const messagesEndRef = useRef(null)
  const heartbeatRef = useRef(null)
  const timerRef = useRef(null)

  // Load session + global stats
  useEffect(() => {
    fetch('/api/stats').then(r => r.json()).then(d => setGlobalStats(d)).catch(() => {})
    fetch(`/api/session/${sessionId}`).then(r => r.json()).then(d => {
      setRemaining(d.remaining)
      if (!d.can_chat && d.requests_used > 0) setShowFeedback(true)
    }).catch(() => {})
  }, [sessionId])

  // Reset agreement checkbox when EULA modal opens
  useEffect(() => { if (showEula) setAgreed(false) }, [showEula])

  // Heartbeat + timer
  useEffect(() => {
    if (step !== 'chat' || showFeedback) return
    const heartbeat = async () => {
      try {
        const res = await fetch(`/api/heartbeat/${sessionId}`)
        const data = await res.json()
        if (!data.active) {
          setShowFeedback(true)
          clearInterval(heartbeatRef.current)
          clearInterval(timerRef.current)
        } else {
          setTimer(data.remaining_seconds)
        }
      } catch {}
    }
    heartbeat()
    heartbeatRef.current = setInterval(heartbeat, 15000)
    timerRef.current = setInterval(() => {
      setTimer(t => {
        if (t <= WARN_BEFORE + 1 && t > 0) setShowTimerWarn(true)
        if (t <= 0) { setShowFeedback(true); return 0 }
        return t - 1
      })
    }, 1000)
    return () => { clearInterval(heartbeatRef.current); clearInterval(timerRef.current) }
  }, [step, showFeedback, sessionId])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const extendTimer = async () => {
    try { await fetch(`/api/session-extend/${sessionId}`, { method: 'POST' }) } catch {}
    setTimer(SESSION_TIMEOUT)
    setShowTimerWarn(false)
  }

  const exitSession = async () => {
    try { await fetch(`/api/session-expire/${sessionId}`, { method: 'POST' }) } catch {}
    setShowFeedback(true)
    setShowTimerWarn(false)
  }

  const acceptTerms = () => {
    if (!agreed) return
    setTermsAccepted()
    setShowEula(false)
    setStep('chat')
    setMessages([{
      role: 'assistant',
      content: '# 🤖 Welcome to SaberBot\n\nYour UAE law guide. Ask any question about UAE federal laws — labor, tax, criminal, family, business, and more.\n\n**Important:** This is an educational AI tool. Always verify with a qualified UAE lawyer for legal decisions.\n\nYou have **5 free queries**. Session will timeout after **3 minutes of inactivity**.',
      sources: [],
    }])
  }

  const sendMessage = useCallback(async () => {
    if (!input.trim() || loading || remaining <= 0) return
    const userMsg = { role: 'user', content: input, sources: [] }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)
    try {
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMsg.content, mode: 'auto', session_id: sessionId }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setRemaining(data.remaining ?? 0)
      setTimer(SESSION_TIMEOUT)
      setMessages(prev => [...prev, {
        role: 'assistant', content: data.response || 'No response.',
        sources: data.sources || [], mode: data.mode || 'unknown',
        model: data.model || '', timing_ms: data.timing_ms || 0,
      }])
      if (data.remaining <= 0) setShowFeedback(true)
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant', content: `⚠️ **Error:** ${err.message}`, sources: [], mode: 'error',
      }])
    } finally { setLoading(false) }
  }, [input, loading, remaining, sessionId])

  const submitFeedback = async (npsScore, comment) => {
    try {
      await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, nps_score: npsScore, comment }),
      })
      setFeedbackSent(true)
      setShowFeedback(false)
    } catch { alert('Failed to submit feedback.') }
  }

  const toggleSources = (idx) => setExpandedSources(p => ({ ...p, [idx]: !p[idx] }))
  const handleKeyDown = (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage() } }

  const goToLanding = () => {
    setStep('landing')
    setMessages([])
    setInput('')
    setLoading(false)
    setExpandedSources({})
  }

  const startChat = () => {
    if (hasAcceptedTerms()) {
      setStep('chat')
      setMessages([{
        role: 'assistant',
        content: '# 🤖 Welcome to SaberBot\n\nYour UAE law guide. Ask any question about UAE federal laws — labor, tax, criminal, family, business, and more.\n\n**Important:** This is an educational AI tool. Always verify with a qualified UAE lawyer for legal decisions.\n\nYou have **5 free queries**. Session will timeout after **3 minutes of inactivity**.',
        sources: [],
      }])
    } else {
      setShowEula(true)
    }
  }

  // ---- Landing Page ----
  if (step === 'landing') {
    return (
      <div className="landing-page">
        <nav className="landing-nav">
          <div className="nav-inner">
            <span className="logo-text-lg">🤖 SaberBot</span>
            <div className="nav-links">
              <button className="nav-link" onClick={() => setShowEula(true)}>Legal</button>
              <button className="btn-primary" onClick={startChat}>Get Started</button>
            </div>
          </div>
        </nav>
        <section className="hero">
          <div className="hero-bg" />
          <div className="hero-content">
            <span className="hero-badge">🔒 Privacy First</span>
            <h1>UAE Law. <span className="gradient-text">AI-Powered Guide.</span></h1>
            <p className="hero-sub">
              Educational AI tool for exploring UAE federal laws. Every answer cites specific articles.
              Zero data retention. No legal advice — just legal information to guide your research.
            </p>
            <div className="hero-actions">
              <button className="btn-primary btn-lg" onClick={startChat}>
                Try SaberBot →
              </button>
            </div>
            <div className="hero-stats">
              <div className="hero-stat"><strong>{globalStats.total_requests?.toLocaleString() || 0}</strong> Queries Answered</div>
              <div className="hero-stat"><strong>{globalStats.feedback_count || 0}</strong> Reviews</div>
              <div className="hero-stat"><strong>7,005</strong> Laws Indexed</div>
            </div>
          </div>
        </section>
        <section className="features">
          <div className="features-grid">
            <div className="feature-card"><span className="feature-icon">📚</span><h3>Citation-First Answers</h3><p>Every response includes the exact law name, article number, and clause — no guessing.</p></div>
            <div className="feature-card"><span className="feature-icon">🛡️</span><h3>Zero Data Retention</h3><p>Session auto-expires after 3 minutes. No tracking, no cookies, no storage of your queries.</p></div>
            <div className="feature-card"><span className="feature-icon">⚖️</span><h3>7,005 UAE Laws</h3><p>Federal laws across labor, tax, criminal, family, business, tenancy, visa, and data privacy.</p></div>
            <div className="feature-card"><span className="feature-icon">🎯</span><h3>5 Free Queries</h3><p>Try before you commit. After 5 queries, share feedback to help me improve the project.</p></div>
          </div>
        </section>
        <footer className="landing-footer">
          <p>🤖 SaberBot · Created by <a href="https://linkedin.com/in/syed-imran-abbas" target="_blank" rel="noopener noreferrer" style={{color:'#60a5fa'}}>Syed Imran Abbas</a> · <a href="https://github.com/imranabbas22/saberbot" target="_blank" rel="noopener noreferrer" style={{color:'#60a5fa'}}>GitHub</a> · Portfolio Project · Not legal advice</p>
        </footer>

        {showEula && (
          <div className="modal-overlay" onClick={() => setShowEula(false)}>
            <div className="modal" onClick={e => e.stopPropagation()}>
              <div className="modal-tabs">
                {['terms', 'privacy', 'eula'].map(tab => (
                  <button key={tab} className={`modal-tab ${eulaTab === tab ? 'active' : ''}`} onClick={() => setEulaTab(tab)}>
                    {tab === 'terms' ? 'Terms of Use' : tab === 'privacy' ? 'Privacy Policy' : 'EULA'}
                  </button>
                ))}
              </div>
              <div className="modal-body">
                {eulaTab === 'terms' && <TermsContent />}
                {eulaTab === 'privacy' && <PrivacyContent />}
                {eulaTab === 'eula' && <EulaContent />}
              </div>
              <div className="modal-footer">
                <label className="agree-check">
                  <input type="checkbox" checked={agreed} onChange={e => setAgreed(e.target.checked)} />
                  <span>I have read and agree to the Terms of Use, Privacy Policy, and EULA</span>
                </label>
                <button className="btn-primary btn-lg" onClick={acceptTerms} disabled={!agreed} style={{opacity: agreed ? 1 : 0.5}}>Accept & Continue</button>
              </div>
            </div>
          </div>
        )}
      </div>
    )
  }

  // ---- Timer Warning Overlay ----
  if (showTimerWarn) {
    return (
      <div className="timer-warn-overlay">
        <div className="timer-warn-card">
          <div className="timer-warn-icon">⏰</div>
          <h2>Session Expiring Soon</h2>
          <p>Your session will expire in <strong>{timer}s</strong>. Any uploaded files will be permanently deleted.</p>
          <div className="timer-warn-actions">
            <button className="btn-primary btn-lg" onClick={extendTimer}>Continue Session →</button>
            <button className="btn-secondary btn-lg" onClick={exitSession}>Exit & Delete Data</button>
          </div>
        </div>
      </div>
    )
  }

  // ---- Chat Interface ----
  return (
    <div className="app-container">
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="logo">
            <span className="logo-icon">🤖</span>
            <div className="logo-text"><h1>SaberBot</h1><span className="tagline">UAE Law Guide</span></div>
          </div>
          <div className="sidebar-stats">
            <div className={`remaining-badge ${remaining <= 0 ? 'used-up' : ''}`}>
              <span className="remaining-num">{remaining}</span>
              <span className="remaining-label">queries left</span>
            </div>
            <div className="timer-badge">
              <span className="timer-icon">⏱️</span>
              <span className={`timer-num ${timer < WARN_BEFORE ? 'timer-warn' : ''}`}>{Math.floor(timer / 60)}:{(timer % 60).toString().padStart(2, '0')}</span>
            </div>
          </div>
        </div>
        <nav className="sidebar-nav">
          <div className="nav-section"><h3>Global Stats</h3>
            <div className="global-stats">
              <div className="stat"><span className="stat-value">{globalStats.total_requests?.toLocaleString() || 0}</span><span className="stat-label">Queries</span></div>
              <div className="stat"><span className="stat-value">{globalStats.feedback_count || 0}</span><span className="stat-label">Reviews</span></div>
            </div>
          </div>
          <div className="nav-section"><h3>Security</h3>
            <ul className="tips-list">
              <li>Session auto-expires in 3 min</li>
              <li>No personal data stored</li>
              <li>Zero cookies or tracking</li>
            </ul>
          </div>
          <div className="nav-section"><h3>Legal Areas</h3>
            <div className="topic-chips">
              {['Labor Law','Tax & VAT','Criminal','Family Law','Business','Tenancy','Visa','Data Privacy'].map(t => (
                <span key={t} className="topic-chip">{t}</span>
              ))}
            </div>
          </div>
          <div className="nav-section"><h3>Legal Documents</h3>
            <button className="nav-link" onClick={() => setShowEula(true)}>Terms · Privacy · EULA</button>
          </div>
        </nav>
        <div className="sidebar-footer">
          <button className="back-btn" onClick={goToLanding}>← Back to Home</button>
          <p style={{fontSize:'10px', lineHeight:'1.4'}}>
            Created by <a href="https://linkedin.com/in/syed-imran-abbas" target="_blank" rel="noopener noreferrer" style={{color:'#60a5fa'}}>Syed Imran Abbas</a><br/>
            7,005 laws · Portfolio · Not legal advice
          </p>
        </div>
      </aside>

      <main className="main-content">
        <div className="chat-container">
          <div className="messages-area">
            {messages.map((msg, idx) => (
              <div key={idx} className={`message ${msg.role}`}>
                <div className="message-avatar">{msg.role === 'user' ? '👤' : '🤖'}</div>
                <div className="message-body">
                  <div className="message-header">
                    <span className="message-author">{msg.role === 'user' ? 'You' : 'SaberBot'}</span>
                    {msg.mode && <span className={`mode-tag ${modeClass(msg.mode)}`}>{modeLabel(msg.mode)}</span>}
                    {msg.timing_ms > 0 && <span className="timing">{(msg.timing_ms / 1000).toFixed(1)}s</span>}
                  </div>
                  <div className="message-content" dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.content) }} />
                  {msg.sources?.length > 0 && (
                    <div className="sources-section">
                      <button className="sources-toggle" onClick={() => toggleSources(idx)}>
                        📚 {msg.sources.length} Sources {expandedSources[idx] ? '▲' : '▼'}
                      </button>
                      {expandedSources[idx] && (
                        <div className="sources-list">
                          {msg.sources.map((src, si) => (
                            <div key={si} className="source-item">
                              <span className="source-title">{src.title || 'Unknown'}</span>
                              <div className="source-details">
                                {src.law_number && <span>Law No. {src.law_number}</span>}
                                {src.law_year && <span> · {src.law_year}</span>}
                                {src.article && <span> · Article {src.article}</span>}
                                {src.retrieval_method && <span className="retrieval-tag">{src.retrieval_method}</span>}
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))}
            {loading && (
              <div className="message assistant">
                <div className="message-avatar">🤖</div>
                <div className="message-body">
                  <div className="message-header"><span className="message-author">SaberBot</span></div>
                  <div className="typing-indicator"><span></span><span></span><span></span></div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="input-area">
            {showFeedback ? (
              <FeedbackForm onSubmit={submitFeedback} sent={feedbackSent} />
            ) : (
              <>
                <div className="input-wrapper">
                  <textarea className="chat-input" placeholder={remaining > 0 ? "Ask about UAE law..." : "All queries used"} value={input}
                    onChange={e => setInput(e.target.value)} onKeyDown={handleKeyDown} rows={1} disabled={loading || remaining <= 0} maxLength={4096} />
                  <button className="send-btn" onClick={sendMessage} disabled={!input.trim() || loading || remaining <= 0}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 2L11 13"/><path d="M22 2L15 22L11 13L2 9L22 2Z"/></svg>
                  </button>
                </div>
                <p className="input-hint">{remaining > 0 ? `${remaining} query${remaining !== 1 ? 'ies' : 'y'} · ${Math.floor(timer / 60)}:${(timer % 60).toString().padStart(2, '0')} remaining` : 'Session complete'}</p>
              </>
            )}
          </div>
        </div>
      </main>

      {showEula && (
        <div className="modal-overlay" onClick={() => setShowEula(false)}>
          <div className="modal modal-sm" onClick={e => e.stopPropagation()}>
            <div className="modal-tabs">
              {['terms', 'privacy', 'eula'].map(tab => (
                <button key={tab} className={`modal-tab ${eulaTab === tab ? 'active' : ''}`} onClick={() => setEulaTab(tab)}>
                  {tab === 'terms' ? 'Terms' : tab === 'privacy' ? 'Privacy' : 'EULA'}
                </button>
              ))}
            </div>
            <div className="modal-body">
              {eulaTab === 'terms' && <TermsContent />}
              {eulaTab === 'privacy' && <PrivacyContent />}
              {eulaTab === 'eula' && <EulaContent />}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function FeedbackForm({ onSubmit, sent }) {
  const [score, setScore] = useState(null)
  const [comment, setComment] = useState('')
  const [submitting, setSubmitting] = useState(false)
  if (sent) return (
    <div className="feedback-thanks"><div className="thanks-icon">🙏</div><h3>Thank You!</h3><p>Your session is now complete. All temporary data has been deleted. Your feedback helps me improve this portfolio project!</p></div>
  )
  const handleSubmit = async () => { if (score === null) return; setSubmitting(true); await onSubmit(score, comment); setSubmitting(false) }
  return (
    <div className="feedback-form">
      <h3>Session Complete 🙏</h3>
      <p className="feedback-intro">How likely would you recommend SaberBot to someone interested in UAE law?</p>
      <div className="nps-scale">
        {[0,1,2,3,4,5,6,7,8,9,10].map(n => (
          <button key={n} className={`nps-btn ${score === n ? 'selected' : ''} ${n >= 9 ? 'promoter' : n >= 7 ? 'passive' : 'detractor'}`} onClick={() => setScore(n)}>{n}</button>
        ))}
      </div>
      <div className="nps-labels"><span>Not likely</span><span>Very likely</span></div>
      {score !== null && (
        <div className="nps-comment-area">
          <textarea className="feedback-comment" placeholder="Any feedback? (optional)" value={comment} onChange={e => setComment(e.target.value)} rows={3} />
          <button className="send-btn feedback-submit" onClick={handleSubmit} disabled={submitting}>{submitting ? 'Sending...' : 'Submit Feedback'}</button>
        </div>
      )}
    </div>
  )
}

const modeLabel = (m) => ({ answer: 'Answered', clarify: 'Clarification', clarify_found: 'Partial', refuse: 'No Match', error: 'Error', limit_reached: 'Limit' }[m] || m)
const modeClass = (m) => ({ answer: 'mode-answer', clarify: 'mode-clarify', clarify_found: 'mode-clarify', refuse: 'mode-refuse', error: 'mode-error' }[m] || '')

const renderMarkdown = (text) => {
  if (!text) return ''
  let html = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/^### (.+)$/gm, '<h4>$1</h4>').replace(/^## (.+)$/gm, '<h3>$1</h3>').replace(/^# (.+)$/gm, '<h2>$1</h2>')
    .replace(/^- (.+)$/gm, '<li>$1</li>').replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>')
    .replace(/^(\d+)\. (.+)$/gm, '<li value="$1">$2</li>').replace(/\n\n/g, '</p><p>').replace(/\n/g, '<br>')
  return `<p>${html}</p>`
}

const TermsContent = () => (
  <div className="legal-text">
    <h2>Terms of Use</h2>
    <p><strong>Last updated:</strong> May 2026</p>
    <h3>1. Nature of Service</h3>
    <p>SaberBot is a <strong>portfolio project and educational tool</strong> created for demonstration purposes. It provides AI-generated legal <strong>information</strong> based on UAE federal laws. It is NOT a substitute for professional legal advice, does not constitute legal counsel, and is not a law firm or legal service.</p>
    <h3>2. No Legal Advice</h3>
    <p>SaberBot does NOT provide legal advice, legal opinions, or attorney-client representation. <strong>Always consult a qualified, licensed UAE lawyer</strong> for any legal decisions or actions. Do not rely on AI-generated responses for legal matters.</p>
    <h3>3. No Attorney-Client Relationship</h3>
    <p>Use of SaberBot does not create an attorney-client relationship. Responses are for informational and educational purposes only. No confidential or privileged relationship is formed.</p>
    <h3>4. Acceptable Use</h3>
    <p>You agree not to: (a) misuse the service for illegal purposes, (b) attempt to extract the underlying database, (c) rely on responses as legal advice, (d) use automated scripts to bypass session limits.</p>
    <h3>5. Limitation of Liability</h3>
    <p><strong>SABERBOT IS PROVIDED "AS IS" WITHOUT ANY WARRANTIES, EXPRESS OR IMPLIED.</strong> The creator of this portfolio project shall not be liable for any damages, losses, or legal consequences arising from use of or reliance on this tool. Use entirely at your own risk.</p>
    <h3>6. Session & Data</h3>
    <p>Sessions expire after 3 minutes of inactivity. No personal data is retained after session expiry. This is a portfolio project — use with that understanding.</p>
    <h3>7. Feedback</h3>
    <p>Any feedback submitted may be used as anonymous testimonials for the creator's portfolio. No identifying information will be published without explicit consent.</p>
  </div>
)

const PrivacyContent = () => (
  <div className="legal-text">
    <h2>Privacy Policy</h2>
    <p><strong>Last updated:</strong> May 2026</p>
    <h3>1. Data We Collect</h3>
    <p><strong>We collect ZERO personal data.</strong> No cookies, no trackers, no analytics, no advertising. Session IDs are randomly generated and stored only in your browser's local storage on your device.</p>
    <h3>2. Chat Queries</h3>
    <p>Your questions are sent to Groq (Llama 3.3 70B) for AI processing. We do not log, store, or retain your questions after the response is delivered. Groq's privacy policy applies to the API transit layer.</p>
    <h3>3. Session Tracking</h3>
    <p>We track only: anonymous session ID, request count, and NPS feedback score (if you choose to submit it). No names, emails, IP addresses, or browser fingerprints are stored.</p>
    <h3>4. Feedback/Testimonials</h3>
    <p>NPS scores and comments are stored anonymously. If you include identifying information in feedback, it will not be published without your explicit consent.</p>
    <h3>5. Third Parties</h3>
    <p>We use Groq for AI inference. No other third-party services have access to your data.</p>
    <h3>6. Data Deletion</h3>
    <p>All session data is automatically deleted after session expiry. No permanent records of your usage are maintained.</p>
  </div>
)

const EulaContent = () => (
  <div className="legal-text">
    <h2>End User License Agreement</h2>
    <p><strong>Last updated:</strong> May 2026</p>
    <h3>1. License Grant</h3>
    <p>We grant you a limited, non-exclusive, non-transferable license to use SaberBot for personal, non-commercial evaluation and educational purposes only.</p>
    <h3>2. Restrictions</h3>
    <p>You may not: (a) copy, modify, or distribute the software, (b) reverse engineer or attempt to extract the source code, (c) rely on outputs as legal advice, (d) exceed rate limits or circumvent session restrictions.</p>
    <h3>3. Intellectual Property</h3>
    <p>All rights, title, and interest in SaberBot remain with the creator. The UAE law database is compiled from publicly available federal laws.</p>
    <h3>4. Disclaimer of Warranties</h3>
    <p><strong>THE SERVICE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.</strong> THE AI-GENERATED LEGAL INFORMATION MAY CONTAIN ERRORS, INACCURACIES, OR BE OUTDATED. THIS IS A PORTFOLIO PROJECT — DO NOT RELY ON IT FOR LEGAL DECISIONS.</p>
    <h3>5. Limitation of Liability</h3>
    <p>IN NO EVENT SHALL THE CREATOR BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY ARISING FROM THE USE OF THIS TOOL. THIS INCLUDES, BUT IS NOT LIMITED TO, LEGAL CONSEQUENCES, FINANCIAL LOSSES, OR MISSED DEADLINES RESULTING FROM RELIANCE ON AI-GENERATED INFORMATION.</p>
    <h3>6. Governing Law</h3>
    <p>This agreement is governed by the laws of the United Arab Emirates.</p>
  </div>
)

export default App
