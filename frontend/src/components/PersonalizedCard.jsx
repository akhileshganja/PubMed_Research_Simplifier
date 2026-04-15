import { useState } from 'react'

const TABS = [
  { key: 'patient', label: '👤 Patient' },
  { key: 'student', label: '🎓 Student' },
  { key: 'doctor', label: '👨‍⚕️ Doctor' }
]

const LEVEL_BADGE = {
  simple:    { label: 'Simple',    color: '#dcfce7', text: '#166534' },
  moderate:  { label: 'Moderate',  color: '#fef9c3', text: '#854d0e' },
  technical: { label: 'Technical', color: '#ede9fe', text: '#5b21b6' },
}

// The backend prepends a template prefix like "Here's what you need to know:\n\n"
// Strip it so we only show the actual content
function stripPrefix(text) {
  if (!text) return ''
  const idx = text.indexOf('\n\n')
  return idx !== -1 ? text.slice(idx + 2).trim() : text.trim()
}

export default function PersonalizedCard({ personalized }) {
  const [active, setActive] = useState('patient')
  const data = personalized?.[active]
  const badge = LEVEL_BADGE[data?.technical_level]

  return (
    <div className="card personalized-card">

      {/* Header */}
      <div className="pc-header">
        <div className="pc-title-row">
          <h2>Personalized Output</h2>
          {badge && (
            <span className="pc-level-badge" style={{ background: badge.color, color: badge.text }}>
              {badge.label}
            </span>
          )}
        </div>
        <div className="tabs">
          {TABS.map(t => (
            <button
              key={t.key}
              className={`tab-btn${active === t.key ? ' active' : ''}`}
              onClick={() => setActive(t.key)}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="personalized-content">
        {!data ? (
          <p style={{ color: 'var(--text-muted)' }}>No personalized content available.</p>
        ) : (
          <>
            {/* Summary */}
            <p className="pc-summary">{stripPrefix(data.summary)}</p>

            {/* Key Points */}
            {data.key_points?.length > 0 && (
              <div className="pc-section">
                <p className="pc-section-label">📌 Key Points</p>
                <div className="pc-key-points">
                  {data.key_points.map((kp, i) => (
                    <div key={i} className="pc-key-point">
                      <span className="pc-kp-dot" />
                      <span>{kp}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Warnings */}
            {data.warnings?.length > 0 && (
              <div className="pc-section">
                <p className="pc-section-label">⚠️ Warnings</p>
                <div className="pc-warnings">
                  {data.warnings.map((w, i) => (
                    <div key={i} className="pc-warning-item">{w}</div>
                  ))}
                </div>
              </div>
            )}

            {/* Recommended Actions */}
            {data.recommended_actions?.length > 0 && (
              <div className="pc-section">
                <p className="pc-section-label">✅ Recommended Actions</p>
                <div className="pc-actions">
                  {data.recommended_actions.map((a, i) => (
                    <div key={i} className="pc-action-item">
                      <span className="pc-action-num">{i + 1}</span>
                      <span>{a}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* References */}
            {data.references?.length > 0 && (
              <div className="pc-section">
                <p className="pc-section-label">📚 References</p>
                <div className="pc-references">
                  {data.references.slice(0, 5).map((r, i) => {
                    const url = r.match(/https?:\/\/\S+/)?.[0]
                    const label = r.replace(/https?:\/\/\S+/, '').trim()
                    return (
                      <div key={i} className="pc-ref-item">
                        <span className="pc-ref-label">{label || r}</span>
                        {url && (
                          <a href={url} target="_blank" rel="noreferrer" className="pc-ref-link">
                            View ↗
                          </a>
                        )}
                      </div>
                    )
                  })}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
