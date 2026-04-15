// Splits "Article 1: Title.. body. Article 2: Title.. body." into individual chunks
function parseArticles(text) {
  if (!text) return []
  const parts = text.split(/Article\s+\d+:/i).map(s => s.trim()).filter(Boolean)
  if (parts.length <= 1) return null // not article-formatted, return null to render as plain text
  return parts.map(part => {
    // First sentence (up to "..") is the title, rest is body
    const dotDot = part.indexOf('..')
    if (dotDot !== -1) {
      return {
        title: part.slice(0, dotDot).trim(),
        body: part.slice(dotDot + 2).trim()
      }
    }
    const firstDot = part.indexOf('.')
    return {
      title: part.slice(0, firstDot + 1).trim(),
      body: part.slice(firstDot + 1).trim()
    }
  })
}

export default function SummaryCard({ summary }) {
  if (!summary) return null

  const articles = parseArticles(summary.summary)

  return (
    <div className="card summary-card">
      <div className="summary-header">
        <span className="summary-icon">📋</span>
        <div>
          <h2>Research Summary</h2>
          {summary.method && (
            <span className="summary-badge">{summary.method}</span>
          )}
        </div>
      </div>

      {articles ? (
        <div className="summary-articles">
          {articles.map((a, i) => (
            <div key={i} className="summary-article-item">
              <div className="summary-article-header">
                <span className="summary-article-num">{i + 1}</span>
                <p className="summary-article-title">{a.title}</p>
              </div>
              {a.body && <p className="summary-article-body">{a.body}</p>}
            </div>
          ))}
        </div>
      ) : (
        <p className="summary-content">{summary.summary}</p>
      )}

      {summary.key_points?.length > 0 && (
        <div className="key-points">
          <p className="key-points-label">Key Points</p>
          <div className="key-points-grid">
            {summary.key_points.map((kp, i) => (
              <div key={i} className="key-point-item">
                <span className="key-point-num">{i + 1}</span>
                <span>{kp}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
