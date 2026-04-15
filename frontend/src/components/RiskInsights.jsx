export default function RiskInsights({ riskFactors, insights }) {
  return (
    <div className="card">
      <h3>⚠️ Risk Factors</h3>
      <div className="risk-list">
        {riskFactors.length === 0 ? (
          <p style={{ color: 'var(--text-muted)' }}>No risk factors identified in analyzed articles.</p>
        ) : (
          riskFactors.slice(0, 8).map((r, i) => (
            <div key={i} className="risk-item">
              <span className="risk-factor">{r.factor}</span>
              <span className="risk-relation">{r.relation}</span>
              <span>{r.outcome}</span>
              <span className="risk-confidence">{(r.confidence * 100).toFixed(0)}%</span>
            </div>
          ))
        )}
      </div>

      <h3 style={{ marginTop: '1.5rem' }}>💡 Key Insights</h3>
      <ul className="insights-list">
        {insights?.key_findings?.map((f, i) => <li key={i}>{f}</li>)}
      </ul>
    </div>
  )
}
