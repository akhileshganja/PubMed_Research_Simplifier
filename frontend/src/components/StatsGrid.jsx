export default function StatsGrid({ result }) {
  const stats = [
    { id: 'articles', value: result.article_count, label: 'Articles Analyzed' },
    { id: 'entities', value: result.entities.length, label: 'Entities Found' },
    { id: 'trend', value: result.trends?.trajectory ?? '-', label: 'Research Trend' },
    {
      id: 'evidence',
      value: result.evidence_scores.length > 0 ? result.evidence_scores[0].evidence_level : '-',
      label: 'Evidence Quality'
    }
  ]

  return (
    <div className="stats-grid">
      {stats.map(s => (
        <div key={s.id} className="stat-card">
          <div className="stat-value">{s.value}</div>
          <div className="stat-label">{s.label}</div>
        </div>
      ))}
    </div>
  )
}
