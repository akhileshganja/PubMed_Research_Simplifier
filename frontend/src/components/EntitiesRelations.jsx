function getEntityClass(label) {
  const l = label.toLowerCase()
  if (l.includes('disease') || l.includes('cancer') || l.includes('condition')) return 'disease'
  if (l.includes('chemical') || l.includes('drug')) return 'chemical'
  if (l.includes('gene') || l.includes('protein')) return 'gene'
  return ''
}

function uniqueEntities(entities) {
  const seen = new Set()
  return entities.filter(e => {
    const key = `${e.text.toLowerCase()}_${e.label}`
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
}

export default function EntitiesRelations({ entities, relations }) {
  const unique = uniqueEntities(entities).slice(0, 20)

  return (
    <div className="card">
      <h3>🏷️ Key Entities</h3>
      <div className="entities-list">
        {unique.map((e, i) => (
          <span key={i} className={`entity-tag ${getEntityClass(e.label)}`}>
            {e.text} <small>({e.label})</small>
          </span>
        ))}
      </div>

      <h3 style={{ marginTop: '1.5rem' }}>🔗 Relations</h3>
      <div className="relations-list">
        {relations.slice(0, 10).map((r, i) => (
          <div key={i} className="relation-item">
            <span className="relation-subject">{r.subject}</span>
            <span className="relation-predicate">→ {r.relation_type} →</span>
            <span className="relation-object">{r.object}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
