const STEPS = [
  { id: 1, label: '📚 Retrieving articles' },
  { id: 2, label: '🔬 Extracting entities' },
  { id: 3, label: '🔗 Finding relations' },
  { id: 4, label: '📝 Summarizing' },
  { id: 5, label: '📊 Generating insights' }
]

export default function LoadingState({ activeStep }) {
  return (
    <div className="loading-state">
      <div className="spinner" />
      <p>Analyzing biomedical literature...</p>
      <div className="progress-steps">
        {STEPS.map(s => (
          <div key={s.id} className={`step${activeStep === s.id ? ' active' : ''}`}>
            {s.label}
          </div>
        ))}
      </div>
    </div>
  )
}
