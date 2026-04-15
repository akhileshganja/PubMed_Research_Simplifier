export default function QAAnswer({ answer }) {
  if (!answer) return null

  const confidence = answer.confidence != null ? (answer.confidence * 100).toFixed(0) : 'N/A'
  const sources = answer.sources?.slice(0, 3).map(s => s.title).filter(Boolean).join(', ')

  return (
    <div className="card qa-answer">
      <h4>❓ Question: {answer.question}</h4>
      <p>{answer.answer}</p>
      <div className="qa-sources">
        Confidence: {confidence}% • Based on: {sources || 'Retrieved articles'}
      </div>
    </div>
  )
}
