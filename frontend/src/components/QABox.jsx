import { useState } from 'react'

export default function QABox({ onAsk, loading }) {
  const [question, setQuestion] = useState('')

  function handleAsk() {
    if (!question.trim()) return
    onAsk(question.trim())
    setQuestion('')
  }

  return (
    <section className="qa-section">
      <div className="qa-box">
        <input
          type="text"
          value={question}
          onChange={e => setQuestion(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleAsk()}
          placeholder="Ask a specific question about the research..."
          autoComplete="off"
        />
        <button className="btn-secondary" onClick={handleAsk} disabled={loading}>
          {loading ? 'Thinking...' : 'Ask AI'}
        </button>
      </div>
    </section>
  )
}
