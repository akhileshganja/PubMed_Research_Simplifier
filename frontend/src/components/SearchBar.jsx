import { useState } from 'react'

export default function SearchBar({ onSearch, loading }) {
  const [query, setQuery] = useState('')
  const [userType, setUserType] = useState('patient')
  const [maxArticles, setMaxArticles] = useState(30)

  function handleSubmit() {
    if (!query.trim()) return
    onSearch(query.trim(), maxArticles, userType)
  }

  return (
    <section className="search-section">
      <div className="search-box">
        <input
          type="text"
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSubmit()}
          placeholder="Enter medical topic (e.g., 'metformin diabetes treatment')..."
          autoComplete="off"
        />
        <div className="search-options">
          <select value={userType} onChange={e => setUserType(e.target.value)}>
            <option value="patient">Patient (Simple)</option>
            <option value="student">Student (Moderate)</option>
            <option value="doctor">Doctor (Technical)</option>
          </select>
          <select value={maxArticles} onChange={e => setMaxArticles(Number(e.target.value))}>
            <option value={20}>20 articles</option>
            <option value={30}>30 articles</option>
            <option value={50}>50 articles</option>
          </select>
          <button className="btn-primary" onClick={handleSubmit} disabled={loading}>
            {loading ? 'Analyzing...' : 'Analyze'}
          </button>
        </div>
      </div>
    </section>
  )
}
