import { useEffect, useRef, useState } from 'react'
import { useSearch } from './hooks/useSearch'
import { api } from './api/client'

import Header from './components/Header'
import Footer from './components/Footer'
import SearchBar from './components/SearchBar'
import QABox from './components/QABox'
import LoadingState from './components/LoadingState'
import StatsGrid from './components/StatsGrid'
import SummaryCard from './components/SummaryCard'
import PersonalizedCard from './components/PersonalizedCard'
import EntitiesRelations from './components/EntitiesRelations'
import RiskInsights from './components/RiskInsights'
import ArticlesList from './components/ArticlesList'
import QAAnswer from './components/QAAnswer'

export default function App() {
  const { result, qaAnswer, loading, qaLoading, error, search, ask } = useSearch()
  const [activeStep, setActiveStep] = useState(0)
  const [apiWarning, setApiWarning] = useState(false)
  const resultsRef = useRef(null)

  // Health check on mount
  useEffect(() => {
    api.health().catch(() => setApiWarning(true))
  }, [])

  // Animate loading steps
  useEffect(() => {
    if (!loading) { setActiveStep(0); return }
    let step = 1
    setActiveStep(step)
    const interval = setInterval(() => {
      step = step < 5 ? step + 1 : 5
      setActiveStep(step)
    }, 2000)
    return () => clearInterval(interval)
  }, [loading])

  // Scroll to results when they arrive
  useEffect(() => {
    if (result && resultsRef.current) {
      resultsRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [result])

  function handleAsk(question) {
    ask(question, result?.query ?? null)
  }

  // Merge rag_answer from search result into qaAnswer display
  const displayAnswer = qaAnswer ?? (result?.rag_answer
    ? { ...result.rag_answer, question: result.query }
    : null)

  return (
    <>
      {apiWarning && (
        <div className="api-warning">
          ⚠️ API connection failed. Please ensure the backend is running on port 8000.
        </div>
      )}

      <Header />

      <main className="main">
        <div className="container">
          <SearchBar onSearch={search} loading={loading} />
          <QABox onAsk={handleAsk} loading={qaLoading} />

          {error && <div className="error-banner">{error}</div>}

          {loading && <LoadingState activeStep={activeStep} />}

          {result && !loading && (
            <section className="results-section" ref={resultsRef}>
              {displayAnswer && <QAAnswer answer={displayAnswer} />}
              <SummaryCard summary={result.summary} />
              <PersonalizedCard personalized={result.personalized} />
              <StatsGrid result={result} />
              <div className="two-column">
                <EntitiesRelations entities={result.entities} relations={result.relations} />
                <RiskInsights riskFactors={result.risk_factors} insights={result.insights} />
              </div>
              <ArticlesList articleCount={result.article_count} />
            </section>
          )}
        </div>
      </main>

      <Footer />
    </>
  )
}
