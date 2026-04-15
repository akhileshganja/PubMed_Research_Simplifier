import { useState, useCallback } from 'react'
import { api } from '../api/client'

export function useSearch() {
  const [result, setResult] = useState(null)
  const [qaAnswer, setQaAnswer] = useState(null)
  const [loading, setLoading] = useState(false)
  const [qaLoading, setQaLoading] = useState(false)
  const [error, setError] = useState(null)

  const search = useCallback(async (query, maxArticles, userType) => {
    setLoading(true)
    setError(null)
    setQaAnswer(null)
    try {
      const data = await api.search(query, maxArticles, userType)
      setResult(data)
    } catch (e) {
      setError('Error analyzing research. Please try again.')
    } finally {
      setLoading(false)
    }
  }, [])

  const ask = useCallback(async (question, contextQuery) => {
    setQaLoading(true)
    try {
      const data = await api.ask(question, contextQuery)
      setQaAnswer(data)
    } catch (e) {
      setError('Error getting answer. Please try again.')
    } finally {
      setQaLoading(false)
    }
  }, [])

  return { result, qaAnswer, loading, qaLoading, error, search, ask }
}
