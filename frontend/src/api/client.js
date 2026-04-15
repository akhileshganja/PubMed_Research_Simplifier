const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export const api = {
  health: () => request('/health'),

  search: (query, maxArticles, userType) =>
    request('/search', {
      method: 'POST',
      body: JSON.stringify({
        query,
        max_articles: maxArticles,
        user_type: userType,
        enable_qa: true
      })
    }),

  ask: (question, contextQuery) =>
    request('/ask', {
      method: 'POST',
      body: JSON.stringify({ question, context_query: contextQuery })
    })
}
