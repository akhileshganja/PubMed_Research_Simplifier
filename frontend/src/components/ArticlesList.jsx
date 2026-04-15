export default function ArticlesList({ articleCount }) {
  return (
    <div className="card articles-card">
      <h3>📄 Source Articles</h3>
      <div className="articles-list">
        <div className="article-item">
          <div className="article-title">{articleCount} articles analyzed</div>
          <div className="article-meta">
            Articles are processed but not stored in response. Check server logs for details.
          </div>
        </div>
      </div>
    </div>
  )
}
