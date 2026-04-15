/**
 * PubMed Research Simplifier - Frontend JavaScript
 */

const API_BASE_URL = 'http://localhost:8000';

// DOM Elements
const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const userTypeSelect = document.getElementById('userType');
const maxArticlesSelect = document.getElementById('maxArticles');
const qaInput = document.getElementById('qaInput');
const qaBtn = document.getElementById('qaBtn');
const loadingState = document.getElementById('loadingState');
const resultsSection = document.getElementById('resultsSection');

// Tab buttons
const tabBtns = document.querySelectorAll('.tab-btn');

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    searchBtn.addEventListener('click', handleSearch);
    qaBtn.addEventListener('click', handleQA);
    
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleSearch();
    });
    
    qaInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleQA();
    });
    
    // Tab switching
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });
});

let currentResult = null;
let activeUserType = 'patient';

async function handleSearch() {
    const query = searchInput.value.trim();
    if (!query) {
        alert('Please enter a search query');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                max_articles: parseInt(maxArticlesSelect.value),
                user_type: userTypeSelect.value,
                enable_qa: true
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        currentResult = result;
        activeUserType = userTypeSelect.value;
        
        displayResults(result);
        
    } catch (error) {
        console.error('Search error:', error);
        alert('Error analyzing research. Please try again.');
    } finally {
        hideLoading();
    }
}

async function handleQA() {
    const question = qaInput.value.trim();
    if (!question) {
        alert('Please enter a question');
        return;
    }
    
    const contextQuery = currentResult ? currentResult.query : null;
    
    try {
        qaBtn.disabled = true;
        qaBtn.textContent = 'Thinking...';
        
        const response = await fetch(`${API_BASE_URL}/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                context_query: contextQuery
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const answer = await response.json();
        displayQAAnswer(answer);
        
    } catch (error) {
        console.error('QA error:', error);
        alert('Error getting answer. Please try again.');
    } finally {
        qaBtn.disabled = false;
        qaBtn.textContent = 'Ask AI';
    }
}

function showLoading() {
    resultsSection.style.display = 'none';
    loadingState.style.display = 'block';
    
    // Animate steps
    const steps = document.querySelectorAll('.step');
    steps.forEach((step, index) => {
        setTimeout(() => {
            steps.forEach(s => s.classList.remove('active'));
            step.classList.add('active');
        }, index * 2000);
    });
}

function hideLoading() {
    loadingState.style.display = 'none';
}

function displayResults(result) {
    resultsSection.style.display = 'block';
    
    // Summary
    const summaryContent = document.getElementById('summaryContent');
    const keyPoints = document.getElementById('keyPoints');
    
    if (result.summary) {
        summaryContent.textContent = result.summary.summary;
        
        if (result.summary.key_points && result.summary.key_points.length > 0) {
            keyPoints.innerHTML = `
                <h4>Key Points</h4>
                <ul>
                    ${result.summary.key_points.map(kp => `<li>${kp}</li>`).join('')}
                </ul>
            `;
        } else {
            keyPoints.style.display = 'none';
        }
    }
    
    // Personalized output
    updatePersonalizedContent();
    
    // Stats
    document.getElementById('articleCount').textContent = result.article_count;
    document.getElementById('entityCount').textContent = result.entities.length;
    document.getElementById('trendIndicator').textContent = 
        result.trends ? result.trends.trajectory : '-';
    document.getElementById('evidenceLevel').textContent = 
        result.evidence_scores.length > 0 ? result.evidence_scores[0].evidence_level : '-';
    
    // Entities
    const entitiesList = document.getElementById('entitiesList');
    const uniqueEntities = getUniqueEntities(result.entities);
    entitiesList.innerHTML = uniqueEntities.slice(0, 20).map(e => {
        const labelClass = getEntityClass(e.label);
        return `<span class="entity-tag ${labelClass}">${e.text} <small>(${e.label})</small></span>`;
    }).join('');
    
    // Relations
    const relationsList = document.getElementById('relationsList');
    relationsList.innerHTML = result.relations.slice(0, 10).map(r => `
        <div class="relation-item">
            <span class="relation-subject">${r.subject}</span>
            <span class="relation-predicate">→ ${r.relation_type} →</span>
            <span class="relation-object">${r.object}</span>
        </div>
    `).join('');
    
    // Risk Factors
    const riskList = document.getElementById('riskFactorsList');
    if (result.risk_factors.length > 0) {
        riskList.innerHTML = result.risk_factors.slice(0, 8).map(r => `
            <div class="risk-item">
                <span class="risk-factor">${r.factor}</span>
                <span class="risk-relation">${r.relation}</span>
                <span>${r.outcome}</span>
                <span class="risk-confidence">${(r.confidence * 100).toFixed(0)}%</span>
            </div>
        `).join('');
    } else {
        riskList.innerHTML = '<p style="color: var(--text-muted);">No risk factors identified in analyzed articles.</p>';
    }
    
    // Insights
    const insightsList = document.getElementById('insightsList');
    if (result.insights && result.insights.key_findings) {
        insightsList.innerHTML = result.insights.key_findings.map(f => `
            <li>${f}</li>
        `).join('');
    }
    
    // Articles - API returns count only, not full articles array
    const articlesList = document.getElementById('articlesList');
    articlesList.innerHTML = `
        <div class="article-item">
            <div class="article-title">${result.article_count} articles analyzed</div>
            <div class="article-meta">Articles are processed but not stored in response. Check server logs for details.</div>
        </div>
    `;
    
    // RAG Answer
    if (result.rag_answer) {
        displayQAAnswer({
            question: result.query,
            answer: result.rag_answer.answer,
            confidence: result.rag_answer.confidence,
            sources: result.rag_answer.sources
        });
    }
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function switchTab(tabName) {
    activeUserType = tabName;
    
    // Update button states
    tabBtns.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });
    
    updatePersonalizedContent();
}

function updatePersonalizedContent() {
    if (!currentResult || !currentResult.personalized) return;
    
    const content = document.getElementById('personalizedContent');
    const data = currentResult.personalized[activeUserType];
    
    if (!data) {
        content.innerHTML = '<p>No personalized content available.</p>';
        return;
    }
    
    let html = `<p>${data.summary}</p>`;
    
    if (data.warnings && data.warnings.length > 0) {
        html += data.warnings.map(w => `
            <div class="warning">⚠️ ${w}</div>
        `).join('');
    }
    
    if (data.recommended_actions && data.recommended_actions.length > 0) {
        html += `<h4 style="margin: 1rem 0 0.5rem;">Recommended Actions:</h4>`;
        html += data.recommended_actions.map(a => `
            <div class="action">→ ${a}</div>
        `).join('');
    }
    
    if (data.references && data.references.length > 0) {
        html += `<h4 style="margin: 1rem 0 0.5rem;">References:</h4>`;
        html += `<ul style="font-size: 0.8125rem; color: var(--text-muted);">`;
        html += data.references.slice(0, 5).map(r => `<li>${r}</li>`).join('');
        html += `</ul>`;
    }
    
    content.innerHTML = html;
}

function displayQAAnswer(answer) {
    // Create or update QA answer section
    let qaSection = document.getElementById('qaAnswerSection');
    
    if (!qaSection) {
        qaSection = document.createElement('div');
        qaSection.id = 'qaAnswerSection';
        qaSection.className = 'card qa-answer';
        resultsSection.insertBefore(qaSection, resultsSection.firstChild);
    }
    
    const confidence = answer.confidence ? (answer.confidence * 100).toFixed(0) : 'N/A';
    const sources = answer.sources ? answer.sources.slice(0, 3).map(s => s.title).join(', ') : '';
    
    qaSection.innerHTML = `
        <h4>❓ Question: ${answer.question || qaInput.value}</h4>
        <p>${answer.answer}</p>
        <div class="qa-sources">
            Confidence: ${confidence}% • Based on: ${sources || 'Retrieved articles'}
        </div>
    `;
    
    // Clear input
    qaInput.value = '';
}

function getUniqueEntities(entities) {
    const seen = new Set();
    return entities.filter(e => {
        const key = `${e.text.toLowerCase()}_${e.label}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
    });
}

function getEntityClass(label) {
    const label_lower = label.toLowerCase();
    if (label_lower.includes('disease') || label_lower.includes('cancer') || label_lower.includes('condition')) {
        return 'disease';
    } else if (label_lower.includes('chemical') || label_lower.includes('drug')) {
        return 'chemical';
    } else if (label_lower.includes('gene') || label_lower.includes('protein')) {
        return 'gene';
    }
    return '';
}

// Health check on load
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            console.log('API is healthy');
        } else {
            console.warn('API health check failed');
        }
    } catch (error) {
        console.warn('Could not connect to API:', error);
        // Show warning to user
        const warning = document.createElement('div');
        warning.style.cssText = `
            background: #fef3c7;
            padding: 1rem;
            text-align: center;
            font-size: 0.875rem;
            color: #92400e;
        `;
        warning.innerHTML = '⚠️ API connection failed. Please ensure the backend is running on port 8000.';
        document.body.insertBefore(warning, document.body.firstChild);
    }
}

checkHealth();
