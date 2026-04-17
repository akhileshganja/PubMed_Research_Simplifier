# 🔬 PubMed Research Simplifier

A domain-specific NLP pipeline for biomedical literature analysis using PubMed as the source.

## 🎯 Key Differentiator

> **"I built a domain-specific NLP pipeline using biomedical models instead of generic LLMs to ensure factual accuracy."**

Unlike generic LLM approaches, this system uses:
- **SciSpacy** for biomedical entity recognition (not generic spaCy)
- **BioBERT** embeddings trained on biomedical text
- **Hybrid summarization** (extractive → abstractive) to reduce hallucinations
- **Contradiction detection** using NLI models
- **Evidence scoring** based on journal quality and study design

## 🏗️ System Architecture

```
User Query → PubMed Retrieval → Filtering → NLP Processing → Insight Engine → Output
                                      ↓
                    ┌─────────────────┴─────────────────┐
                    ↓                                   ↓
            Preprocessing                      Named Entity Recognition
            (SciSpacy)                         (Diseases, Drugs, Genes)
                    ↓                                   ↓
            Information Extraction               Contradiction Detection
            (Keyphrases, Relations)              (Embeddings + NLI)
                    ↓                                   ↓
            Hybrid Summarization                 RAG-based QA
            (TextRank + Transformers)            (Vector Search)
                    ↓
            Personalization Layer
            (Patient / Student / Doctor)
```

## 📦 Features

### 1. Data Ingestion (PubMed API)
- `esearch` → Article IDs
- `efetch` → Full abstracts + metadata
- Stores: Title, Abstract, Authors, Year, MeSH terms

### 2. Preprocessing (Critical NLP Step)
- Lowercasing, stopword removal
- Sentence segmentation
- Lemmatization
- **Biomedical-specific**: Uses **SciSpacy** (en_core_sci_lg)

### 3. Named Entity Recognition
- **Diseases** (cancer, diabetes, etc.)
- **Drugs/Chemicals** (metformin, aspirin, etc.)
- **Genes/Proteins**
- **Cell types, Species, Anatomy**

Uses: `en_ner_bc5cdr_md` + `en_ner_jnlpba_md`

### 4. Information Extraction
- **Relation Extraction**: Drug → treats → Disease
- **Keyphrase Extraction**: KeyBERT + TF-IDF
- **Topic Modeling**: BERTopic for trend detection

### 5. Summarization Layer (Hybrid - BEST)
1. Extractive: TextRank for key sentences
2. Abstractive: Medical summarization model
3. Validation: ROUGE + entity overlap checks

### 6. Contradiction Detection
- Sentence embeddings with **BioBERT**
- NLI classification: entailment / contradiction / neutral
- Cosine similarity for clustering

### 7. Insight Engine
- **Trend Detection**: Publication frequency over years
- **Evidence Scoring**: Citation count + Journal quality + Recency + Study type
- **Risk Factor Mining**: Aggregates relations (e.g., Smoking → increases → Lung Cancer)

### 8. Personalization Layer
| User Type | Output Characteristics |
|-----------|----------------------|
| **Patient** | Simple language, actionable info, prominent warnings |
| **Student** | Moderate detail, educational context, key concepts |
| **Doctor** | Full technical detail, evidence focus, statistics |

### 9. RAG-based QA
- Vector store: ChromaDB
- Embeddings: BioBERT
- Context-aware answers grounded in PubMed data

## 🛠️ Tech Stack

### NLP Core
- `scispacy` (0.5.3) - Biomedical NLP
- `transformers` (4.35+) - HuggingFace models
- `sentence-transformers` (2.2+) - Semantic search
- `bertopic` (0.15+) - Topic modeling
- `keybert` (0.8+) - Keyphrase extraction

### Backend
- `fastapi` (0.104+) - API framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation

### Database
- `chromadb` - Vector store
- `pymongo` - Optional document store

### Evaluation
- `rouge-score` - Summarization metrics
- `seqeval` - NER metrics

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repo-url>
cd pubmed-research-simplifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download SciSpacy models
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_lg-0.5.3.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_ner_bc5cdr_md-0.5.3.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_ner_jnlpba_md-0.5.3.tar.gz
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
PUBMED_EMAIL=your.email@example.com
PUBMED_API_KEY=your_ncbi_api_key  # Optional but recommended
```

### 3. Run API Server

```bash
# Start FastAPI server
python -m api.main

# Server runs at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### 4. Run Frontend

```bash
# Open frontend/index.html in browser
# Or serve with Python
python -m http.server 3000 --directory frontend

# Frontend at http://localhost:3000
```

## 📖 API Usage

### Search & Analyze

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "metformin type 2 diabetes treatment",
    "max_articles": 30,
    "user_type": "patient",
    "enable_qa": true
  }'
```

### Ask a Question (RAG)

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the side effects of metformin?",
    "context_query": "metformin diabetes"
  }'
```

### Quick Search

```bash
curl "http://localhost:8000/quick-search?q=aspirin+cardiovascular&limit=10"
```

## 🧪 Python API Usage

```python
from pubmed_nlp.pipeline import PubMedNLPPipeline
from pubmed_nlp.personalization import UserType

# Initialize pipeline
pipeline = PubMedNLPPipeline(
    enable_rag=True,
    enable_contradiction=True,
    device='cpu'  # or 'cuda'
)

# Process a query
result = pipeline.process(
    query="metformin type 2 diabetes",
    max_articles=30,
    user_types=[UserType.PATIENT, UserType.DOCTOR],
    enable_qa=True
)

# Access results
print(f"Found {len(result.articles)} articles")
print(f"Entities: {len(result.entities)}")
print(f"Summary: {result.summary.summary}")
print(f"Risk factors: {result.risk_factors}")

# Ask a follow-up question
answer = pipeline.answer_question(
    question="What are common side effects?",
    context_query="metformin diabetes"
)
print(answer.answer)
```

## 📊 Evaluation

```python
from evaluation import Evaluator, run_benchmark

# Initialize evaluator
evaluator = Evaluator()

# Run benchmark
from pubmed_nlp.pipeline import PubMedNLPPipeline
pipeline = PubMedNLPPipeline()

benchmark_results = run_benchmark(pipeline, [
    "metformin diabetes",
    "aspirin cardiovascular",
    "covid-19 vaccine"
])

print(f"Avg processing time: {benchmark_results['avg_time']:.2f}s")
print(f"Avg entities found: {benchmark_results['avg_entities']:.1f}")
```

## 📁 Project Structure

```
pubmed-research-simplifier/
├── pubmed_nlp/              # Core NLP modules
│   ├── __init__.py
│   ├── config.py            # Configuration
│   ├── pubmed_client.py     # PubMed API client
│   ├── preprocessing.py     # SciSpacy preprocessing
│   ├── named_entity_recognition.py  # NER
│   ├── information_extraction.py    # Keyphrases, Relations, Topics
│   ├── summarization.py     # Hybrid summarization
│   ├── contradiction_detection.py     # NLI-based
│   ├── insight_engine.py    # Trends, Evidence, Risk factors
│   ├── personalization.py # User type adaptation
│   ├── rag_system.py        # RAG QA system
│   └── pipeline.py          # Main orchestrator
├── api/                     # FastAPI backend
│   ├── __init__.py
│   └── main.py
├── frontend/                # React-like vanilla JS frontend
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── evaluation.py            # Metrics (ROUGE, NER)
├── requirements.txt
├── .env.example
└── README.md
```

## ⚠️ Important Notes

### ❌ Common Mistakes to Avoid

1. **Over-relying on LLM** → This pipeline uses domain-specific models first
2. **Skipping preprocessing** → SciSpacy handles biomedical terms better than generic NLP
3. **No domain-specific models** → We use BC5CDR, JNLPBA for accurate entity detection
4. **No evaluation** → Included: ROUGE for summarization, precision/recall for NER

### ✅ Best Practices Implemented

- **Hybrid summarization**: Extractive first (reduces hallucination), then abstractive
- **MeSH term processing**: Critical for biomedical context
- **Contradiction detection**: NLI models for statement comparison
- **Evidence scoring**: Journal impact + study design + recency
- **RAG grounding**: All answers tied to retrieved PubMed abstracts

## 🏥 Clinical Disclaimer

> **⚠️ This tool is for research purposes only. Always consult healthcare professionals for medical decisions.**

## 📚 References

- [SciSpacy](https://allenai.github.io/scispacy/) - Biomedical NLP
- [BioBERT](https://github.com/dmis-lab/biobert) - Biomedical BERT
- [PubMed E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25500/) - NCBI API
- [BERTopic](https://maartengr.github.io/BERTopic/) - Topic modeling

## 📝 License

MIT License - See LICENSE file for details.

## 🙋‍♂️ Interview Talking Points

When discussing this project:

1. **Domain-specific approach**: "I used SciSpacy instead of regular spaCy because standard NLP fails on medical terms"

2. **Hallucination prevention**: "Hybrid summarization with validation - we constrain the LLM input to reduce fabrication"

3. **Evidence-based**: "We score articles by journal impact, citations, and study design - not all papers are equal"

4. **Contradiction awareness**: "The system detects conflicting findings across papers, which is critical in biomedical research"

5. **Multi-audience**: "Personalization layer adapts the same research findings for patients, students, or doctors"
#   P u b M e d _ R e a s e a c h _ S i m p l i f i e r  
 