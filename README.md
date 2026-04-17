🔬 PubMed Research Simplifier

A domain-specific NLP pipeline for analyzing biomedical literature using PubMed data.

🎯 Key Differentiator

“A domain-specific NLP pipeline using biomedical models instead of generic LLMs to ensure factual accuracy.”

Unlike generic approaches, this system uses:

SciSpacy → Biomedical entity recognition
BioBERT → Domain-trained embeddings
Hybrid Summarization → Extractive + Abstractive
Contradiction Detection → NLI models
Evidence Scoring → Journal quality + study design
🏗️ System Architecture
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
📦 Features
1. 📡 Data Ingestion
Uses PubMed API (esearch, efetch)
Extracts:
Title
Abstract
Authors
Year
MeSH terms
2. 🧹 Preprocessing
Lowercasing, stopword removal
Sentence segmentation
Lemmatization
SciSpacy-based biomedical preprocessing
3. 🧬 Named Entity Recognition

Detects:

Diseases
Drugs/Chemicals
Genes/Proteins
Cell types, Species, Anatomy

Models used:

en_ner_bc5cdr_md
en_ner_jnlpba_md
4. 🔗 Information Extraction
Relation Extraction (Drug → Disease)
Keyphrase Extraction (KeyBERT + TF-IDF)
Topic Modeling (BERTopic)
5. 🧠 Summarization (Hybrid Approach)
Extractive → TextRank
Abstractive → Transformer model
Validation → ROUGE + entity overlap
6. ⚖️ Contradiction Detection
BioBERT embeddings
NLI classification:
Entailment
Contradiction
Neutral
7. 📊 Insight Engine
Trend detection (publication frequency)
Evidence scoring:
Citation count
Journal quality
Recency
Study type
Risk factor mining
8. 👤 Personalization Layer
User Type	Output Style
Patient	Simple + actionable
Student	Moderate detail
Doctor	Technical + evidence-based
9. 🤖 RAG-based QA
Vector DB: ChromaDB
Embeddings: BioBERT
Context-aware grounded answers
🛠️ Tech Stack
🔍 NLP
scispacy
transformers
sentence-transformers
bertopic
keybert
⚙️ Backend
fastapi
uvicorn
pydantic
🗄️ Database
chromadb
pymongo
📈 Evaluation
rouge-score
seqeval
🚀 Quick Start
1. Installation
git clone <repo-url>
cd pubmed-research-simplifier

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt

Install SciSpacy models:

pip install <scispacy-model-links>
2. Configuration
cp .env.example .env

Add:

PUBMED_EMAIL=your.email@example.com
PUBMED_API_KEY=your_api_key
3. Run Backend
python -m api.main
API → http://localhost:8000
Docs → http://localhost:8000/docs
4. Run Frontend
python -m http.server 3000 --directory frontend
Frontend → http://localhost:3000
📖 API Usage
🔎 Search
POST /search
❓ Ask Question
POST /ask
⚡ Quick Search
GET /quick-search
🧪 Python Usage
from pubmed_nlp.pipeline import PubMedNLPPipeline

pipeline = PubMedNLPPipeline()

result = pipeline.process("metformin diabetes")

print(result.summary.summary)
📁 Project Structure
pubmed-research-simplifier/
├── pubmed_nlp/
├── api/
├── frontend/
├── evaluation.py
├── requirements.txt
├── .env.example
└── README.md
