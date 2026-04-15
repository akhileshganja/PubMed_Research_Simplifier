# PubMed Research Simplifier — Presentation Outline

---

## Slide 1 — Title

**🔬 PubMed Research Simplifier**
AI-powered Biomedical Literature Analysis

> "Making medical research accessible to everyone — patients, students, and doctors."

---

## Slide 2 — The Problem

**Medical research is hard to consume.**

- 35+ million articles on PubMed, growing daily
- Written for specialists, not general readers
- No single tool that extracts, summarizes, and explains findings for different audiences
- Generic AI (ChatGPT etc.) hallucinates on medical facts

**The gap:** People need accurate, grounded, audience-aware insights from research — not just a chatbot answer.

---

## Slide 3 — The Solution

**A domain-specific NLP pipeline built on biomedical models.**

- Search any medical topic → get structured insights in seconds
- Summaries adapted for Patients, Students, or Doctors
- Every answer is grounded in real PubMed abstracts
- Detects contradictions across papers
- Scores evidence quality — not all papers are equal

---

## Slide 4 — Why Not Just Use ChatGPT?

| Generic LLM | This System |
|---|---|
| Trained on general text | Uses biomedical-specific models |
| Can hallucinate medical facts | Grounded in retrieved PubMed abstracts |
| No source attribution | Every answer tied to real articles |
| No contradiction awareness | NLI-based contradiction detection |
| One-size-fits-all output | Personalized for Patient / Student / Doctor |

> Key differentiator: **SciSpacy + BioBERT + RAG** instead of a black-box LLM.

---

## Slide 5 — System Architecture

```
User Query
    ↓
PubMed API  →  Fetch up to 50 articles (titles, abstracts, MeSH terms)
    ↓
Preprocessing  →  SciSpacy tokenization, lemmatization, stopword removal
    ↓
┌──────────────────────────┬──────────────────────────┐
↓                          ↓                          ↓
NER                  Info Extraction           Contradiction Detection
(Diseases, Drugs,    (Relations, Keyphrases,   (BioBERT embeddings
 Genes, Proteins)     Topic Modeling)           + NLI classifier)
└──────────────────────────┴──────────────────────────┘
    ↓
Hybrid Summarization  →  TextRank (extractive) → Transformer (abstractive)
    ↓
Insight Engine  →  Trends · Evidence Scoring · Risk Factors
    ↓
Personalization Layer  →  Patient / Student / Doctor output
    ↓
RAG-based QA  →  ChromaDB vector search + BioBERT answer generation
    ↓
React Frontend  +  FastAPI Backend
```

---

## Slide 6 — Core NLP Modules (1/2)

**1. Named Entity Recognition**
- Models: `en_ner_bc5cdr_md` (diseases & chemicals) + `en_ner_jnlpba_md` (genes & proteins)
- Extracts: Diseases, Drugs, Genes, Cell types, Anatomy
- Why SciSpacy: Generic spaCy fails on terms like "metformin", "BRCA1", "glioblastoma"

**2. Information Extraction**
- Relation extraction: Drug → treats → Disease
- Keyphrase extraction: KeyBERT + TF-IDF
- Topic modeling: BERTopic for trend detection

**3. Hybrid Summarization**
- Step 1 — Extractive: TextRank picks the most important sentences
- Step 2 — Abstractive: Medical summarization transformer refines them
- Step 3 — Validation: ROUGE score + entity overlap check
- Result: Reduces hallucination by constraining what the model can say

---

## Slide 7 — Core NLP Modules (2/2)

**4. Contradiction Detection**
- BioBERT sentence embeddings → cosine similarity clustering
- NLI classifier labels pairs: entailment / contradiction / neutral
- Critical in biomedical research where studies often conflict

**5. Insight Engine**
- Trend detection: publication frequency over years → trajectory (rising/stable/declining)
- Evidence scoring: citation count + journal quality + recency + study design type
- Risk factor mining: aggregates relations (e.g., Smoking → increases → Lung Cancer)

**6. RAG-based QA**
- Vector store: ChromaDB
- Embeddings: BioBERT
- User asks a question → nearest abstracts retrieved → answer generated and grounded

---

## Slide 8 — Personalization Layer

Same research, three different outputs:

| Audience | What they get |
|---|---|
| 🧑 Patient | Simple language, clear warnings, actionable steps |
| 🎓 Student | Moderate detail, key concepts explained, references |
| 👨‍⚕️ Doctor | Full technical detail, statistics, evidence levels, study types |

> One pipeline, three audiences — no separate models needed.

---

## Slide 9 — Tech Stack

**NLP Core**
- SciSpacy 0.5.3 — biomedical NLP
- HuggingFace Transformers 4.35+ — summarization & NLI
- Sentence-Transformers 2.2+ — semantic search
- BERTopic 0.15+ — topic modeling
- KeyBERT 0.8+ — keyphrase extraction

**Backend**
- FastAPI + Uvicorn — REST API
- Pydantic — data validation
- ChromaDB — vector store

**Frontend**
- React 18 + Vite
- No UI library — custom CSS design system

**Evaluation**
- ROUGE score — summarization quality
- Precision / Recall / F1 — NER accuracy

---

## Slide 10 — API Endpoints

| Method | Endpoint | What it does |
|---|---|---|
| GET | `/health` | Check if backend is running |
| POST | `/search` | Full NLP pipeline on a topic |
| POST | `/ask` | RAG-based question answering |
| GET | `/quick-search` | Raw article fetch, no NLP |

**Example request:**
```json
POST /search
{
  "query": "metformin type 2 diabetes",
  "max_articles": 30,
  "user_type": "patient",
  "enable_qa": true
}
```

---

## Slide 11 — Frontend Demo

**What the user sees:**

- Search bar → topic input + audience selector + article count
- Loading state → animated 5-step progress (Retrieving → Extracting → Relations → Summarizing → Insights)
- Research Summary → each article broken out with title + body
- Personalized tab → switches between Patient / Student / Doctor view
- Stats grid → articles analyzed, entities found, research trend, evidence quality
- Entities & Relations → color-coded tags (disease = red, chemical = green, gene = purple)
- Risk Factors → factor → relation → outcome with confidence %
- AI Q&A → ask any follow-up question, answer grounded in retrieved articles

---

## Slide 12 — Evaluation

**Summarization**
- ROUGE-1, ROUGE-2, ROUGE-L scores against reference summaries
- Entity overlap validation — ensures key medical terms are preserved

**NER**
- Precision, Recall, F1 per entity type
- Benchmarked on BC5CDR dataset

**RAG QA**
- Confidence score per answer
- Source attribution — every answer links back to PubMed articles

---

## Slide 13 — Challenges & How They Were Solved

| Challenge | Solution |
|---|---|
| Generic NLP fails on medical terms | SciSpacy with biomedical-trained models |
| LLMs hallucinate medical facts | Hybrid summarization + RAG grounding |
| Conflicting research findings | NLI-based contradiction detection |
| One output doesn't fit all users | Personalization layer with 3 audience modes |
| Not all papers are equally reliable | Evidence scoring (citations + journal + recency) |

---

## Slide 14 — Key Takeaways

1. **Domain-specific beats generic** — SciSpacy + BioBERT outperform general NLP on medical text
2. **Hallucination prevention** — Hybrid summarization + RAG keeps answers grounded in real data
3. **Evidence-aware** — Not all papers are equal; the system scores and ranks them
4. **Contradiction-aware** — Conflicting findings are flagged, not hidden
5. **Multi-audience** — Same pipeline, adapted output for patients, students, and doctors

---

## Slide 15 — Thank You

**🔬 PubMed Research Simplifier**

GitHub: `github.com/your-repo`
API Docs: `localhost:8000/docs`
Frontend: `localhost:3000`

> "Built to make biomedical research accessible, accurate, and trustworthy."

---

*Tip: Use a dark blue / white color scheme to match the app's gradient header. Add screenshots of the UI on slides 11 and the architecture diagram on slide 5.*
