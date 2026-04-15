"""FastAPI Backend for PubMed NLP Pipeline."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal
import uvicorn
from datetime import datetime
import json

from pubmed_nlp.pipeline import PubMedNLPPipeline, PipelineResult
from pubmed_nlp.personalization import UserType
from pubmed_nlp.rag_system import RAGAnswer


# API Models
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, description="PubMed search query")
    max_articles: int = Field(default=30, ge=5, le=100)
    user_type: Literal["patient", "student", "doctor"] = "patient"
    enable_qa: bool = True


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=5, description="Question to answer")
    context_query: Optional[str] = Field(None, description="Optional topic to search for context")


class EntityResponse(BaseModel):
    text: str
    label: str
    source: str


class RelationResponse(BaseModel):
    subject: str
    subject_type: str
    predicate: str
    object: str
    object_type: str
    relation_type: str
    confidence: float


class SummaryResponse(BaseModel):
    summary: str
    method: str
    compression_ratio: float
    key_points: List[str]


class TrendPointResponse(BaseModel):
    year: int
    count: int
    percentage: float
    top_terms: List[tuple]


class TrendResponse(BaseModel):
    topic: str
    trajectory: str
    growth_rate: float
    peak_year: int


class EvidenceResponse(BaseModel):
    pmid: str
    citation_count: int
    overall_score: float
    evidence_level: str


class RiskFactorResponse(BaseModel):
    factor: str
    outcome: str
    relation: str
    confidence: float
    evidence_count: int


class PersonalizedOutputResponse(BaseModel):
    summary: str
    key_points: List[str]
    technical_level: str
    warnings: List[str]
    recommended_actions: List[str]
    references: List[str]


class PipelineResponse(BaseModel):
    query: str
    article_count: int
    entities: List[EntityResponse]
    relations: List[RelationResponse]
    summary: SummaryResponse
    trends: Optional[TrendResponse]
    evidence_scores: List[EvidenceResponse]
    risk_factors: List[RiskFactorResponse]
    personalized: Dict[str, PersonalizedOutputResponse]
    insights: Dict
    rag_answer: Optional[Dict]
    processing_time: float


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str


class SearchResponse(BaseModel):
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    year: Optional[int]
    journal: str


# Initialize FastAPI app
app = FastAPI(
    title="PubMed NLP Research Simplifier API",
    description="Domain-specific NLP pipeline for biomedical literature analysis",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (lazy loaded)
_pipeline: Optional[PubMedNLPPipeline] = None


def get_pipeline() -> PubMedNLPPipeline:
    """Get or initialize pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = PubMedNLPPipeline(
            enable_rag=True,
            enable_contradiction=True,
            device='cpu'
        )
    return _pipeline


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return await root()


@app.post("/search", response_model=PipelineResponse)
async def search_and_analyze(request: SearchRequest):
    """
    Search PubMed and run full NLP analysis pipeline.
    """
    import time
    start_time = time.time()
    
    try:
        pipeline = get_pipeline()
        
        # Map user type string to enum
        user_type_map = {
            "patient": UserType.PATIENT,
            "student": UserType.STUDENT,
            "doctor": UserType.DOCTOR
        }
        user_type = user_type_map.get(request.user_type, UserType.PATIENT)
        
        # Run pipeline
        result = pipeline.process(
            query=request.query,
            max_articles=request.max_articles,
            user_types=[user_type],
            enable_qa=request.enable_qa
        )
        
        processing_time = time.time() - start_time
        
        # Build response
        return PipelineResponse(
            query=result.query,
            article_count=len(result.articles),
            entities=[EntityResponse(**e) for e in result.entities[:50]],
            relations=[RelationResponse(
                subject=r.subject,
                subject_type=r.subject_type,
                predicate=r.predicate,
                object=r.object,
                object_type=r.object_type,
                relation_type=r.relation_type,
                confidence=r.confidence
            ) for r in result.relations[:20]],
            summary=SummaryResponse(
                summary=result.summary.summary if result.summary else "",
                method=result.summary.method if result.summary else "none",
                compression_ratio=result.summary.compression_ratio if result.summary else 0.0,
                key_points=result.summary.key_points if result.summary else []
            ),
            trends=TrendResponse(
                topic=result.trends.topic,
                trajectory=result.trends.current_trajectory,
                growth_rate=result.trends.growth_rate,
                peak_year=result.trends.peak_year
            ) if result.trends else None,
            evidence_scores=[EvidenceResponse(
                pmid=e.pmid,
                citation_count=e.citation_count,
                overall_score=e.overall_score,
                evidence_level=e.evidence_level
            ) for e in result.evidence_scores[:10]],
            risk_factors=[RiskFactorResponse(
                factor=r.factor,
                outcome=r.outcome,
                relation=r.relation,
                confidence=r.confidence,
                evidence_count=r.evidence_count
            ) for r in result.risk_factors[:10]],
            personalized={
                k: PersonalizedOutputResponse(**v) 
                for k, v in result.personalized.items()
            },
            insights=result.insights,
            rag_answer={
                "answer": result.rag_answer.answer,
                "confidence": result.rag_answer.confidence,
                "sources": [
                    {"title": s.title, "relevance": s.relevance_score}
                    for s in result.rag_answer.sources
                ]
            } if result.rag_answer else None,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/ask", response_model=Dict)
async def ask_question(request: QuestionRequest):
    """
    Answer a question using RAG.
    """
    try:
        pipeline = get_pipeline()
        answer = pipeline.answer_question(
            question=request.question,
            context_query=request.context_query
        )
        
        return {
            "question": request.question,
            "answer": answer.answer,
            "confidence": answer.confidence,
            "method": answer.method,
            "sources": [
                {
                    "pmid": s.source,
                    "title": s.title,
                    "relevance_score": s.relevance_score
                }
                for s in answer.sources
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QA error: {str(e)}")


@app.get("/quick-search")
async def quick_search(
    q: str = Query(..., min_length=3, description="Search query"),
    limit: int = Query(default=10, ge=1, le=50)
):
    """
    Quick search without full NLP analysis.
    Returns raw articles only.
    """
    try:
        from pubmed_nlp.pubmed_client import PubMedClient
        client = PubMedClient()
        articles = client.search_and_fetch(q, limit)
        
        return {
            "query": q,
            "count": len(articles),
            "articles": [
                {
                    "pmid": a.pmid,
                    "title": a.title,
                    "abstract": a.abstract[:300] + "..." if len(a.abstract) > 300 else a.abstract,
                    "authors": a.authors[:3],
                    "year": a.year,
                    "journal": a.journal
                }
                for a in articles
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
