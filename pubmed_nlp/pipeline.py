"""Main NLP Pipeline Orchestrator."""

from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
from tqdm import tqdm

from pubmed_nlp.config import get_settings
from pubmed_nlp.pubmed_client import PubMedClient, PubMedArticle
from pubmed_nlp.preprocessing import BiomedicalPreprocessor, PreprocessedText
from pubmed_nlp.named_entity_recognition import BiomedicalNER, NERDocument
from pubmed_nlp.information_extraction import InformationExtractor, ExtractedKeyphrase, ExtractedRelation
from pubmed_nlp.summarization import HybridSummarizer, SummaryResult
from pubmed_nlp.contradiction_detection import ContradictionDetector, ContradictionResult
from pubmed_nlp.insight_engine import InsightEngine, TrendAnalysis, EvidenceScore, RiskFactor
from pubmed_nlp.personalization import PersonalizationEngine, UserType, PersonalizedOutput
from pubmed_nlp.rag_system import RAGSystem, RAGAnswer


@dataclass
class PipelineResult:
    """Complete pipeline output."""
    query: str
    articles: List[PubMedArticle]
    entities: List[Dict]
    relations: List[ExtractedRelation]
    keyphrases: List[ExtractedKeyphrase]
    summary: SummaryResult
    contradictions: List[ContradictionResult]
    trends: Optional[TrendAnalysis]
    evidence_scores: List[EvidenceScore]
    risk_factors: List[RiskFactor]
    personalized: Dict[str, PersonalizedOutput]
    rag_answer: Optional[RAGAnswer]
    insights: Dict


class PubMedNLPPipeline:
    """
    Complete NLP Pipeline for PubMed Research Simplification.
    
    High-level flow:
    User Query → PubMed Retrieval → Preprocessing → NER → 
    Info Extraction → Summarization → Contradiction Detection → 
    Insight Engine → Personalization → Output
    """
    
    def __init__(self,
                 enable_rag: bool = True,
                 enable_contradiction: bool = True,
                 device: str = None):
        """
        Initialize the complete pipeline.
        
        Args:
            enable_rag: Whether to enable RAG-based QA
            enable_contradiction: Whether to enable contradiction detection
            device: 'cuda' or 'cpu'
        """
        print("=" * 60)
        print("Initializing PubMed NLP Pipeline")
        print("=" * 60)
        
        # Data ingestion
        self.pubmed_client = PubMedClient()
        
        # NLP Core
        print("\n[1/7] Loading preprocessing module...")
        self.preprocessor = BiomedicalPreprocessor(enable_linker=False)
        
        print("\n[2/7] Loading NER module...")
        self.ner = BiomedicalNER(models=['bc5cdr', 'jnlpba'])
        
        print("\n[3/7] Loading information extraction module...")
        self.info_extractor = InformationExtractor(device=device)
        
        print("\n[4/7] Loading summarization module...")
        self.summarizer = HybridSummarizer(device=-1 if device == 'cpu' else 0)
        
        # Advanced modules
        if enable_contradiction:
            print("\n[5/7] Loading contradiction detection module...")
            self.contradiction_detector = ContradictionDetector(device=device)
        else:
            self.contradiction_detector = None
        
        print("\n[6/7] Loading insight engine...")
        self.insight_engine = InsightEngine()
        
        if enable_rag:
            print("\n[7/7] Loading RAG system...")
            self.rag_system = RAGSystem(device=device)
        else:
            self.rag_system = None
        
        # Personalization
        self.personalization = PersonalizationEngine()
        
        print("\n" + "=" * 60)
        print("Pipeline initialization complete!")
        print("=" * 60)
    
    def process(self,
               query: str,
               max_articles: int = 50,
               user_types: List[UserType] = None,
               enable_qa: bool = False) -> PipelineResult:
        """
        Process a query through the complete pipeline.
        
        Args:
            query: Search query for PubMed
            max_articles: Maximum articles to retrieve
            user_types: List of user types for personalization
            enable_qa: Whether to generate RAG answer
            
        Returns:
            Complete PipelineResult
        """
        print(f"\n{'='*60}")
        print(f"Processing query: {query}")
        print(f"{'='*60}\n")
        
        # Step 1: Data Ingestion
        print("[Step 1/8] Retrieving articles from PubMed...")
        articles = self.pubmed_client.search_and_fetch(query, max_articles)
        print(f"  → Retrieved {len(articles)} articles")
        
        if not articles:
            return self._empty_result(query)
        
        # Step 2: Preprocessing
        print("\n[Step 2/8] Preprocessing articles...")
        texts = [f"{a.title}. {a.abstract}" for a in articles]
        preprocessed = self.preprocessor.preprocess_batch(texts, batch_size=16)
        print(f"  → Processed {len(preprocessed)} documents")
        
        # Step 3: Named Entity Recognition
        print("\n[Step 3/8] Extracting biomedical entities...")
        ner_docs = [self.ner.extract_entities(text) for text in tqdm(texts, desc="NER")]
        all_entities = []
        for doc in ner_docs:
            all_entities.extend([{
                'text': e.text,
                'label': e.label,
                'source': e.source
            } for e in doc.entities])
        print(f"  → Extracted {len(all_entities)} entities")
        
        # Step 4: Information Extraction
        print("\n[Step 4/8] Extracting keyphrases and relations...")
        all_keyphrases = []
        all_relations = []
        
        for i, (doc, ner_doc) in enumerate(zip(tqdm(texts, desc="Info Extraction"), ner_docs)):
            # Keyphrases
            kps = self.info_extractor.extract_keyphrases(doc, method='keybert', top_n=5)
            all_keyphrases.extend(kps)
            
            # Relations
            rels = self.info_extractor.extract_relations(doc, [{
                'text': e.text,
                'label': e.label,
                'start': e.start,
                'end': e.end
            } for e in ner_doc.entities])
            
            # Add PMID to relations
            for rel in rels:
                rel_dict = {
                    'subject': rel.subject,
                    'subject_type': rel.subject_type,
                    'predicate': rel.predicate,
                    'object': rel.object,
                    'object_type': rel.object_type,
                    'relation_type': rel.relation_type,
                    'confidence': rel.confidence,
                    'evidence': rel.evidence,
                    'pmid': articles[i].pmid if i < len(articles) else ''
                }
                all_relations.append(rel_dict)
        
        print(f"  → Extracted {len(all_keyphrases)} keyphrases, {len(all_relations)} relations")
        
        # Step 5: Summarization
        print("\n[Step 5/8] Generating hybrid summary...")
        combined_text = "\n\n".join([f"Article {i+1}: {t}" for i, t in enumerate(texts[:10])])
        summary = self.summarizer.summarize(combined_text, method='hybrid', compression_ratio=0.2)
        print(f"  → Summary length: {len(summary.summary)} chars (compression: {summary.compression_ratio:.1%})")
        
        # Step 6: Contradiction Detection
        contradictions = []
        if self.contradiction_detector and len(texts) > 1:
            print("\n[Step 6/8] Detecting contradictions...")
            # Extract key statements for contradiction detection
            statements = []
            for text in texts[:10]:  # Limit to first 10
                extracted = self.contradiction_detector.extract_statements_from_text(text)
                statements.extend(extracted[:3])  # Top 3 per article
            
            contradictions = self.contradiction_detector.detect_contradictions(statements[:30])
            print(f"  → Found {len(contradictions)} contradictions/relations")
        
        # Step 7: Insight Engine
        print("\n[Step 7/8] Generating insights...")
        
        # Trend analysis
        article_dicts = [{
            'pmid': a.pmid,
            'year': a.year,
            'mesh_terms': a.mesh_terms,
            'keywords': a.keywords,
            'journal': a.journal,
            'publication_types': a.publication_types,
            'abstract': a.abstract
        } for a in articles]
        
        trends = self.insight_engine.analyze_trends(article_dicts, query)
        
        # Evidence scoring
        evidence_scores = self.insight_engine.score_evidence(article_dicts)
        
        # Risk factor mining
        risk_factors = self.insight_engine.mine_risk_factors(all_relations)
        
        print(f"  → Trends: {trends.current_trajectory}, Risk factors: {len(risk_factors)}")
        
        # Step 8: Personalization
        print("\n[Step 8/8] Personalizing output...")
        user_types = user_types or [UserType.PATIENT, UserType.STUDENT, UserType.DOCTOR]
        personalized = {}
        
        evidence_level = evidence_scores[0].evidence_level if evidence_scores else 'unknown'
        
        for ut in user_types:
            personalized[ut.value] = self.personalization.to_dict(
                self.personalization.personalize(
                    summary=summary.summary,
                    key_points=summary.key_points or [],
                    entities=all_entities[:20],
                    relations=all_relations[:10],
                    evidence_level=evidence_level,
                    user_type=ut
                )
            )
        
        # Generate insights summary
        insights = self.insight_engine.generate_insights_summary(
            trends, evidence_scores, risk_factors
        )
        
        # RAG Answer (if enabled)
        rag_answer = None
        if enable_qa and self.rag_system:
            print("\n[Bonus] Generating RAG answer...")
            # Add to RAG
            self.rag_system.add_documents(
                texts=[a.abstract for a in articles if a.abstract],
                ids=[a.pmid for a in articles if a.abstract],
                metadatas=[{'title': a.title, 'year': a.year} for a in articles if a.abstract]
            )
            rag_answer = self.rag_system.answer(query, top_k=5)
        
        print(f"\n{'='*60}")
        print("Pipeline processing complete!")
        print(f"{'='*60}\n")
        
        return PipelineResult(
            query=query,
            articles=articles,
            entities=all_entities,
            relations=[ExtractedRelation(**{k: v for k, v in r.items() if k != 'pmid'}) for r in all_relations],
            keyphrases=all_keyphrases,
            summary=summary,
            contradictions=contradictions,
            trends=trends,
            evidence_scores=evidence_scores,
            risk_factors=risk_factors,
            personalized=personalized,
            rag_answer=rag_answer,
            insights=insights
        )
    
    def answer_question(self, question: str, context_query: str = None) -> RAGAnswer:
        """
        Answer a question using RAG.
        
        If context_query provided, first retrieves articles on that topic.
        """
        if not self.rag_system:
            raise ValueError("RAG system not enabled")
        
        # If context needed, fetch and add
        if context_query:
            articles = self.pubmed_client.search_and_fetch(context_query, 30)
            self.rag_system.add_documents(
                texts=[a.abstract for a in articles if a.abstract],
                ids=[a.pmid for a in articles if a.abstract],
                metadatas=[{'title': a.title, 'year': a.year} for a in articles if a.abstract]
            )
        
        return self.rag_system.answer(question, top_k=5)
    
    def _empty_result(self, query: str) -> PipelineResult:
        """Return empty result for failed queries."""
        return PipelineResult(
            query=query,
            articles=[],
            entities=[],
            relations=[],
            keyphrases=[],
            summary=None,
            contradictions=[],
            trends=None,
            evidence_scores=[],
            risk_factors=[],
            personalized={},
            rag_answer=None,
            insights={'error': 'No articles found for query'}
        )
    
    def save_result(self, result: PipelineResult, filepath: str) -> None:
        """Save pipeline result to JSON."""
        # Convert dataclasses to dicts
        data = {
            'query': result.query,
            'articles': [self._article_to_dict(a) for a in result.articles],
            'entities': result.entities,
            'relations': [self._relation_to_dict(r) for r in result.relations],
            'keyphrases': [{'text': k.text, 'score': k.score, 'method': k.method} for k in result.keyphrases],
            'summary': {
                'summary': result.summary.summary,
                'method': result.summary.method,
                'compression_ratio': result.summary.compression_ratio,
                'key_points': result.summary.key_points
            } if result.summary else None,
            'contradictions': [self._contradiction_to_dict(c) for c in result.contradictions],
            'trends': self._trend_to_dict(result.trends) if result.trends else None,
            'evidence_scores': [self._evidence_to_dict(e) for e in result.evidence_scores],
            'risk_factors': [self._risk_to_dict(r) for r in result.risk_factors],
            'personalized': result.personalized,
            'insights': result.insights
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _article_to_dict(self, a: PubMedArticle) -> Dict:
        return {
            'pmid': a.pmid,
            'title': a.title,
            'abstract': a.abstract[:500] + '...' if len(a.abstract) > 500 else a.abstract,
            'authors': a.authors[:5],
            'year': a.year,
            'mesh_terms': a.mesh_terms[:10],
            'journal': a.journal
        }
    
    def _relation_to_dict(self, r: ExtractedRelation) -> Dict:
        return {
            'subject': r.subject,
            'subject_type': r.subject_type,
            'predicate': r.predicate,
            'object': r.object,
            'object_type': r.object_type,
            'relation_type': r.relation_type,
            'confidence': r.confidence
        }
    
    def _contradiction_to_dict(self, c: ContradictionResult) -> Dict:
        return {
            'statement1': c.statement1,
            'statement2': c.statement2,
            'relation': c.relation,
            'confidence': c.confidence,
            'similarity': c.similarity_score
        }
    
    def _trend_to_dict(self, t: TrendAnalysis) -> Dict:
        return {
            'topic': t.topic,
            'trajectory': t.current_trajectory,
            'growth_rate': t.growth_rate,
            'peak_year': t.peak_year,
            'data_points': len(t.trend_points)
        }
    
    def _evidence_to_dict(self, e: EvidenceScore) -> Dict:
        return {
            'pmid': e.pmid,
            'overall_score': e.overall_score,
            'evidence_level': e.evidence_level
        }
    
    def _risk_to_dict(self, r: RiskFactor) -> Dict:
        return {
            'factor': r.factor,
            'outcome': r.outcome,
            'relation': r.relation,
            'confidence': r.confidence,
            'evidence_count': r.evidence_count
        }
