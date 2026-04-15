"""Insight Engine: Trend detection, evidence scoring, and risk factor mining."""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import Counter, defaultdict
import numpy as np
from datetime import datetime
import re


@dataclass
class TrendPoint:
    """Single point in a trend."""
    year: int
    count: int
    percentage: float
    top_terms: List[Tuple[str, int]]


@dataclass
class TrendAnalysis:
    """Trend analysis results."""
    topic: str
    trend_points: List[TrendPoint]
    growth_rate: float
    peak_year: int
    current_trajectory: str  # 'rising', 'stable', 'declining'


@dataclass
class EvidenceScore:
    """Evidence quality scoring."""
    pmid: str
    citation_count: int
    journal_impact: float
    recency_score: float
    study_type_score: float
    sample_size_score: float
    overall_score: float
    evidence_level: str  # 'high', 'moderate', 'low'


@dataclass
class RiskFactor:
    """Extracted risk factor relationship."""
    factor: str
    factor_type: str
    outcome: str
    outcome_type: str
    relation: str  # 'increases', 'decreases', 'associated_with'
    confidence: float
    evidence_count: int
    supporting_pmids: List[str]


class InsightEngine:
    """
    Insight generation from processed PubMed data.
    
    Features:
    - Trend detection over time
    - Evidence quality scoring
    - Risk factor mining from relations
    """
    
    # Journal impact factors (simplified - in production use official JCR data)
    JOURNAL_TIERS = {
        'high': [
            'nature', 'science', 'cell', 'lancet', 'nejm', 'jama', 'bmj',
            'nature medicine', 'nature reviews', 'cell host microbe'
        ],
        'medium': [
            'plos', 'bmc', 'journal of', 'clinical', 'american journal',
            'british journal', 'international journal'
        ],
    }
    
    STUDY_TYPE_HIERARCHY = {
        'systematic review': 1.0,
        'meta-analysis': 1.0,
        'randomized controlled trial': 0.9,
        'clinical trial': 0.8,
        'cohort study': 0.7,
        'case-control study': 0.6,
        'cross-sectional study': 0.5,
        'case series': 0.4,
        'case report': 0.3,
        'observational study': 0.5,
    }
    
    def __init__(self, current_year: int = None):
        """Initialize insight engine."""
        self.current_year = current_year or datetime.now().year
    
    def analyze_trends(self,
                      articles: List[Dict],
                      topic_query: str,
                      entity_field: str = 'mesh_terms') -> TrendAnalysis:
        """
        Analyze publication trends for a topic over time.
        
        Args:
            articles: List of article dictionaries with 'year' and entity_field
            topic_query: Topic to analyze
            entity_field: Field containing entities (mesh_terms, keywords, etc.)
        """
        # Group by year
        year_data = defaultdict(lambda: {'count': 0, 'terms': Counter()})
        
        for article in articles:
            year = article.get('year')
            if year and 2000 <= year <= self.current_year:
                year_data[year]['count'] += 1
                
                # Count terms
                terms = article.get(entity_field, [])
                if isinstance(terms, list):
                    year_data[year]['terms'].update([t.lower() for t in terms])
        
        if not year_data:
            return TrendAnalysis(
                topic=topic_query,
                trend_points=[],
                growth_rate=0.0,
                peak_year=self.current_year,
                current_trajectory='unknown'
            )
        
        # Calculate trend points
        total_articles = sum(d['count'] for d in year_data.values())
        trend_points = []
        
        for year in sorted(year_data.keys()):
            data = year_data[year]
            percentage = (data['count'] / total_articles * 100) if total_articles > 0 else 0
            top_terms = data['terms'].most_common(5)
            
            trend_points.append(TrendPoint(
                year=year,
                count=data['count'],
                percentage=percentage,
                top_terms=top_terms
            ))
        
        # Calculate growth rate (last 3 years vs previous 3)
        recent_years = [y for y in year_data.keys() if y >= self.current_year - 3]
        earlier_years = [y for y in year_data.keys() 
                        if self.current_year - 6 <= y < self.current_year - 3]
        
        recent_count = sum(year_data[y]['count'] for y in recent_years)
        earlier_count = sum(year_data[y]['count'] for y in earlier_years)
        
        if earlier_count > 0:
            growth_rate = (recent_count - earlier_count) / earlier_count
        else:
            growth_rate = float('inf') if recent_count > 0 else 0.0
        
        # Find peak year
        peak_year = max(year_data.keys(), key=lambda y: year_data[y]['count'])
        
        # Determine trajectory
        if len(trend_points) >= 2:
            recent_avg = np.mean([p.count for p in trend_points[-3:]])
            earlier_avg = np.mean([p.count for p in trend_points[:3]]) if len(trend_points) >= 6 else recent_avg
            
            if recent_avg > earlier_avg * 1.2:
                trajectory = 'rising'
            elif recent_avg < earlier_avg * 0.8:
                trajectory = 'declining'
            else:
                trajectory = 'stable'
        else:
            trajectory = 'insufficient_data'
        
        return TrendAnalysis(
            topic=topic_query,
            trend_points=trend_points,
            growth_rate=growth_rate,
            peak_year=peak_year,
            current_trajectory=trajectory
        )
    
    def score_evidence(self, 
                      articles: List[Dict],
                      citation_data: Dict[str, int] = None) -> List[EvidenceScore]:
        """
        Score evidence quality for articles.
        
        Factors:
        - Citation count (if available)
        - Journal impact tier
        - Recency
        - Study type
        - Sample size (if mentioned)
        """
        scores = []
        
        for article in articles:
            pmid = article.get('pmid', '')
            
            # Citation score
            citations = citation_data.get(pmid, 0) if citation_data else 0
            citation_score = min(citations / 100, 1.0)  # Cap at 100 citations
            
            # Journal impact
            journal = article.get('journal', '').lower()
            journal_score = self._score_journal(journal)
            
            # Recency score (exponential decay)
            year = article.get('year')
            if year:
                years_old = self.current_year - year
                recency_score = np.exp(-years_old / 10)  # Half-life of 10 years
            else:
                recency_score = 0.5
            
            # Study type score
            pub_types = article.get('publication_types', [])
            study_type_score = self._score_study_type(pub_types)
            
            # Sample size (from abstract if mentioned)
            abstract = article.get('abstract', '')
            sample_score = self._extract_sample_size_score(abstract)
            
            # Weighted overall score
            weights = {
                'citation': 0.2,
                'journal': 0.25,
                'recency': 0.15,
                'study_type': 0.25,
                'sample': 0.15
            }
            
            overall = (
                weights['citation'] * citation_score +
                weights['journal'] * journal_score +
                weights['recency'] * recency_score +
                weights['study_type'] * study_type_score +
                weights['sample'] * sample_score
            )
            
            # Evidence level
            if overall >= 0.7:
                level = 'high'
            elif overall >= 0.4:
                level = 'moderate'
            else:
                level = 'low'
            
            scores.append(EvidenceScore(
                pmid=pmid,
                citation_count=citations,
                journal_impact=journal_score,
                recency_score=recency_score,
                study_type_score=study_type_score,
                sample_size_score=sample_score,
                overall_score=overall,
                evidence_level=level
            ))
        
        # Sort by overall score
        scores.sort(key=lambda x: x.overall_score, reverse=True)
        
        return scores
    
    def _score_journal(self, journal_name: str) -> float:
        """Score journal based on tier."""
        journal_lower = journal_name.lower()
        
        for high_journal in self.JOURNAL_TIERS['high']:
            if high_journal in journal_lower:
                return 1.0
        
        for med_journal in self.JOURNAL_TIERS['medium']:
            if med_journal in journal_lower:
                return 0.6
        
        return 0.4  # Default
    
    def _score_study_type(self, pub_types: List[str]) -> float:
        """Score based on publication/study type."""
        if not pub_types:
            return 0.5
        
        best_score = 0.0
        for pt in pub_types:
            pt_lower = pt.lower()
            for study_type, score in self.STUDY_TYPE_HIERARCHY.items():
                if study_type in pt_lower:
                    best_score = max(best_score, score)
        
        return best_score if best_score > 0 else 0.5
    
    def _extract_sample_size_score(self, abstract: str) -> float:
        """Extract sample size from abstract and score it."""
        if not abstract:
            return 0.5
        
        # Look for patterns like "n=500", "500 patients", "sample of 500"
        patterns = [
            r'[Nn]\s*=\s*(\d{1,6})',
            r'(\d{1,6})\s+(?:patients?|subjects?|participants?|samples?|cases?)',
            r'(?:enrolled|recruited|included)\s+(\d{1,6})',
            r'total\s+(?:of\s+)?(\d{1,6})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, abstract)
            if match:
                n = int(match.group(1))
                # Score based on size
                if n >= 10000:
                    return 1.0
                elif n >= 1000:
                    return 0.9
                elif n >= 500:
                    return 0.8
                elif n >= 100:
                    return 0.7
                elif n >= 50:
                    return 0.6
                else:
                    return 0.5
        
        return 0.5  # Unknown
    
    def mine_risk_factors(self, 
                         relations: List[Dict],
                         min_confidence: float = 0.5,
                         min_evidence: int = 2) -> List[RiskFactor]:
        """
        Mine risk factor relationships from extracted relations.
        
        Aggregates relations to identify significant risk factors.
        """
        # Group by factor-outcome pair
        factor_groups = defaultdict(list)
        
        for rel in relations:
            relation_type = rel.get('relation_type', '')
            if relation_type not in ['causes', 'associated_with', 'treats', 'prevents']:
                continue
            
            subj = rel.get('subject', '')
            obj = rel.get('object', '')
            conf = rel.get('confidence', 0.0)
            pmid = rel.get('pmid', '')
            
            if conf < min_confidence:
                continue
            
            # Create key (factor, outcome)
            if relation_type in ['causes', 'associated_with']:
                key = (subj, obj)
                factor_type = rel.get('subject_type', 'unknown')
                outcome_type = rel.get('object_type', 'unknown')
            else:  # treats, prevents
                key = (obj, subj)  # Reverse for risk interpretation
                factor_type = rel.get('object_type', 'unknown')
                outcome_type = rel.get('subject_type', 'unknown')
            
            factor_groups[key].append({
                'relation': relation_type,
                'confidence': conf,
                'pmid': pmid,
                'factor_type': factor_type,
                'outcome_type': outcome_type
            })
        
        # Aggregate
        risk_factors = []
        
        for (factor, outcome), evidence_list in factor_groups.items():
            if len(evidence_list) < min_evidence:
                continue
            
            # Determine relation type
            relations_types = [e['relation'] for e in evidence_list]
            
            if 'causes' in relations_types or relations_types.count('causes') > len(evidence_list) / 2:
                relation = 'increases'
            elif 'treats' in relations_types or 'prevents' in relations_types:
                relation = 'decreases'
            else:
                relation = 'associated_with'
            
            # Average confidence
            avg_confidence = np.mean([e['confidence'] for e in evidence_list])
            
            # Get unique PMIDs
            pmids = list(set([e['pmid'] for e in evidence_list if e['pmid']]))
            
            # Determine types from evidence
            factor_type = evidence_list[0]['factor_type']
            outcome_type = evidence_list[0]['outcome_type']
            
            risk_factors.append(RiskFactor(
                factor=factor,
                factor_type=factor_type,
                outcome=outcome,
                outcome_type=outcome_type,
                relation=relation,
                confidence=min(avg_confidence * len(evidence_list) / 10, 1.0),  # Boost with evidence count
                evidence_count=len(evidence_list),
                supporting_pmids=pmids
            ))
        
        # Sort by confidence
        risk_factors.sort(key=lambda x: x.confidence, reverse=True)
        
        return risk_factors
    
    def generate_insights_summary(self,
                                  trend_analysis: TrendAnalysis,
                                  evidence_scores: List[EvidenceScore],
                                  risk_factors: List[RiskFactor]) -> Dict:
        """
        Generate human-readable insights summary.
        """
        insights = {
            'trend_summary': {},
            'evidence_summary': {},
            'risk_factors_summary': [],
            'key_findings': []
        }
        
        # Trend insights
        if trend_analysis.trend_points:
            recent = trend_analysis.trend_points[-3:]
            insights['trend_summary'] = {
                'topic': trend_analysis.topic,
                'trajectory': trend_analysis.current_trajectory,
                'peak_year': trend_analysis.peak_year,
                'growth_rate': f"{trend_analysis.growth_rate:.1%}",
                'recent_publications': sum(p.count for p in recent),
                'recent_top_terms': recent[-1].top_terms if recent else []
            }
        
        # Evidence insights
        if evidence_scores:
            high_evidence = [e for e in evidence_scores if e.evidence_level == 'high']
            insights['evidence_summary'] = {
                'total_articles': len(evidence_scores),
                'high_quality_count': len(high_evidence),
                'high_quality_percentage': f"{len(high_evidence) / len(evidence_scores):.1%}",
                'top_evidence_pmids': [e.pmid for e in evidence_scores[:5]]
            }
        
        # Risk factor insights
        for rf in risk_factors[:5]:
            insights['risk_factors_summary'].append({
                'factor': rf.factor,
                'outcome': rf.outcome,
                'relation': rf.relation,
                'confidence': f"{rf.confidence:.1%}",
                'evidence_count': rf.evidence_count
            })
        
        # Generate key findings
        if trend_analysis.current_trajectory == 'rising':
            insights['key_findings'].append(
                f"Research on '{trend_analysis.topic}' is growing rapidly ({trend_analysis.growth_rate:.1%} increase)"
            )
        
        if risk_factors:
            top_rf = risk_factors[0]
            insights['key_findings'].append(
                f"{top_rf.factor} is a significant {top_rf.relation} risk factor for {top_rf.outcome} "
                f"(confidence: {top_rf.confidence:.1%})"
            )
        
        high_count = len([e for e in evidence_scores if e.evidence_level == 'high'])
        if high_count > 0:
            insights['key_findings'].append(
                f"Found {high_count} high-quality evidence sources"
            )
        
        return insights
