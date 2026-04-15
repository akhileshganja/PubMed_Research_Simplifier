"""Contradiction Detection using Embeddings and NLI models."""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re


@dataclass
class ContradictionResult:
    """Contradiction detection result."""
    statement1: str
    statement2: str
    similarity_score: float
    contradiction_score: float
    entailment_score: float
    neutral_score: float
    relation: str  # 'contradiction', 'entailment', 'neutral', 'similar'
    confidence: float


@dataclass
class StatementCluster:
    """Cluster of similar statements."""
    statements: List[str]
    representative: str
    avg_similarity: float
    sources: List[str]  # Article IDs


class ContradictionDetector:
    """
    Contradiction detection using sentence embeddings and NLI.
    
    Two-stage approach:
    1. Embedding similarity to find candidate pairs
    2. NLI classification for contradiction/entailment/neutral
    """
    
    def __init__(self,
                 embedding_model: str = 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb',
                 nli_model: str = None,
                 similarity_threshold: float = 0.7,
                 contradiction_threshold: float = 0.5,
                 device: str = None):
        """
        Initialize contradiction detector.
        
        Args:
            embedding_model: Model for sentence embeddings
            nli_model: Model for NLI (defaults to embedding_model if supports NLI)
            similarity_threshold: Threshold for semantic similarity
            contradiction_threshold: Threshold for contradiction classification
            device: 'cuda' or 'cpu'
        """
        print(f"Loading embedding model: {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model, device=device)
        
        self.similarity_threshold = similarity_threshold
        self.contradiction_threshold = contradiction_threshold
        
        # Check if model supports NLI (has 'mnli' or 'snli' in name)
        self.nli_enabled = 'nli' in embedding_model.lower()
        
        if not self.nli_enabled and nli_model:
            print(f"Loading NLI model: {nli_model}...")
            self.nli_model = SentenceTransformer(nli_model, device=device)
            self.nli_enabled = True
        else:
            self.nli_model = self.embedding_model if self.nli_enabled else None
    
    def encode_statements(self, statements: List[str]) -> np.ndarray:
        """Encode statements to embeddings."""
        return self.embedding_model.encode(statements, show_progress_bar=True)
    
    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity matrix."""
        return cosine_similarity(embeddings)
    
    def find_similar_pairs(self, 
                          statements: List[str],
                          embeddings: np.ndarray = None,
                          top_k: int = 5) -> List[Tuple[int, int, float]]:
        """
        Find most similar statement pairs.
        
        Returns: List of (idx1, idx2, similarity)
        """
        if embeddings is None:
            embeddings = self.encode_statements(statements)
        
        sim_matrix = self.compute_similarity_matrix(embeddings)
        
        pairs = []
        n = len(statements)
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = sim_matrix[i, j]
                if sim >= self.similarity_threshold:
                    pairs.append((i, j, sim))
        
        # Sort by similarity
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        return pairs[:top_k] if top_k else pairs
    
    def classify_relation(self, statement1: str, statement2: str) -> Dict[str, float]:
        """
        Classify relation between two statements using NLI.
        
        Returns scores for: entailment, contradiction, neutral
        """
        if not self.nli_enabled:
            # Fallback to similarity-based classification
            embeddings = self.encode_statements([statement1, statement2])
            sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            # Heuristic mapping
            if sim > 0.85:
                return {'entailment': sim, 'contradiction': 0.0, 'neutral': 1 - sim}
            elif sim < 0.5:
                return {'entailment': 0.0, 'contradiction': 0.5, 'neutral': 0.5}
            else:
                return {'entailment': 0.0, 'contradiction': 0.0, 'neutral': sim}
        
        # Use NLI model
        # NLI models typically take (premise, hypothesis) pairs
        # For bi-directional NLI, check both directions
        
        scores1 = self._nli_predict(statement1, statement2)
        scores2 = self._nli_predict(statement2, statement1)
        
        # Average the bi-directional scores
        return {
            'entailment': (scores1['entailment'] + scores2['entailment']) / 2,
            'contradiction': (scores1['contradiction'] + scores2['contradiction']) / 2,
            'neutral': (scores1['neutral'] + scores2['neutral']) / 2,
        }
    
    def _nli_predict(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """
        Predict NLI label for premise-hypothesis pair.
        
        Returns softmax scores for entailment, contradiction, neutral.
        """
        # Tokenize and encode
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        # Try to use cached tokenizer/model
        if not hasattr(self, '_nli_tokenizer'):
            # Use a standard NLI model
            model_name = 'roberta-large-mnli'
            self._nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._nli_classifier = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._nli_classifier.eval()
        
        # Encode pair
        inputs = self._nli_tokenizer(
            premise, hypothesis,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Predict
        import torch
        with torch.no_grad():
            outputs = self._nli_classifier(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
        
        # RoBERTa-MNLI: 0=contradiction, 1=neutral, 2=entailment
        return {
            'contradiction': float(probs[0]),
            'neutral': float(probs[1]),
            'entailment': float(probs[2]),
        }
    
    def detect_contradictions(self, 
                              statements: List[str],
                              sources: List[str] = None) -> List[ContradictionResult]:
        """
        Detect contradictions within a set of statements.
        
        Args:
            statements: List of statements to compare
            sources: Optional list of source article IDs
            
        Returns:
            List of ContradictionResult for detected relations
        """
        if len(statements) < 2:
            return []
        
        # Encode all statements
        embeddings = self.encode_statements(statements)
        
        # Find candidate pairs
        similar_pairs = self.find_similar_pairs(statements, embeddings)
        
        results = []
        
        for idx1, idx2, sim in similar_pairs:
            stmt1 = statements[idx1]
            stmt2 = statements[idx2]
            
            # Classify relation
            nli_scores = self.classify_relation(stmt1, stmt2)
            
            # Determine relation type
            max_label = max(nli_scores, key=nli_scores.get)
            max_score = nli_scores[max_label]
            
            # Map to output relation
            if max_label == 'contradiction' and max_score > self.contradiction_threshold:
                relation = 'contradiction'
                confidence = max_score
            elif max_label == 'entailment' and max_score > 0.5:
                relation = 'entailment'
                confidence = max_score
            elif sim > 0.8:
                relation = 'similar'
                confidence = sim
            else:
                relation = 'neutral'
                confidence = nli_scores['neutral']
            
            # Only report contradictions, entailments, and high-similarity pairs
            if relation in ['contradiction', 'entailment'] or (relation == 'similar' and sim > 0.85):
                results.append(ContradictionResult(
                    statement1=stmt1,
                    statement2=stmt2,
                    similarity_score=sim,
                    contradiction_score=nli_scores['contradiction'],
                    entailment_score=nli_scores['entailment'],
                    neutral_score=nli_scores['neutral'],
                    relation=relation,
                    confidence=confidence
                ))
        
        # Sort by confidence
        results.sort(key=lambda x: x.confidence, reverse=True)
        
        return results
    
    def cluster_statements(self, 
                          statements: List[str],
                          sources: List[str] = None,
                          similarity_threshold: float = 0.75) -> List[StatementCluster]:
        """
        Cluster similar statements together.
        
        Uses hierarchical clustering based on similarity threshold.
        """
        if len(statements) == 0:
            return []
        
        if len(statements) == 1:
            return [StatementCluster(
                statements=statements,
                representative=statements[0],
                avg_similarity=1.0,
                sources=sources or []
            )]
        
        embeddings = self.encode_statements(statements)
        sim_matrix = self.compute_similarity_matrix(embeddings)
        
        # Greedy clustering
        n = len(statements)
        assigned = set()
        clusters = []
        
        for i in range(n):
            if i in assigned:
                continue
            
            # Find all similar statements
            cluster_indices = [i]
            for j in range(n):
                if j != i and j not in assigned:
                    if sim_matrix[i, j] >= similarity_threshold:
                        cluster_indices.append(j)
            
            if len(cluster_indices) > 1:
                # Mark as assigned
                for idx in cluster_indices:
                    assigned.add(idx)
                
                # Calculate average similarity
                cluster_sims = []
                for ii in cluster_indices:
                    for jj in cluster_indices:
                        if ii < jj:
                            cluster_sims.append(sim_matrix[ii, jj])
                
                avg_sim = np.mean(cluster_sims) if cluster_sims else 1.0
                
                # Choose representative (most central)
                cluster_embeddings = embeddings[cluster_indices]
                centroid = np.mean(cluster_embeddings, axis=0)
                distances = [np.linalg.norm(embeddings[idx] - centroid) for idx in cluster_indices]
                rep_idx = cluster_indices[np.argmin(distances)]
                
                clusters.append(StatementCluster(
                    statements=[statements[idx] for idx in cluster_indices],
                    representative=statements[rep_idx],
                    avg_similarity=float(avg_sim),
                    sources=[sources[idx] for idx in cluster_indices] if sources else []
                ))
        
        # Add remaining unassigned as single clusters
        for i in range(n):
            if i not in assigned:
                clusters.append(StatementCluster(
                    statements=[statements[i]],
                    representative=statements[i],
                    avg_similarity=1.0,
                    sources=[sources[i]] if sources else []
                ))
        
        return clusters
    
    def check_article_consistency(self,
                                  claim: str,
                                  article_statements: List[str]) -> Dict:
        """
        Check if a claim is consistent with article statements.
        
        Returns consistency analysis with supporting/contradicting evidence.
        """
        all_statements = [claim] + article_statements
        
        # Encode
        embeddings = self.encode_statements(all_statements)
        
        # Compare claim to each article statement
        claim_embedding = embeddings[0]
        article_embeddings = embeddings[1:]
        
        similarities = cosine_similarity([claim_embedding], article_embeddings)[0]
        
        supporting = []
        contradicting = []
        
        for i, (stmt, sim) in enumerate(zip(article_statements, similarities)):
            nli_scores = self.classify_relation(claim, stmt)
            
            if nli_scores['entailment'] > 0.6 or sim > 0.85:
                supporting.append({
                    'statement': stmt,
                    'similarity': float(sim),
                    'entailment_score': nli_scores['entailment'],
                    'index': i
                })
            elif nli_scores['contradiction'] > 0.5:
                contradicting.append({
                    'statement': stmt,
                    'similarity': float(sim),
                    'contradiction_score': nli_scores['contradiction'],
                    'index': i
                })
        
        # Determine overall consistency
        if contradicting and not supporting:
            consistency = 'contradicted'
            confidence = max(c['contradiction_score'] for c in contradicting)
        elif supporting and not contradicting:
            consistency = 'supported'
            confidence = max(s['entailment_score'] for s in supporting)
        elif supporting and contradicting:
            consistency = 'mixed'
            confidence = 0.5
        else:
            consistency = 'neutral'
            confidence = 0.5
        
        return {
            'consistency': consistency,
            'confidence': confidence,
            'supporting_evidence': supporting,
            'contradicting_evidence': contradicting,
            'total_statements_checked': len(article_statements)
        }
    
    def extract_statements_from_text(self, text: str, 
                                     sentence_level: bool = True) -> List[str]:
        """
        Extract factual statements from text.
        
        For sentence-level: split into sentences
        For claim-level: extract statements with claim-like patterns
        """
        if sentence_level:
            # Split on sentence boundaries
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if len(s.strip()) > 20]
        else:
            # Extract claim-like statements
            patterns = [
                r'[^.]*?(?:is|are|was|were|has|have|shows?|demonstrates?|suggests?|indicates?|confirms?|proves?)[^.]*?\.',
                r'[^.]*?(?:treatment|therapy|drug|medication|intervention)[^.]*?(?:effective|efficacious|beneficial|helpful)[^.]*?\.',
                r'[^.]*?(?:risk|association|correlation|link)[^.]*?(?:increase|decrease|higher|lower)[^.]*?\.',
            ]
            
            statements = []
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                statements.extend([m.strip() for m in matches if len(m) > 20])
            
            # Deduplicate
            seen = set()
            unique = []
            for s in statements:
                key = s.lower()
                if key not in seen:
                    seen.add(key)
                    unique.append(s)
            
            return unique if unique else self.extract_statements_from_text(text, sentence_level=True)
