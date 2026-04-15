"""Evaluation Metrics for PubMed NLP Pipeline."""

from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np
from rouge_score import rouge_scorer
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from seqeval.scheme import IOB2


@dataclass
class NERMetrics:
    """NER evaluation metrics."""
    precision: float
    recall: float
    f1: float
    per_entity: Dict[str, Dict[str, float]]


@dataclass
class SummarizationMetrics:
    """Summarization evaluation metrics."""
    rouge1: float
    rouge2: float
    rougeL: float
    avg_compression: float


class Evaluator:
    """
    Comprehensive evaluation suite for the PubMed NLP pipeline.
    
    Metrics:
    - NER: Precision, Recall, F1 (per entity type)
    - Summarization: ROUGE-1, ROUGE-2, ROUGE-L
    """
    
    def __init__(self):
        """Initialize evaluator."""
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def evaluate_ner(self,
                    predictions: List[List[Tuple[str, str]]],
                    references: List[List[Tuple[str, str]]]) -> NERMetrics:
        """
        Evaluate NER performance.
        
        Args:
            predictions: List of (token, label) sequences
            references: List of (token, label) sequences
            
        Returns:
            NERMetrics with precision, recall, f1
        """
        # Convert to seqeval format
        y_true = [[label for _, label in seq] for seq in references]
        y_pred = [[label for _, label in seq] for seq in predictions]
        
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # Per-entity metrics
        report = classification_report(y_true, y_pred, output_dict=True)
        per_entity = {
            k: {'precision': v['precision'], 'recall': v['recall'], 'f1': v['f1-score']}
            for k, v in report.items()
            if k not in ['micro avg', 'macro avg', 'weighted avg']
        }
        
        return NERMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            per_entity=per_entity
        )
    
    def evaluate_summarization(self,
                              predictions: List[str],
                              references: List[str],
                              sources: List[str] = None) -> SummarizationMetrics:
        """
        Evaluate summarization performance using ROUGE.
        
        Args:
            predictions: Generated summaries
            references: Reference (gold) summaries
            sources: Optional source texts for compression calculation
            
        Returns:
            SummarizationMetrics with ROUGE scores
        """
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        # Compression ratio
        avg_compression = 0.0
        if sources:
            compressions = [len(p) / len(s) for p, s in zip(predictions, sources)]
            avg_compression = np.mean(compressions)
        
        return SummarizationMetrics(
            rouge1=np.mean(rouge1_scores),
            rouge2=np.mean(rouge2_scores),
            rougeL=np.mean(rougeL_scores),
            avg_compression=avg_compression
        )
    
    def evaluate_retrieval(self,
                          retrieved: List[List[str]],
                          relevant: List[List[str]],
                          k: int = 10) -> Dict:
        """
        Evaluate retrieval performance.
        
        Metrics:
        - Precision@K
        - Recall@K
        - MRR (Mean Reciprocal Rank)
        - NDCG@K
        
        Args:
            retrieved: List of retrieved document IDs per query
            relevant: List of relevant document IDs per query
            k: Cutoff for metrics
            
        Returns:
            Dictionary of metrics
        """
        precisions = []
        recalls = []
        mrrs = []
        ndcgs = []
        
        for ret, rel in zip(retrieved, relevant):
            rel_set = set(rel)
            
            # Precision@K
            retrieved_k = ret[:k]
            relevant_retrieved = len([r for r in retrieved_k if r in rel_set])
            precision = relevant_retrieved / k if k > 0 else 0
            precisions.append(precision)
            
            # Recall@K
            recall = relevant_retrieved / len(rel_set) if rel_set else 0
            recalls.append(recall)
            
            # MRR
            mrr = 0
            for i, r in enumerate(retrieved_k):
                if r in rel_set:
                    mrr = 1 / (i + 1)
                    break
            mrrs.append(mrr)
            
            # NDCG@K
            dcg = sum([1 / np.log2(i + 2) for i, r in enumerate(retrieved_k) if r in rel_set])
            ideal_dcg = sum([1 / np.log2(i + 2) for i in range(min(k, len(rel_set)))])
            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
            ndcgs.append(ndcg)
        
        return {
            f'precision@{k}': np.mean(precisions),
            f'recall@{k}': np.mean(recalls),
            'mrr': np.mean(mrrs),
            f'ndcg@{k}': np.mean(ndcgs)
        }
    
    def evaluate_contradiction_detection(self,
                                        predictions: List[str],
                                        labels: List[str]) -> Dict:
        """
        Evaluate contradiction detection accuracy.
        
        Args:
            predictions: Predicted relations ('contradiction', 'entailment', 'neutral')
            labels: Gold labels
            
        Returns:
            Accuracy and per-class metrics
        """
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        accuracy = accuracy_score(labels, predictions)
        report = classification_report(labels, predictions, output_dict=True)
        cm = confusion_matrix(labels, predictions, labels=['contradiction', 'entailment', 'neutral'])
        
        return {
            'accuracy': accuracy,
            'per_class': report,
            'confusion_matrix': cm.tolist()
        }
    
    def evaluate_rag(self,
                    answers: List[str],
                    references: List[str],
                    contexts: List[List[str]]) -> Dict:
        """
        Evaluate RAG answer quality.
        
        Metrics:
        - ROUGE vs reference
        - Faithfulness (answer supported by context)
        
        Args:
            answers: Generated answers
            references: Reference answers
            contexts: Retrieved contexts per question
            
        Returns:
            Dictionary of metrics
        """
        # ROUGE
        rouge_scores = self.evaluate_summarization(answers, references)
        
        # Faithfulness (simplified - check if answer entities appear in context)
        faithful_scores = []
        for ans, ctx_list in zip(answers, contexts):
            ctx_text = ' '.join(ctx_list).lower()
            # Extract simple entities (capitalized phrases)
            import re
            entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', ans))
            if entities:
                matched = sum(1 for e in entities if e.lower() in ctx_text)
                faithful_scores.append(matched / len(entities))
            else:
                faithful_scores.append(1.0)  # No entities to check
        
        return {
            'rouge1': rouge_scores.rouge1,
            'rougeL': rouge_scores.rougeL,
            'faithfulness': np.mean(faithful_scores)
        }


def run_benchmark(pipeline, test_queries: List[str]) -> Dict:
    """
    Run benchmark evaluation on test queries.
    
    This is a simplified benchmark. For full evaluation,
    use annotated gold standard datasets.
    """
    results = {
        'queries': test_queries,
        'article_counts': [],
        'entity_counts': [],
        'processing_times': [],
    }
    
    import time
    
    for query in test_queries:
        start = time.time()
        result = pipeline.process(query, max_articles=20)
        elapsed = time.time() - start
        
        results['article_counts'].append(len(result.articles))
        results['entity_counts'].append(len(result.entities))
        results['processing_times'].append(elapsed)
    
    results['avg_articles'] = np.mean(results['article_counts'])
    results['avg_entities'] = np.mean(results['entity_counts'])
    results['avg_time'] = np.mean(results['processing_times'])
    
    return results
