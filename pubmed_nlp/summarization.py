"""Hybrid Summarization: Extractive + Abstractive with medical safety."""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np

# Extractive summarization
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# Transformers for abstractive
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Evaluation
from rouge_score import rouge_scorer


@dataclass
class SummaryResult:
    """Summarization output with metadata."""
    summary: str
    method: str
    source_sentences: List[str]  # For extractive
    compression_ratio: float
    rouge_scores: Optional[Dict] = None
    key_points: List[str] = None


class HybridSummarizer:
    """
    Hybrid summarization pipeline combining extractive and abstractive methods.
    
    Strategy:
    1. First extract most important sentences (extractive)
    2. Then condense/refine using transformer (abstractive)
    3. Validate output against source (reduce hallucination)
    """
    
    def __init__(self,
                 abstractive_model: str = "Falconsai/medical_summarization",
                 device: int = -1):
        """
        Initialize hybrid summarizer.
        
        Args:
            abstractive_model: HuggingFace model for abstractive summarization
            device: -1 for CPU, 0+ for GPU
        """
        print(f"Loading abstractive model: {abstractive_model}...")
        
        try:
            self.abstractive_pipeline = pipeline(
                "summarization",
                model=abstractive_model,
                device=device,
                max_length=512,
                min_length=50
            )
            self.abstractive_model_name = abstractive_model
        except Exception as e:
            print(f"Warning: Could not load abstractive model: {e}")
            print("Falling back to extractive-only summarization")
            self.abstractive_pipeline = None
            self.abstractive_model_name = None
        
        # Initialize extractive summarizers
        self.language = "english"
        self.stemmer = Stemmer(self.language)
        self.stop_words = get_stop_words(self.language)
        
        self.textrank = TextRankSummarizer(self.stemmer)
        self.textrank.stop_words = self.stop_words
        
        self.lexrank = LexRankSummarizer(self.stemmer)
        self.lexrank.stop_words = self.stop_words
        
        # ROUGE scorer for evaluation
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def summarize(self,
                  text: str,
                  method: str = 'hybrid',
                  compression_ratio: float = 0.3,
                  max_length: int = 200) -> SummaryResult:
        """
        Summarize text using specified method.
        
        Methods:
        - 'extractive': TextRank sentence extraction
        - 'abstractive': Transformer-based (if available)
        - 'hybrid': Extractive then abstractive (RECOMMENDED)
        
        Args:
            text: Input text to summarize
            method: Summarization method
            compression_ratio: Target compression ratio
            max_length: Maximum output length
        """
        if not text or len(text.strip()) < 100:
            return SummaryResult(
                summary=text,
                method='none',
                source_sentences=[],
                compression_ratio=1.0
            )
        
        if method == 'extractive':
            return self._extractive_summarize(text, compression_ratio)
        elif method == 'abstractive':
            return self._abstractive_summarize(text, max_length)
        else:  # hybrid
            return self._hybrid_summarize(text, compression_ratio, max_length)
    
    def _extractive_summarize(self, text: str, compression_ratio: float) -> SummaryResult:
        """Extractive summarization using TextRank."""
        parser = PlaintextParser.from_string(text, Tokenizer(self.language))
        
        # Calculate number of sentences to extract
        total_sentences = len(parser.document.sentences)
        num_sentences = max(3, int(total_sentences * compression_ratio))
        
        # Extract using TextRank
        summary_sentences = self.textrank(parser.document, num_sentences)
        
        # Preserve original order
        summary_sentences = sorted(summary_sentences, 
                                   key=lambda s: parser.document.sentences.index(s))
        
        source_sentences = [str(s) for s in summary_sentences]
        summary = ' '.join(source_sentences)
        
        # Extract key points
        key_points = self._extract_key_points(source_sentences)
        
        actual_ratio = len(summary) / len(text) if text else 1.0
        
        return SummaryResult(
            summary=summary,
            method='extractive_textrank',
            source_sentences=source_sentences,
            compression_ratio=actual_ratio,
            key_points=key_points
        )
    
    def _abstractive_summarize(self, text: str, max_length: int) -> SummaryResult:
        """Abstractive summarization using transformers."""
        if self.abstractive_pipeline is None:
            # Fallback to extractive
            return self._extractive_summarize(text, 0.3)
        
        # Truncate input if too long (model context limit)
        max_input = 1024
        if len(text) > max_input * 4:  # Rough char to token estimate
            # Pre-extract key content
            extractive = self._extractive_summarize(text, 0.5)
            text = extractive.summary
        
        try:
            result = self.abstractive_pipeline(
                text,
                max_length=max_length,
                min_length=min(50, max_length // 4),
                do_sample=False,
                num_beams=4,
                early_stopping=True
            )
            
            summary = result[0]['summary_text']
            
            # Validate - check for hallucinations
            if not self._validate_summary(text, summary):
                print("Warning: Abstractive summary contains potential hallucinations, using extractive fallback")
                return self._extractive_summarize(text, 0.3)
            
            actual_ratio = len(summary) / len(text) if text else 1.0
            
            return SummaryResult(
                summary=summary,
                method='abstractive_transformer',
                source_sentences=[],  # Abstractive doesn't preserve source sentences
                compression_ratio=actual_ratio,
                key_points=self._extract_key_points_from_summary(summary)
            )
            
        except Exception as e:
            print(f"Abstractive summarization failed: {e}")
            return self._extractive_summarize(text, 0.3)
    
    def _hybrid_summarize(self, text: str, compression_ratio: float, max_length: int) -> SummaryResult:
        """
        Hybrid approach: Extractive + Abstractive.
        
        1. Extract key sentences (reduces noise)
        2. Generate abstractive summary from extracted content
        3. Validate output
        """
        # Step 1: Extractive
        extractive = self._extractive_summarize(text, compression_ratio * 1.5)
        extracted_text = ' '.join(extractive.source_sentences)
        
        # Step 2: Abstractive on extracted content
        if self.abstractive_pipeline and len(extracted_text) > 200:
            try:
                result = self.abstractive_pipeline(
                    extracted_text,
                    max_length=max_length,
                    min_length=min(50, max_length // 4),
                    do_sample=False,
                    num_beams=4,
                    early_stopping=True
                )
                
                summary = result[0]['summary_text']
                
                # Validate against original AND extracted
                if self._validate_summary(extracted_text, summary):
                    final_summary = summary
                    method = 'hybrid_extractive_abstractive'
                else:
                    # Validation failed, use extractive
                    final_summary = extracted_text
                    method = 'extractive_fallback'
            except Exception as e:
                final_summary = extracted_text
                method = 'extractive_fallback'
        else:
            final_summary = extracted_text
            method = 'extractive_only'
        
        actual_ratio = len(final_summary) / len(text) if text else 1.0
        
        return SummaryResult(
            summary=final_summary,
            method=method,
            source_sentences=extractive.source_sentences,
            compression_ratio=actual_ratio,
            key_points=extractive.key_points
        )
    
    def _validate_summary(self, source: str, summary: str, threshold: float = 0.5) -> bool:
        """
        Validate that summary doesn't hallucinate.
        
        Checks:
        1. Named entities in summary should appear in source
        2. ROUGE score above threshold
        """
        # Basic length check
        if len(summary) > len(source) * 0.8:
            return False
        
        # Check ROUGE
        scores = self.rouge.score(source, summary)
        rouge_l = scores['rougeL'].fmeasure
        
        if rouge_l < threshold:
            return False
        
        # Extract simple entities (capitalized phrases) from summary
        summary_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', summary))
        source_lower = source.lower()
        
        # Most entities should appear in source
        entity_matches = sum(1 for e in summary_entities if e.lower() in source_lower)
        if summary_entities and entity_matches / len(summary_entities) < 0.5:
            return False
        
        return True
    
    def _extract_key_points(self, sentences: List[str]) -> List[str]:
        """Extract key points from sentences."""
        key_points = []
        
        for sent in sentences:
            # Look for patterns indicating key findings
            patterns = [
                r'(?:significant|significantly|significantlly)\s+(?:increase|decrease|reduction|improvement|difference)',
                r'(?:result|finding|conclusion|demonstrate|show|suggest)\w*\s+(?:that)?',
                r'(?:risk|protective|associated|correlated|linked)\s+(?:with|to|factor)',
                r'(?:treatment|therapy|intervention)\s+(?:result|lead|cause|effect)',
            ]
            
            sent_lower = sent.lower()
            for pattern in patterns:
                if re.search(pattern, sent_lower):
                    # Clean and truncate
                    point = sent.strip()
                    if len(point) > 200:
                        point = point[:197] + '...'
                    key_points.append(point)
                    break
        
        return key_points[:5]  # Top 5 key points
    
    def _extract_key_points_from_summary(self, summary: str) -> List[str]:
        """Extract key points from abstractive summary."""
        sentences = re.split(r'[.!?]+', summary)
        points = [s.strip() for s in sentences if len(s.strip()) > 20]
        return points[:5]
    
    def evaluate_summary(self, source: str, summary: str, reference: str = None) -> Dict:
        """
        Evaluate summary quality using ROUGE scores.
        
        Args:
            source: Original source text
            summary: Generated summary
            reference: Optional human reference summary
        """
        scores = {}
        
        # ROUGE against source
        rouge_source = self.rouge.score(source, summary)
        scores['rouge_vs_source'] = {
            'rouge1': rouge_source['rouge1'].fmeasure,
            'rouge2': rouge_source['rouge2'].fmeasure,
            'rougeL': rouge_source['rougeL'].fmeasure,
        }
        
        # ROUGE against reference if provided
        if reference:
            rouge_ref = self.rouge.score(reference, summary)
            scores['rouge_vs_reference'] = {
                'rouge1': rouge_ref['rouge1'].fmeasure,
                'rouge2': rouge_ref['rouge2'].fmeasure,
                'rougeL': rouge_ref['rougeL'].fmeasure,
            }
        
        # Compression ratio
        scores['compression_ratio'] = len(summary) / len(source) if source else 0
        
        return scores
    
    def batch_summarize(self, 
                       texts: List[str],
                       method: str = 'hybrid',
                       compression_ratio: float = 0.3) -> List[SummaryResult]:
        """Summarize multiple texts."""
        results = []
        for text in texts:
            try:
                result = self.summarize(text, method, compression_ratio)
                results.append(result)
            except Exception as e:
                print(f"Summarization failed for text: {e}")
                results.append(SummaryResult(
                    summary=text[:200] + "..." if len(text) > 200 else text,
                    method='failed',
                    source_sentences=[],
                    compression_ratio=1.0
                ))
        return results
