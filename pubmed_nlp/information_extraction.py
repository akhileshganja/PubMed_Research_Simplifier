"""Advanced Information Extraction: Keyphrases, Relations, and Topic Modeling."""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter, defaultdict
import re

# Keyphrase extraction
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer

# Topic Modeling
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN

# Sentence transformers for embeddings
from sentence_transformers import SentenceTransformer

# TextRank
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer


@dataclass
class ExtractedKeyphrase:
    """Keyphrase with score and metadata."""
    text: str
    score: float
    method: str  # 'keybert', 'tfidf', 'textrank'


@dataclass
class ExtractedRelation:
    """Semantic relation between entities."""
    subject: str
    subject_type: str
    predicate: str
    object: str
    object_type: str
    relation_type: str
    confidence: float
    evidence: str  # Sentence/context


@dataclass
class TopicInfo:
    """Topic model output."""
    topic_id: int
    topic_words: List[Tuple[str, float]]
    document_count: int
    representative_docs: List[str]


class InformationExtractor:
    """
    Comprehensive information extraction pipeline.
    Combines multiple techniques: KeyBERT, TF-IDF, Topic Modeling, Relations.
    """
    
    # Biomedical patterns for relation extraction
    RELATION_PATTERNS = {
        'treats': [
            r'(\w+(?:\s+\w+){0,3})\s+(?:treat|treatment|therapy|therapeutic|administer|prescribe|give)\w*\s+(?:for|in|to)?\s*(\w+(?:\s+\w+){0,3})',
            r'(\w+(?:\s+\w+){0,3})\s+(?:improve|reduce|decrease|alleviate|relieve)\w*\s+(?:in)?\s*(\w+(?:\s+\w+){0,3})',
        ],
        'causes': [
            r'(\w+(?:\s+\w+){0,3})\s+(?:cause|induce|trigger|lead\s+to|result\s+in)\w*\s+(\w+(?:\s+\w+){0,3})',
            r'(\w+(?:\s+\w+){0,3})\s+(?:increase|elevate)\w*\s+risk\s+(?:of)?\s*(\w+(?:\s+\w+){0,3})',
        ],
        'prevents': [
            r'(\w+(?:\s+\w+){0,3})\s+(?:prevent|protect|reduce\s+risk|lower\s+risk)\w*\s+(?:of|against)?\s*(\w+(?:\s+\w+){0,3})',
        ],
        'associated_with': [
            r'(\w+(?:\s+\w+){0,3})\s+(?:associate|link|correlate|relate)\w*\s+(?:with|to)\s+(\w+(?:\s+\w+){0,3})',
            r'(\w+(?:\s+\w+){0,3})\s+(?:risk\s+factor|marker|indicator)\s+(?:for|of)?\s*(\w+(?:\s+\w+){0,3})',
        ],
    }
    
    def __init__(self, 
                 embedding_model: str = 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb',
                 keyphrase_model: str = 'all-MiniLM-L6-v2',
                 device: str = None):
        """
        Initialize information extractor.
        
        Args:
            embedding_model: SentenceTransformer model for semantic embeddings
            keyphrase_model: Model for keyphrase extraction
            device: 'cuda' or 'cpu'
        """
        print(f"Loading embedding model: {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model, device=device)
        
        print(f"Loading keyphrase model: {keyphrase_model}...")
        self.keybert = KeyBERT(model=keyphrase_model)
        
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        
        self.textrank_summarizer = TextRankSummarizer()
        self._topic_model = None
        
    def extract_keyphrases(self, 
                          text: str, 
                          method: str = 'keybert',
                          top_n: int = 10,
                          keyphrase_ngram_range: Tuple[int, int] = (1, 2)) -> List[ExtractedKeyphrase]:
        """
        Extract keyphrases from text using specified method.
        
        Methods:
        - 'keybert': Neural keyphrase extraction (default, best quality)
        - 'tfidf': Statistical TF-IDF
        - 'textrank': Graph-based ranking
        - 'hybrid': Combine all methods
        """
        if method == 'keybert' or method == 'hybrid':
            keybert_kp = self._extract_keybert(text, top_n, keyphrase_ngram_range)
        else:
            keybert_kp = []
        
        if method == 'tfidf' or method == 'hybrid':
            tfidf_kp = self._extract_tfidf(text, top_n)
        else:
            tfidf_kp = []
        
        if method == 'textrank' or method == 'hybrid':
            textrank_kp = self._extract_textrank(text, top_n)
        else:
            textrank_kp = []
        
        if method == 'hybrid':
            # Merge and deduplicate
            all_kp = keybert_kp + tfidf_kp + textrank_kp
            seen = set()
            unique = []
            for kp in sorted(all_kp, key=lambda x: x.score, reverse=True):
                key = kp.text.lower()
                if key not in seen:
                    seen.add(key)
                    unique.append(kp)
            return unique[:top_n]
        
        if method == 'keybert':
            return keybert_kp
        elif method == 'tfidf':
            return tfidf_kp
        else:
            return textrank_kp
    
    def _extract_keybert(self, text: str, top_n: int, 
                        ngram_range: Tuple[int, int]) -> List[ExtractedKeyphrase]:
        """Extract keyphrases using KeyBERT."""
        keywords = self.keybert.extract_keywords(
            text,
            keyphrase_ngram_range=ngram_range,
            stop_words='english',
            top_n=top_n,
            use_maxsum=True,
            nr_candidates=20,
            diversity=0.5
        )
        
        return [ExtractedKeyphrase(text=k[0], score=k[1], method='keybert') 
                for k in keywords]
    
    def _extract_tfidf(self, text: str, top_n: int) -> List[ExtractedKeyphrase]:
        """Extract keyphrases using TF-IDF (requires corpus, single-doc fallback)."""
        # For single document, use term frequency with length penalty
        words = re.findall(r'\b[a-zA-Z][a-zA-Z\-]+\b', text.lower())
        
        # Filter short words and common stopwords
        stopwords = ['this', 'that', 'with', 'from', 'they', 'have', 'been']
        filtered = [w for w in words if len(w) > 3 and w not in stopwords]
        
        # Bigrams
        bigrams = [f"{filtered[i]} {filtered[i+1]}" for i in range(len(filtered)-1)]
        
        # Score by frequency and position (earlier = more important)
        scores = Counter(filtered + bigrams)
        
        # Normalize by max frequency
        if scores:
            max_score = max(scores.values())
            normalized = [(phrase, count/max_score) for phrase, count in scores.most_common(top_n)]
        else:
            normalized = []
        
        return [ExtractedKeyphrase(text=p, score=s, method='tfidf') 
                for p, s in normalized]
    
    def _extract_textrank(self, text: str, top_n: int) -> List[ExtractedKeyphrase]:
        """Extract keyphrases using TextRank."""
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        
        # Get top sentences as representative
        summary = self.textrank_summarizer(parser.document, min(top_n, 5))
        sentences = [str(s) for s in summary]
        
        # Extract noun phrases from summary sentences
        phrases = []
        for sent in sentences:
            # Simple noun phrase extraction
            matches = re.findall(r'\b(?:[A-Z][a-z]+\s*){1,3}\b|\b(?:[a-z]+\s+){1,2}[a-z]+\b', sent)
            phrases.extend([m.strip() for m in matches if len(m.strip()) > 5])
        
        # Score by frequency in top sentences
        scores = Counter(phrases)
        max_score = max(scores.values()) if scores else 1
        
        return [ExtractedKeyphrase(text=p, score=c/max_score, method='textrank') 
                for p, c in scores.most_common(top_n)]
    
    def extract_relations(self, text: str, entities: List[Dict]) -> List[ExtractedRelation]:
        """
        Extract semantic relations using pattern matching.
        
        This is a baseline approach. For production, consider:
        - BioBERT for relation classification
        - Custom transformer models
        """
        relations = []
        sentences = re.split(r'[.!?;]+', text)
        
        entity_texts = {e['text'].lower(): e for e in entities}
        
        for sent in sentences:
            sent_lower = sent.lower().strip()
            if not sent_lower:
                continue
            
            for rel_type, patterns in self.RELATION_PATTERNS.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, sent_lower)
                    for match in matches:
                        subj_text = match.group(1)
                        obj_text = match.group(2)
                        
                        # Find matching entities
                        subj_ent = None
                        obj_ent = None
                        
                        for ent_text, ent in entity_texts.items():
                            if ent_text in subj_text or subj_text in ent_text:
                                subj_ent = ent
                            if ent_text in obj_text or obj_text in ent_text:
                                obj_ent = ent
                        
                        if subj_ent and obj_ent and subj_ent['text'] != obj_ent['text']:
                            relations.append(ExtractedRelation(
                                subject=subj_ent['text'],
                                subject_type=subj_ent.get('label', 'unknown'),
                                predicate=self._get_predicate_from_pattern(rel_type, match.group(0)),
                                object=obj_ent['text'],
                                object_type=obj_ent.get('label', 'unknown'),
                                relation_type=rel_type,
                                confidence=0.6,  # Pattern-based confidence
                                evidence=sent.strip()
                            ))
        
        return relations
    
    def _get_predicate_from_pattern(self, rel_type: str, match_text: str) -> str:
        """Extract predicate verb from matched text."""
        verbs = {
            'treats': ['treat', 'treatment', 'therapy', 'administer', 'prescribe', 'improve', 'reduce'],
            'causes': ['cause', 'induce', 'trigger', 'lead', 'result'],
            'prevents': ['prevent', 'protect', 'reduce risk', 'lower risk'],
            'associated_with': ['associate', 'link', 'correlate', 'relate', 'risk factor'],
        }
        
        match_lower = match_text.lower()
        for verb in verbs.get(rel_type, []):
            if verb in match_lower:
                return verb
        return rel_type
    
    def fit_topic_model(self, 
                      documents: List[str],
                      min_topic_size: int = 5,
                      nr_topics: str = 'auto') -> List[TopicInfo]:
        """
        Fit BERTopic model on documents.
        
        Args:
            documents: List of texts
            min_topic_size: Minimum documents per topic
            nr_topics: Number of topics ('auto' or int)
        """
        print(f"Fitting topic model on {len(documents)} documents...")
        
        # Custom HDBSCAN for better control
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        # c-TF-IDF for better topic representations
        ctfidf_model = ClassTfidfTransformer()
        
        self._topic_model = BERTopic(
            embedding_model=self.embedding_model,
            hdbscan_model=hdbscan_model,
            ctfidf_model=ctfidf_model,
            nr_topics=nr_topics,
            verbose=True
        )
        
        topics, probs = self._topic_model.fit_transform(documents)
        
        # Extract topic info
        topic_info = self._topic_model.get_topic_info()
        results = []
        
        for _, row in topic_info.iterrows():
            if row['Topic'] == -1:  # Skip outliers
                continue
            
            topic_words = self._topic_model.get_topic(row['Topic'])
            if topic_words:
                results.append(TopicInfo(
                    topic_id=row['Topic'],
                    topic_words=[(w, float(s)) for w, s in topic_words[:10]],
                    document_count=row['Count'],
                    representative_docs=[]  # Could be added with c-TF-IDF
                ))
        
        return results
    
    def predict_topics(self, documents: List[str]) -> List[int]:
        """Predict topic assignments for new documents."""
        if self._topic_model is None:
            raise ValueError("Topic model not fitted. Call fit_topic_model first.")
        
        topics, _ = self._topic_model.transform(documents)
        return topics
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for texts."""
        return self.embedding_model.encode(texts, show_progress_bar=True)
    
    def find_similar_documents(self, 
                              query: str, 
                              documents: List[str],
                              top_k: int = 5) -> List[Tuple[int, str, float]]:
        """
        Find most similar documents to query using embeddings.
        
        Returns: List of (index, document, similarity_score)
        """
        # Encode
        query_embedding = self.embedding_model.encode([query])
        doc_embeddings = self.embedding_model.encode(documents)
        
        # Compute cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        # Get top k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(int(i), documents[i], float(similarities[i])) for i in top_indices]
