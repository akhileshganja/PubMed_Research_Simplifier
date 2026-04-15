"""Biomedical text preprocessing pipeline using SciSpacy."""

import re
import string
from typing import List, Dict, Optional
from dataclasses import dataclass
import spacy
import scispacy
from scispacy.linking import EntityLinker
import nltk
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


@dataclass
class PreprocessedText:
    """Preprocessed text with all NLP annotations."""
    original_text: str
    cleaned_text: str
    tokens: List[str]
    sentences: List[str]
    lemmas: List[str]
    entities: List[Dict]  # Named entities
    dependencies: List[Dict]  # Dependency parse
    pos_tags: List[tuple]  # Part-of-speech tags


class BiomedicalPreprocessor:
    """Domain-specific text preprocessor for biomedical literature."""
    
    # Biomedical stopwords to keep (domain-specific terms)
    KEEP_TERMS = {
        'no', 'not', 'none', 'neither', 'never', 'without', 'against',
        'in', 'at', 'on', 'over', 'under', 'between', 'through', 'during',
        'mg', 'ml', 'kg', 'mm', 'cm', 'percent', '%'
    }
    
    def __init__(self, 
                 scispacy_model: str = "en_core_sci_lg",
                 enable_linker: bool = False):
        """
        Initialize preprocessor with SciSpacy models.
        
        Args:
            scispacy_model: SciSpacy model name
            enable_linker: Enable UMLS linker (slower but adds entity normalization)
        """
        print(f"Loading SciSpacy model: {scispacy_model}...")
        self.nlp = spacy.load(scispacy_model)
        
        if enable_linker:
            print("Enabling UMLS entity linker...")
            self.nlp.add_pipe("scispacy_linker", 
                             config={"resolve_abbreviations": True,
                                    "linker_name": "umls"})
        
        # Get standard stopwords but keep biomedical terms
        self.stopwords = set(stopwords.words('english')) - self.KEEP_TERMS
        
        # Add custom biomedical stopwords
        self.stopwords.update(['et', 'al', 'fig', 'table', 'supplementary', 
                              'appendix', 'doi', 'pmid', 'copyright'])
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize biomedical text.
        
        Steps:
        1. Lowercase
        2. Fix encoding issues
        3. Remove extra whitespace
        4. Normalize punctuation
        5. Handle special biomedical characters
        """
        if not text:
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove DOI patterns
        text = re.sub(r'10\.\d{4,}/\S+', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Handle hyphenated words (common in biomedical)
        text = re.sub(r'(\w+)-(\w+)', r'\1_\2', text)
        
        # Normalize numbers (keep as tokens)
        text = re.sub(r'\b\d+\.?\d*\b', ' NUM ', text)
        
        return text.strip()
    
    def segment_sentences(self, doc) -> List[str]:
        """Extract sentences from spacy doc."""
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords while preserving domain terms."""
        return [t for t in tokens if t.lower() not in self.stopwords and len(t) > 1]
    
    def lemmatize(self, doc) -> List[str]:
        """Extract lemmas, preserving biomedical entities."""
        return [token.lemma_ for token in doc if not token.is_space]
    
    def extract_entities(self, doc) -> List[Dict]:
        """Extract named entities with positions and labels."""
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'start_token': ent.start,
                'end_token': ent.end
            })
        return entities
    
    def extract_dependencies(self, doc) -> List[Dict]:
        """Extract dependency parse relationships."""
        deps = []
        for token in doc:
            deps.append({
                'token': token.text,
                'lemma': token.lemma_,
                'pos': token.pos_,
                'dep': token.dep_,
                'head': token.head.text,
                'head_pos': token.head.pos_,
                'children': [child.text for child in token.children]
            })
        return deps
    
    def preprocess(self, text: str, 
                   remove_stopwords: bool = True,
                   extract_deps: bool = False) -> PreprocessedText:
        """
        Full preprocessing pipeline.
        
        Args:
            text: Raw input text
            remove_stopwords: Whether to filter stopwords
            extract_deps: Whether to extract dependency parse
            
        Returns:
            PreprocessedText with all annotations
        """
        # Clean text
        cleaned = self.clean_text(text)
        
        # Process with SciSpacy
        doc = self.nlp(cleaned)
        
        # Extract components
        sentences = self.segment_sentences(doc)
        tokens = [token.text for token in doc if not token.is_space]
        lemmas = self.lemmatize(doc)
        entities = self.extract_entities(doc)
        pos_tags = [(token.text, token.pos_, token.tag_) for token in doc]
        
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        dependencies = self.extract_dependencies(doc) if extract_deps else []
        
        return PreprocessedText(
            original_text=text,
            cleaned_text=cleaned,
            tokens=tokens,
            sentences=sentences,
            lemmas=lemmas,
            entities=entities,
            dependencies=dependencies,
            pos_tags=pos_tags
        )
    
    def preprocess_batch(self, texts: List[str], 
                        batch_size: int = 32,
                        **kwargs) -> List[PreprocessedText]:
        """Process multiple texts efficiently using pipe."""
        results = []
        for doc, text in zip(self.nlp.pipe([self.clean_text(t) for t in texts], 
                                          batch_size=batch_size), texts):
            tokens = [token.text for token in doc if not token.is_space]
            if kwargs.get('remove_stopwords', True):
                tokens = self.remove_stopwords(tokens)
            
            results.append(PreprocessedText(
                original_text=text,
                cleaned_text=doc.text,
                tokens=tokens,
                sentences=[sent.text.strip() for sent in doc.sents],
                lemmas=[token.lemma_ for token in doc],
                entities=self.extract_entities(doc),
                dependencies=self.extract_dependencies(doc) if kwargs.get('extract_deps') else [],
                pos_tags=[(t.text, t.pos_, t.tag_) for t in doc]
            ))
        return results
