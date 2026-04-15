"""Biomedical Named Entity Recognition using specialized SciSpacy models."""

import spacy
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import warnings


@dataclass
class BiomedicalEntity:
    """Structured biomedical entity."""
    text: str
    label: str
    start: int
    end: int
    confidence: Optional[float] = None
    normalized_id: Optional[str] = None  # UMLS CUI or similar
    source: str = "scispacy"  # Which model detected this


@dataclass
class NERDocument:
    """Document with all extracted entities."""
    text: str
    entities: List[BiomedicalEntity] = field(default_factory=list)
    
    def get_by_type(self, entity_type: str) -> List[BiomedicalEntity]:
        """Filter entities by type."""
        return [e for e in self.entities if e.label == entity_type]
    
    def get_diseases(self) -> List[BiomedicalEntity]:
        """Get all disease entities."""
        disease_labels = {'DISEASE', 'CANCER', 'PATHOLOGY', 'CONDITION'}
        return [e for e in self.entities if any(d in e.label.upper() for d in disease_labels)]
    
    def get_chemicals_drugs(self) -> List[BiomedicalEntity]:
        """Get all chemical/drug entities."""
        chem_labels = {'CHEMICAL', 'DRUG', 'PHARMACOLOGIC', 'SUBSTANCE'}
        return [e for e in self.entities if any(c in e.label.upper() for c in chem_labels)]
    
    def get_genes_proteins(self) -> List[BiomedicalEntity]:
        """Get all gene/protein entities."""
        gene_labels = {'GENE', 'PROTEIN', 'DNA', 'RNA', 'CELL_TYPE', 'CELL_LINE'}
        return [e for e in self.entities if any(g in e.label.upper() for g in gene_labels)]
    
    def get_symptoms(self) -> List[BiomedicalEntity]:
        """Get symptom-related entities."""
        # Symptoms often have disease-like labels but context-specific
        return [e for e in self.entities if 'SYMPTOM' in e.label.upper()]


class BiomedicalNER:
    """
    Multi-model biomedical NER system.
    Combines multiple SciSpacy models for comprehensive entity detection.
    """
    
    # Model mappings
    MODELS = {
        'bc5cdr': 'en_ner_bc5cdr_md',  # Chemicals & Diseases
        'jnlpba': 'en_ner_jnlpba_md',  # Genes, Proteins, Cell types
        'craft': 'en_ner_craft_md',    # Concepts (optional)
    }
    
    # Entity type mappings
    ENTITY_CATEGORIES = {
        'DISEASE': 'disease',
        'CHEMICAL': 'chemical_drug',
        'DRUG': 'chemical_drug',
        'GENE': 'gene_protein',
        'PROTEIN': 'gene_protein',
        'DNA': 'gene_protein',
        'RNA': 'gene_protein',
        'CELL_TYPE': 'cell',
        'CELL_LINE': 'cell',
        'SPECIES': 'organism',
        'TAXON': 'organism',
        'ANATOMY': 'anatomy',
        'ORGAN': 'anatomy',
        'TISSUE': 'anatomy',
        'SYMPTOM': 'symptom',
        'PROCEDURE': 'procedure',
    }
    
    def __init__(self, 
                 models: Optional[List[str]] = None,
                 resolve_overlaps: bool = True):
        """
        Initialize NER pipeline.
        
        Args:
            models: List of model names to load (default: ['bc5cdr', 'jnlpba'])
            resolve_overlaps: Whether to merge overlapping entity spans
        """
        self.models = models or ['bc5cdr', 'jnlpba']
        self.resolve_overlaps = resolve_overlaps
        self._nlp_pipelines = {}
        
        self._load_models()
    
    def _load_models(self):
        """Load all specified SciSpacy models."""
        for model_key in self.models:
            model_name = self.MODELS.get(model_key, model_key)
            try:
                print(f"Loading NER model: {model_name}...")
                self._nlp_pipelines[model_key] = spacy.load(model_name)
            except OSError as e:
                warnings.warn(f"Could not load model {model_name}: {e}")
                print(f"Install with: pip install {model_name}")
    
    def extract_entities(self, text: str) -> NERDocument:
        """
        Extract all biomedical entities from text using all loaded models.
        
        Args:
            text: Input text
            
        Returns:
            NERDocument with all extracted entities
        """
        all_entities = []
        
        for model_name, nlp in self._nlp_pipelines.items():
            doc = nlp(text)
            
            for ent in doc.ents:
                entity = BiomedicalEntity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    source=model_name
                )
                all_entities.append(entity)
        
        # Resolve overlaps if enabled
        if self.resolve_overlaps:
            all_entities = self._resolve_entity_overlaps(all_entities)
        
        return NERDocument(text=text, entities=all_entities)
    
    def _resolve_entity_overlaps(self, entities: List[BiomedicalEntity]) -> List[BiomedicalEntity]:
        """
        Resolve overlapping entity spans using longest match strategy.
        
        When entities overlap, keep the longer one (more specific).
        """
        if not entities:
            return entities
        
        # Sort by start position, then by length (descending)
        sorted_entities = sorted(entities, key=lambda e: (e.start, -(e.end - e.start)))
        
        resolved = []
        last_end = -1
        
        for ent in sorted_entities:
            if ent.start >= last_end:
                resolved.append(ent)
                last_end = ent.end
            # If overlapping and longer, replace the previous one
            elif ent.end > last_end and resolved:
                # Check if this entity extends further
                if ent.end > resolved[-1].end:
                    resolved[-1] = ent
                    last_end = ent.end
        
        return resolved
    
    def extract_relations(self, text: str, entities: List[BiomedicalEntity]) -> List[Dict]:
        """
        Simple relation extraction based on dependency parsing.
        
        This is a baseline approach. For advanced relation extraction,
        use specialized models like BioBERT or transformer-based RE.
        """
        if not self._nlp_pipelines:
            return []
        
        # Use the first available model for dependency parsing
        nlp = list(self._nlp_pipelines.values())[0]
        doc = nlp(text)
        
        relations = []
        
        # Map entity positions to tokens
        entity_spans = {(e.start, e.end): e for e in entities}
        
        # Find subject-verb-object patterns
        for token in doc:
            if token.dep_ in ('nsubj', 'nsubjpass') and token.head.pos_ == 'VERB':
                # Find subject entity
                subj_ent = self._find_entity_at_span(token.idx, token.idx + len(token.text), 
                                                     entity_spans, entities)
                
                # Find object (dobj, pobj)
                for child in token.head.children:
                    if child.dep_ in ('dobj', 'pobj', 'attr'):
                        obj_ent = self._find_entity_at_span(child.idx, 
                                                            child.idx + len(child.text),
                                                            entity_spans, entities)
                        
                        if subj_ent and obj_ent:
                            relations.append({
                                'subject': subj_ent.text,
                                'subject_type': subj_ent.label,
                                'predicate': token.head.lemma_,
                                'object': obj_ent.text,
                                'object_type': obj_ent.label,
                                'relation_type': self._classify_relation(subj_ent, obj_ent, token.head.lemma_),
                                'confidence': 0.7  # Placeholder
                            })
        
        return relations
    
    def _find_entity_at_span(self, start: int, end: int, 
                            entity_spans: Dict, entities: List[BiomedicalEntity]) -> Optional[BiomedicalEntity]:
        """Find entity that contains or overlaps with given span."""
        for (es, ee), ent in entity_spans.items():
            if es <= start < ee or es < end <= ee:
                return ent
        return None
    
    def _classify_relation(self, subj: BiomedicalEntity, obj: BiomedicalEntity, 
                          predicate: str) -> str:
        """Classify the type of relation."""
        subj_cat = self.ENTITY_CATEGORIES.get(subj.label, 'unknown')
        obj_cat = self.ENTITY_CATEGORIES.get(obj.label, 'unknown')
        
        # Common biomedical relations
        if subj_cat == 'chemical_drug' and obj_cat == 'disease':
            if any(w in predicate for w in ['treat', 'therap', 'administer', 'give', 'prescribe']):
                return 'treats'
            elif any(w in predicate for w in ['cause', 'induce', 'trigger']):
                return 'causes'
            elif any(w in predicate for w in ['prevent', 'protect', 'reduce']):
                return 'prevents'
        
        if subj_cat == 'disease' and obj_cat == 'symptom':
            return 'presents_with'
        
        if subj_cat == 'gene_protein' and obj_cat == 'disease':
            if any(w in predicate for w in ['cause', 'induce', 'lead', 'result']):
                return 'causes'
            elif any(w in predicate for w in ['associate', 'link', 'relate', 'correlate']):
                return 'associated_with'
        
        return 'related_to'
    
    def batch_extract(self, texts: List[str], batch_size: int = 32) -> List[NERDocument]:
        """Extract entities from multiple texts."""
        results = []
        for text in texts:
            results.append(self.extract_entities(text))
        return results
    
    def get_entity_statistics(self, documents: List[NERDocument]) -> Dict:
        """Get statistics about entities across documents."""
        stats = defaultdict(lambda: defaultdict(int))
        
        for doc in documents:
            for ent in doc.entities:
                category = self.ENTITY_CATEGORIES.get(ent.label, 'other')
                stats[category][ent.text.lower()] += 1
        
        return dict(stats)
