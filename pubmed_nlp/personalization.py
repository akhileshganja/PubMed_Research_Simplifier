"""Personalization Layer: Adapt output based on user type (Patient/Student/Doctor)."""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import re


class UserType(Enum):
    """User types with different information needs."""
    PATIENT = "patient"
    STUDENT = "student"
    DOCTOR = "doctor"


@dataclass
class PersonalizedOutput:
    """Personalized output for a specific user type."""
    summary: str
    key_points: List[str]
    technical_level: str
    warnings: List[str]
    recommended_actions: List[str]
    references: List[str]


class PersonalizationEngine:
    """
    Personalization engine that adapts medical information to user needs.
    
    Strategies by user type:
    - PATIENT: Simple language, focus on actionable info, prominent warnings
    - STUDENT: Moderate detail, educational context, key concepts highlighted
    - DOCTOR: Full technical detail, focus on evidence, statistics preserved
    """
    
    # Simplification mappings
    MEDICAL_TERMS = {
        'hypertension': 'high blood pressure',
        'myocardial infarction': 'heart attack',
        'cerebrovascular accident': 'stroke',
        'malignancy': 'cancer',
        'neoplasm': 'tumor/cancer',
        'anticoagulant': 'blood thinner',
        'antihypertensive': 'blood pressure medication',
        'analgesic': 'pain reliever',
        'antiemetic': 'anti-nausea medication',
        'bronchodilator': 'asthma medication',
        'hyperlipidemia': 'high cholesterol',
        'diabetes mellitus': 'diabetes',
        'gastroesophageal reflux': 'acid reflux/heartburn',
        'upper respiratory infection': 'cold/flu',
        'pneumonia': 'lung infection',
        'pyrexia': 'fever',
        'dyspnea': 'shortness of breath',
        'pruritus': 'itching',
        'edema': 'swelling',
        'fatigue': 'tiredness',
        'nausea': 'feeling sick to stomach',
        'efficacy': 'effectiveness',
        'adverse effects': 'side effects',
        'contraindication': 'reason not to use',
        'placebo-controlled': 'compared to inactive treatment',
        'double-blind': 'neither patients nor doctors knew treatment',
        'randomized': 'randomly assigned',
        'meta-analysis': 'combined analysis of multiple studies',
        'cohort study': 'study following a group over time',
        'cross-sectional': 'snapshot study at one point in time',
        'retrospective': 'looking back at past records',
        'prospective': 'following forward over time',
    }
    
    # Templates for different user types
    TEMPLATES = {
        UserType.PATIENT: {
            'intro': "Here's what you need to know:",
            'evidence_prefix': "This is based on",
            'warning_prefix': "⚠️ Important:",
            'action_prefix': "What this means for you:",
            'reference_intro': "For more information, see these studies:",
        },
        UserType.STUDENT: {
            'intro': "Key learning points:",
            'evidence_prefix': "Evidence base:",
            'warning_prefix': "⚠️ Clinical considerations:",
            'action_prefix': "Clinical implications:",
            'reference_intro': "Primary literature:",
        },
        UserType.DOCTOR: {
            'intro': "Clinical Summary:",
            'evidence_prefix': "Evidence:",
            'warning_prefix': "⚠️ Cautions:",
            'action_prefix': "Clinical recommendations:",
            'reference_intro': "References:",
        }
    }
    
    def __init__(self):
        """Initialize personalization engine."""
        pass
    
    def personalize(self,
                   summary: str,
                   key_points: List[str],
                   entities: List[Dict],
                   relations: List[Dict],
                   evidence_level: str,
                   user_type: UserType = UserType.PATIENT) -> PersonalizedOutput:
        """
        Personalize content for specific user type.
        
        Args:
            summary: Original summary text
            key_points: Key points from analysis
            entities: Extracted entities
            relations: Extracted relations
            evidence_level: 'high', 'moderate', or 'low'
            user_type: Target user type
            
        Returns:
            PersonalizedOutput tailored to user type
        """
        templates = self.TEMPLATES[user_type]
        
        # Process summary based on user type
        if user_type == UserType.PATIENT:
            processed_summary = self._simplify_for_patient(summary)
            processed_key_points = [self._simplify_for_patient(kp) for kp in key_points]
            tech_level = 'simple'
        elif user_type == UserType.STUDENT:
            processed_summary = self._simplify_for_student(summary)
            processed_key_points = [self._simplify_for_student(kp) for kp in key_points]
            tech_level = 'moderate'
        else:  # DOCTOR
            processed_summary = summary
            processed_key_points = key_points
            tech_level = 'technical'
        
        # Generate warnings
        warnings = self._generate_warnings(relations, evidence_level, user_type)
        
        # Generate recommended actions
        actions = self._generate_actions(relations, entities, user_type)
        
        # Format references (PMIDs)
        references = self._format_references(relations, user_type)
        
        # Apply templates
        final_summary = self._apply_template(
            processed_summary, templates['intro'], user_type
        )
        
        return PersonalizedOutput(
            summary=final_summary,
            key_points=processed_key_points[:5],  # Top 5
            technical_level=tech_level,
            warnings=warnings,
            recommended_actions=actions,
            references=references
        )
    
    def _simplify_for_patient(self, text: str) -> str:
        """Simplify medical text for patient understanding."""
        simplified = text
        
        # Replace medical terms with simpler equivalents
        for term, simple in self.MEDICAL_TERMS.items():
            pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            simplified = pattern.sub(simple, simplified)
        
        # Remove or simplify statistical jargon
        simplified = re.sub(r'p\s*[<>=]\s*0\.\d+', '', simplified)
        simplified = re.sub(r'\b(?:95%\s+CI|confidence interval|CI)\b', '', simplified, flags=re.IGNORECASE)
        simplified = re.sub(r'\bOR\s*=\s*\d+\.?\d*', 'risk ratio mentioned', simplified)
        simplified = re.sub(r'\bHR\s*=\s*\d+\.?\d*', 'risk over time mentioned', simplified)
        simplified = re.sub(r'\bRR\s*=\s*\d+\.?\d*', 'risk ratio mentioned', simplified)
        
        # Simplify numbers
        simplified = re.sub(r'\b\d+\.\d+%\b', lambda m: f"about {round(float(m.group().rstrip('%')))}%", simplified)
        
        # Clean up extra spaces
        simplified = re.sub(r'\s+', ' ', simplified).strip()
        
        return simplified
    
    def _simplify_for_student(self, text: str) -> str:
        """Simplify for students - keep educational value but clearer."""
        simplified = text
        
        # Replace some medical terms but keep key terminology
        basic_terms = {
            'myocardial infarction': 'myocardial infarction (heart attack)',
            'cerebrovascular accident': 'cerebrovascular accident (stroke)',
            'malignancy': 'malignancy (cancer)',
            'hyperlipidemia': 'hyperlipidemia (high cholesterol)',
            'anticoagulant': 'anticoagulant (blood thinner)',
            'efficacy': 'efficacy (effectiveness)',
            'adverse effects': 'adverse effects (side effects)',
        }
        
        for term, explanation in basic_terms.items():
            pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            simplified = pattern.sub(explanation, simplified)
        
        # Keep stats but format better
        simplified = re.sub(r'p\s*(<)\s*(0\.\d+)', r'p\1\2 (statistically significant)', simplified)
        simplified = re.sub(r'p\s*(>)\s*(0\.\d+)', r'p\1\2 (not statistically significant)', simplified)
        
        return simplified
    
    def _generate_warnings(self, 
                          relations: List[Dict], 
                          evidence_level: str,
                          user_type: UserType) -> List[str]:
        """Generate appropriate warnings based on relations and evidence."""
        warnings = []
        
        if evidence_level == 'low':
            if user_type == UserType.PATIENT:
                warnings.append("This information is based on limited research. Please consult your doctor before making health decisions.")
            elif user_type == UserType.STUDENT:
                warnings.append("Limited evidence quality - findings should be interpreted cautiously.")
            else:
                warnings.append("Limited evidence base - findings require further validation.")
        
        # Check for side effects or risks
        for rel in relations:
            if rel.get('relation_type') == 'causes':
                obj = rel.get('object', '')
                subj = rel.get('subject', '')
                
                if any(term in obj.lower() for term in ['side effect', 'adverse', 'toxicity', 'risk']):
                    if user_type == UserType.PATIENT:
                        warnings.append(f"{subj} may cause: {obj}")
                    else:
                        warnings.append(f"Known adverse effect: {subj} → {obj}")
        
        return warnings
    
    def _generate_actions(self,
                         relations: List[Dict],
                         entities: List[Dict],
                         user_type: UserType) -> List[str]:
        """Generate recommended actions based on analysis."""
        actions = []
        
        if user_type == UserType.PATIENT:
            # Focus on actionable patient information
            for rel in relations:
                if rel.get('relation_type') == 'treats':
                    drug = rel.get('subject', '')
                    condition = rel.get('object', '')
                    actions.append(f"Talk to your doctor about {drug} if you have {condition}")
                    break  # One is enough
            
            if not actions:
                actions.append("Discuss these findings with your healthcare provider")
            
            actions.append("Always consult a doctor before starting or stopping any treatment")
        
        elif user_type == UserType.STUDENT:
            actions.append("Review primary literature for detailed methodology")
            actions.append("Consider limitations and confounding factors")
            
            # Add study design suggestions
            study_types = set()
            for ent in entities:
                if ent.get('label', '').lower() in ['study', 'trial']:
                    study_types.add(ent.get('text', ''))
            
            if study_types:
                actions.append(f"Key study types to understand: {', '.join(list(study_types)[:3])}")
        
        else:  # DOCTOR
            actions.append("Evaluate applicability to your patient population")
            actions.append("Consider individual patient factors and contraindications")
            
            # Add evidence-based recommendations
            for rel in relations:
                if rel.get('relation_type') == 'treats':
                    drug = rel.get('subject', '')
                    condition = rel.get('object', '')
                    actions.append(f"Evidence supports {drug} for {condition}")
                    break
        
        return actions
    
    def _format_references(self, relations: List[Dict], user_type: UserType) -> List[str]:
        """Format reference list based on user type."""
        # Extract PMIDs from relations
        pmids = set()
        for rel in relations:
            pmid = rel.get('pmid')
            if pmid:
                pmids.add(pmid)
        
        if not pmids:
            return []
        
        refs = list(pmids)[:10]  # Limit to 10
        
        if user_type == UserType.PATIENT:
            # Simple links
            return [f"Study {pmid}: https://pubmed.ncbi.nlm.nih.gov/{pmid}/" for pmid in refs]
        elif user_type == UserType.STUDENT:
            return [f"PMID: {pmid} - https://pubmed.ncbi.nlm.nih.gov/{pmid}/" for pmid in refs]
        else:
            return [f"PMID: {pmid}" for pmid in refs]
    
    def _apply_template(self, content: str, template_prefix: str, user_type: UserType) -> str:
        """Apply template formatting to content."""
        return f"{template_prefix}\n\n{content}"
    
    def to_dict(self, output: PersonalizedOutput) -> Dict:
        """Convert PersonalizedOutput to dictionary."""
        return {
            'summary': output.summary,
            'key_points': output.key_points,
            'technical_level': output.technical_level,
            'warnings': output.warnings,
            'recommended_actions': output.recommended_actions,
            'references': output.references
        }
