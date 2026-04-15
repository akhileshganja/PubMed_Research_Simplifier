"""RAG-based QA System for PubMed research."""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb


@dataclass
class RetrievedContext:
    """Retrieved context for RAG."""
    text: str
    source: str  # PMID
    title: str
    relevance_score: float
    metadata: Dict


@dataclass
class RAGAnswer:
    """RAG-generated answer."""
    answer: str
    sources: List[RetrievedContext]
    confidence: float
    method: str


class RAGSystem:
    """
    Retrieval-Augmented Generation system for PubMed QA.
    
    Pipeline:
    1. Retrieve relevant abstracts using vector similarity
    2. Rank by relevance
    3. Generate answer based on retrieved context
    """
    
    def __init__(self,
                 embedding_model: str = 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb',
                 persist_dir: str = './chroma_db',
                 device: str = None):
        """
        Initialize RAG system.
        
        Args:
            embedding_model: Model for embedding queries and documents
            persist_dir: Directory for ChromaDB persistence
            device: 'cuda' or 'cpu'
        """
        print(f"Loading embedding model: {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model, device=device)
        
        # Initialize ChromaDB
        print(f"Initializing ChromaDB at: {persist_dir}...")
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="pubmed_abstracts",
            metadata={"hnsw:space": "cosine"}
        )
        
        self._initialized = True
    
    def add_documents(self,
                     texts: List[str],
                     ids: List[str],
                     metadatas: List[Dict] = None,
                     batch_size: int = 100) -> None:
        """
        Add documents to the vector store.
        
        Args:
            texts: List of document texts (abstracts)
            ids: List of document IDs (PMIDs)
            metadatas: Optional metadata for each document
            batch_size: Batch size for embedding
        """
        print(f"Adding {len(texts)} documents to vector store...")
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size] if metadatas else None
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(batch_texts, show_progress_bar=False)
            embeddings = embeddings.tolist()
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=batch_texts,
                ids=batch_ids,
                metadatas=batch_metadatas
            )
        
        print(f"Added {len(texts)} documents successfully")
    
    def retrieve(self,
                query: str,
                top_k: int = 5,
                filter_dict: Dict = None) -> List[RetrievedContext]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            filter_dict: Optional filter (e.g., {'year': {'$gte': 2020}})
            
        Returns:
            List of RetrievedContext
        """
        # Encode query
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Query collection
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=filter_dict
        )
        
        contexts = []
        
        for i in range(len(results['ids'][0])):
            doc_id = results['ids'][0][i]
            doc_text = results['documents'][0][i]
            distance = results['distances'][0][i]
            metadata = results['metadatas'][0][i] if results['metadatas'] else {}
            
            # Convert distance to similarity (cosine distance -> similarity)
            similarity = 1 - distance
            
            contexts.append(RetrievedContext(
                text=doc_text,
                source=doc_id,
                title=metadata.get('title', ''),
                relevance_score=similarity,
                metadata=metadata
            ))
        
        return contexts
    
    def answer(self,
              question: str,
              top_k: int = 5,
              filter_dict: Dict = None,
              generate_answer: bool = True) -> RAGAnswer:
        """
        Answer a question using RAG.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            filter_dict: Optional filters
            generate_answer: Whether to generate an answer or just return context
            
        Returns:
            RAGAnswer with answer and sources
        """
        # Retrieve relevant documents
        contexts = self.retrieve(question, top_k, filter_dict)
        
        if not contexts:
            return RAGAnswer(
                answer="No relevant documents found for this question.",
                sources=[],
                confidence=0.0,
                method='retrieval_failed'
            )
        
        if not generate_answer:
            # Just return the most relevant context as answer
            best = contexts[0]
            return RAGAnswer(
                answer=best.text,
                sources=contexts,
                confidence=best.relevance_score,
                method='direct_retrieval'
            )
        
        # Generate answer from context
        answer_text = self._generate_answer(question, contexts)
        
        # Calculate confidence based on retrieval scores
        avg_score = np.mean([c.relevance_score for c in contexts])
        confidence = min(avg_score * 1.2, 1.0)  # Scale up slightly
        
        return RAGAnswer(
            answer=answer_text,
            sources=contexts,
            confidence=confidence,
            method='rag_generated'
        )
    
    def _generate_answer(self, question: str, contexts: List[RetrievedContext]) -> str:
        """
        Generate answer from retrieved contexts.
        
        Uses template-based generation (no LLM required).
        For production, integrate with OpenAI/HuggingFace LLM.
        """
        # Build context string
        context_parts = []
        for i, ctx in enumerate(contexts, 1):
            context_parts.append(
                f"[{i}] Title: {ctx.title}\nAbstract: {ctx.text[:500]}..."
            )
        
        context_str = "\n\n".join(context_parts)
        
        # Template-based answer generation
        # In production, replace with actual LLM call
        answer = self._template_answer(question, contexts)
        
        return answer
    
    def _template_answer(self, question: str, contexts: List[RetrievedContext]) -> str:
        """Generate template-based answer."""
        # Extract key sentences from contexts
        key_sentences = []
        
        # Question type detection
        question_lower = question.lower()
        is_what = any(w in question_lower for w in ['what', 'define', 'explain'])
        is_why = any(w in question_lower for w in ['why', 'cause', 'reason'])
        is_how = any(w in question_lower for w in ['how', 'treatment', 'manage', 'therapy'])
        is_what_is_the_risk = any(w in question_lower for w in ['risk', 'association', 'link'])
        
        # Extract relevant sentences based on question type
        for ctx in contexts:
            sentences = ctx.text.split('. ')
            
            for sent in sentences:
                sent_lower = sent.lower()
                
                # Score sentence relevance
                score = 0
                
                # Keyword overlap with question
                question_words = set(question_lower.split())
                sent_words = set(sent_lower.split())
                overlap = len(question_words & sent_words)
                score += overlap
                
                # Boost for certain patterns
                if is_what and any(w in sent_lower for w in ['is a', 'defined as', 'refers to']):
                    score += 2
                if is_why and any(w in sent_lower for w in ['cause', 'due to', 'because', 'associated', 'linked']):
                    score += 2
                if is_how and any(w in sent_lower for w in ['treatment', 'therapy', 'intervention', 'improved', 'reduced']):
                    score += 2
                if is_what_is_the_risk and any(w in sent_lower for w in ['risk', 'increase', 'decrease', 'hazard', 'odds']):
                    score += 2
                
                if score > 1:
                    key_sentences.append((sent.strip(), score, ctx.title))
        
        # Sort by score and take top
        key_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = key_sentences[:5]
        
        # Build answer
        if not top_sentences:
            return f"Based on the retrieved literature, I found {len(contexts)} relevant articles, but no specific information addressing your question directly. Please try rephrasing your question."
        
        # Group by source
        by_source = {}
        for sent, score, title in top_sentences:
            if title not in by_source:
                by_source[title] = []
            by_source[title].append(sent)
        
        # Format answer
        answer_parts = []
        
        if is_what:
            answer_parts.append(f"Based on the literature, {top_sentences[0][0]}")
        elif is_why:
            answer_parts.append(f"Research suggests that {top_sentences[0][0]}")
        elif is_how:
            answer_parts.append(f"Studies indicate that {top_sentences[0][0]}")
        else:
            answer_parts.append(f"According to the literature: {top_sentences[0][0]}")
        
        # Add supporting information
        if len(top_sentences) > 1:
            answer_parts.append("\nAdditional context:")
            for sent, score, title in top_sentences[1:3]:
                answer_parts.append(f"- {sent}")
        
        answer_parts.append(f"\n\nThis answer is based on {len(contexts)} relevant research articles from PubMed.")
        
        return "\n".join(answer_parts)
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the vector collection."""
        count = self.collection.count()
        return {
            'total_documents': count,
            'collection_name': self.collection.name,
            'embedding_model': self.embedding_model.get_sentence_embedding_dimension()
        }
    
    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents from the collection."""
        self.collection.delete(ids=ids)
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        self.chroma_client.delete_collection("pubmed_abstracts")
        self.collection = self.chroma_client.get_or_create_collection(
            name="pubmed_abstracts",
            metadata={"hnsw:space": "cosine"}
        )


class SimpleRAGPipeline:
    """
    Simplified RAG pipeline without external vector DB.
    Uses in-memory embeddings for smaller datasets.
    """
    
    def __init__(self, embedding_model: str = 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb'):
        """Initialize simple RAG."""
        self.embedding_model = SentenceTransformer(embedding_model)
        self.documents = []
        self.embeddings = None
        self.ids = []
        self.metadatas = []
    
    def add_documents(self,
                     texts: List[str],
                     ids: List[str],
                     metadatas: List[Dict] = None) -> None:
        """Add documents to in-memory store."""
        self.documents.extend(texts)
        self.ids.extend(ids)
        if metadatas:
            self.metadatas.extend(metadatas)
        else:
            self.metadatas.extend([{}] * len(texts))
        
        # Compute embeddings
        new_embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedContext]:
        """Retrieve documents using cosine similarity."""
        if not self.documents:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        
        # Compute similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append(RetrievedContext(
                text=self.documents[idx],
                source=self.ids[idx],
                title=self.metadatas[idx].get('title', ''),
                relevance_score=float(similarities[idx]),
                metadata=self.metadatas[idx]
            ))
        
        return results
    
    def answer(self, question: str, top_k: int = 5) -> RAGAnswer:
        """Answer question using simple RAG."""
        contexts = self.retrieve(question, top_k)
        
        if not contexts:
            return RAGAnswer(
                answer="No relevant documents found.",
                sources=[],
                confidence=0.0,
                method='simple_retrieval_failed'
            )
        
        # Simple answer extraction
        best_contexts = [c.text[:500] for c in contexts[:3]]
        answer = f"Based on the literature:\n\n" + "\n\n".join(best_contexts)
        
        confidence = np.mean([c.relevance_score for c in contexts])
        
        return RAGAnswer(
            answer=answer,
            sources=contexts,
            confidence=confidence,
            method='simple_rag'
        )
