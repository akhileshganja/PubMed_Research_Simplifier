"""
PubMed NLP Pipeline - Example Usage

This script demonstrates the full pipeline with a sample query.
Run this to see the system in action!
"""

import sys
import json
from datetime import datetime

# Import pipeline components
from pubmed_nlp.pipeline import PubMedNLPPipeline
from pubmed_nlp.personalization import UserType


def print_section(title, char="="):
    """Print a formatted section header."""
    print(f"\n{char * 70}")
    print(f" {title}")
    print(f"{char * 70}\n")


def main():
    """Run example pipeline processing."""
    
    # Example queries
    QUERIES = [
        "metformin type 2 diabetes treatment",
        "aspirin cardiovascular disease prevention",
        "covid-19 vaccine efficacy",
    ]
    
    print_section("PubMed NLP Research Simplifier - Demo", "=")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis demo will:")
    print("  1. Search PubMed for articles")
    print("  2. Extract biomedical entities (diseases, drugs, genes)")
    print("  3. Find relationships between entities")
    print("  4. Generate a hybrid summary")
    print("  5. Detect contradictions")
    print("  6. Analyze trends and evidence quality")
    print("  7. Personalize output for different users")
    print("  8. Answer questions using RAG")
    
    # Initialize pipeline
    print_section("Initializing Pipeline", "-")
    print("Loading SciSpacy models, transformers, and embedding models...")
    print("(This may take 1-2 minutes on first run)\n")
    
    try:
        pipeline = PubMedNLPPipeline(
            enable_rag=True,
            enable_contradiction=True,
            device='cpu'  # Use 'cuda' if GPU available
        )
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure all requirements are installed: pip install -r requirements.txt")
        print("  2. Download SciSpacy models (see README)")
        print("  3. Check that you have sufficient RAM (8GB+)")
        sys.exit(1)
    
    # Process example query
    query = QUERIES[0]  # Use first query
    print_section(f"Processing Query: '{query}'", "-")
    
    try:
        result = pipeline.process(
            query=query,
            max_articles=20,  # Limit for demo
            user_types=[UserType.PATIENT, UserType.STUDENT, UserType.DOCTOR],
            enable_qa=True
        )
        
        # Display results
        print_section("Results Overview")
        print(f"Articles retrieved: {len(result.articles)}")
        print(f"Entities extracted: {len(result.entities)}")
        print(f"Relations found: {len(result.relations)}")
        print(f"Contradictions detected: {len(result.contradictions)}")
        print(f"Risk factors identified: {len(result.risk_factors)}")
        
        # Show sample entities
        print_section("Sample Entities (Top 10)")
        seen = set()
        for entity in result.entities[:20]:
            key = f"{entity['text']}_{entity['label']}"
            if key not in seen:
                seen.add(key)
                print(f"  • {entity['text']} ({entity['label']}) [from {entity['source']}]")
                if len(seen) >= 10:
                    break
        
        # Show sample relations
        print_section("Sample Relations (Top 5)")
        for rel in result.relations[:5]:
            print(f"  • {rel.subject} --[{rel.relation_type}]--> {rel.object}")
            print(f"    (confidence: {rel.confidence:.2f})")
        
        # Show summary
        if result.summary:
            print_section("Generated Summary")
            print(f"Method: {result.summary.method}")
            print(f"Compression: {result.summary.compression_ratio:.1%}")
            print(f"\n{result.summary.summary}")
            
            if result.summary.key_points:
                print("\nKey Points:")
                for i, point in enumerate(result.summary.key_points[:5], 1):
                    print(f"  {i}. {point}")
        
        # Show trends
        if result.trends:
            print_section("Trend Analysis")
            print(f"Topic: {result.trends.topic}")
            print(f"Current trajectory: {result.trends.current_trajectory}")
            print(f"Growth rate: {result.trends.growth_rate:.1%}")
            print(f"Peak year: {result.trends.peak_year}")
            
            if result.trends.trend_points:
                print("\nRecent trend data:")
                for point in result.trends.trend_points[-3:]:
                    print(f"  {point.year}: {point.count} articles ({point.percentage:.1f}%)")
        
        # Show evidence scores
        if result.evidence_scores:
            print_section("Evidence Quality (Top 5)")
            for score in result.evidence_scores[:5]:
                print(f"  PMID {score.pmid}:")
                print(f"    Overall score: {score.overall_score:.2f} ({score.evidence_level} quality)")
                print(f"    Citations: {score.citation_count}, Journal impact: {score.journal_impact:.2f}")
        
        # Show risk factors
        if result.risk_factors:
            print_section("Risk Factors (Top 5)")
            for rf in result.risk_factors[:5]:
                print(f"  • {rf.factor} → {rf.relation} → {rf.outcome}")
                print(f"    Confidence: {rf.confidence:.1%}, Evidence: {rf.evidence_count} studies")
        
        # Show personalized outputs
        print_section("Personalized Outputs")
        for user_type, content in result.personalized.items():
            print(f"\n{'='*40}")
            print(f"User Type: {user_type.upper()}")
            print(f"{'='*40}")
            print(f"Technical Level: {content['technical_level']}")
            print(f"\n{content['summary'][:300]}...")
            
            if content['warnings']:
                print("\n⚠️ Warnings:")
                for w in content['warnings'][:2]:
                    print(f"  - {w}")
            
            if content['recommended_actions']:
                print("\n→ Recommended Actions:")
                for a in content['recommended_actions'][:2]:
                    print(f"  - {a}")
        
        # Show RAG answer
        if result.rag_answer:
            print_section("RAG-based Q&A")
            print(f"Query: {result.query}")
            print(f"Answer: {result.rag_answer.answer}")
            print(f"Confidence: {result.rag_answer.confidence:.1%}")
            print(f"Method: {result.rag_answer.method}")
        
        # Show insights summary
        if result.insights:
            print_section("Key Insights")
            for finding in result.insights.get('key_findings', []):
                print(f"  💡 {finding}")
        
        # Save results to file
        output_file = f"result_{query.replace(' ', '_')[:30]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        pipeline.save_result(result, output_file)
        print_section(f"Results saved to: {output_file}")
        
        # Interactive QA
        print_section("Interactive Q&A", "-")
        print("You can now ask questions about this research!")
        print("(Type 'quit' to exit)\n")
        
        while True:
            question = input("Ask a question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            try:
                answer = pipeline.answer_question(
                    question=question,
                    context_query=None  # Use existing context
                )
                
                print(f"\n🤖 Answer (confidence: {answer.confidence:.1%}):")
                print(f"{answer.answer}\n")
                
                if answer.sources:
                    print("Sources:")
                    for src in answer.sources[:3]:
                        print(f"  - {src.title}")
                print()
                
            except Exception as e:
                print(f"Error: {e}\n")
        
        print_section("Demo Complete!", "=")
        print("Thank you for trying the PubMed NLP Research Simplifier.")
        print("\nNext steps:")
        print("  - Start the API: python -m api.main")
        print("  - Open frontend/index.html in your browser")
        print("  - Explore the full web interface!")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
