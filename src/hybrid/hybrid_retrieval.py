"""Hybrid Retrieval System - Two-Stage Retrieval"""
import pickle
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.indexing.boolean_indexer import BooleanRetrieval
from src.indexing.bm25_indexer import BM25Retrieval

class HybridRetrieval:
    """
    Two-stage hybrid retrieval:
    1. Stage 1: Fast Boolean retrieval (broad recall)
    2. Stage 2: BM25 re-ranking (high precision)
    """
    
    def __init__(self, boolean_retriever, bm25_retriever, first_stage_k=100):
        """
        Args:
            boolean_retriever: BooleanRetrieval instance
            bm25_retriever: BM25Retrieval instance
            first_stage_k: Number of candidates from Boolean stage
        """
        self.boolean_retriever = boolean_retriever
        self.bm25_retriever = bm25_retriever
        self.first_stage_k = first_stage_k
    
    def search(self, query, top_k=10):
        """
        Two-stage search
        Args:
            query: Search query string
            top_k: Final number of results to return
        Returns:
            list: List of (doc_id, score) tuples
        """
        # Stage 1: Boolean retrieval (fast, broad)
        candidates = self.boolean_retriever.search(query, operator='OR')
        
        # Limit candidates
        if len(candidates) > self.first_stage_k:
            candidates = candidates[:self.first_stage_k]
        
        if not candidates:
            return []
        
        # Stage 2: BM25 re-ranking
        # Get BM25 scores for candidate documents
        query_tokens = self.bm25_retriever.preprocessor.preprocess(query)
        all_scores = self.bm25_retriever.bm25.get_scores(query_tokens)
        
        # Score only the candidates
        candidate_scores = [(doc_id, all_scores[doc_id]) for doc_id in candidates]
        
        # Sort by BM25 score and return top-k
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        
        return candidate_scores[:top_k]
    
    def get_document(self, doc_id):
        """Get document by ID"""
        return self.bm25_retriever.get_document(doc_id)


class ScoreFusionRetrieval:
    """
    Score fusion: Combine TF-IDF and BM25 scores
    """
    
    def __init__(self, tfidf_retriever, bm25_retriever, tfidf_weight=0.5, bm25_weight=0.5):
        """
        Args:
            tfidf_retriever: TFIDFRetrieval instance
            bm25_retriever: BM25Retrieval instance
            tfidf_weight: Weight for TF-IDF scores
            bm25_weight: Weight for BM25 scores
        """
        self.tfidf_retriever = tfidf_retriever
        self.bm25_retriever = bm25_retriever
        self.tfidf_weight = tfidf_weight
        self.bm25_weight = bm25_weight
    
    def normalize_scores(self, scores):
        """Normalize scores to [0, 1] range"""
        if not scores or max(scores.values()) == 0:
            return scores
        
        max_score = max(scores.values())
        min_score = min(scores.values())
        
        if max_score == min_score:
            return {doc_id: 1.0 for doc_id in scores}
        
        normalized = {}
        for doc_id, score in scores.items():
            normalized[doc_id] = (score - min_score) / (max_score - min_score)
        
        return normalized
    
    def search(self, query, top_k=10):
        """
        Fuse scores from TF-IDF and BM25
        Args:
            query: Search query string
            top_k: Number of results to return
        Returns:
            list: List of (doc_id, score) tuples
        """
        # Get results from both systems
        tfidf_results = self.tfidf_retriever.search(query, top_k=50)
        bm25_results = self.bm25_retriever.search(query, top_k=50)
        
        # Convert to dictionaries
        tfidf_scores = {doc_id: score for doc_id, score in tfidf_results}
        bm25_scores = {doc_id: score for doc_id, score in bm25_results}
        
        # Normalize scores
        tfidf_scores = self.normalize_scores(tfidf_scores)
        bm25_scores = self.normalize_scores(bm25_scores)
        
        # Combine scores
        all_doc_ids = set(tfidf_scores.keys()) | set(bm25_scores.keys())
        fused_scores = {}
        
        for doc_id in all_doc_ids:
            tfidf_score = tfidf_scores.get(doc_id, 0.0)
            bm25_score = bm25_scores.get(doc_id, 0.0)
            
            fused_scores[doc_id] = (
                self.tfidf_weight * tfidf_score + 
                self.bm25_weight * bm25_score
            )
        
        # Sort and return top-k
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_results[:top_k]
    
    def get_document(self, doc_id):
        """Get document by ID"""
        return self.bm25_retriever.get_document(doc_id)


# Test and evaluation script
if __name__ == "__main__":
    import pandas as pd
    import json
    from src.indexing.tfidf_indexer import TFIDFRetrieval
    from src.evaluation.evaluator import IREvaluator
    
    print("="*60)
    print("HYBRID RETRIEVAL SYSTEMS")
    print("="*60)
    
    # Load data
    print("\n1. Loading dataset...")
    df = pd.read_csv('data/raw/Articles.csv', encoding='latin-1')
    
    # Load base retrieval systems
    print("\n2. Loading base retrieval systems...")
    
    boolean_retriever = BooleanRetrieval()
    boolean_retriever.load_index('data/processed/boolean_index.pkl')
    
    tfidf_retriever = TFIDFRetrieval()
    tfidf_retriever.load_index('data/processed/tfidf_index.pkl')
    
    bm25_retriever = BM25Retrieval()
    bm25_retriever.load_index('data/processed/bm25_index.pkl')
    
    # Create hybrid systems
    print("\n3. Creating hybrid systems...")
    
    # Hybrid 1: Two-stage (Boolean + BM25)
    hybrid_two_stage = HybridRetrieval(
        boolean_retriever, 
        bm25_retriever, 
        first_stage_k=100
    )
    print("   ✓ Two-Stage Retrieval (Boolean → BM25)")
    
    # Hybrid 2: Score Fusion (TF-IDF + BM25)
    hybrid_fusion = ScoreFusionRetrieval(
        tfidf_retriever,
        bm25_retriever,
        tfidf_weight=0.5,
        bm25_weight=0.5
    )
    print("   ✓ Score Fusion (TF-IDF + BM25)")
    
    # Test queries
    print("\n4. Testing hybrid systems...")
    
    test_queries = {
        'q1': 'cricket match',
        'q2': 'stock market',
        'q3': 'football world cup'
    }
    
    print("\n--- Two-Stage Retrieval ---")
    for query_id, query_text in test_queries.items():
        results = hybrid_two_stage.search(query_text, top_k=3)
        print(f"\nQuery: '{query_text}'")
        for i, (doc_id, score) in enumerate(results, 1):
            doc = hybrid_two_stage.get_document(doc_id)
            print(f"  {i}. [Score: {score:.4f}] [{doc['news_type']}] {doc['heading'][:60]}...")
    
    print("\n--- Score Fusion ---")
    for query_id, query_text in test_queries.items():
        results = hybrid_fusion.search(query_text, top_k=3)
        print(f"\nQuery: '{query_text}'")
        for i, (doc_id, score) in enumerate(results, 1):
            doc = hybrid_fusion.get_document(doc_id)
            print(f"  {i}. [Score: {score:.4f}] [{doc['news_type']}] {doc['heading'][:60]}...")
    
    # Evaluate hybrid systems
    print("\n" + "="*60)
    print("EVALUATING HYBRID SYSTEMS")
    print("="*60)
    
    # Load test queries and relevance judgments
    with open('experiments/results/test_queries.json', 'r') as f:
        queries = json.load(f)
    
    with open('experiments/results/relevance_judgments.json', 'r') as f:
        relevance_judgments = json.load(f)
    
    evaluator = IREvaluator()
    
    # Evaluate two-stage hybrid
    evaluator.evaluate_system(
        hybrid_two_stage, 
        queries, 
        relevance_judgments,
        'Hybrid-TwoStage',
        k_values=[5, 10, 20]
    )
    
    # Evaluate score fusion
    evaluator.evaluate_system(
        hybrid_fusion,
        queries,
        relevance_judgments,
        'Hybrid-Fusion',
        k_values=[5, 10, 20]
    )
    
    # Compare all systems (including base systems)
    print("\n" + "="*60)
    print("LOADING PREVIOUS RESULTS FOR COMPARISON")
    print("="*60)
    
    # Load previous results
    prev_results = pd.read_csv('experiments/results/system_comparison.csv', index_col=0)
    
    # Add new results
    for system_name, metrics in evaluator.results.items():
        prev_results.loc[system_name] = metrics
    
    print("\n", prev_results.to_string())
    
    # Save updated comparison
    prev_results.to_csv('experiments/results/system_comparison_with_hybrid.csv')
    print("\n✓ Results saved to experiments/results/system_comparison_with_hybrid.csv")
    
    print("\n" + "="*60)
    print("✅ Hybrid systems evaluation complete!")
    print("="*60)