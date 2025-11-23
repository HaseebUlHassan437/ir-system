"""Evaluation System for IR Systems"""
import json
import time
import os
import numpy as np
from collections import defaultdict
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

class IREvaluator:
    """Evaluate Information Retrieval Systems"""
    
    def __init__(self):
        self.results = {}
    
    def precision_at_k(self, retrieved, relevant, k):
        """
        Calculate Precision@K
        Precision@K = (# of relevant documents in top K) / K
        """
        retrieved_at_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_at_k) & set(relevant))
        return relevant_retrieved / k if k > 0 else 0.0
    
    def recall_at_k(self, retrieved, relevant, k):
        """
        Calculate Recall@K
        Recall@K = (# of relevant documents in top K) / (total # of relevant documents)
        """
        if len(relevant) == 0:
            return 0.0
        retrieved_at_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_at_k) & set(relevant))
        return relevant_retrieved / len(relevant)
    
    def average_precision(self, retrieved, relevant):
        """
        Calculate Average Precision (AP)
        AP = (sum of P@k for each relevant doc) / (total # of relevant docs)
        """
        if len(relevant) == 0:
            return 0.0
        
        score = 0.0
        num_relevant = 0
        
        for k, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                num_relevant += 1
                score += num_relevant / k
        
        return score / len(relevant) if len(relevant) > 0 else 0.0
    
    def mean_average_precision(self, all_retrieved, all_relevant):
        """
        Calculate Mean Average Precision (MAP)
        MAP = average of AP across all queries
        """
        ap_scores = []
        for query_id in all_retrieved.keys():
            retrieved = all_retrieved[query_id]
            relevant = all_relevant.get(query_id, [])
            ap = self.average_precision(retrieved, relevant)
            ap_scores.append(ap)
        
        return np.mean(ap_scores) if ap_scores else 0.0
    
    def ndcg_at_k(self, retrieved, relevant, k):
        """
        Calculate Normalized Discounted Cumulative Gain @ K
        Assumes binary relevance (relevant=1, not relevant=0)
        """
        retrieved_at_k = retrieved[:k]
        
        # DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_at_k, 1):
            if doc_id in relevant:
                dcg += 1.0 / np.log2(i + 1)
        
        # IDCG (ideal DCG)
        idcg = 0.0
        for i in range(1, min(len(relevant), k) + 1):
            idcg += 1.0 / np.log2(i + 1)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_system(self, retriever, queries, relevance_judgments, 
                       system_name, k_values=[5, 10, 20]):
        """
        Evaluate a retrieval system
        Args:
            retriever: Retrieval system object with search() method
            queries: Dict of {query_id: query_text}
            relevance_judgments: Dict of {query_id: [relevant_doc_ids]}
            system_name: Name of the system
            k_values: List of k values for P@K, R@K, NDCG@K
        """
        print(f"\n{'='*60}")
        print(f"Evaluating: {system_name}")
        print('='*60)
        
        all_retrieved = {}
        all_metrics = defaultdict(list)
        query_times = []
        
        for query_id, query_text in queries.items():
            # Measure query time
            start_time = time.time()
            
            # Search
            # Search - handle different retrieval systems
            if system_name == 'Boolean':
                # Boolean returns list of doc IDs
                retrieved_ids = retriever.search(query_text, operator='OR')[:max(k_values)]
            else:
                # TF-IDF and BM25 return list of (doc_id, score) tuples
                results = retriever.search(query_text, top_k=max(k_values))
                retrieved_ids = [doc_id for doc_id, score in results]
            
            query_time = (time.time() - start_time) * 1000  # Convert to ms
            query_times.append(query_time)
            
            all_retrieved[query_id] = retrieved_ids
            relevant_ids = relevance_judgments.get(query_id, [])
            
            # Calculate metrics for different k values
            for k in k_values:
                p_at_k = self.precision_at_k(retrieved_ids, relevant_ids, k)
                r_at_k = self.recall_at_k(retrieved_ids, relevant_ids, k)
                ndcg_k = self.ndcg_at_k(retrieved_ids, relevant_ids, k)
                
                all_metrics[f'P@{k}'].append(p_at_k)
                all_metrics[f'R@{k}'].append(r_at_k)
                all_metrics[f'NDCG@{k}'].append(ndcg_k)
            
            # Calculate AP
            ap = self.average_precision(retrieved_ids, relevant_ids)
            all_metrics['AP'].append(ap)
        
        # Calculate MAP
        map_score = np.mean(all_metrics['AP'])
        
        # Average metrics
        avg_metrics = {
            'System': system_name,
            'MAP': map_score,
            'Avg Query Time (ms)': np.mean(query_times),
            'Std Query Time (ms)': np.std(query_times)
        }
        
        for metric_name, values in all_metrics.items():
            if metric_name != 'AP':
                avg_metrics[metric_name] = np.mean(values)
        
        # Print results
        print(f"\nResults:")
        print(f"  MAP: {map_score:.4f}")
        for k in k_values:
            print(f"  P@{k}: {avg_metrics[f'P@{k}']:.4f}")
            print(f"  R@{k}: {avg_metrics[f'R@{k}']:.4f}")
            print(f"  NDCG@{k}: {avg_metrics[f'NDCG@{k}']:.4f}")
        print(f"\n  Avg Query Time: {avg_metrics['Avg Query Time (ms)']:.2f} ms")
        
        self.results[system_name] = avg_metrics
        
        return avg_metrics
    
    def compare_systems(self, output_path=None):
        """
        Compare all evaluated systems
        """
        if not self.results:
            print("No systems evaluated yet!")
            return
        
        print("\n" + "="*80)
        print("SYSTEM COMPARISON")
        print("="*80)
        
        # Create comparison DataFrame
        df = pd.DataFrame(self.results).T
        df = df.round(4)
        
        print("\n", df.to_string())
        
        if output_path:
            df.to_csv(output_path)
            print(f"\n✓ Results saved to {output_path}")
        
        return df
    
    def get_memory_usage(self, index_path):
        """Get index file size in MB"""
        if os.path.exists(index_path):
            size_mb = os.path.getsize(index_path) / (1024 * 1024)
            return size_mb
        return 0.0


def create_test_queries():
    """Create test queries for evaluation"""
    queries = {
        'q1': 'cricket match',
        'q2': 'stock market',
        'q3': 'business economy',
        'q4': 'football world cup',
        'q5': 'pakistan team',
        'q6': 'company profit',
        'q7': 'championship tournament',
        'q8': 'financial crisis',
        'q9': 'player performance',
        'q10': 'trade investment'
    }
    return queries


def create_pseudo_relevance_judgments(df, queries):
    """
    Create pseudo relevance judgments based on NewsType and keywords
    """
    print("\nCreating pseudo-relevance judgments...")
    
    from src.preprocessing.pipeline import TextPreprocessor
    preprocessor = TextPreprocessor(use_stemming=True, remove_stopwords=True)
    
    relevance = {}
    
    # Define query categories
    query_categories = {
        'q1': 'sports',  # cricket match
        'q2': 'business',  # stock market
        'q3': 'business',  # business economy
        'q4': 'sports',  # football world cup
        'q5': 'sports',  # pakistan team
        'q6': 'business',  # company profit
        'q7': 'sports',  # championship tournament
        'q8': 'business',  # financial crisis
        'q9': 'sports',  # player performance
        'q10': 'business'  # trade investment
    }
    
    for query_id, query_text in queries.items():
        relevant_docs = []
        expected_category = query_categories[query_id]
        
        # Preprocess query
        query_tokens = set(preprocessor.preprocess(query_text))
        
        for idx, row in df.iterrows():
            # Check if document is in the right category
            if row['NewsType'] == expected_category:
                # Check if document contains query terms
                doc_text = str(row['Article']) + " " + str(row['Heading'])
                doc_tokens = set(preprocessor.preprocess(doc_text))
                
                # If document contains at least one query term, mark as relevant
                if query_tokens & doc_tokens:  # Intersection
                    relevant_docs.append(idx)
        
        relevance[query_id] = relevant_docs
        print(f"  {query_id}: {len(relevant_docs)} relevant documents")
    
    return relevance


if __name__ == "__main__":
    print("="*60)
    print("IR SYSTEM EVALUATION")
    print("="*60)
    
    # Load dataset
    print("\n1. Loading dataset...")
    df = pd.read_csv('data/raw/Articles.csv', encoding='latin-1')
    print(f"   Loaded {len(df)} articles")
    
    # Create test queries
    print("\n2. Creating test queries...")
    queries = create_test_queries()
    print(f"   Created {len(queries)} test queries")
    
    # Create relevance judgments
    print("\n3. Creating relevance judgments...")
    relevance_judgments = create_pseudo_relevance_judgments(df, queries)
    
    # Save queries and judgments
    os.makedirs('experiments/results', exist_ok=True)
    with open('experiments/results/test_queries.json', 'w') as f:
        json.dump(queries, f, indent=2)
    with open('experiments/results/relevance_judgments.json', 'w') as f:
        json.dump(relevance_judgments, f, indent=2)
    print("   ✓ Saved to experiments/results/")
    
    # Load all retrieval systems
    print("\n4. Loading retrieval systems...")
    
    from src.indexing.boolean_indexer import BooleanRetrieval
    from src.indexing.tfidf_indexer import TFIDFRetrieval
    from src.indexing.bm25_indexer import BM25Retrieval
    
    boolean_retriever = BooleanRetrieval()
    boolean_retriever.load_index('data/processed/boolean_index.pkl')
    
    tfidf_retriever = TFIDFRetrieval()
    tfidf_retriever.load_index('data/processed/tfidf_index.pkl')
    
    bm25_retriever = BM25Retrieval()
    bm25_retriever.load_index('data/processed/bm25_index.pkl')
    
    # Evaluate all systems
    print("\n5. Evaluating systems...")
    evaluator = IREvaluator()
    
    evaluator.evaluate_system(boolean_retriever, queries, relevance_judgments, 
                             'Boolean', k_values=[5, 10, 20])
    
    evaluator.evaluate_system(tfidf_retriever, queries, relevance_judgments, 
                             'TF-IDF', k_values=[5, 10, 20])
    
    evaluator.evaluate_system(bm25_retriever, queries, relevance_judgments, 
                             'BM25', k_values=[5, 10, 20])
    
    # Get memory usage
    print("\n6. Analyzing memory usage...")
    boolean_size = evaluator.get_memory_usage('data/processed/boolean_index.pkl')
    tfidf_size = evaluator.get_memory_usage('data/processed/tfidf_index.pkl')
    bm25_size = evaluator.get_memory_usage('data/processed/bm25_index.pkl')
    
    print(f"   Boolean Index: {boolean_size:.2f} MB")
    print(f"   TF-IDF Index: {tfidf_size:.2f} MB")
    print(f"   BM25 Index: {bm25_size:.2f} MB")
    
    # Compare systems
    print("\n7. Comparing all systems...")
    comparison_df = evaluator.compare_systems('experiments/results/system_comparison.csv')
    
    print("\n" + "="*60)
    print("✅ Evaluation complete!")
    print("="*60)