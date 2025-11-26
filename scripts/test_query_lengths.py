"""
Test retrieval systems with varying query lengths
File: scripts/test_query_lengths.py
"""
import pandas as pd
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.indexing.boolean_indexer import BooleanRetrieval
from src.indexing.tfidf_indexer import TFIDFRetrieval
from src.indexing.bm25_indexer import BM25Retrieval
from src.evaluation.evaluator import IREvaluator, create_pseudo_relevance_judgments

def create_varied_length_queries():
    """Create test queries of different lengths"""
    queries = {
        # Short queries (1-2 words)
        'short_1': 'cricket',
        'short_2': 'stock market',
        'short_3': 'economy',
        'short_4': 'football',
        
        # Medium queries (3-4 words)
        'medium_1': 'pakistan cricket team performance',
        'medium_2': 'stock market business news',
        'medium_3': 'economy growth financial crisis',
        'medium_4': 'football world cup tournament',
        
        # Long queries (5-7 words)
        'long_1': 'pakistan cricket team wins championship match',
        'long_2': 'stock market business economy financial growth',
        'long_3': 'international football tournament world cup final',
        'long_4': 'company profit trade investment business economy',
        
        # Very long queries (8+ words)
        'vlong_1': 'pakistan cricket team performance in international championship tournament match',
        'vlong_2': 'stock market business economy financial crisis trade investment company profit growth',
        'vlong_3': 'international football world cup tournament championship final match team performance',
    }
    
    # Categorize by length
    query_categories = {
        'short': ['short_1', 'short_2', 'short_3', 'short_4'],
        'medium': ['medium_1', 'medium_2', 'medium_3', 'medium_4'],
        'long': ['long_1', 'long_2', 'long_3', 'long_4'],
        'very_long': ['vlong_1', 'vlong_2', 'vlong_3']
    }
    
    return queries, query_categories

def assign_query_categories(queries):
    """Assign expected category (sports/business) to each query"""
    categories = {}
    
    # Define based on keywords
    sports_keywords = ['cricket', 'football', 'tournament', 'championship', 'match', 'team', 'performance']
    business_keywords = ['stock', 'market', 'economy', 'financial', 'business', 'trade', 'investment', 'profit', 'company', 'growth']
    
    for qid, query_text in queries.items():
        query_lower = query_text.lower()
        
        # Count keyword matches
        sports_count = sum(1 for kw in sports_keywords if kw in query_lower)
        business_count = sum(1 for kw in business_keywords if kw in query_lower)
        
        # Assign category based on majority
        if sports_count > business_count:
            categories[qid] = 'sports'
        else:
            categories[qid] = 'business'
    
    return categories

# def test_system_by_query_length(retriever, system_name, queries, query_categories, relevance_judgments):
    """Test a system on queries of different lengths"""
    
    evaluator = IREvaluator()
    
    print(f"\n{'='*70}")
    print(f"Testing {system_name} with Varying Query Lengths")
    print('='*70)
    
    results_by_length = {}
    
    for length_category, query_ids in query_categories.items():
        print(f"\n--- {length_category.upper()} Queries ---")
        
        # Filter queries for this length category
        length_queries = {qid: queries[qid] for qid in query_ids}
        length_relevance = {qid: relevance_judgments[qid] for qid in query_ids}
        
        # Evaluate
        metrics = evaluator.evaluate_system(
            retriever,
            length_queries,
            length_relevance,
            f"{system_name}_{length_category}",
            k_values=[5, 10, 20]
        )
        
        results_by_length[length_category] = metrics
        
        # Show sample results for first query
        first_qid = query_ids[0]
        first_query = length_queries[first_qid]
        print(f"\nSample Query: '{first_query}'")
        
        # Evaluate
        if system_name == 'Boolean':
            # Create a wrapper to make Boolean compatible with evaluator
            class BooleanWrapper:
                def __init__(self, retriever):
                    self.retriever = retriever
                
                def search(self, query, top_k=10):
                    results = self.retriever.search(query, operator='OR')[:top_k]
                    # Return as list of tuples (doc_id, dummy_score) for compatibility
                    return [(doc_id, 1.0) for doc_id in results]
                
                def get_document(self, doc_id):
                    return self.retriever.get_document(doc_id)
            
            wrapped_retriever = BooleanWrapper(retriever)
            metrics = evaluator.evaluate_system(
                wrapped_retriever,
                length_queries,
                length_relevance,
                f"{system_name}_{length_category}",
                k_values=[5, 10, 20]
            )
        else:
            metrics = evaluator.evaluate_system(
                retriever,
                length_queries,
                length_relevance,
                f"{system_name}_{length_category}",
                k_values=[5, 10, 20]
            )
    
    return results_by_length
def test_system_by_query_length(retriever, system_name, queries, query_categories, relevance_judgments):
    """Test a system on queries of different lengths"""
    
    evaluator = IREvaluator()
    
    print(f"\n{'='*70}")
    print(f"Testing {system_name} with Varying Query Lengths")
    print('='*70)
    
    results_by_length = {}
    
    for length_category, query_ids in query_categories.items():
        print(f"\n--- {length_category.upper()} Queries ---")
        
        # Filter queries for this length category
        length_queries = {qid: queries[qid] for qid in query_ids}
        length_relevance = {qid: relevance_judgments[qid] for qid in query_ids}
        
        # Handle Boolean separately
        if system_name == 'Boolean':
            # Create wrapper for Boolean
            class BooleanWrapper:
                def __init__(self, retriever):
                    self.retriever = retriever
                
                def search(self, query, top_k=10):
                    results = self.retriever.search(query, operator='OR')[:top_k]
                    return [(doc_id, 1.0) for doc_id in results]
                
                def get_document(self, doc_id):
                    return self.retriever.get_document(doc_id)
            
            wrapped = BooleanWrapper(retriever)
            metrics = evaluator.evaluate_system(
                wrapped,
                length_queries,
                length_relevance,
                f"{system_name}_{length_category}",
                k_values=[5, 10, 20]
            )
        else:
            # TF-IDF and BM25 work directly
            metrics = evaluator.evaluate_system(
                retriever,
                length_queries,
                length_relevance,
                f"{system_name}_{length_category}",
                k_values=[5, 10, 20]
            )
        
        results_by_length[length_category] = metrics
        
        # Show sample results for first query
        first_qid = query_ids[0]
        first_query = length_queries[first_qid]
        print(f"\nSample Query: '{first_query}'")
        
        if system_name == 'Boolean':
            results = retriever.search(first_query, operator='OR')[:5]
            for i, doc_id in enumerate(results, 1):
                doc = retriever.get_document(doc_id)
                print(f"  {i}. [{doc['news_type']}] {doc['heading'][:70]}...")
        else:
            results = retriever.search(first_query, top_k=5)
            for i, (doc_id, score) in enumerate(results, 1):
                doc = retriever.get_document(doc_id)
                print(f"  {i}. [Score: {score:.4f}] [{doc['news_type']}] {doc['heading'][:60]}...")
    
    return results_by_length


def compare_systems_by_length(all_results):
    """Compare all systems across query lengths"""
    
    print("\n" + "="*70)
    print("COMPARISON: Performance by Query Length")
    print("="*70)
    
    # Extract length categories
    length_categories = list(next(iter(all_results.values())).keys())
    
    for length_cat in length_categories:
        print(f"\n{length_cat.upper()} QUERIES:")
        print("-" * 70)
        print(f"{'System':<20} {'MAP':>8} {'P@5':>8} {'P@10':>8} {'R@10':>8} {'Time(ms)':>10}")
        print("-" * 70)
        
        for system_name, results in all_results.items():
            metrics = results[length_cat]
            print(f"{system_name:<20} "
                  f"{metrics['MAP']:>8.4f} "
                  f"{metrics['P@5']:>8.3f} "
                  f"{metrics['P@10']:>8.3f} "
                  f"{metrics['R@10']:>8.4f} "
                  f"{metrics['Avg Query Time (ms)']:>10.2f}")

def analyze_length_impact(all_results):
    """Analyze how query length affects each system"""
    
    print("\n" + "="*70)
    print("ANALYSIS: Impact of Query Length on Each System")
    print("="*70)
    
    for system_name, results in all_results.items():
        print(f"\n{system_name}:")
        
        length_order = ['short', 'medium', 'long', 'very_long']
        
        print(f"  {'Length':<12} {'MAP':>8} {'P@10':>8} {'Time(ms)':>10}")
        print(f"  {'-'*40}")
        
        for length_cat in length_order:
            if length_cat in results:
                metrics = results[length_cat]
                print(f"  {length_cat:<12} "
                      f"{metrics['MAP']:>8.4f} "
                      f"{metrics['P@10']:>8.3f} "
                      f"{metrics['Avg Query Time (ms)']:>10.2f}")

if __name__ == "__main__":
    
    print("="*70)
    print("QUERY LENGTH VARIATION TEST")
    print("="*70)
    
    # Load dataset
    print("\n1. Loading dataset...")
    df = pd.read_csv('data/raw/Articles.csv', encoding='latin-1')
    print(f"   Loaded {len(df)} articles")
    
    # Create varied queries
    print("\n2. Creating varied-length queries...")
    queries, query_categories = create_varied_length_queries()
    
    print(f"   Created {len(queries)} queries:")
    for category, qids in query_categories.items():
        print(f"     - {category}: {len(qids)} queries")
    
    # Assign categories
    print("\n3. Assigning query categories...")
    query_cats = assign_query_categories(queries)
    
    # Create relevance judgments
    print("\n4. Creating relevance judgments...")
    
    # Temporarily create them with proper category assignment
    from src.preprocessing.pipeline import TextPreprocessor
    preprocessor = TextPreprocessor()
    
    relevance_judgments = {}
    for query_id, query_text in queries.items():
        relevant_docs = []
        expected_category = query_cats[query_id]
        query_tokens = set(preprocessor.preprocess(query_text))
        
        for idx, row in df.iterrows():
            if row['NewsType'] == expected_category:
                doc_text = str(row['Article']) + " " + str(row['Heading'])
                doc_tokens = set(preprocessor.preprocess(doc_text))
                
                if query_tokens & doc_tokens:
                    relevant_docs.append(idx)
        
        relevance_judgments[query_id] = relevant_docs
        print(f"   {query_id}: {len(relevant_docs)} relevant documents")
    
    # Save queries and relevance
    os.makedirs('experiments/results', exist_ok=True)
    
    with open('experiments/results/varied_queries.json', 'w') as f:
        json.dump(queries, f, indent=2)
    
    with open('experiments/results/varied_relevance.json', 'w') as f:
        json.dump(relevance_judgments, f, indent=2)
    
    print("   ✓ Saved to experiments/results/")
    
    # Load retrieval systems
    print("\n5. Loading retrieval systems...")
    
    boolean_retriever = BooleanRetrieval()
    boolean_retriever.load_index('data/processed/boolean_index.pkl')
    
    tfidf_retriever = TFIDFRetrieval()
    tfidf_retriever.load_index('data/processed/tfidf_index.pkl')
    
    bm25_retriever = BM25Retrieval()
    bm25_retriever.load_index('data/processed/bm25_index.pkl')
    
    # Test each system
    print("\n6. Testing systems with varied query lengths...")
    
    all_results = {}
    
    all_results['Boolean'] = test_system_by_query_length(
        boolean_retriever, 'Boolean', queries, query_categories, relevance_judgments
    )
    
    all_results['TF-IDF'] = test_system_by_query_length(
        tfidf_retriever, 'TF-IDF', queries, query_categories, relevance_judgments
    )
    
    all_results['BM25'] = test_system_by_query_length(
        bm25_retriever, 'BM25', queries, query_categories, relevance_judgments
    )
    
    # Compare systems
    compare_systems_by_length(all_results)
    
    # Analyze length impact
    analyze_length_impact(all_results)
    
    # Save detailed results
    print("\n7. Saving detailed results...")
    
    # Convert to flat structure for CSV
    rows = []
    for system_name, results in all_results.items():
        for length_cat, metrics in results.items():
            row = {'System': system_name, 'Query_Length': length_cat}
            row.update(metrics)
            rows.append(row)
    
    results_df = pd.DataFrame(rows)
    results_df.to_csv('experiments/results/query_length_analysis.csv', index=False)
    print("   ✓ Saved to experiments/results/query_length_analysis.csv")
    
    print("\n" + "="*70)
    print("✅ Query length testing complete!")
    print("="*70)