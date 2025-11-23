"""BM25 Retrieval System"""
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.preprocessing.pipeline import TextPreprocessor

class BM25Retrieval:
    """BM25 probabilistic retrieval model"""
    
    def __init__(self, k1=1.5, b=0.75):
        """
        Args:
            k1: Term frequency saturation parameter (default 1.5)
            b: Length normalization parameter (default 0.75)
        """
        self.k1 = k1
        self.b = b
        self.preprocessor = TextPreprocessor()
        self.bm25 = None
        self.documents = {}
        self.tokenized_corpus = []
    
    def build_index(self, documents_df):
        """
        Build BM25 index from documents
        Args:
            documents_df: DataFrame with columns ['Article', 'Heading', 'NewsType']
        """
        print("Building BM25 index...")
        print(f"Parameters: k1={self.k1}, b={self.b}")
        
        # Tokenize all documents
        corpus = []
        for idx, row in documents_df.iterrows():
            doc_id = idx
            text = str(row['Article']) + " " + str(row['Heading'])
            
            # Preprocess and tokenize
            tokens = self.preprocessor.preprocess(text)
            self.tokenized_corpus.append(tokens)
            
            # Store original document
            self.documents[doc_id] = {
                'heading': row['Heading'],
                'article': row['Article'],
                'news_type': row['NewsType']
            }
            
            if (idx + 1) % 500 == 0:
                print(f"  Processed {idx + 1} documents...")
        
        # Build BM25 index
        print("  Computing BM25 scores...")
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        
        print(f"✓ Indexed {len(self.documents)} documents")
        print(f"✓ Average document length: {self.bm25.avgdl:.2f} terms")
        print(f"✓ Total corpus size: {sum(self.bm25.doc_len)} terms")
    
    def search(self, query, top_k=10):
        """
        Search for documents using BM25
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
        Returns:
            list: List of (doc_id, score) tuples sorted by relevance
        """
        # Tokenize query
        query_tokens = self.preprocessor.preprocess(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Return doc_id and scores (filter out zero scores)
        results = [(idx, scores[idx]) for idx in top_indices if scores[idx] > 0]
        
        return results
    
    def get_document(self, doc_id):
        """Get document by ID"""
        return self.documents.get(doc_id)
    
    def save_index(self, filepath):
        """Save index to disk"""
        data = {
            'bm25': self.bm25,
            'documents': self.documents,
            'tokenized_corpus': self.tokenized_corpus,
            'k1': self.k1,
            'b': self.b
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Index saved to {filepath}")
    
    def load_index(self, filepath):
        """Load index from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.bm25 = data['bm25']
        self.documents = data['documents']
        self.tokenized_corpus = data['tokenized_corpus']
        self.k1 = data['k1']
        self.b = data['b']
        
        print(f"✓ Index loaded from {filepath}")
        print(f"  Documents: {len(self.documents)}")
        print(f"  Parameters: k1={self.k1}, b={self.b}")


# Test and build script
if __name__ == "__main__":
    import pandas as pd
    
    print("="*60)
    print("BM25 RETRIEVAL SYSTEM - BUILD INDEX")
    print("="*60)
    
    # Load data
    print("\n1. Loading dataset...")
    df = pd.read_csv('data/raw/Articles.csv', encoding='latin-1')
    print(f"   Loaded {len(df)} articles")
    
    # Build index
    print("\n2. Building BM25 index...")
    bm25_retriever = BM25Retrieval(k1=1.5, b=0.75)
    bm25_retriever.build_index(df)
    
    # Save index
    print("\n3. Saving index...")
    os.makedirs('data/processed', exist_ok=True)
    bm25_retriever.save_index('data/processed/bm25_index.pkl')
    
    # Test queries
    print("\n" + "="*60)
    print("TESTING QUERIES")
    print("="*60)
    
    test_queries = [
        "cricket match",
        "business economy",
        "stock market",
        "football world cup"
    ]
    
    for query in test_queries:
        results = bm25_retriever.search(query, top_k=5)
        print(f"\nQuery: '{query}'")
        print(f"Results: {len(results)} documents")
        
        if results:
            print("Top 5 matches:")
            for i, (doc_id, score) in enumerate(results, 1):
                doc = bm25_retriever.get_document(doc_id)
                print(f"  {i}. [Score: {score:.4f}] [{doc['news_type']}] {doc['heading'][:70]}...")
    
    print("\n" + "="*60)
    print("✅ BM25 retrieval system ready!")
    print("="*60)
    
    # Compare with TF-IDF for one query
    print("\n" + "="*60)
    print("COMPARISON: BM25 vs TF-IDF")
    print("="*60)
    print("\nLoading TF-IDF index for comparison...")
    
    try:
        from src.indexing.tfidf_indexer import TFIDFRetrieval
        tfidf_retriever = TFIDFRetrieval()
        tfidf_retriever.load_index('data/processed/tfidf_index.pkl')
        
        test_query = "cricket match"
        print(f"\nQuery: '{test_query}'")
        
        print("\n--- BM25 Results ---")
        bm25_results = bm25_retriever.search(test_query, top_k=3)
        for i, (doc_id, score) in enumerate(bm25_results, 1):
            doc = bm25_retriever.get_document(doc_id)
            print(f"{i}. [BM25: {score:.4f}] {doc['heading'][:60]}...")
        
        print("\n--- TF-IDF Results ---")
        tfidf_results = tfidf_retriever.search(test_query, top_k=3)
        for i, (doc_id, score) in enumerate(tfidf_results, 1):
            doc = tfidf_retriever.get_document(doc_id)
            print(f"{i}. [TF-IDF: {score:.4f}] {doc['heading'][:60]}...")
    
    except Exception as e:
        print(f"Could not load TF-IDF for comparison: {e}")