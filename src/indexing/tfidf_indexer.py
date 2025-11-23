"""TF-IDF Retrieval System"""
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.preprocessing.pipeline import TextPreprocessor

class TFIDFRetrieval:
    """TF-IDF based retrieval with cosine similarity"""
    
    def __init__(self, max_features=None, min_df=2, max_df=0.95):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            tokenizer=self.preprocessor.preprocess,
            lowercase=False,  # Already done in preprocessor
            token_pattern=None
        )
        self.tfidf_matrix = None
        self.documents = {}
    
    def build_index(self, documents_df):
        """
        Build TF-IDF index from documents
        Args:
            documents_df: DataFrame with columns ['Article', 'Heading', 'NewsType']
        """
        print("Building TF-IDF index...")
        
        # Combine article and heading
        corpus = []
        for idx, row in documents_df.iterrows():
            doc_id = idx
            text = str(row['Article']) + " " + str(row['Heading'])
            corpus.append(text)
            
            # Store original document
            self.documents[doc_id] = {
                'heading': row['Heading'],
                'article': row['Article'],
                'news_type': row['NewsType']
            }
            
            if (idx + 1) % 500 == 0:
                print(f"  Processed {idx + 1} documents...")
        
        # Build TF-IDF matrix
        print("  Computing TF-IDF vectors...")
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        
        print(f"✓ Indexed {len(self.documents)} documents")
        print(f"✓ Vocabulary size: {len(self.vectorizer.vocabulary_)} unique terms")
        print(f"✓ TF-IDF matrix shape: {self.tfidf_matrix.shape}")
    
    def search(self, query, top_k=10):
        """
        Search for documents using TF-IDF and cosine similarity
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
        Returns:
            list: List of (doc_id, score) tuples sorted by relevance
        """
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity with all documents
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return doc_id and scores
        results = [(idx, similarities[idx]) for idx in top_indices if similarities[idx] > 0]
        
        return results
    
    def get_document(self, doc_id):
        """Get document by ID"""
        return self.documents.get(doc_id)
    
    def save_index(self, filepath):
        """Save index to disk"""
        data = {
            'vectorizer': self.vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'documents': self.documents
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Index saved to {filepath}")
    
    def load_index(self, filepath):
        """Load index from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.vectorizer = data['vectorizer']
        self.tfidf_matrix = data['tfidf_matrix']
        self.documents = data['documents']
        
        print(f"✓ Index loaded from {filepath}")
        print(f"  Documents: {len(self.documents)}")
        print(f"  Vocabulary: {len(self.vectorizer.vocabulary_)} terms")


# Test and build script
if __name__ == "__main__":
    import pandas as pd
    
    print("="*60)
    print("TF-IDF RETRIEVAL SYSTEM - BUILD INDEX")
    print("="*60)
    
    # Load data
    print("\n1. Loading dataset...")
    df = pd.read_csv('data/raw/Articles.csv', encoding='latin-1')
    print(f"   Loaded {len(df)} articles")
    
    # Build index
    print("\n2. Building TF-IDF index...")
    tfidf_retriever = TFIDFRetrieval(min_df=2, max_df=0.95)
    tfidf_retriever.build_index(df)
    
    # Save index
    print("\n3. Saving index...")
    os.makedirs('data/processed', exist_ok=True)
    tfidf_retriever.save_index('data/processed/tfidf_index.pkl')
    
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
        results = tfidf_retriever.search(query, top_k=5)
        print(f"\nQuery: '{query}'")
        print(f"Results: {len(results)} documents")
        
        if results:
            print("Top 5 matches:")
            for i, (doc_id, score) in enumerate(results, 1):
                doc = tfidf_retriever.get_document(doc_id)
                print(f"  {i}. [Score: {score:.4f}] [{doc['news_type']}] {doc['heading'][:70]}...")
    
    print("\n" + "="*60)
    print("✅ TF-IDF retrieval system ready!")
    print("="*60)