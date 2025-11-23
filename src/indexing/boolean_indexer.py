# """Boolean retrieval indexer"""
# from .base_indexer import BaseIndexer

# class BooleanIndexer(BaseIndexer):
#     """Boolean inverted index"""
    
#     def __init__(self):
#         self.inverted_index = {}
    
#     def build_index(self, documents):
#         # TODO: Build inverted index
#         pass
    
#     def save_index(self, path):
#         # TODO: Save index
#         pass
    
#     def load_index(self, path):
#         # TODO: Load index
#         pass


"""Boolean Retrieval System"""
import pickle
from collections import defaultdict
# from src.preprocessing.pipeline import 
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.preprocessing.pipeline import TextPreprocessor

class BooleanRetrieval:
    """Simple Boolean retrieval with inverted index"""
    
    def __init__(self):
        self.inverted_index = defaultdict(set)
        self.documents = {}
        self.preprocessor = TextPreprocessor()
    
    def build_index(self, documents_df):
        """
        Build inverted index from documents
        Args:
            documents_df: DataFrame with columns ['Article', 'Heading', 'NewsType']
        """
        print("Building Boolean inverted index...")
        
        for idx, row in documents_df.iterrows():
            doc_id = idx
            text = str(row['Article']) + " " + str(row['Heading'])
            
            # Store original document
            self.documents[doc_id] = {
                'heading': row['Heading'],
                'article': row['Article'],
                'news_type': row['NewsType']
            }
            
            # Preprocess and index
            tokens = self.preprocessor.preprocess(text)
            
            # Add to inverted index
            for token in set(tokens):  # Use set to avoid duplicate doc_ids
                self.inverted_index[token].add(doc_id)
            
            if (idx + 1) % 500 == 0:
                print(f"  Indexed {idx + 1} documents...")
        
        print(f"✓ Indexed {len(self.documents)} documents")
        print(f"✓ Vocabulary size: {len(self.inverted_index)} unique terms")
    
    def search(self, query, operator='AND'):
        """
        Search for documents matching query
        Args:
            query (str): Search query
            operator (str): 'AND' or 'OR'
        Returns:
            list: List of matching document IDs
        """
        # Preprocess query
        query_tokens = self.preprocessor.preprocess(query)
        
        if not query_tokens:
            return []
        
        # Get posting lists for each term
        posting_lists = []
        for token in query_tokens:
            if token in self.inverted_index:
                posting_lists.append(self.inverted_index[token])
            else:
                posting_lists.append(set())
        
        # Combine posting lists based on operator
        if operator.upper() == 'AND':
            # Intersection of all posting lists
            if not posting_lists:
                result = set()
            else:
                result = posting_lists[0]
                for posting_list in posting_lists[1:]:
                    result = result.intersection(posting_list)
        
        elif operator.upper() == 'OR':
            # Union of all posting lists
            result = set()
            for posting_list in posting_lists:
                result = result.union(posting_list)
        
        else:
            raise ValueError("Operator must be 'AND' or 'OR'")
        
        return list(result)
    
    def get_document(self, doc_id):
        """Get document by ID"""
        return self.documents.get(doc_id)
    
    def save_index(self, filepath):
        """Save index to disk"""
        data = {
            'inverted_index': dict(self.inverted_index),
            'documents': self.documents
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Index saved to {filepath}")
    
    def load_index(self, filepath):
        """Load index from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.inverted_index = defaultdict(set, data['inverted_index'])
        self.documents = data['documents']
        print(f"✓ Index loaded from {filepath}")
        print(f"  Documents: {len(self.documents)}")
        print(f"  Vocabulary: {len(self.inverted_index)} terms")


# Test and build script
if __name__ == "__main__":
    import pandas as pd
    import os
    
    print("="*60)
    print("BOOLEAN RETRIEVAL SYSTEM - BUILD INDEX")
    print("="*60)
    
    # Load data
    print("\n1. Loading dataset...")
    df = pd.read_csv('data/raw/Articles.csv', encoding='latin-1')
    print(f"   Loaded {len(df)} articles")
    
    # Build index
    print("\n2. Building inverted index...")
    boolean_retriever = BooleanRetrieval()
    boolean_retriever.build_index(df)
    
    # Save index
    print("\n3. Saving index...")
    os.makedirs('data/processed', exist_ok=True)
    boolean_retriever.save_index('data/processed/boolean_index.pkl')
    
    # Test queries
    print("\n" + "="*60)
    print("TESTING QUERIES")
    print("="*60)
    
    test_queries = [
        ("cricket match", "AND"),
        ("business economy", "AND"),
        ("cricket football", "OR"),
        ("stock market", "AND")
    ]
    
    for query, operator in test_queries:
        results = boolean_retriever.search(query, operator)
        print(f"\nQuery: '{query}' ({operator})")
        print(f"Results: {len(results)} documents")
        
        # Show top 3 results
        if results:
            print("Top 3 matches:")
            for i, doc_id in enumerate(results[:3], 1):
                doc = boolean_retriever.get_document(doc_id)
                print(f"  {i}. [{doc['news_type']}] {doc['heading'][:80]}...")
    
    print("\n" + "="*60)
    print("✅ Boolean retrieval system ready!")
    print("="*60)
