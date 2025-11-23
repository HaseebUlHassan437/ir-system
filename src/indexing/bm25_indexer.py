"""BM25 indexer"""
from .base_indexer import BaseIndexer

class BM25Indexer(BaseIndexer):
    """BM25 probabilistic model"""
    
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.bm25 = None
    
    def build_index(self, documents):
        # TODO: Build BM25 index
        pass
    
    def save_index(self, path):
        # TODO: Save index
        pass
    
    def load_index(self, path):
        # TODO: Load index
        pass
