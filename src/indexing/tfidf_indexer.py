"""TF-IDF indexer"""
from .base_indexer import BaseIndexer

class TFIDFIndexer(BaseIndexer):
    """TF-IDF vector space model"""
    
    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
    
    def build_index(self, documents):
        # TODO: Build TF-IDF index
        pass
    
    def save_index(self, path):
        # TODO: Save index
        pass
    
    def load_index(self, path):
        # TODO: Load index
        pass
