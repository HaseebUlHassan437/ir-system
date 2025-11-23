"""Boolean retrieval indexer"""
from .base_indexer import BaseIndexer

class BooleanIndexer(BaseIndexer):
    """Boolean inverted index"""
    
    def __init__(self):
        self.inverted_index = {}
    
    def build_index(self, documents):
        # TODO: Build inverted index
        pass
    
    def save_index(self, path):
        # TODO: Save index
        pass
    
    def load_index(self, path):
        # TODO: Load index
        pass
