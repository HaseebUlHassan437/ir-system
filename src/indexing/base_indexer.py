"""Base class for all indexers"""
from abc import ABC, abstractmethod

class BaseIndexer(ABC):
    """Abstract base class for indexers"""
    
    @abstractmethod
    def build_index(self, documents):
        """Build index from documents"""
        pass
    
    @abstractmethod
    def save_index(self, path):
        """Save index to disk"""
        pass
    
    @abstractmethod
    def load_index(self, path):
        """Load index from disk"""
        pass
