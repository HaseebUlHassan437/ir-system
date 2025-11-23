"""Main evaluator class"""

class Evaluator:
    """Evaluate IR system performance"""
    
    def __init__(self, retriever, queries, relevance_judgments):
        self.retriever = retriever
        self.queries = queries
        self.relevance_judgments = relevance_judgments
    
    def evaluate(self):
        """Run full evaluation"""
        # TODO: Implement evaluation pipeline
        pass
