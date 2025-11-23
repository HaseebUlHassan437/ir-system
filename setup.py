import os

def create_project_structure():
    """
    Creates the complete IR system project structure
    """
    
    # Define the directory structure
    directories = [
        "data/raw",
        "data/processed",
        "data/queries",
        "src/preprocessing",
        "src/indexing",
        "src/retrieval",
        "src/hybrid",
        "src/evaluation",
        "src/utils",
        "experiments/results",
        "experiments/plots",
        "experiments/logs",
        "notebooks",
        "tests",
        "scripts",
        "configs",
        "docs/ai_usage",
        "report/draft/sections",
        "report/draft/figures",
    ]
    
    # Define files to create (path: content)
    files = {
        # Root files
        "README.md": """# Information Retrieval System

## Project Overview
A local IR system implementing multiple retrieval strategies.

## Setup Instructions
1. Install dependencies: `pip install -r requirements.txt`
2. Download dataset: `python scripts/download_data.py`
3. Build indices: `python scripts/build_index.py`
4. Run experiments: `python scripts/run_experiments.py`

## System Architecture
[Add diagram here]

## Evaluation Results
[Add results here]
""",
        
        "requirements.txt": """# Core dependencies
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0

# IR libraries
rank-bm25>=0.2.2
gensim>=4.3.0

# NLP
nltk>=3.8.0

# Utilities
pyyaml>=6.0
tqdm>=4.65.0

# Jupyter
jupyter>=1.0.0
ipykernel>=6.25.0
""",
        
        ".gitignore": """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
*.egg-info/
dist/
build/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Data (large files)
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep

# Experiments
experiments/logs/*
!experiments/logs/.gitkeep

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Reports (don't commit drafts)
report/draft/
""",
        
        # Source __init__ files
        "src/__init__.py": "",
        "src/preprocessing/__init__.py": "",
        "src/indexing/__init__.py": "",
        "src/retrieval/__init__.py": "",
        "src/hybrid/__init__.py": "",
        "src/evaluation/__init__.py": "",
        "src/utils/__init__.py": "",
        "tests/__init__.py": "",
        
        # Preprocessing files
        "src/preprocessing/cleaner.py": """\"\"\"Text cleaning and normalization\"\"\"

def clean_text(text):
    \"\"\"
    Clean and normalize text
    Args:
        text (str): Raw text
    Returns:
        str: Cleaned text
    \"\"\"
    # TODO: Implement cleaning logic
    pass
""",
        
        "src/preprocessing/tokenizer.py": """\"\"\"Tokenization utilities\"\"\"

def tokenize(text):
    \"\"\"
    Tokenize text into words
    Args:
        text (str): Input text
    Returns:
        list: List of tokens
    \"\"\"
    # TODO: Implement tokenization
    pass
""",
        
        "src/preprocessing/stopwords.py": """\"\"\"Stopword removal\"\"\"

def remove_stopwords(tokens):
    \"\"\"
    Remove stopwords from token list
    Args:
        tokens (list): List of tokens
    Returns:
        list: Filtered tokens
    \"\"\"
    # TODO: Implement stopword removal
    pass
""",
        
        # Indexing files
        "src/indexing/base_indexer.py": """\"\"\"Base class for all indexers\"\"\"
from abc import ABC, abstractmethod

class BaseIndexer(ABC):
    \"\"\"Abstract base class for indexers\"\"\"
    
    @abstractmethod
    def build_index(self, documents):
        \"\"\"Build index from documents\"\"\"
        pass
    
    @abstractmethod
    def save_index(self, path):
        \"\"\"Save index to disk\"\"\"
        pass
    
    @abstractmethod
    def load_index(self, path):
        \"\"\"Load index from disk\"\"\"
        pass
""",
        
        "src/indexing/boolean_indexer.py": """\"\"\"Boolean retrieval indexer\"\"\"
from .base_indexer import BaseIndexer

class BooleanIndexer(BaseIndexer):
    \"\"\"Boolean inverted index\"\"\"
    
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
""",
        
        "src/indexing/tfidf_indexer.py": """\"\"\"TF-IDF indexer\"\"\"
from .base_indexer import BaseIndexer

class TFIDFIndexer(BaseIndexer):
    \"\"\"TF-IDF vector space model\"\"\"
    
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
""",
        
        "src/indexing/bm25_indexer.py": """\"\"\"BM25 indexer\"\"\"
from .base_indexer import BaseIndexer

class BM25Indexer(BaseIndexer):
    \"\"\"BM25 probabilistic model\"\"\"
    
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
""",
        
        # Evaluation files
        "src/evaluation/metrics.py": """\"\"\"Evaluation metrics\"\"\"

def precision_at_k(retrieved, relevant, k):
    \"\"\"Calculate Precision@K\"\"\"
    # TODO: Implement
    pass

def recall_at_k(retrieved, relevant, k):
    \"\"\"Calculate Recall@K\"\"\"
    # TODO: Implement
    pass

def mean_average_precision(all_retrieved, all_relevant):
    \"\"\"Calculate MAP\"\"\"
    # TODO: Implement
    pass
""",
        
        "src/evaluation/evaluator.py": """\"\"\"Main evaluator class\"\"\"

class Evaluator:
    \"\"\"Evaluate IR system performance\"\"\"
    
    def __init__(self, retriever, queries, relevance_judgments):
        self.retriever = retriever
        self.queries = queries
        self.relevance_judgments = relevance_judgments
    
    def evaluate(self):
        \"\"\"Run full evaluation\"\"\"
        # TODO: Implement evaluation pipeline
        pass
""",
        
        # Utils files
        "src/utils/config.py": """\"\"\"Configuration management\"\"\"
import yaml

def load_config(config_path):
    \"\"\"Load configuration from YAML file\"\"\"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
""",
        
        "src/utils/file_handler.py": """\"\"\"File reading and writing utilities\"\"\"

def read_documents(data_dir):
    \"\"\"Read documents from directory\"\"\"
    # TODO: Implement
    pass

def save_results(results, output_path):
    \"\"\"Save results to JSON\"\"\"
    # TODO: Implement
    pass
""",
        
        # Scripts
        "scripts/download_data.py": """\"\"\"Download dataset\"\"\"

def download_dataset():
    \"\"\"Download and extract the dataset\"\"\"
    print("Downloading dataset...")
    # TODO: Implement download logic
    print("Dataset downloaded to data/raw/")

if __name__ == "__main__":
    download_dataset()
""",
        
        "scripts/build_index.py": """\"\"\"Build all indices\"\"\"

def build_all_indices():
    \"\"\"Build Boolean, TF-IDF, and BM25 indices\"\"\"
    print("Building indices...")
    # TODO: Implement
    print("All indices built successfully!")

if __name__ == "__main__":
    build_all_indices()
""",
        
        "scripts/run_experiments.py": """\"\"\"Run all experiments and evaluations\"\"\"

def run_all_experiments():
    \"\"\"Run experiments for all retrieval systems\"\"\"
    print("Running experiments...")
    # TODO: Implement
    print("Experiments complete! Results saved to experiments/results/")

if __name__ == "__main__":
    run_all_experiments()
""",
        
        # Config files
        "configs/boolean_config.yaml": """# Boolean Retrieval Configuration
system_name: "Boolean"
preprocessing:
  lowercase: true
  remove_stopwords: true
  stemming: false
""",
        
        "configs/tfidf_config.yaml": """# TF-IDF Configuration
system_name: "TF-IDF"
preprocessing:
  lowercase: true
  remove_stopwords: true
  stemming: true
tfidf:
  max_features: null
  min_df: 2
  max_df: 0.95
""",
        
        "configs/bm25_config.yaml": """# BM25 Configuration
system_name: "BM25"
preprocessing:
  lowercase: true
  remove_stopwords: true
  stemming: true
bm25:
  k1: 1.5
  b: 0.75
""",
        
        # Data files
        "data/queries/queries.txt": """# Sample queries (one per line)
information retrieval systems
text mining techniques
""",
        
        "data/queries/relevance.txt": """# Relevance judgments
# Format: query_id doc_id relevance_score
""",
        
        # Keep files for empty directories
        "data/raw/.gitkeep": "",
        "data/processed/.gitkeep": "",
        "experiments/logs/.gitkeep": "",
    }
    
    print("Creating IR System Project Structure...")
    print("=" * 50)
    
    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Create files
    for filepath, content in files.items():
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Created file: {filepath}")
    
    print("=" * 50)
    print("✅ Project structure created successfully!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Start with data exploration")
    print("3. Implement preprocessing pipeline")

if __name__ == "__main__":
    create_project_structure()