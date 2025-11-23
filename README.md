# Information Retrieval System

A comprehensive local information retrieval (IR) system implementing multiple retrieval strategies: Boolean, TF-IDF, BM25, and Hybrid approaches. Built for CS 516 - Information Retrieval and Text Mining course at ITU.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Installation & Setup](#installation--setup)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Evaluation Results](#evaluation-results)
- [Key Findings](#key-findings)
- [Contributors](#contributors)

---

## 🎯 Project Overview

This project implements and evaluates five information retrieval systems:

1. **Boolean Retrieval** - Inverted index with AND/OR operators
2. **TF-IDF Retrieval** - Vector space model with cosine similarity
3. **BM25 Retrieval** - Probabilistic ranking model
4. **Hybrid Two-Stage** - Boolean + BM25 re-ranking
5. **Hybrid Fusion** - TF-IDF + BM25 score combination

### Key Features
- ✅ Local implementation (no cloud dependencies)
- ✅ Fully reproducible pipeline
- ✅ Comprehensive evaluation with multiple metrics
- ✅ Professional visualizations
- ✅ Comparative analysis

---

## 🏗️ System Architecture

```
Raw Documents → Preprocessing → Indexing → Retrieval → Ranking → Results
                    ↓              ↓          ↓
              [Tokenize]    [Boolean]    [Boolean Search]
              [Lowercase]   [TF-IDF]     [TF-IDF Search]
              [Stopwords]   [BM25]       [BM25 Search]
              [Stemming]    [Hybrid]     [Hybrid Search]
                                              ↓
                                         Evaluation
                                         (P@K, R@K, MAP, NDCG)
```

---

## 📊 Dataset

**Source:** [News Articles Dataset from Kaggle](https://www.kaggle.com/datasets/asad1m9a9h6mood/news-articles)

**Description:**
- 2,692 news articles from The News Pakistan (2015-present)
- Categories: Business (50%) and Sports (50%)
- Columns: Article, Date, Heading, NewsType

**Statistics:**
- Total documents: 2,692
- Average article length: 179 words
- Vocabulary size: 37,375 unique terms (Boolean), 13,531 (TF-IDF/BM25)

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/HaseebUlHassan437/ir-system.git
cd ir-system
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- numpy>=1.24.0
- pandas>=2.0.0
- scikit-learn>=1.3.0
- matplotlib>=3.7.0
- seaborn>=0.12.0
- rank-bm25>=0.2.2
- gensim>=4.3.0
- nltk>=3.8.0
- kagglehub>=0.2.0
- pyyaml>=6.0
- tqdm>=4.65.0

### Step 4: Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
```

---

## 📁 Project Structure

```
ir-system/
├── data/
│   ├── raw/                          # Original dataset
│   │   └── Articles.csv
│   ├── processed/                    # Processed indices
│   │   ├── boolean_index.pkl
│   │   ├── tfidf_index.pkl
│   │   └── bm25_index.pkl
│   └── queries/                      # Test queries
│       ├── queries.txt
│       └── relevance.txt
│
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── pipeline.py              # Text preprocessing
│   │
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── boolean_indexer.py       # Boolean retrieval
│   │   ├── tfidf_indexer.py         # TF-IDF retrieval
│   │   └── bm25_indexer.py          # BM25 retrieval
│   │
│   ├── hybrid/
│   │   ├── __init__.py
│   │   └── hybrid_retrieval.py      # Hybrid systems
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py             # Evaluation metrics
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── file_handler.py
│
├── experiments/
│   ├── results/
│   │   ├── test_queries.json
│   │   ├── relevance_judgments.json
│   │   ├── system_comparison.csv
│   │   └── system_comparison_with_hybrid.csv
│   │
│   └── plots/                        # Generated visualizations
│       ├── precision_comparison.png
│       ├── recall_comparison.png
│       ├── map_ndcg_comparison.png
│       └── [6 more charts]
│
├── notebooks/
│   └── 01_data_exploration.ipynb    # Data analysis
│
├── scripts/
│   ├── download_data.py             # Dataset download
│   └── generate_visualizations.py   # Create charts
│
├── configs/
│   ├── boolean_config.yaml
│   ├── tfidf_config.yaml
│   ├── bm25_config.yaml
│   └── hybrid_config.yaml
│
├── docs/
│   └── ai_usage/                    # AI assistance screenshots
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📖 Usage Guide

### Step 1: Download Dataset

```bash
python scripts/download_data.py
```

**Output:**
- Downloads dataset from Kaggle
- Saves to `data/raw/Articles.csv`
- Displays dataset statistics

### Step 2: Explore Data (Optional)

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

**What it does:**
- Loads and analyzes the dataset
- Shows statistics, distributions, samples
- Generates exploratory visualizations

### Step 3: Build Retrieval Indices

#### Boolean Index
```bash
python src/indexing/boolean_indexer.py
```
- Builds inverted index
- Saves to `data/processed/boolean_index.pkl`
- Index size: ~6.19 MB

#### TF-IDF Index
```bash
python src/indexing/tfidf_indexer.py
```
- Computes TF-IDF vectors
- Saves to `data/processed/tfidf_index.pkl`
- Index size: ~8.59 MB

#### BM25 Index
```bash
python src/indexing/bm25_indexer.py
```
- Builds BM25 scoring model
- Saves to `data/processed/bm25_index.pkl`
- Index size: ~11.45 MB

### Step 4: Evaluate Base Systems

```bash
python src/evaluation/evaluator.py
```

**What it does:**
- Creates 10 test queries
- Generates pseudo-relevance judgments
- Evaluates Boolean, TF-IDF, and BM25
- Calculates: MAP, P@K, R@K, NDCG@K, Query Time
- Saves results to `experiments/results/system_comparison.csv`

### Step 5: Build and Evaluate Hybrid Systems

```bash
python src/hybrid/hybrid_retrieval.py
```

**What it does:**
- Creates two hybrid approaches:
  - **Two-Stage:** Boolean → BM25 re-ranking
  - **Score Fusion:** Weighted TF-IDF + BM25
- Evaluates hybrid systems
- Saves results to `experiments/results/system_comparison_with_hybrid.csv`

### Step 6: Generate Visualizations

```bash
python scripts/generate_visualizations.py
```

**Generates 8 charts:**
1. Precision comparison (P@5, P@10, P@20)
2. Recall comparison (R@5, R@10, R@20)
3. MAP and NDCG comparison
4. Query time comparison
5. Precision-Recall trade-off
6. Performance radar chart
7. Speed vs Accuracy scatter plot
8. Performance summary table

**Output location:** `experiments/plots/`

---

## 📊 Evaluation Results

### Performance Summary

| System | MAP | P@10 | R@10 | NDCG@10 | Query Time (ms) |
|--------|-----|------|------|---------|-----------------|
| **TF-IDF** | **0.0394** | **1.000** | 0.0200 | **1.000** | 6.60 |
| **Hybrid-Fusion** | 0.0387 | 0.990 | 0.0197 | 0.9915 | 9.32 |
| **BM25** | 0.0385 | 0.980 | 0.0193 | 0.9827 | **1.92** |
| Hybrid-TwoStage | 0.0331 | 0.790 | 0.0172 | 0.7915 | 2.35 |
| Boolean | 0.0286 | 0.660 | 0.0155 | 0.6622 | **0.10** |

### Key Metrics Explained

- **MAP (Mean Average Precision):** Overall ranking quality (0-1, higher is better)
- **P@K (Precision at K):** Percentage of relevant docs in top K results
- **R@K (Recall at K):** Percentage of all relevant docs retrieved in top K
- **NDCG@K:** Normalized ranking quality considering position (0-1, higher is better)
- **Query Time:** Average time to process a query in milliseconds

### Memory Usage

| System | Index Size |
|--------|------------|
| Boolean | 6.19 MB |
| TF-IDF | 8.59 MB |
| BM25 | 11.45 MB |

---

## 🔍 Key Findings

### 1. Accuracy
- **Winner:** TF-IDF (100% precision at all K values)
- BM25 and Hybrid-Fusion are close seconds (98-99%)
- Boolean significantly lower (66%)

### 2. Speed
- **Winner:** Boolean (0.10 ms - 60x faster than TF-IDF)
- BM25 offers best balance (1.92 ms with 98% precision)
- Hybrid-Fusion is slowest (9.32 ms - sum of both components)

### 3. Trade-offs
- **TF-IDF:** Best accuracy, moderate speed, medium memory
- **BM25:** Excellent balance of speed and accuracy
- **Boolean:** Extremely fast but less accurate
- **Hybrid-Fusion:** Slight improvement over base systems but slower

### 4. Recommendations
- **For accuracy-critical applications:** Use TF-IDF
- **For real-time search:** Use BM25 (best speed/accuracy balance)
- **For large-scale systems:** Consider Boolean first-pass with BM25 re-ranking

---

## 🛠️ Implementation Details

### Preprocessing Pipeline
```python
from src.preprocessing.pipeline import TextPreprocessor

preprocessor = TextPreprocessor(
    use_stemming=True,        # Porter Stemmer
    remove_stopwords=True     # NLTK English stopwords
)

tokens = preprocessor.preprocess("Your text here")
```

**Steps:**
1. Lowercase conversion
2. Special character removal
3. Tokenization (NLTK word_tokenize)
4. Stopword removal
5. Stemming (Porter Stemmer)

### Boolean Retrieval
- **Method:** Inverted index with set operations
- **Operators:** AND, OR
- **Time Complexity:** O(n) where n = number of postings

### TF-IDF Retrieval
- **Implementation:** Sklearn TfidfVectorizer
- **Similarity:** Cosine similarity
- **Parameters:** 
  - min_df=2 (ignore terms in < 2 docs)
  - max_df=0.95 (ignore terms in > 95% docs)

### BM25 Retrieval
- **Implementation:** rank-bm25 library
- **Parameters:**
  - k1=1.5 (term frequency saturation)
  - b=0.75 (length normalization)

### Hybrid Systems
- **Two-Stage:** Boolean retrieves 100 candidates → BM25 ranks them
- **Score Fusion:** Normalized TF-IDF + BM25 scores (50% weight each)

---

## 🔬 Evaluation Methodology

### Test Queries (10 queries)
```
q1: cricket match
q2: stock market
q3: business economy
q4: football world cup
q5: pakistan team
q6: company profit
q7: championship tournament
q8: financial crisis
q9: player performance
q10: trade investment
```

### Relevance Judgments
- **Method:** Pseudo-relevance using NewsType labels + keyword matching
- **Logic:** Document is relevant if:
  1. Matches query category (sports/business)
  2. Contains at least one query term (after preprocessing)

### Metrics Calculated
- **MAP:** Mean Average Precision across all queries
- **P@K:** Precision at K=5, 10, 20
- **R@K:** Recall at K=5, 10, 20
- **NDCG@K:** Normalized Discounted Cumulative Gain at K=5, 10, 20
- **Query Time:** Average execution time per query
- **Memory:** Index file size on disk

---

## 🧪 Running Custom Queries

### Boolean Search
```python
from src.indexing.boolean_indexer import BooleanRetrieval

retriever = BooleanRetrieval()
retriever.load_index('data/processed/boolean_index.pkl')

results = retriever.search("cricket match", operator='AND')
print(f"Found {len(results)} documents")
```

### TF-IDF Search
```python
from src.indexing.tfidf_indexer import TFIDFRetrieval

retriever = TFIDFRetrieval()
retriever.load_index('data/processed/tfidf_index.pkl')

results = retriever.search("stock market", top_k=10)
for doc_id, score in results:
    doc = retriever.get_document(doc_id)
    print(f"Score: {score:.4f} - {doc['heading']}")
```

### BM25 Search
```python
from src.indexing.bm25_indexer import BM25Retrieval

retriever = BM25Retrieval()
retriever.load_index('data/processed/bm25_index.pkl')

results = retriever.search("football world cup", top_k=10)
for doc_id, score in results:
    doc = retriever.get_document(doc_id)
    print(f"Score: {score:.4f} - {doc['heading']}")
```

---

## 📈 Reproducing Results

To completely reproduce the evaluation:

```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Download data
python scripts/download_data.py

# 3. Build all indices
python src/indexing/boolean_indexer.py
python src/indexing/tfidf_indexer.py
python src/indexing/bm25_indexer.py

# 4. Evaluate base systems
python src/evaluation/evaluator.py

# 5. Evaluate hybrid systems
python src/hybrid/hybrid_retrieval.py

# 6. Generate visualizations
python scripts/generate_visualizations.py
```

**Expected runtime:** ~5-10 minutes total

**Results location:**
- CSV files: `experiments/results/`
- Charts: `experiments/plots/`

---

## 🐛 Troubleshooting

### Issue: Module not found error
```
ModuleNotFoundError: No module named 'src'
```
**Solution:** Run scripts from project root directory, not from subdirectories.

### Issue: Encoding error when reading CSV
```
UnicodeDecodeError: 'utf-8' codec can't decode...
```
**Solution:** Already handled in code with `encoding='latin-1'` parameter.

### Issue: NLTK data not found
```
LookupError: Resource punkt not found
```
**Solution:** 
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Issue: Kaggle API authentication
**Solution:** The kagglehub library handles authentication automatically on first use.

---

## 📚 References

1. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.

2. Robertson, S. E., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333-389.

3. Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, 24(5), 513-523.

4. Scikit-learn: Machine Learning in Python. Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

5. rank-bm25 Python library: https://github.com/dorianbrown/rank_bm25

6. NLTK: Bird, Steven, Edward Loper and Ewan Klein (2009). *Natural Language Processing with Python*. O'Reilly Media Inc.

---

## 👤 Contributors

- **Haseeb Ul Hassan**
  - Email: [mscs25003@itu.edu.pk]
  - GitHub: [@HaseebUlHassan437](https://github.com/HaseebUlHassan437)

---

## 📄 License

This project is submitted as coursework for CS 516 at ITU. All rights reserved.

---

## 🙏 Acknowledgments

- Information Technology University (ITU)
- Dr. Ahmad Mustafa (Course Instructor)
- The News Pakistan (Dataset source)
- Kaggle community
- Open-source contributors (NLTK, scikit-learn, matplotlib, pandas)

---

## 📝 AI Usage Disclosure

This project used AI assistance (Claude by Anthropic) for:
- Code structure and implementation guidance
- Documentation writing
- Debugging assistance

All AI interactions are documented with screenshots in `docs/ai_usage/` directory as required by the assignment guidelines.

---

**Last Updated:** November 23, 2025

**Course:** CS 516 - Information Retrieval and Text Mining  
**Institution:** Information Technology University (ITU)  
**Semester:** Fall 2025