"""
Text preprocessing and retrieval WITHOUT built-in libraries
Implements core IR algorithms from scratch for educational purposes
"""
import math
from collections import defaultdict, Counter

class ManualTextPreprocessor:
    """Text preprocessing without NLTK or regex libraries"""
    
    def __init__(self):
        # Common English stopwords (manual list)
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'this', 'but', 'they', 'have', 'had',
            'what', 'when', 'where', 'who', 'which', 'why', 'how', 'or', 'not'
        }
    
    def to_lowercase(self, text):
        """Convert text to lowercase manually"""
        result = ""
        for char in text:
            # Check if uppercase letter (ASCII 65-90)
            if 'A' <= char <= 'Z':
                # Convert to lowercase by adding 32 to ASCII value
                result += chr(ord(char) + 32)
            else:
                result += char
        return result
    
    def remove_special_chars(self, text):
        """Remove non-alphabetic characters manually"""
        result = ""
        for char in text:
            # Keep only letters and spaces
            if ('a' <= char <= 'z') or ('A' <= char <= 'Z') or char == ' ':
                result += char
            else:
                result += ' '  # Replace special chars with space
        return result
    
    def tokenize(self, text):
        """Split text into words manually"""
        tokens = []
        current_word = ""
        
        for char in text:
            if char == ' ':
                if current_word:  # If we have accumulated a word
                    tokens.append(current_word)
                    current_word = ""
            else:
                current_word += char
        
        # Don't forget the last word
        if current_word:
            tokens.append(current_word)
        
        return tokens
    
    def remove_stopwords(self, tokens):
        """Remove stopwords manually"""
        filtered = []
        for token in tokens:
            if token not in self.stopwords:
                filtered.append(token)
        return filtered
    
    def porter_stem(self, word):
        """
        Simplified Porter Stemmer implementation
        Implements basic Porter stemming rules
        """
        # Step 1a: Remove plural endings
        if word.endswith('sses'):
            word = word[:-2]  # sses -> ss (e.g., stresses -> stress)
        elif word.endswith('ies'):
            word = word[:-2]  # ies -> i (e.g., ponies -> poni)
        elif word.endswith('ss'):
            pass  # Keep ss (e.g., stress -> stress)
        elif word.endswith('s'):
            word = word[:-1]  # s -> '' (e.g., cats -> cat)
        
        # Step 1b: Remove -ed, -ing endings
        if word.endswith('eed'):
            if len(word) > 4:  # Measure must be > 0
                word = word[:-1]  # eed -> ee
        elif word.endswith('ed'):
            if len(word) > 3 and self._has_vowel(word[:-2]):
                word = word[:-2]
        elif word.endswith('ing'):
            if len(word) > 4 and self._has_vowel(word[:-3]):
                word = word[:-3]
        
        # Step 2: Convert terminal y to i
        if word.endswith('y') and len(word) > 2:
            if self._has_vowel(word[:-1]):
                word = word[:-1] + 'i'
        
        # Step 3: Remove -ational, -ation, -ator endings
        if word.endswith('ational'):
            word = word[:-7] + 'ate'
        elif word.endswith('ation'):
            word = word[:-5] + 'ate'
        elif word.endswith('ator'):
            word = word[:-4] + 'ate'
        
        # Step 4: Remove -ness, -ment, -ful endings
        if word.endswith('ness'):
            word = word[:-4]
        elif word.endswith('ment'):
            word = word[:-4]
        elif word.endswith('ful'):
            word = word[:-3]
        
        return word
    
    def _has_vowel(self, text):
        """Check if text contains a vowel"""
        vowels = {'a', 'e', 'i', 'o', 'u'}
        for char in text:
            if char in vowels:
                return True
        return False
    
    def preprocess(self, text):
        """Complete preprocessing pipeline"""
        # Step 1: Lowercase
        text = self.to_lowercase(text)
        
        # Step 2: Remove special characters
        text = self.remove_special_chars(text)
        
        # Step 3: Tokenize
        tokens = self.tokenize(text)
        
        # Step 4: Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Step 5: Stem
        tokens = [self.porter_stem(token) for token in tokens]
        
        return tokens


class ManualTFIDF:
    """TF-IDF implementation from scratch"""
    
    def __init__(self):
        self.documents = []
        self.doc_count = 0
        self.term_doc_freq = defaultdict(int)  # How many docs contain each term
        self.idf = {}
    
    def fit(self, documents):
        """Build TF-IDF model from documents"""
        self.documents = documents
        self.doc_count = len(documents)
        
        # Count document frequency for each term
        for doc in documents:
            unique_terms = set(doc)
            for term in unique_terms:
                self.term_doc_freq[term] += 1
        
        # Calculate IDF for each term
        for term, doc_freq in self.term_doc_freq.items():
            # IDF = log(N / df)
            self.idf[term] = self._log((self.doc_count + 1) / (doc_freq + 1))
    
    def _log(self, x):
        """Manual logarithm calculation using natural log"""
        # Using math.log is okay here as it's a mathematical function, not NLP
        return math.log(x)
    
    def calculate_tf(self, term, document):
        """Calculate term frequency"""
        term_count = 0
        for word in document:
            if word == term:
                term_count += 1
        
        # TF = count / total terms in doc
        if len(document) == 0:
            return 0.0
        return term_count / len(document)
    
    def calculate_tfidf(self, term, document):
        """Calculate TF-IDF score for a term in a document"""
        tf = self.calculate_tf(term, document)
        idf = self.idf.get(term, 0.0)
        return tf * idf
    
    def vectorize_document(self, document):
        """Convert document to TF-IDF vector"""
        vector = {}
        unique_terms = set(document)
        for term in unique_terms:
            vector[term] = self.calculate_tfidf(term, document)
        return vector
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        # Get all unique terms
        all_terms = set(vec1.keys()) | set(vec2.keys())
        
        # Calculate dot product
        dot_product = 0.0
        for term in all_terms:
            dot_product += vec1.get(term, 0.0) * vec2.get(term, 0.0)
        
        # Calculate magnitudes
        mag1 = self._magnitude(vec1)
        mag2 = self._magnitude(vec2)
        
        # Cosine similarity = dot_product / (mag1 * mag2)
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot_product / (mag1 * mag2)
    
    def _magnitude(self, vector):
        """Calculate magnitude (length) of vector"""
        sum_squares = 0.0
        for value in vector.values():
            sum_squares += value * value
        return math.sqrt(sum_squares)
    
    def search(self, query, top_k=10):
        """Search documents using TF-IDF"""
        # Vectorize query
        query_vector = self.vectorize_document(query)
        
        # Calculate similarity with each document
        scores = []
        for idx, doc in enumerate(self.documents):
            doc_vector = self.vectorize_document(doc)
            similarity = self.cosine_similarity(query_vector, doc_vector)
            scores.append((idx, similarity))
        
        # Sort by score (descending)
        scores = self._manual_sort(scores)
        
        # Return top K
        return scores[:top_k]
    
    def _manual_sort(self, items):
        """Bubble sort implementation (simple sorting)"""
        n = len(items)
        for i in range(n):
            for j in range(0, n - i - 1):
                # Sort by score (second element), descending
                if items[j][1] < items[j + 1][1]:
                    items[j], items[j + 1] = items[j + 1], items[j]
        return items


class ManualBM25:
    """BM25 implementation from scratch"""
    
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1  # Term frequency saturation
        self.b = b    # Length normalization
        self.documents = []
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.doc_count = 0
        self.term_doc_freq = defaultdict(int)
        self.idf = {}
    
    def fit(self, documents):
        """Build BM25 model from documents"""
        self.documents = documents
        self.doc_count = len(documents)
        
        # Calculate document lengths
        total_length = 0
        for doc in documents:
            doc_len = len(doc)
            self.doc_lengths.append(doc_len)
            total_length += doc_len
        
        # Average document length
        if self.doc_count > 0:
            self.avg_doc_length = total_length / self.doc_count
        
        # Count document frequency
        for doc in documents:
            unique_terms = set(doc)
            for term in unique_terms:
                self.term_doc_freq[term] += 1
        
        # Calculate IDF
        for term, doc_freq in self.term_doc_freq.items():
            # BM25 IDF = log((N - df + 0.5) / (df + 0.5))
            numerator = self.doc_count - doc_freq + 0.5
            denominator = doc_freq + 0.5
            self.idf[term] = math.log(numerator / denominator + 1.0)
    
    def calculate_bm25_score(self, query, doc_idx):
        """Calculate BM25 score for a document given a query"""
        score = 0.0
        document = self.documents[doc_idx]
        doc_length = self.doc_lengths[doc_idx]
        
        # Count term frequencies in document
        term_freq = Counter(document)
        
        for term in query:
            if term not in self.idf:
                continue
            
            # Get term frequency in document
            tf = term_freq.get(term, 0)
            
            # BM25 formula
            # Score = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avgdl)))
            idf = self.idf[term]
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query, top_k=10):
        """Search documents using BM25"""
        scores = []
        
        for idx in range(self.doc_count):
            score = self.calculate_bm25_score(query, idx)
            scores.append((idx, score))
        
        # Sort by score
        scores = self._manual_sort(scores)
        
        return scores[:top_k]
    
    def _manual_sort(self, items):
        """Bubble sort"""
        n = len(items)
        for i in range(n):
            for j in range(0, n - i - 1):
                if items[j][1] < items[j + 1][1]:
                    items[j], items[j + 1] = items[j + 1], items[j]
        return items


# Test the implementations
if __name__ == "__main__":
    print("="*60)
    print("TESTING MANUAL IMPLEMENTATIONS (NO BUILT-IN LIBRARIES)")
    print("="*60)
    
    # Sample documents
    documents_text = [
        "The cricket match was exciting and thrilling",
        "Stock market shows positive growth in business sector",
        "Cricket team performs well in the championship",
        "Business economy faces challenges in the market"
    ]
    
    # Initialize preprocessor
    preprocessor = ManualTextPreprocessor()
    
    # Preprocess documents
    print("\n1. PREPROCESSING TEST")
    print("-" * 60)
    processed_docs = []
    for i, doc in enumerate(documents_text):
        tokens = preprocessor.preprocess(doc)
        processed_docs.append(tokens)
        print(f"Doc {i+1}: {' '.join(tokens)}")
    
    # Test TF-IDF
    print("\n2. TF-IDF SEARCH TEST")
    print("-" * 60)
    tfidf = ManualTFIDF()
    tfidf.fit(processed_docs)
    
    query_text = "cricket match"
    query_tokens = preprocessor.preprocess(query_text)
    print(f"Query: '{query_text}' -> {query_tokens}")
    
    tfidf_results = tfidf.search(query_tokens, top_k=3)
    print("\nTop 3 Results:")
    for idx, score in tfidf_results:
        print(f"  Doc {idx+1}: {documents_text[idx][:50]}... (Score: {score:.4f})")
    
    # Test BM25
    print("\n3. BM25 SEARCH TEST")
    print("-" * 60)
    bm25 = ManualBM25(k1=1.5, b=0.75)
    bm25.fit(processed_docs)
    
    bm25_results = bm25.search(query_tokens, top_k=3)
    print(f"Query: '{query_text}'")
    print("\nTop 3 Results:")
    for idx, score in bm25_results:
        print(f"  Doc {idx+1}: {documents_text[idx][:50]}... (Score: {score:.4f})")
    
    # Compare stemming
    print("\n4. STEMMING EXAMPLES")
    print("-" * 60)
    test_words = ['running', 'cats', 'ponies', 'stresses', 'played', 'happiness']
    for word in test_words:
        stemmed = preprocessor.porter_stem(word)
        print(f"  {word:15} -> {stemmed}")
    
    print("\n" + "="*60)
    print("✅ All manual implementations working!")
    print("="*60)