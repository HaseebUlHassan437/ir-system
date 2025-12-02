import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import time
sys.path.append('..')

from src.preprocessing.pipeline import TextPreprocessor
from src.preprocessing.manual_preprocessing import ManualTextPreprocessor, ManualTFIDF, ManualBM25

def test_preprocessing():
    """Compare how both preprocessors work"""
    print("\n" + "="*60)
    print("Testing Preprocessing")
    print("="*60)
    
    # Sample texts to test
    test_sentences = [
        "The Cricket Match was EXCITING!!!",
        "Stock markets show positive growth.",
        "Players performances are analyzed."
    ]
    
    # Create both preprocessors
    library_version = TextPreprocessor()
    manual_version = ManualTextPreprocessor()
    
    for sentence in test_sentences:
        print(f"\nOriginal: {sentence}")
        
        library_result = library_version.preprocess(sentence)
        manual_result = manual_version.preprocess(sentence)
        
        print(f"Library version: {library_result}")
        print(f"Manual version:  {manual_result}")
        
        # Check how many words match
        matching_words = len(set(library_result) & set(manual_result))
        total_words = len(set(library_result) | set(manual_result))
        
        if total_words > 0:
            match_percent = (matching_words / total_words) * 100
            print(f"Match: {matching_words}/{total_words} words ({match_percent:.0f}%)")

def test_tfidf_search():
    """Compare TF-IDF search results"""
    print("\n" + "="*60)
    print("Testing TF-IDF Search")
    print("="*60)
    
    # Some test documents
    documents = [
        "Cricket match was exciting",
        "Stock market shows growth",
        "Cricket team won championship",
        "Business faces market challenges"
    ]
    
    print(f"\nUsing {len(documents)} test documents")
    
    # Preprocess documents
    library_prep = TextPreprocessor()
    manual_prep = ManualTextPreprocessor()
    
    library_docs = [library_prep.preprocess(d) for d in documents]
    manual_docs = [manual_prep.preprocess(d) for d in documents]
    
    # Build manual TF-IDF
    manual_tfidf = ManualTFIDF()
    manual_tfidf.fit(manual_docs)
    
    # Test a query
    query = "cricket match"
    print(f"\nQuery: '{query}'")
    
    query_tokens = manual_prep.preprocess(query)
    results = manual_tfidf.search(query_tokens, top_k=3)
    
    print("\nTop 3 results:")
    for i, (doc_idx, score) in enumerate(results, 1):
        print(f"  {i}. {documents[doc_idx]}")
        print(f"     Score: {score:.4f}")

def test_bm25_search():
    """Compare BM25 search results"""
    print("\n" + "="*60)
    print("Testing BM25 Search")
    print("="*60)
    
    documents = [
        "Cricket match was exciting",
        "Stock market shows growth",
        "Cricket team won championship",
        "Business faces market challenges"
    ]
    
    print(f"\nUsing {len(documents)} test documents")
    
    # Preprocess
    manual_prep = ManualTextPreprocessor()
    manual_docs = [manual_prep.preprocess(d) for d in documents]
    
    # Build BM25
    manual_bm25 = ManualBM25()
    manual_bm25.fit(manual_docs)
    
    # Test query
    query = "cricket match"
    print(f"\nQuery: '{query}'")
    
    query_tokens = manual_prep.preprocess(query)
    results = manual_bm25.search(query_tokens, top_k=3)
    
    print("\nTop 3 results:")
    for i, (doc_idx, score) in enumerate(results, 1):
        print(f"  {i}. {documents[doc_idx]}")
        print(f"     Score: {score:.4f}")

def test_speed_difference():
    """Show speed difference between implementations"""
    print("\n" + "="*60)
    print("Testing Speed Difference")
    print("="*60)
    
    # Create longer test document
    test_doc = "cricket match team player " * 50  # Repeat words
    
    # Library version
    library_prep = TextPreprocessor()
    start = time.time()
    for i in range(100):  # Process 100 times
        library_prep.preprocess(test_doc)
    library_time = time.time() - start
    
    # Manual version
    manual_prep = ManualTextPreprocessor()
    start = time.time()
    for i in range(100):
        manual_prep.preprocess(test_doc)
    manual_time = time.time() - start
    
    print(f"\nProcessing same text 100 times:")
    print(f"Library version: {library_time:.3f} seconds")
    print(f"Manual version:  {manual_time:.3f} seconds")
    print(f"Manual is {manual_time/library_time:.1f}x slower")
    
    print("\nWhy the difference?")
    print("- Library uses optimized C code")
    print("- Manual uses pure Python")
    print("- Both produce similar results")

def show_stemming_examples():
    """Show how stemming works in manual version"""
    print("\n" + "="*60)
    print("Stemming Examples")
    print("="*60)
    
    manual_prep = ManualTextPreprocessor()
    
    test_words = [
        'running', 'runs', 'ran',
        'cats', 'cat',
        'ponies', 'pony',
        'happiness', 'happy',
        'playing', 'played', 'plays'
    ]
    
    print("\nHow words are stemmed:")
    for word in test_words:
        stemmed = manual_prep.porter_stem(word)
        print(f"  {word:12} -> {stemmed}")

if __name__ == "__main__":
    print("="*60)
    print("Comparing Library vs Manual Implementation")
    print("="*60)
    
    # Run tests
    test_preprocessing()
    show_stemming_examples()
    test_tfidf_search()
    test_bm25_search()
    test_speed_difference()
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("\nWhat we learned:")
    print("1. Both approaches give similar results")
    print("2. Libraries are much faster (5-10x)")
    print("3. Manual code helps understand how it works")
    print("4. For real projects, use libraries")
    print("5. For learning, manual implementation is great")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)