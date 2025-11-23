"""Simple preprocessing pipeline"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

class TextPreprocessor:
    def __init__(self, use_stemming=True, remove_stopwords=True):
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess(self, text):
        """Clean and preprocess text"""
        # Lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        
        # Stem
        if self.use_stemming:
            tokens = [self.stemmer.stem(t) for t in tokens]
        
        return tokens