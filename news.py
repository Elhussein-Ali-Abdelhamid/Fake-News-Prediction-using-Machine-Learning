import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk

# Download stopwords if not already downloaded (only once)
nltk.download('stopwords', quiet=True)

def load_and_preprocess_data():
    """Load and preprocess the data efficiently."""
    # Load data and drop unnecessary columns
    df = pd.read_csv("news.csv").drop(columns=["Unnamed: 0"], errors='ignore')
    
    # Map labels to numerical values
    df['label'] = df['label'].map({"FAKE": 0, "REAL": 1})
    
    return df['text'], df['label'].values

def preprocess_text(text_series):
    """Optimized text preprocessing function."""
    # Precompile regex and load resources once
    stop_words = set(stopwords.words('english'))
    port_stem = PorterStemmer()
    non_alpha_re = re.compile('[^a-zA-Z]')
    
    def process_single_text(content):
        """Process a single text document."""
        # Remove non-alphabetic characters
        stemmed_content = non_alpha_re.sub(' ', content)
        stemmed_content = stemmed_content.lower()
        
        # Stem words and remove stopwords in one pass
        return ' '.join(
            port_stem.stem(word) 
            for word in stemmed_content.split() 
            if word not in stop_words and len(word) > 2  # Added length filter
        )
    
    # Use parallel processing if dataset is large
    if len(text_series) > 10000:
        import swifter  # Requires pip install swifter
        return text_series.swifter.apply(process_single_text)
    return text_series.apply(process_single_text)

def train_and_evaluate(X, y):
    """Train and evaluate the model with optimized parameters."""
    # Vectorize text data
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_vectorized = vectorizer.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, stratify=y, random_state=2
    )
    
    # Train model with optimized parameters
    model = LogisticRegression(
        solver='liblinear',  # Faster for small datasets
        max_iter=1000,
        C=0.5,  # Regularization parameter
        n_jobs=-1  # Use all cores
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    print(f'Training accuracy: {train_acc:.4f}')
    print(f'Test accuracy: {test_acc:.4f}')
    
    return model, vectorizer

def main():
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    X_processed = preprocess_text(X)
    
    # Train and evaluate
    model, vectorizer = train_and_evaluate(X_processed, y)

if __name__ == "__main__":
    main()