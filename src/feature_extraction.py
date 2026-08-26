from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import joblib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

def extract_features(texts):
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=3000)
    # Fit and transform the texts
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

if __name__ == "__main__":
    # Load the preprocessed data
    df = pd.read_csv(PROJECT_ROOT / 'data' / 'preprocessed_emails.csv')
    
    # Extract features from the 'text' column
    X, vectorizer = extract_features(df['text'])
    
    # Save the vectorizer and feature matrix
    joblib.dump(vectorizer, PROJECT_ROOT / 'tfidf_vectorizer.pkl')
    joblib.dump(X, PROJECT_ROOT / 'X_features.pkl')
    
    # Print the shape of the feature matrix
    print("Feature matrix shape:", X.shape)
