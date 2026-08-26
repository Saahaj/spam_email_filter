from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

def train_model(X, y):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the Naive Bayes classifier
    model = MultinomialNB()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Save the model and test data
    joblib.dump(model, PROJECT_ROOT / 'spam_detector_model.pkl')
    joblib.dump(X_test, PROJECT_ROOT / 'X_test.pkl')
    joblib.dump(y_test, PROJECT_ROOT / 'y_test.pkl')
    
    return model, X_test, y_test

if __name__ == "__main__":
    # Load the preprocessed data
    df = pd.read_csv(PROJECT_ROOT / 'data' / 'preprocessed_emails.csv')
    
    # Load the feature matrix
    X = joblib.load(PROJECT_ROOT / 'X_features.pkl')
    
    # Extract labels
    y = df['spam']
    
    # Train the model and save it
    model, X_test, y_test = train_model(X, y)
    
    print("Model trained and saved.")
