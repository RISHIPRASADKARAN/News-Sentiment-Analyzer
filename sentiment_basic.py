# Import libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# Download stopwords (first time only)
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load dataset
df = pd.read_csv("all-data.csv", encoding="latin1", header=None)
df.columns = ["sentiment", "news_text"]  # Rename columns
print("Dataset Loaded:")
print(df.head())

# Clean text function
def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())  # Remove punctuation & lowercase
    text = " ".join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text

# Apply cleaning
df["cleaned_text"] = df["news_text"].apply(clean_text)
print("\nCleaned Data:")
print(df[["news_text", "cleaned_text"]].head())

# Vectorize text
vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df["cleaned_text"])
y = df["sentiment"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("\nLabel Encoding:")
print(f"Classes: {label_encoder.classes_}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
print("\nData Split:")
print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# Train SVM model
model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)
print("\nModel Trained: SVM with Linear Kernel")

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save model, vectorizer, and label encoder
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("\nModel and components saved as 'sentiment_model.pkl', 'tfidf_vectorizer.pkl', 'label_encoder.pkl'")

# Test prediction function
def predict_sentiment(text):
    cleaned_text = clean_text(text)
    text_tfidf = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_tfidf)
    return label_encoder.inverse_transform(prediction)[0]

# Example predictions
examples = [
    "The company reported a significant increase in revenue this quarter.",
    "Economic downturns have negatively impacted the stock market.",
    "The weather forecast predicts a neutral impact on farming production."
]
print("\nExample Predictions:")
for text in examples:
    sentiment = predict_sentiment(text)
    print(f"Text: {text}")
    print(f"Predicted Sentiment: {sentiment}\n")