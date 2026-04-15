# 03_tfidf_baselines.py
# Traditional ML Baselines: TF-IDF + Classifiers on tweet_eval dataset
# March 2026

import re
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ── Configuration ──
SAVE_FOLDER_DATA   = 'data'
SAVE_FOLDER_VISUAL = 'visuals'
os.makedirs(SAVE_FOLDER_DATA,   exist_ok=True)
os.makedirs(SAVE_FOLDER_VISUAL, exist_ok=True)

# ── Load data ──
print("Loading TweetEval sentiment dataset from Hugging Face...")
dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")
df = dataset['train'].to_pandas()
print(f"Loaded {len(df):,} tweets\n")

# ── Basic cleaning function ──
def basic_clean(text):
    """Simple cleaning: remove URLs, mentions, hashtags, normalize spaces"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

# Apply cleaning
print("Applying basic cleaning...")
df['clean_text'] = df['text'].apply(basic_clean)

# ── Prepare features and labels ──
X = df['clean_text']
y = df['label']

# Train-test split (stratified for class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}\n")

# ── TF-IDF Vectorization ──
print("Fitting TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(
    max_features=10000,      # Limit vocabulary size
    ngram_range=(1, 2),      # Unigrams + bigrams
    stop_words='english',
    min_df=2                 # Ignore very rare terms
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

print(f"TF-IDF feature dimension: {X_train_tfidf.shape[1]:,}\n")

# ── Models to evaluate ──
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    "Linear SVC": LinearSVC(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}

results = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_tfidf, y_train)
    
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"{name} Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    print(classification_report(y_test, y_pred, 
                                target_names=['Negative', 'Neutral', 'Positive'], 
                                digits=4))
    
    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Neg', 'Neu', 'Pos'],
                yticklabels=['Neg', 'Neu', 'Pos'])
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    cm_path = f'{SAVE_FOLDER_VISUAL}/{name.lower().replace(" ", "_")}_confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved: {cm_path}\n")
    
    results[name] = acc

print("All traditional ML baselines completed!")
print("Summary of accuracies:")
for name, acc in results.items():
    print(f"  {name}: {acc:.4f}")