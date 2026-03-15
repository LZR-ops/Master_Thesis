# Preprocessing + VADER Baseline on full tweet_eval sentiment dataset

import re
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, classification_report
from datasets import load_dataset

# One-time NLTK download
nltk.download('vader_lexicon', quiet=True)

print("Loading full TweetEval sentiment dataset from Hugging Face...")
dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")
df = dataset['train'].to_pandas()  # ~45k rows

print(f"Loaded {len(df):,} tweets")

# ── Preprocessing function ──
def clean_tweet(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove @mentions
    text = re.sub(r'@\w+', '', text)
    # Remove # symbol but keep the word
    text = re.sub(r'#', '', text)
    # Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# Apply cleaning (this may take 10–30 seconds on full dataset)
print("Cleaning tweets...")
df['clean_text'] = df['text'].apply(clean_tweet)

# ── VADER sentiment analysis ──
print("Running VADER...")
sia = SentimentIntensityAnalyzer()

# Compound score: -1 (very negative) to +1 (very positive)
df['vader_compound'] = df['clean_text'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Convert compound score to class labels (0=neg, 1=neu, 2=pos)
def vader_to_class(score):
    if score >= 0.05:
        return 2   # Positive
    elif score <= -0.05:
        return 0   # Negative
    else:
        return 1   # Neutral

df['vader_pred'] = df['vader_compound'].apply(vader_to_class)

# ── Evaluation ──
true_labels = df['label']
pred_labels = df['vader_pred']

accuracy = accuracy_score(true_labels, pred_labels)
print(f"\nVADER Accuracy on full dataset: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(
    true_labels,
    pred_labels,
    target_names=['Negative', 'Neutral', 'Positive'],
    digits=4
))

# Optional: Save predictions for later analysis
print("Saving results...")
df[['text', 'label', 'clean_text', 'vader_compound', 'vader_pred']].to_csv(
    'data/vader_full_results.csv',
    index=False
)
print("Results saved to: data/vader_full_results.csv")

# Add after classification_report

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Neg', 'Neu', 'Pos'],
            yticklabels=['Neg', 'Neu', 'Pos'])
plt.title('VADER Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('data/vader_confusion_matrix.png')
plt.show()

# Bar plot of F1-scores
report = classification_report(true_labels, pred_labels, target_names=['Negative', 'Neutral', 'Positive'], output_dict=True)
f1_scores = [report[c]['f1-score'] for c in ['Negative', 'Neutral', 'Positive']]
plt.figure(figsize=(8, 5))
sns.barplot(x=['Negative', 'Neutral', 'Positive'], y=f1_scores)
plt.title('VADER F1-Score per Class')
plt.ylabel('F1-Score')
plt.ylim(0, 1)
plt.savefig('data/vader_f1_per_class.png')
plt.show()

# ── Visualization: Confusion Matrix & F1-score bar plot ──
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Make sure visuals folder exists
os.makedirs('visuals', exist_ok=True)

# 1. Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(true_labels, pred_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'])
plt.title('VADER Confusion Matrix (Full Dataset)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
cm_path = 'visuals/vader_confusion_matrix.png'
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()  # Close figure to free memory
print(f"Confusion matrix saved to: {os.path.abspath(cm_path)}")

# 2. F1-score per class bar plot
report_dict = classification_report(
    true_labels,
    pred_labels,
    target_names=['Negative', 'Neutral', 'Positive'],
    output_dict=True
)

f1_scores = [
    report_dict['Negative']['f1-score'],
    report_dict['Neutral']['f1-score'],
    report_dict['Positive']['f1-score']
]

plt.figure(figsize=(8, 5))
sns.barplot(x=['Negative', 'Neutral', 'Positive'], y=f1_scores, palette='viridis')
plt.title('VADER F1-Score per Class (Full Dataset)')
plt.ylabel('F1-Score')
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
f1_path = 'visuals/vader_f1_per_class.png'
plt.savefig(f1_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"F1-score bar plot saved to: {os.path.abspath(f1_path)}")