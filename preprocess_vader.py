# preprocess_vader.py
# Preprocessing + VADER Baseline on full tweet_eval sentiment dataset
# Updated March 2026 - with versioned cleaning (v1 basic vs v2 improved)

import re
import pandas as pd
import nltk
import emoji
import contractions  # pip install contractions
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset
import os

# ── One-time downloads ──
nltk.download('vader_lexicon', quiet=True)

# ── Configuration ──
SAVE_FOLDER_DATA   = 'data'
SAVE_FOLDER_VISUAL = 'visuals'
os.makedirs(SAVE_FOLDER_DATA,   exist_ok=True)
os.makedirs(SAVE_FOLDER_VISUAL, exist_ok=True)

# ── Load full dataset ──
print("Loading full TweetEval sentiment dataset from Hugging Face...")
dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")
df = dataset['train'].to_pandas()  # ~45,615 rows
print(f"Loaded {len(df):,} tweets\n")

# ── Preprocessing functions ──

def clean_tweet_v1(text):
    """V1: Basic cleaning - original version"""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def clean_tweet_v2(text):
    """V2: Improved - emoji to text, contractions, better normalization"""
    text = emoji.demojize(text, delimiters=("", ""))
    text = contractions.fix(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z\s!?.,]', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


# ── Choose which cleaning version to use ──
clean_function = clean_tweet_v2   # ← switch to clean_tweet_v1 for original
version_tag    = "v2" if clean_function == clean_tweet_v2 else "v1"

print(f"Cleaning tweets with version {version_tag.upper()}...")
df['clean_text'] = df['text'].apply(clean_function)

# ── VADER sentiment analysis ──
print("Running VADER...")
sia = SentimentIntensityAnalyzer()

df['vader_compound'] = df['clean_text'].apply(lambda x: sia.polarity_scores(x)['compound'])

def vader_to_class(score):
    if score >= 0.05:    return 2   # Positive
    elif score <= -0.05: return 0   # Negative
    else:                return 1   # Neutral

df['vader_pred'] = df['vader_compound'].apply(vader_to_class)

# ── Evaluation ──
true_labels = df['label']
pred_labels = df['vader_pred']

accuracy = accuracy_score(true_labels, pred_labels)
print(f"\nVADER Accuracy ({version_tag.upper()}): {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(
    true_labels,
    pred_labels,
    target_names=['Negative', 'Neutral', 'Positive'],
    digits=4
))

# ── Save predictions ──
results_path = f'{SAVE_FOLDER_DATA}/vader_{version_tag}_results.csv'
df[['text', 'label', 'clean_text', 'vader_compound', 'vader_pred']].to_csv(results_path, index=False)
print(f"Results saved to: {os.path.abspath(results_path)}")

# ── Visualizations ──
print("Generating visualizations...")

# Confusion Matrix
cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'])
plt.title(f'VADER Confusion Matrix ({version_tag.upper()}) - Full Dataset')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
cm_path = f'{SAVE_FOLDER_VISUAL}/vader_{version_tag}_confusion_matrix.png'
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Confusion matrix saved to: {os.path.abspath(cm_path)}")

# Create report_dict for F1 scores (this line was missing!)
report_dict = classification_report(
    true_labels,
    pred_labels,
    target_names=['Negative', 'Neutral', 'Positive'],
    output_dict=True
)

# F1-score bar plot (fixed FutureWarning with hue)
plt.figure(figsize=(8, 5))
sns.barplot(x=['Negative', 'Neutral', 'Positive'], 
            y=[report_dict['Negative']['f1-score'],
               report_dict['Neutral']['f1-score'],
               report_dict['Positive']['f1-score']],
            hue=['Negative', 'Neutral', 'Positive'], 
            palette='viridis', 
            legend=False)
plt.title(f'VADER F1-Score per Class ({version_tag.upper()})')
plt.ylabel('F1-Score')
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
f1_path = f'{SAVE_FOLDER_VISUAL}/vader_{version_tag}_f1_per_class.png'
plt.savefig(f1_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"F1-score bar plot saved to: {os.path.abspath(f1_path)}")

# ── Quick comparison reminder ──
print("\nComparison reminder:")
print(" - Previous run with v1 (basic cleaning): ~55.14%")
print(f" - Current ({version_tag.upper()}): {accuracy:.4f}")