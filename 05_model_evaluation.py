# 05_model_evaluation.py
# Load saved DistilBERT model and generate detailed evaluation

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Load model and tokenizer
model_path = "models/distilbert_sentiment_final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Create pipeline for easy inference
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)  # -1 = CPU

# Load test data
dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")
test_df = dataset['test'].to_pandas().head(3000)

print("Running predictions on test set...")

# Get predictions
predictions = []
for text in test_df['text']:
    result = classifier(text)[0]
    # Map label: LABEL_0 = negative, LABEL_1 = neutral, LABEL_2 = positive
    label_map = {'LABEL_0': 0, 'LABEL_1': 1, 'LABEL_2': 2}
    pred_label = label_map[result['label']]
    predictions.append(pred_label)

test_df['predicted'] = predictions

# Evaluation
print("\nDistilBERT Detailed Classification Report:")
print(classification_report(test_df['label'], test_df['predicted'],
                            target_names=['Negative', 'Neutral', 'Positive'], digits=4))

# Confusion Matrix
cm = confusion_matrix(test_df['label'], test_df['predicted'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'])
plt.title('DistilBERT Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('visuals/distilbert_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("Confusion matrix saved to visuals/distilbert_confusion_matrix.png")

# Save wrong predictions for analysis
wrong = test_df[test_df['label'] != test_df['predicted']]
wrong.to_csv('data/distilbert_wrong_predictions.csv', index=False)
print(f"Saved {len(wrong)} wrong predictions for manual analysis")