# 04_distilbert_finetune.py
# DistilBERT Fine-tuning for Sentiment Analysis
# Updated March 2026

import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import os

# Create folders
os.makedirs('models', exist_ok=True)
os.makedirs('visuals', exist_ok=True)

print("Loading TweetEval sentiment dataset...")
dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")

# Use a manageable sample for faster training (you can increase later)
sample_size = 12000   # Good balance between speed and performance
train_df = dataset['train'].to_pandas().sample(n=sample_size, random_state=42)
test_df  = dataset['test'].to_pandas().head(3000)

print(f"Using {len(train_df)} training samples and {len(test_df)} test samples")

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
test_dataset  = Dataset.from_pandas(test_df[['text', 'label']])

# ── Tokenizer ──
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

print("Tokenizing datasets...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test  = test_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# ── Model ──
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# ── Training Arguments ──
training_args = TrainingArguments(
    output_dir="./results_distilbert",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",           # Disable wandb / external logging
    fp16=False,                 # Set True if you have GPU
)

# ── Metrics ──
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

# ── Trainer ──
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

print("Starting DistilBERT fine-tuning... (this may take 10-30 minutes)")
trainer.train()

# ── Final Evaluation ──
print("\nFinal evaluation on test set...")
eval_results = trainer.evaluate()
print(f"DistilBERT Test Accuracy: {eval_results['eval_accuracy']:.4f}")

# Save the fine-tuned model
model_path = "models/distilbert_sentiment_final"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print(f"Model saved to: {model_path}")

print("DistilBERT fine-tuning completed successfully!")