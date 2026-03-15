import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading TweetEval sentiment dataset...")

# Load from Hugging Face (internet required first time)
dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")

# Convert train split to pandas DataFrame
df = dataset['train'].to_pandas()

print("\nDataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# Label mapping (0=negative, 1=neutral, 2=positive)
label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
df['sentiment'] = df['label'].map(label_map)

# Basic stats
print("\nSentiment Distribution:")
print(df['sentiment'].value_counts(normalize=True) * 100)

# Plot distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='sentiment', data=df, order=['Negative', 'Neutral', 'Positive'])
plt.title('Sentiment Distribution in TweetEval (Train Split)')
plt.xlabel('Sentiment Class')
plt.ylabel('Number of Tweets')
plt.savefig('sentiment_distribution.png')  # Save for README or report
plt.show()
