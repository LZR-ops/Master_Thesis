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
plt.savefig('sentiment_distribution.png') 
plt.show()

import os
os.makedirs('data', exist_ok=True)

df_sample = df.sample(1000, random_state=42)  # 1000 random rows
sample_path = 'data/tweet_eval_sample.csv'
df_sample.to_csv(sample_path, index=False)
print(f"Sample saved successfully to: {os.path.abspath(sample_path)}")

# Tweet length distribution (helps understand text complexity)
df['text_length'] = df['text'].apply(len)
plt.figure(figsize=(10, 6))
sns.histplot(df['text_length'], bins=50, kde=True, color='skyblue')
plt.title('Distribution of Tweet Lengths (Characters) in TweetEval')
plt.xlabel('Length')
plt.ylabel('Count')
plt.savefig('tweet_length_distribution.png')
plt.show()

# Word cloud for visual overview (requires wordcloud library - already in your pip install)
from wordcloud import WordCloud

all_text = ' '.join(df['text'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=200).generate(all_text)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of All Tweets in Dataset')
plt.savefig('wordcloud_all_tweets.png')
plt.show()

# Sample tweets per class (print 3 examples each)
print("\nSample Negative Tweets:")
print(df[df['label'] == 0]['text'].head(3).to_string(index=False))

print("\nSample Neutral Tweets:")
print(df[df['label'] == 1]['text'].head(3).to_string(index=False))

print("\nSample Positive Tweets:")
print(df[df['label'] == 2]['text'].head(3).to_string(index=False))