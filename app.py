# app.py - Final Polished Version
import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.express as px
from bertopic import BERTopic

st.set_page_config(page_title="Brand Reputation Monitor", layout="wide")

st.title("📊 Brand Reputation Sentiment Analysis Dashboard")
st.markdown("**Master Thesis** — Data Science for Society and Business")

# Load models
@st.cache_resource
def load_models():
    sentiment_model = pipeline("sentiment-analysis", 
                               model="models/distilbert_sentiment_final", 
                               device=-1)
    topic_model = BERTopic(
        embedding_model="all-MiniLM-L6-v2",
        min_topic_size=3,      # Lower threshold
        verbose=False
    )
    return sentiment_model, topic_model

classifier, topic_model = load_models()

# Sidebar
st.sidebar.header("Settings")
brand_name = st.sidebar.text_input("Brand Name (optional)", "Nike")
analysis_mode = st.sidebar.radio("Analysis Mode", ["Paste Text", "Upload CSV"])

# Main content
st.header("Real-time Sentiment Analysis")

if analysis_mode == "Paste Text":
    text_input = st.text_area("Paste tweets or text here (one per line)", height=180)
    num_tweets = st.slider("Number of tweets to analyze", 5, 50, 15)
else:
    uploaded_file = st.file_uploader("Upload CSV file (must have 'text' column)", type=["csv"])
    num_tweets = st.slider("Number of tweets to analyze", 5, 100, 30)

if st.button("🚀 Analyze Sentiment", type="primary"):
    # Get texts
    if analysis_mode == "Paste Text" and text_input:
        texts = [line.strip() for line in text_input.split('\n') if line.strip()]
    elif analysis_mode == "Upload CSV" and uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        if 'text' not in df_upload.columns:
            st.error("CSV must contain a 'text' column")
            st.stop()
        texts = df_upload['text'].astype(str).tolist()
    else:
        st.warning("Please provide input data")
        st.stop()

    texts = texts[:num_tweets]

    with st.spinner("Analyzing with DistilBERT..."):
        results = []
        for text in texts:
            if not text.strip():
                continue
            result = classifier(text)[0]
            label_map = {'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 'Positive'}
            sentiment = label_map[result['label']]
            confidence = round(result['score'], 4)
            results.append({
                "Text": text[:150] + "..." if len(text) > 150 else text,
                "Sentiment": sentiment,
                "Confidence": confidence
            })

        df_results = pd.DataFrame(results)

    # Layout
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Analysis Results")
        st.dataframe(df_results, use_container_width=True)

    with col2:
        st.subheader("Sentiment Distribution")
        if not df_results.empty:
            sentiment_counts = df_results['Sentiment'].value_counts()
            fig = px.pie(
                names=sentiment_counts.index,
                values=sentiment_counts.values,
                color_discrete_sequence=['#ef4444', '#64748b', '#22c55e'],
                title=f'Sentiment Distribution for "{brand_name}"'
            )
            st.plotly_chart(fig, use_container_width=True)

    # Alert
    neg_count = (df_results['Sentiment'] == 'Negative').sum()
    if neg_count > 0:
        st.error(f"⚠️ **{neg_count} negative sentiments detected** — Potential reputation risk for {brand_name}!")
    else:
        st.success("✅ No major negative sentiment detected.")

    # Topic Modeling - More robust
    negative_texts = df_results[df_results['Sentiment'] == 'Negative']['Text'].tolist()
    
    if len(negative_texts) >= 4:
        st.subheader("🔍 Key Topics from Negative Feedback")
        with st.spinner("Extracting topics..."):
            try:
                topics, probs = topic_model.fit_transform(negative_texts)
                topic_info = topic_model.get_topic_info()
                
                if len(topic_info) > 1:
                    st.dataframe(topic_info.head(8)[['Topic', 'Count', 'Name']], use_container_width=True)
                else:
                    st.info("Not enough varied negative tweets to generate meaningful topics.")
            except:
                st.info("Topic modeling could not generate stable topics with this data.")
    else:
        st.info("Add more negative tweets to see key topics.")

st.markdown("---")
st.caption("Master's Thesis — Sentiment Analysis for Brand Reputation Management | Zerong")