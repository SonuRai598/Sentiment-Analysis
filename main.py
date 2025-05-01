import streamlit as st
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import seaborn as sns
import pandas as pd
from model import load_and_train_model, predict_sentiment


st.set_page_config(layout="wide")
st.title("🎬 IMDB Movie Review Sentiment Analyzer")
st.write("This app uses a machine learning model to predict whether a movie review is positive or negative.")

# Loading model
with st.spinner("🔄 Training model..."):
    model, tfidf, df, acc, report, matrix = load_and_train_model()

st.sidebar.header("💬 Predict Sentiment")
user_input = st.sidebar.text_area("Enter a movie review here:", height=150)

if st.sidebar.button("🔍 Predict"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input, model, tfidf)
        st.sidebar.success(f"Predicted Sentiment: {sentiment}")
    else:
        st.sidebar.warning("⚠️ Please enter a review.")

# Displaying Accuracy
st.header("📈 Model Evaluation")

st.markdown("### 🤖 Model Used: `Support Vector Machine (SVM)`")
st.metric(label="Accuracy", value=f"{acc * 100:.2f}%")

# Classification report
st.subheader("📄 Classification Report")
report_df = pd.DataFrame(report).transpose()
report_df = report_df.drop(['accuracy'], errors='ignore').round(2)
st.dataframe(report_df.style.highlight_max(axis=1), use_container_width=True)

# Confusion matrix
st.subheader("🧩 Confusion Matrix")
fig_cm, ax_cm = plt.subplots()
sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'], ax=ax_cm)
ax_cm.set_xlabel('Predicted')
ax_cm.set_ylabel('Actual')
st.pyplot(fig_cm)


# Word Cloud Section
st.header("🌥️ Word Clouds")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Positive Reviews")
    pos_text = ' '.join(df[df['sentiment'] == 'positive']['processed_review'])
    pos_cloud = WordCloud(width=400, height=300, background_color='white').generate(pos_text)
    st.image(pos_cloud.to_array(), use_column_width=True)

with col2:
    st.subheader("Negative Reviews")
    neg_text = ' '.join(df[df['sentiment'] == 'negative']['processed_review'])
    neg_cloud = WordCloud(width=400, height=300, background_color='white').generate(neg_text)
    st.image(neg_cloud.to_array(), use_column_width=True)

# Frequency Distribution
st.header("📊 Most Frequent Words")
all_words = ' '.join(df['processed_review'])
tokens = word_tokenize(all_words)
fdist = FreqDist(tokens)

fig_freq, ax_freq = plt.subplots()
fdist.plot(20, cumulative=False)
plt.title("Top 20 Most Frequent Words")
st.pyplot(fig_freq)



