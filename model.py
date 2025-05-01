
import re
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(tokens)

def load_and_train_model():
    df = pd.read_csv("IMDB Dataset.csv")
    df = df.sample(n=5000, random_state=42)
    df['processed_review'] = df['review'].apply(preprocess_text)
    df['sentiment_label'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

    tfidf = TfidfVectorizer(max_features=1000)
    X = tfidf.fit_transform(df['processed_review']).toarray()
    y = df['sentiment_label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SVC(kernel='linear', class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)

    return model, tfidf, df, acc, report, matrix

def predict_sentiment(text, model, tfidf):
    preprocessed = preprocess_text(text)
    features = tfidf.transform([preprocessed]).toarray()
    prediction = model.predict(features)[0]
    return "Positive 😊" if prediction == 1 else "Negative 😞"
