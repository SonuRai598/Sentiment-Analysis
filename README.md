🎬 IMDB Movie Review Sentiment Analyzer

This is a Streamlit web application that uses a machine learning model to predict whether a movie review is **positive** or **negative** based on user input. The model is trained on a subset of the IMDB dataset using **Support Vector Machine (SVM)** with TF-IDF vectorization.

🚀 Features

- 🔍 **Real-time Sentiment Prediction** for user-entered movie reviews.
- 📈 **Model Evaluation Metrics** including accuracy, classification report, and confusion matrix.
- 🌥️ **Word Clouds** to visualize commonly used words in positive and negative reviews.
- 📊 **Top 20 Frequent Words** displayed using NLTK's frequency distribution.

📦 Dependencies

Make sure to install the required packages using:
pip install -r requirements.txt

Key packages used:
-streamlit
-scikit-learn
-pandas
-nltk
-matplotlib
-seaborn
-wordcloud

🧠 Model Details
-Algorithm: Support Vector Machine (SVM) with linear kernel
-Vectorization: TF-IDF (max_features=1000)
-Training Set: 5,000 reviews (sampled from the IMDB dataset)

Evaluation Metrics:
-Accuracy score
-Classification report (Precision, Recall, F1-score)
-Confusion matrix

📁 Project Structure
├── model.py                  # Contains model training and prediction logic
├── app.py                   # Main Streamlit app
├── IMDB Dataset.csv         # Input data (ensure this is in the same directory)
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation

🔧 Setup Instructions
Clone the repository:
git clone https://github.com/your-username/imdb-sentiment-analyzer.git
cd imdb-sentiment-analyzer

Install dependencies:
pip install -r requirements.txt

Download NLTK resources (automatically handled in the script):
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

Run the Streamlit app:
streamlit run app.py

✍️ Example Input
"The plot was engaging and the characters were incredibly well developed."
Output: Predicted Sentiment: Positive 😊

📌 Notes
This app samples 5,000 reviews from the IMDB dataset for faster training.
The IMDB Dataset.csv file must be in the root directory.
The model is retrained every time the app runs. For deployment, consider saving the trained model using joblib.

🙋‍♂️ Author
Developed by Sonu Rai

