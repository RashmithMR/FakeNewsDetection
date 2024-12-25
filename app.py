from flask import Flask, request, render_template, jsonify
import pickle
import requests
from bs4 import BeautifulSoup
import traceback

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model and vectorizer
try:
    joblib_model = pickle.load(open('model2.pkl', 'rb'))
    joblib_vect = pickle.load(open('tfidfvect2.pkl', 'rb'))
except Exception as e:
    raise RuntimeError(f"Failed to load model or vectorizer: {str(e)}")

# Function to extract headline from a URL
def extract_headline_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        headline = soup.find('title').text.strip()
        return headline
    except Exception as e:
        return f"Error extracting headline: {str(e)}"

# Preprocess and classify news input
def classify_news(user_input):
    try:
        if not user_input.strip():
            return "Error: Empty input. Please enter a valid news statement."

        # Check if the vectorizer is fitted
        if not hasattr(joblib_vect, 'vocabulary_'):
            return "Error: Vectorizer is not fitted. Please ensure it is trained and saved correctly."

        processed_input = joblib_vect.transform([user_input])

        # Check if dimensions match
        if processed_input.shape[1] != joblib_model.n_features_in_:
            return "Error: Dimension mismatch between vectorizer and model. Ensure both are trained on the same dataset."

        prediction = joblib_model.predict(processed_input)
        return prediction
    except Exception as e:
        return f"Error during classification: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from form
        article_url = request.form.get('article_url', '')
        news_text = request.form.get('news_statement', '')

        if article_url:
            headline = extract_headline_from_url(article_url)
            if headline.startswith("Error"):
                return render_template('index.html', prediction=headline)
            user_input = headline
        elif news_text:
            user_input = news_text
        else:
            return render_template('index.html', prediction="Error: No input provided. Please enter a URL or text.")

        # Classify the input
        prediction = classify_news(user_input)
        if isinstance(prediction, str) and prediction.startswith("Error"):
            return render_template('index.html', prediction=prediction)

        result = "Fake News!" if prediction[0] == 0 else "Real News"
        return render_template('index.html', prediction=f"Input: {user_input}<br>Prediction: {result}")
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        return render_template('index.html', prediction=error_message)

if __name__ == '__main__':
    app.run(debug=True)
