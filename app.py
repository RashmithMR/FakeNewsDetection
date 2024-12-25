from flask import Flask, request, render_template, jsonify
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model and vectorizer
joblib_model = pickle.load(open('model2.pkl', 'rb'))
joblib_vect = pickle.load(open('tfidfvect2.pkl', 'rb'))

# Preprocess and classify news input
def classify_news(user_input):
    if not user_input.strip():
        return "Error: Empty input. Please enter a valid news statement."
    processed_input = joblib_vect.transform([user_input]).toarray()
    prediction = joblib_model.predict(processed_input)
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from form
        user_input = request.form.get('news_statement', '')
        # Classify the input
        prediction = classify_news(user_input)
        result = "Fake News!" if prediction[0] == 0 else "Real News"
        return render_template('index.html', prediction=result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
