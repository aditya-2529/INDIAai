import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize Flask app
app = Flask(__name__)

# Download NLTK resources
nltk.download('punkt',download_dir='C:/Users/ranja/nltk_data')
nltk.download('stopwords',download_dir='C:/Users/ranja/nltk_data')
nltk.download('punkt_tab',download_dir='C:/Users/ranja/nltk_data')

nltk.data.path.append('C:/Users/ranja/nltk_data')

# Load pre-trained model and TF-IDF vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Text Preprocessing Function
def preprocess_text(text):
    # Lowercase conversion
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stop words
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Stemming
    cleaned_tokens = [word for word in tokens if word not in stopwords.words('english')]

    # # Join tokens back into a single string
    cleaned_text = ' '.join(cleaned_tokens)

    return cleaned_text
@app.route('/')
def index():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['GET','POST'])
def predict():
    # Get the request data
    inputText = request.form.get('text')
    if '' == inputText:
        return jsonify({'error': 'No text provided'}), 400
    
    # Preprocess the input text
    cleaned_text = preprocess_text(inputText)
    
    # Transform text using TF-IDF
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    
    # Predict the category
    prediction = model.predict(vectorized_text)[0]
    
    # Return the result as JSON
    return render_template('index.html',pp=prediction)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
