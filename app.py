from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Load data from the uploaded file
    # names = ['comments', 'type']
    names = ['raw']
    data = pd.read_table(file, sep="\t", names=names)
    
    # Preprocessing
    df = pd.DataFrame(data)
    X = df['comments']
    y = df['type']
    
    # Convert to lowercase and remove whitespace
    lower_text = [str(comment).strip().lower() for comment in X]
    
    # Remove punctuation
    punc_text = [re.sub(r'[^\w\s]', '', text) for text in lower_text]
    
    # TF-IDF vectorization
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.7, ngram_range=(1, 2), stop_words='english')
    X_tfidf = tfidf.fit_transform(punc_text)
    
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X_tfidf, y, test_size=0.1, random_state=0)
    
    # Train the SVM model
    clf = LinearSVC()
    clf.fit(X_train, Y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Calculate metrics
    confusion = confusion_matrix(Y_test, y_pred).tolist()  # Convert to list for JSON serialization
    classification = classification_report(Y_test, y_pred, output_dict=True)  # Dict for structured JSON response
    accuracy = accuracy_score(Y_test, y_pred)
    
    # Return JSON response
    return jsonify({
        "confusion_matrix": confusion,
        "classification_matrix": classification,
        "accuracy": accuracy
    })

if __name__ == '__main__':
    app.run(debug=True)
