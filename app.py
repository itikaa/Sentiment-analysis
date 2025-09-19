from flask import Flask, request, render_template
import pickle

# Load model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')   # simple form for input

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['tweet']
        transformed_text = vectorizer.transform([text])
        prediction = model.predict(transformed_text)[0]

        sentiment = "Positive" if prediction == 1 else "Negative" if prediction == -1 else "Neutral"
        return render_template('index.html', prediction_text=f"Sentiment: {sentiment}")

if __name__ == "__main__":
    app.run(debug=True)
