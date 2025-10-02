from flask import Flask, render_template, request
import pickle


tokenizer = pickle.load(open("spam_tokenizer.pkl", "rb"))
classifier = pickle.load(open("spam_classifier.pkl", "rb"))

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    content = ""
    content = request.form.get("sms-content")
    tokenized_content = tokenizer.transform([content])
    prediction = classifier.predict(tokenized_content)

    prediction_info = "SPAM" if prediction == 1 else "HAM"
    return render_template("index.html", text=content, prediction=prediction_info)


# Run the Flask application
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
