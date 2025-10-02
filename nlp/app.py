from flask import Flask, render_template, request, jsonify
from flasgger import Swagger
import pickle


# Load ML artifacts
with open("spam_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("spam_classifier.pkl", "rb") as f:
    classifier = pickle.load(f)


# Initialize Flask app
app = Flask(__name__)


# Swagger configuration
swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "SMS Spam Classifier API",
        "description": "REST API to classify SMS messages as SPAM or HAM using ML model",
        "version": "1.0.0",
    },
    "basePath": "/",
    "schemes": ["http", "https"],
}

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec_1",
            "route": "/apispec_1.json",
            "rule_filter": lambda rule: True,  # include all endpoints
            "model_filter": lambda tag: True,  # include all models
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/",
}

swagger = Swagger(app, template=swagger_template, config=swagger_config)


# --- Utility function ---
def tokenize_predict(content: str) -> str:
    """Transform text and predict spam/ham label."""
    tokenized_content = tokenizer.transform([content])
    prediction = classifier.predict(tokenized_content)[0]
    return "spam" if prediction == 1 else "ham"


# --- Web UI route ---
@app.route("/")
def home():
    return render_template("index.html")


# --- HTML form submission ---
@app.route("/predict", methods=["POST"])
def predict():
    content = request.form.get("sms-content", "")
    prediction_info = tokenize_predict(content)
    return render_template("index.html", text=content, prediction=prediction_info)


# --- REST API endpoint ---
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Predict if a message is SPAM or HAM
    ---
    tags:
      - Prediction
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - content
          properties:
            content:
              type: string
              description: The SMS/text message to classify
              example: "Win a free iPhone now!"
          example:
            content: "Win a free iPhone now!"
    responses:
      200:
        description: Prediction result
        schema:
          type: object
          properties:
            input:
              type: string
              example: "Win a free iPhone now!"
            prediction:
              type: string
              example: "spam"
      400:
        description: Bad Request – missing parameter
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Parameter 'content' is required"
    """
    data = request.get_json()
    if not data or "content" not in data:
        return jsonify({"error": "Parameter 'content' is required"}), 400
    content = data["content"]
    prediction_info = tokenize_predict(content)
    return jsonify({"input": content, "prediction": prediction_info})


# Run the Flask application
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
