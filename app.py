from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app) # Allow cross-origin requests

try:
    model = joblib.load("model.pkl")
    symptoms_list = joblib.load("symptoms.pkl")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route("/model/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500
        
    try:
        data = request.json
        symptoms_input = data.get("symptoms", [])

        # Create a dictionary initialized to 0 for all known symptoms
        input_data = {sym: 0 for sym in symptoms_list}
        
        # Set 1 for the symptoms provided by user
        for sym in symptoms_input:
            if sym in input_data:
                input_data[sym] = 1
                
        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        
        # Predict using the model
        prediction = model.predict(df)[0]
        
        # Calculate probabilities
        probabilities = model.predict_proba(df)[0]
        max_prob = max(probabilities)

        return jsonify({
            "disease": prediction,
            "accuracy": round(max_prob * 100, 2)
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/model/symptoms", methods=["GET"])
def get_symptoms():
    return jsonify({"symptoms": symptoms_list})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
