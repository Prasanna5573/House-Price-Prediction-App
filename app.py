import os
import json
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Global model and metadata variables
model = None
metadata = None
MODEL_PATH = "models/model.pkl"
METADATA_PATH = "models/model_metadata.json"

def load_model_assets():
    global model, metadata
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)

# Initial load
load_model_assets()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/metrics", methods=["GET"])
def get_metrics():
    global metadata
    if metadata is None:
        load_model_assets()
    
    if metadata is not None:
        return jsonify(metadata)
    else:
        return jsonify({"error": "Model metadata not found. Has the model been trained yet?"}), 404

@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        load_model_assets()
        
    if model is None:
        return jsonify({"error": "Model not loaded. Please train the model first."}), 500
    
    try:
        data = request.get_json()
        
        # Extract features from request
        square_feet = float(data.get("SquareFeet", 1500))
        bedrooms = int(data.get("Bedrooms", 3))
        bathrooms = float(data.get("Bathrooms", 2.0))
        neighborhood = data.get("Neighborhood", "Suburbs")
        year_built = int(data.get("YearBuilt", 2000))
        has_garage = int(data.get("HasGarage", 1))
        has_pool = int(data.get("HasPool", 0))
        condition = int(data.get("Condition", 3))
        
        # Apply the exact same feature engineering as during training
        house_age = 2026 - year_built
        sqft_per_room = square_feet / (bedrooms + bathrooms + 0.1)
        
        # Construct DataFrame matching the features expected by the pipeline
        input_data = pd.DataFrame([{
            "SquareFeet": square_feet,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Neighborhood": neighborhood,
            "HasGarage": has_garage,
            "HasPool": has_pool,
            "Condition": condition,
            "HouseAge": house_age,
            "SqFtPerRoom": sqft_per_room
        }])
        
        # Run prediction
        prediction = model.predict(input_data)[0]
        
        # Add some mock comparable listings or standard deviation logic for visual value
        # In a real app we can use model residuals or standard errors, 
        # here we will calculate a realistic comparable confidence range
        std_error = 15000 # derived from standard deviation of noise in training
        confidence_low = max(50000, prediction - (1.96 * std_error))
        confidence_high = prediction + (1.96 * std_error)
        
        response = {
            "prediction": float(prediction),
            "confidence_low": float(confidence_low),
            "confidence_high": float(confidence_high),
            "details": {
                "SquareFeet": square_feet,
                "Bedrooms": bedrooms,
                "Bathrooms": bathrooms,
                "Neighborhood": neighborhood,
                "Condition": condition,
                "HouseAge": house_age,
                "SqFtPerRoom": round(sqft_per_room, 2)
            }
        }
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Ensure templates and static folders exist
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static/css", exist_ok=True)
    os.makedirs("static/js", exist_ok=True)
    
    print("Starting Flask web server on port 5050...")
    app.run(debug=True, host="127.0.0.1", port=5050)
