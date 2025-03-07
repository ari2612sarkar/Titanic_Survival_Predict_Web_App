from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model from the pickle file
model = joblib.load(r"C:\Users\TUF\Desktop\G\titanic_model.pkl")

# Encoders for preprocessing (loaded here for reference, no need to modify your pickle structure)
sex_encoder = LabelEncoder()
sex_encoder.fit(["male", "female"])

embarked_encoder = LabelEncoder()
embarked_encoder.fit(["S", "C", "Q"])

title_encoder = LabelEncoder()
title_encoder.fit(["Mr", "Mrs", "Miss", "Master", "Other"])

def preprocess(data):
    """Preprocess input JSON to match model features."""
    df = pd.DataFrame([data])
    
    # Drop 'PassengerId' from the input data if present
    if 'PassengerId' in df.columns:
        df.drop(columns=['PassengerId'], inplace=True)
    
    # Fill missing values
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna("S", inplace=True)  # Default to 'S' like training
    
    # Apply Label Encoding
    df["Sex"] = sex_encoder.transform([df["Sex"]])[0]
    df["Embarked"] = embarked_encoder.transform([df["Embarked"]])[0]
    df["Title"] = title_encoder.transform([df["Title"]])[0]
    
    # Select features used in training
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title"]
    return df[features]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Validate input data structure
        required_fields = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title"]
        if not all(field in data for field in required_fields):
            return jsonify({"error": f"Missing one or more required fields: {', '.join(required_fields)}"}), 400
        
        # Preprocess input
        input_data = preprocess(data)
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Return prediction result
        return jsonify({"Survived": int(prediction[0])})
    
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
