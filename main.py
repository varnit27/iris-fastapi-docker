from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the model at startup
try:
    model = joblib.load("model.pkl")
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False

# Define the expected JSON payload
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_loaded
    }

@app.post("/predict")
def predict(features: IrisFeatures):
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")
    
    # Convert input data to the format sklearn expects (2D array)
    input_data = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]])
    
    prediction = model.predict(input_data)
    
    # Map the numeric prediction back to the flower name
    classes = ["setosa", "versicolor", "virginica"]
    predicted_class = classes[prediction[0]]
    
    return {"prediction": predicted_class}