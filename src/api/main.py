from fastapi import FastAPI
import pandas as pd
import mlflow
import mlflow.sklearn
from src.api.pydantic_models import PredictRequest, PredictResponse

app = FastAPI(title="Credit Risk API")

# Load model from MLflow registry or local path
MODEL_URI = "models:/credit_risk_model/Production"
model = mlflow.sklearn.load_model(MODEL_URI)

@app.get("/")
def root():
    return {"message": "Credit Risk Prediction API is running"}

@app.post("/predict", response_model=PredictResponse)
def predict(data: PredictRequest):
    # Convert incoming request to DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Predict probability
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[:, 1][0]
    else:
        proba = model.predict(df)[0]
    
    return PredictResponse(risk_probability=proba)
