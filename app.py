from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pickle
import pandas as pd
import time
import json

#load pipeline and schema
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_schema.json") as f:
    feature_schema = json.load(f)

#metrics to track
request_count = 0
total_response_time = 0.0

app = FastAPI(title="Churn Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

#health check endpoint
@app.get("/api/health")
def health():
    return {"status": "ok"}

#input validation
class PredictInput(BaseModel):
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: float
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    tenure_group: str
    num_services: int
    tenure_contract_ratio: float


#predict endpoint
@app.post("/api/predict")
def predict(inputs: List[PredictInput]):
    global request_count, total_response_time
    start = time.time()

    #turn input to DataFrame
    df = pd.DataFrame([i.dict() for i in inputs])

    #DONT FORGET TO APPLY TRANSFORMS YOU DID WHILE TRAINING
    df['TotalCharges'] = df['TotalCharges'] ** 0.5

    #make predictions
    predictions = model.predict(df).tolist()
    probabilities = model.predict_proba(df)[:,1].tolist()

    #update metrics
    response_time = time.time() - start
    request_count += 1
    total_response_time += response_time

    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "response_time": response_time
    }

#metrics endpoint
@app.get("/api/metrics")
def metrics():
    avg_response = total_response_time / request_count if request_count > 0 else 0.0
    return {
        "request_count": request_count,
        "average_response_time": avg_response
    }
