from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="Bike Buyers Inference API")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "global_best_model.pkl"
model = joblib.load(MODEL_PATH)

class BikeBuyerInput(BaseModel):
    gender: str
    age: int
    marital_status: str
    children: int
    income: int
    education_level: str
    occupation_name: str
    region_name: str
    commute_distance: str
    home_owner: str
    cars: int

@app.post("/predict")
def predict(data: BikeBuyerInput):
    df = pd.DataFrame([data.dict()])

    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    return {
        "prediction": prediction,
        "probability": probability
    }
