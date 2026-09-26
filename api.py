import joblib, pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
top_features = joblib.load("top_features.pkl")

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Record(BaseModel):
    data: dict

@app.post("/predict")
def predict(record: Record):
    df = pd.DataFrame([record.data])
    for c, le in encoders.items():
        if c in df.columns:
            val = df[c].astype(str).iloc[0]
            df[c] = le.transform([val if val in le.classes_ else "__unseen__"])
    num_cols = [c for c in df.columns if c not in encoders]
    df[num_cols] = scaler.transform(df[num_cols])
    X = df[top_features]
    prob = model.predict_proba(X)[0][1]
    return {"label": int(prob >= 0.5), "probability": float(prob)}

import io
from fastapi import UploadFile, File

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    df = pd.read_csv(io.BytesIO(await file.read()))
    df = df.drop(columns=["id", "label", "attack_cat"], errors="ignore")
    for c, le in encoders.items():
        if c in df.columns:
            df[c] = df[c].astype(str).apply(lambda v: v if v in le.classes_ else "__unseen__")
            df[c] = le.transform(df[c])
    num_cols = [c for c in df.columns if c not in encoders]
    df[num_cols] = scaler.transform(df[num_cols])
    probs = model.predict_proba(df[top_features])[:, 1]
    return [{"row": i, "label": int(p >= 0.5), "probability": float(p)} for i, p in enumerate(probs)]