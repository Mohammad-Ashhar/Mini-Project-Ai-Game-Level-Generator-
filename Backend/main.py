# main.py

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
import pandas as pd
from db import init_db
from crud import insert_dataframe, fetch_all_data_as_df, save_ml_metadata
from ml_pipeline import train_model, load_model, predict
from typing import Dict
from datetime import datetime
import io

app = FastAPI(title="Project Backend - Core Architecture")

# Initialize DB tables
@app.on_event("startup")
def on_startup():
    init_db()

# --- Simple API: health / schema ---
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

@app.get("/data/schema")
def data_schema():
    # Return expected columns for DataRecord
    return {"expected_columns": ["feature1", "feature2", "label"]}

# --- Data ingestion ---
@app.post("/data/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """
    Accepts a CSV file with columns matching the DataRecord schema.
    """
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    count = insert_dataframe(df)
    return {"inserted_rows": count}

# --- Train model endpoint ---
@app.post("/ml/train")
def ml_train(payload: Dict = None):
    """
    Trains model on data currently in DB.
    payload can include {"target_col":"label"} if different.
    """
    df = fetch_all_data_as_df()
    if df.empty:
        raise HTTPException(status_code=400, detail="No data available for training. Upload data first.")

    target_col = "label"
    if payload and "target_col" in payload:
        target_col = payload["target_col"]

    try:
        model_path, metrics = train_model(df, target_col=target_col)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # persist metadata
    save_ml_metadata(name="random_forest", version="v1", trained_at=datetime.utcnow(), metrics=str(metrics))
    return {"model_path": model_path, "metrics": metrics}

# --- Predict ---
@app.post("/ml/predict")
def ml_predict(payload: Dict):
    """
    Example payload:
    {
      "features": { "feature1": 1.2, "feature2": 3.4 }
    }
    or batch:
    {
      "batch": [ {..}, {..} ]
    }
    """
    if not payload:
        raise HTTPException(status_code=400, detail="Empty payload")

    try:
        model = load_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load model: {e}")

    if "batch" in payload:
        df = pd.DataFrame(payload["batch"])
        preds = predict(model, df)
        return {"predictions": preds}

    if "features" in payload:
        df = pd.DataFrame([payload["features"]])
        preds = predict(model, df)
        return {"predictions": preds}

    raise HTTPException(status_code=400, detail="payload must contain 'features' or 'batch'")

# --- Basic auth placeholder (expand as needed) ---
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import Security
security = HTTPBasic()

def fake_auth(creds: HTTPBasicCredentials = Security(security)):
    # Replace this with proper auth (JWT/OAuth) for production
    if creds.username != "admin" or creds.password != "password":
        raise HTTPException(status_code=401, detail="Unauthorized")
    return creds.username

@app.get("/protected")
def protected_route(user: str = Depends(fake_auth)):
    return {"message": f"Hello {user}"}
