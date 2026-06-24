from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
from pydantic import BaseModel
import pickle
import sqlite3
from datetime import datetime
import numpy as np
from contextlib import asynccontextmanager

# ==================== DB CREATE - OKATI MATRAME ====================
def init_db():
    conn = sqlite3.connect('crop_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            N REAL, P REAL, K REAL,
            temperature REAL, humidity REAL, ph REAL, rainfall REAL,
            state TEXT, season TEXT,
            recommended_crop TEXT,
            predicted_yield REAL
        )
    ''')
    conn.commit()
    conn.close()
    print(">>> Database initialized: crop_data.db created <<<")

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(
    title="Crop Recommendation & Yield Prediction API",
    description="Based on Kaggle Crop Yield Dataset",
    version="2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== LOAD ALL 6 MODELS ====================
try:
    print("Loading models...")
    crop_model = pickle.load(open('crop_recommendation_model.pkl', 'rb'))
    label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
    yield_model = pickle.load(open('yield_model.pkl', 'rb'))
    crop_encoder = pickle.load(open('crop_encoder.pkl', 'rb'))
    season_encoder = pickle.load(open('season_encoder.pkl', 'rb'))
    state_encoder = pickle.load(open('state_encoder.pkl', 'rb'))
    print("✅ All 6 models loaded successfully!")
    print("Valid Seasons:", list(season_encoder.classes_))
    print("Valid States count:", len(state_encoder.classes_))
    print("Valid Crops count:", len(crop_encoder.classes_))
except FileNotFoundError as e:
    print(f"❌ ERROR: {e}. 6.pkl files ee folder lo unnai ani check cheyi")
    exit()

# ==================== 1. CROP RECOMMENDATION ====================
class CropInput(BaseModel):
    N: int
    P: int
    K: int
    temperature: float
    humidity: float
    ph: float
    rainfall: float

@app.post("/recommend-crop", tags=["Crop Recommendation"])
def recommend_crop(data: CropInput):
    try:
        features = np.array([
            data.N, data.P, data.K, data.temperature,
            data.humidity, data.ph, data.rainfall
        ], dtype=float).reshape(1, -1)

        prediction = crop_model.predict(features)
        crop_name = label_encoder.inverse_transform(prediction)[0]

        try:
            conn = sqlite3.connect('crop_data.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions
                (timestamp, N, P, K, temperature, humidity, ph, rainfall, recommended_crop)
                VALUES (?,?,?,?,?,?,?,?,?)
            ''', (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                data.N, data.P, data.K, data.temperature, data.humidity, data.ph, data.rainfall,
                crop_name
            ))
            conn.commit()
            conn.close()
        except Exception as db_error:
            print(f"DB Error: {db_error}")

        return {
            "status": "success",
            "recommended_crop": crop_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crop prediction failed: {str(e)}")

# ==================== 2. YIELD PREDICTION - 100% FIXED ====================
class YieldInput(BaseModel):
    State: str
    Season: str
    Crop: str
    Area: float
    Annual_Rainfall: float
    Fertilizer: float
    Pesticide: float

@app.post("/predict-yield", tags=["Yield Prediction"])
def predict_yield(data: YieldInput):
    try:
        # *** CRITICAL FIX: Dataset format ki convert cheyadam ***
        state_clean = data.State.strip().title()
        season_clean = data.Season.strip().upper()
        crop_clean = data.Crop.strip().capitalize()

        # Encoders tho transform cheyadam
        state = state_encoder.transform([state_clean])[0]
        season = season_encoder.transform([season_clean])[0]
        crop = crop_encoder.transform([crop_clean])[0]

        features = np.array([
            state, season, crop, data.Area,
            data.Annual_Rainfall, data.Fertilizer, data.Pesticide
        ], dtype=float).reshape(1, -1)

        yield_pred = yield_model.predict(features)[0]

        try:
            conn = sqlite3.connect('crop_data.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions
                (timestamp, state, season, recommended_crop, predicted_yield)
                VALUES (?,?,?,?,?)
            ''', (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                state_clean, season_clean, crop_clean,
                round(float(yield_pred), 2)
            ))
            conn.commit()
            conn.close()
        except Exception as db_error:
            print(f"Yield DB Error: {db_error}")

        return {
            "status": "success",
            "matched_values": {
                "State": state_clean,
                "Season": season_clean,
                "Crop": crop_clean
            },
            "predicted_yield": round(float(yield_pred), 2),
            "unit": "ton/hectare"
        }
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: {str(e)}. Check spelling. Valid Crops: {list(crop_encoder.classes_)[:10]}..."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Yield prediction failed: {str(e)}")

@app.get("/")
def home():
    return FileResponse('index.html')

# *** BRACKET FIX CHESA IKKADA ***
@app.get("/history", tags=["History"])
def get_history():
    conn = sqlite3.connect('crop_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 20")
    rows = cursor.fetchall()
    conn.close()

    columns = ['id','timestamp','N','P','K','temperature','humidity','ph','rainfall','state','season','recommended_crop','predicted_yield']
    history = [dict(zip(columns, row)) for row in rows] # ] bracket add chesa
    return history

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
