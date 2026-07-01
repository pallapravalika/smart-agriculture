from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle
import sqlite3
from datetime import datetime
import numpy as np
from contextlib import asynccontextmanager
# --- Global Variables for Models ---
crop_model = None
label_encoder = None
yield_model = None
crop_encoder = None
season_encoder = None
state_encoder = None

# --- Global Variable for Latest Sensor Data ---
latest_sensor_data = {}

# --- DB Init ---
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
            recommended_crop TEXT, predicted_yield REAL
        )
    ''')
    conn.commit()
    conn.close()
    print(">>> Database initialized: crop_data.db created <<<")

# --- Lifespan - OKATE SAARI MATRAMAE ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global crop_model, label_encoder, yield_model, crop_encoder, season_encoder, state_encoder
    init_db()
    try:
        print("Loading models...")
        crop_model = pickle.load(open('crop_recommendation_model.pkl', 'rb'))
        print("1. crop_model done")
        label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
        print("2. label_encoder done")
        yield_model = pickle.load(open('yield_model.pkl', 'rb'))
        print("3. yield_model done")
        crop_encoder = pickle.load(open('crop_encoder.pkl', 'rb'))
        print("4. crop_encoder done")
        season_encoder = pickle.load(open('season_encoder.pkl', 'rb'))
        print("5. season_encoder done")
        state_encoder = pickle.load(open('state_encoder.pkl', 'rb'))
        print("✅ All 6 models loaded!")
    except Exception as e:
        print(f"❌ ASALU ERROR: {e}")
        print(f"❌ ERROR TYPE: {type(e).__name__}")
    yield
    # Shutdown
    print("Shutting down...")

# --- FastAPI App ---
app = FastAPI(
    title="Crop Recommendation & Yield Prediction API",
    version="2.0",
    lifespan=lifespan
)

# --- Templates ---
templates = Jinja2Templates(directory="templates")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class CropInput(BaseModel):
    N: int
    P: int
    K: int
    temperature: float
    humidity: float
    ph: float
    rainfall: float

class YieldInput(BaseModel):
    State: str
    Season: str
    Crop: str
    Area: float
    Annual_Rainfall: float
    Fertilizer: float
    Pesticide: float

class SimulationInput(BaseModel):
    rainfall: float
    fertilizer: float
    current_yield: float
    plant_height: float

# --- NEW: ESP32 Data Model ---
class SensorData(BaseModel):
    temperature: float
    humidity: float
    pH: float
    soilMoisture: int
    turbidity: float
    rainValue: int
    waterFlow: float

# --- API Routes ---
@app.post("/recommend-crop", tags=["Crop Recommendation"])
def recommend_crop(data: CropInput):
    if crop_model is None:
        raise HTTPException(status_code=503, detail="Crop model not loaded")
    try:
        features = np.array([
            data.N, data.P, data.K,
            data.temperature, data.humidity, data.ph, data.rainfall
        ], dtype=float).reshape(1, -1)
        prediction = crop_model.predict(features)
        crop_name = label_encoder.inverse_transform(prediction)[0]
        try:
            conn = sqlite3.connect('crop_data.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions (timestamp, N, P, K, temperature, humidity, ph, rainfall, recommended_crop)
                VALUES (?,?,?,?,?,?,?,?,?)
            ''', (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                data.N, data.P, data.K,
                data.temperature, data.humidity, data.ph, data.rainfall,
                crop_name
            ))
            conn.commit()
            conn.close()
        except Exception as db_error:
            print(f"DB Error: {db_error}")
        return {"status": "success", "recommended_crop": crop_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crop prediction failed: {str(e)}")

@app.post("/predict-yield", tags=["Yield Prediction"])
def predict_yield(data: YieldInput):
    if yield_model is None:
        raise HTTPException(status_code=503, detail="Yield model not loaded")
    try:
        state = state_encoder.transform([data.State.strip().title()])[0]
        season = season_encoder.transform([data.Season.strip().upper()])[0]
        crop = crop_encoder.transform([data.Crop.strip().capitalize()])[0]
        features = np.array([
            state, season, crop,
            data.Area, data.Annual_Rainfall, data.Fertilizer, data.Pesticide
        ], dtype=float).reshape(1, -1)
        yield_pred = yield_model.predict(features)[0]
        try:
            conn = sqlite3.connect('crop_data.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions (timestamp, state, season, recommended_crop, predicted_yield)
                VALUES (?,?,?,?,?)
            ''', (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                data.State, data.Season, data.Crop,
                round(float(yield_pred), 2)
            ))
            conn.commit()
            conn.close()
        except Exception as db_error:
            print(f"Yield DB Error: {db_error}")
        return {
            "status": "success",
            "predicted_yield": round(float(yield_pred), 2),
            "unit": "ton/hectare"
        }
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: {str(e)}. Valid Crops: {list(crop_encoder.classes_)[:10]}..."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Yield prediction failed: {str(e)}")

# --- NEW: ESP32 Routes ---
@app.post("/api/sensor-data")
async def receive_sensor_data(data: SensorData):
    global latest_sensor_data
    latest_sensor_data = data.dict()
    print("ESP32 nundi vachina data:", latest_sensor_data)
    return {"message": "Data received", "status": 200}

@app.get("/api/get-data")
async def get_latest_data():
    return latest_sensor_data

# --- HTML Routes ---
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )

@app.get("/twin", response_class=HTMLResponse)
def digital_twin(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="twin.html"
    )

@app.get("/test")
def test():
    return {"message": "Test route working"}

@app.get("/history", tags=["History"])
def get_history():
    conn = sqlite3.connect('crop_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 20")
    rows = cursor.fetchall()
    conn.close()
    columns = [
        'id','timestamp','N','P','K',
        'temperature','humidity','ph','rainfall',
        'state','season','recommended_crop','predicted_yield'
    ]
    history = [dict(zip(columns, row)) for row in rows]
    return history

@app.post("/simulate")
def simulate(data: SimulationInput):
    new_yield = data.current_yield + (data.rainfall * 0.001) - (data.fertilizer * 0.01)
    new_height = data.plant_height + (data.rainfall * 0.0005)
    new_status = "Healthy" if new_yield > 3.5 else "Weak"
    return {
        "yield": round(new_yield, 2),
        "height": round(new_height, 2),
        "status": new_status
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
