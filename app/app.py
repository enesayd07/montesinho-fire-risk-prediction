from fastapi import FastAPI, Request, BackgroundTasks, Depends, HTTPException, Security
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import Literal
import subprocess
import sys
import os
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from predictor import FireRiskPredictor

app = FastAPI(title="Montesinho Fire Risk API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# --- GÜVENLİK (AUTH) ---
API_KEY = "montesinho-secure-key-2026"
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=True)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Yetkisiz Erişim: Geçersiz API Anahtarı")
    return api_key

# --- PYDANTIC DOĞRULAMA ŞEMALARI (VALIDATION) ---
class FirePredictionSchema(BaseModel):
    X: int = Field(default=7, ge=1, le=9)
    Y: int = Field(default=5, ge=1, le=9)
    month: Literal['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'] = 'aug'
    day: Literal['mon','tue','wed','thu','fri','sat','sun'] = 'fri'
    FFMC: float = Field(default=96.1, ge=0.0, le=100.0)
    DMC: float = Field(default=181.1, ge=0.0)
    DC: float = Field(default=671.2, ge=0.0)
    ISI: float = Field(default=14.3, ge=0.0)
    temp: float = Field(default=28.7)
    RH: float = Field(default=30.0, ge=0.0, le=100.0)
    wind: float = Field(default=4.5, ge=0.0)
    rain: float = Field(default=0.0, ge=0.0)

class FireReportSchema(FirePredictionSchema):
    area: float = Field(..., ge=0.0, description="Yanan alan (hektar) negatif olamaz.")

# --- MODEL YÜKLEME ---
try:
    predictor = FireRiskPredictor()
    predictor._load_model()
    model_loaded = True
except Exception as e:
    predictor = None
    model_loaded = False
    print(f"Model yüklenirken hata oluştu: {e}")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"model_loaded": model_loaded})

@app.post("/predict")
async def predict_fire_risk(payload: FirePredictionSchema):
    if not model_loaded:
        return JSONResponse(status_code=503, content={"error": "Model henüz hazır değil. Lütfen bekleyin."})
    
    try:
        raw_data = payload.model_dump()
        result, margin = predictor.predict(raw_data)
        
        if result <= 5.0:
            risk_level, css_class = "DÜŞÜK RİSK", "risk-low"
        elif result <= 20.0:
            risk_level, css_class = "ORTA RİSK", "risk-medium"
        elif result <= 50.0:
            risk_level, css_class = "YÜKSEK RİSK", "risk-high"
        else:
            risk_level, css_class = "KRİTİK RİSK", "risk-critical"
            
        return JSONResponse({
            "success": True, 
            "area_ha": round(float(result), 2), 
            "margin_ha": round(float(margin), 2),
            "lower_bound": max(0.0, round(float(result - margin), 2)),
            "upper_bound": round(float(result + margin), 2),
            "risk_level": risk_level,
            "css_class": css_class
        })
        
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/health")
async def health_check():
    return JSONResponse({
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded
    })

# --- GÜVENLİ (AUTH KORUMALI) ENDPOINTLER ---

@app.post("/api/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks, api_key: str = Depends(get_api_key)):
    def run_script():
        subprocess.run([sys.executable, "src/retrain.py"], cwd=PROJECT_ROOT)
    
    background_tasks.add_task(run_script)
    return JSONResponse({
        "status": "success",
        "message": "Retrain süreci arka planda güvenli şekilde başlatıldı."
    })

@app.post("/api/report")
async def report_fire_data(payload: FireReportSchema, api_key: str = Depends(get_api_key)):
    try:
        data = payload.model_dump()
        
        row = [
            data["X"], data["Y"], data["month"], data["day"],
            data["FFMC"], data["DMC"], data["DC"], data["ISI"],
            data["temp"], data["RH"], data["wind"], data["rain"],
            data["area"]
        ]
        
        collected_dir = os.path.join(PROJECT_ROOT, "data", "collected")
        os.makedirs(collected_dir, exist_ok=True)
        
        csv_path = os.path.join(collected_dir, "new_data.csv")
        file_exists = os.path.isfile(csv_path)
        
        with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["X", "Y", "month", "day", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain", "area"])
            writer.writerow(row)
            
        return JSONResponse({"success": True, "message": "Rapor doğrulandı ve veri setine eklendi."})
        
    except Exception as e:
        return JSONResponse(status_code=400, content={"success": False, "error": str(e)})