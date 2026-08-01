from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
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

try:
    predictor = FireRiskPredictor()
    model_loaded = True
except Exception as e:
    predictor = None
    model_loaded = False
    print(f"Model yüklenirken hata oluştu: {e}")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"model_loaded": model_loaded})

@app.post("/predict")
async def predict_fire_risk(request: Request):
    if not model_loaded:
        return JSONResponse(status_code=503, content={"error": "Model henüz hazır değil. Lütfen eğitim bitene kadar bekleyin."})
    
    try:
        data = await request.json()
        raw_data = {
            "X": int(data.get("X", 7)), "Y": int(data.get("Y", 5)),
            "month": data.get("month", "aug"), "day": data.get("day", "fri"),
            "FFMC": float(data.get("FFMC", 96.1)), "DMC": float(data.get("DMC", 181.1)),
            "DC": float(data.get("DC", 671.2)), "ISI": float(data.get("ISI", 14.3)),
            "temp": float(data.get("temp", 28.7)), "RH": float(data.get("RH", 30.0)),
            "wind": float(data.get("wind", 4.5)), "rain": float(data.get("rain", 0.0))
        }
        
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

@app.post("/api/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    def run_script():
        subprocess.run([sys.executable, "src/retrain.py"], cwd=PROJECT_ROOT)
    
    background_tasks.add_task(run_script)
    return JSONResponse({
        "status": "success",
        "message": "Retrain süreci arka planda başlatıldı. Sunucu loglarını takip edebilirsiniz."
    })

@app.post("/api/report")
async def report_fire_data(request: Request):
    try:
        data = await request.json()
        
        row = [
            int(data.get("X", 7)), int(data.get("Y", 5)),
            data.get("month", "aug"), data.get("day", "fri"),
            float(data.get("FFMC", 0.0)), float(data.get("DMC", 0.0)),
            float(data.get("DC", 0.0)), float(data.get("ISI", 0.0)),
            float(data.get("temp", 0.0)), float(data.get("RH", 0.0)),
            float(data.get("wind", 0.0)), float(data.get("rain", 0.0)),
            float(data.get("area", 0.0)) # <-- Yeni alanımız
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
            
        return JSONResponse({"success": True, "message": "Rapor başarıyla kaydedildi! Veri setine eklendi."})
        
    except Exception as e:
        return JSONResponse(status_code=400, content={"success": False, "error": str(e)})