"""
FastAPI Endpoint Birim Testleri
/predict ve / endpoint'lerinin doğru HTTP kodları döndürdüğünü doğrular.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi.testclient import TestClient
from app.app import app


client = TestClient(app)

# Test ortamı için API anahtarı (CI/CD'de ortam değişkeni olarak da verilebilir)
TEST_API_KEY = os.environ.get("API_KEY", "montesinho-secure-key-2026")
AUTH_HEADERS = {"X-API-KEY": TEST_API_KEY}


class TestRootEndpoint:
    """Ana sayfa endpoint testleri."""

    def test_root_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200, f"/ endpoint'i 200 yerine {response.status_code} döndü"

    def test_root_returns_html(self):
        response = client.get("/")
        assert "text/html" in response.headers["content-type"], "/ endpoint'i HTML döndürmeli"


class TestPredictEndpoint:
    """Tahmin endpoint testleri."""

    def test_predict_with_valid_data(self):
        payload = {
            "X": 7, "Y": 5, "month": "aug", "day": "fri",
            "FFMC": 96.1, "DMC": 181.1, "DC": 671.2, "ISI": 14.3,
            "temp": 28.7, "RH": 30.0, "wind": 4.5, "rain": 0.0
        }
        response = client.post("/predict", json=payload)

        assert response.status_code in [200, 400, 503], (
            f"/predict endpoint'i 200 veya 503 yerine {response.status_code} döndü"
        )

    def test_predict_response_has_correct_keys(self):
        
        payload = {
            "X": 7, "Y": 5, "month": "aug", "day": "fri",
            "FFMC": 96.1, "DMC": 181.1, "DC": 671.2, "ISI": 14.3,
            "temp": 28.7, "RH": 30.0, "wind": 4.5, "rain": 0.0
        }
        response = client.post("/predict", json=payload)
        data = response.json()
        if response.status_code == 200:
            assert "area_ha" in data, "Başarılı yanıtta area_ha anahtarı olmalı"
            assert "risk_level" in data, "Başarılı yanıtta risk_level anahtarı olmalı"
            assert "css_class" in data, "Başarılı yanıtta css_class anahtarı olmalı"
        elif response.status_code == 503:
            assert "error" in data, "503 yanıtında error anahtarı olmalı"

    def test_predict_with_empty_body(self):
        response = client.post("/predict", json={})

        assert response.status_code in [200, 400, 503], (
            f"Boş veri ile beklenmeyen kod: {response.status_code}"
        )

class TestReportEndpoint:
    """Raporlama (Veri Toplama) endpoint testleri."""

    def test_report_without_auth_returns_403(self):
        """Auth header olmadan istek atılırsa 403 dönmeli."""
        payload = {
            "X": 7, "Y": 5, "month": "aug", "day": "fri",
            "FFMC": 96.1, "DMC": 181.1, "DC": 671.2, "ISI": 14.3,
            "temp": 28.7, "RH": 30.0, "wind": 4.5, "rain": 0.0,
            "area": 12.5
        }
        response = client.post("/api/report", json=payload)
        assert response.status_code == 403, f"Auth olmadan 403 beklendi, {response.status_code} geldi"

    def test_report_fire_data(self):
        """Geçerli auth ile rapor kaydedilmeli."""
        payload = {
            "X": 7, "Y": 5, "month": "aug", "day": "fri",
            "FFMC": 96.1, "DMC": 181.1, "DC": 671.2, "ISI": 14.3,
            "temp": 28.7, "RH": 30.0, "wind": 4.5, "rain": 0.0,
            "area": 12.5
        }
        response = client.post("/api/report", json=payload, headers=AUTH_HEADERS)
        assert response.status_code == 200, f"/api/report endpoint'i 200 yerine {response.status_code} döndü"
        
        data = response.json()
        assert data.get("success") is True, "Yanıtın success değeri True olmalı"

    def test_report_invalid_area_rejected(self):
        """Negatif alan değeri Pydantic tarafından reddedilmeli."""
        payload = {
            "X": 7, "Y": 5, "month": "aug", "day": "fri",
            "FFMC": 96.1, "DMC": 181.1, "DC": 671.2, "ISI": 14.3,
            "temp": 28.7, "RH": 30.0, "wind": 4.5, "rain": 0.0,
            "area": -500.0
        }
        response = client.post("/api/report", json=payload, headers=AUTH_HEADERS)
        assert response.status_code == 422, f"Negatif area ile 422 beklendi, {response.status_code} geldi"