import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from fastapi.testclient import TestClient
from backend.src.api.main import app

def test_prediction():
    client = TestClient(app)

    payload = {
        "etf_name": "Test_Sector_ETF",
        "reporting_date": "2023-10-31",
        "holdings": [
            {"ticker": "RELIANCE.NS", "weight": 0.50},
            {"ticker": "TCS.NS", "weight": 0.50}
        ]
    }

    print("Sending request to /api/v1/risk/predict...")
    response = client.post("/api/v1/risk/predict", json=payload)
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Risk Class: {data['risk_class']}")
        print(f"Metrics: {data['metrics']}")
        print(f"Explanation:\n{data['explanation']}")
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    test_prediction()
