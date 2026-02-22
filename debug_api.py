#!/usr/bin/env python
"""Script para debuggear errores del API"""
import requests
import json

API_URL = "http://localhost:8000"

# Test del endpoint de uploads/process
print("Testing POST /uploads/process...")
payload = {
    "upload_id": "test-123",
    "user_id": "demo@example.com", 
    "project_id": "project-001",
    "filename": "test_data.csv",
    "s3_path": "s3://bucket/test_data.csv"
}

try:
    response = requests.post(
        f"{API_URL}/uploads/process",
        json=payload,
        timeout=10
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*60 + "\n")

# Test del endpoint de forecasts/generate
print("Testing POST /forecasts/generate...")
payload = {
    "upload_id": "test-123",
    "product": "PRODUCTO_PRUEBA",
    "model_type": "ets",
    "forecast_periods": 12
}

try:
    response = requests.post(
        f"{API_URL}/forecasts/generate",
        json=payload,
        timeout=30
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")
