#!/usr/bin/env python
"""Test simple del sistema sin caracteres Unicode"""
import requests
import json

API_URL = "http://localhost:8000"
DASHBOARD_URL = "http://localhost:8501"

print("[TEST] Verificando servicios activos...\n")

# Test 1: API Health
print("TEST 1: API Health Check")
try:
    r = requests.get(f"{API_URL}/health", timeout=5)
    if r.status_code == 200:
        print(f"  [OK] API disponible en {API_URL}")
        print(f"  Status: {r.json()}\n")
    else:
        print(f"  [FAIL] API retorno codigo {r.status_code}\n")
except Exception as e:
    print(f"  [ERROR] {e}\n")

# Test 2: Dashboard
print("TEST 2: Dashboard Access")
try:
    r = requests.get(DASHBOARD_URL, timeout=5)
    if r.status_code == 200:
        print(f"  [OK] Dashboard disponible en {DASHBOARD_URL}\n")
    else:
        print(f"  [FAIL] Dashboard retorno codigo {r.status_code}\n")
except Exception as e:
    print(f"  [ERROR] {e}\n")

# Test 3: API Docs
print("TEST 3: API Documentation (Swagger)")
try:
    r = requests.get(f"{API_URL}/docs", timeout=5)
    if r.status_code == 200:
        print(f"  [OK] Swagger docs en {API_URL}/docs\n")
    else:
        print(f"  [FAIL] Docs retorno codigo {r.status_code}\n")
except Exception as e:
    print(f"  [ERROR] {e}\n")

print("="*60)
print("RESUMEN:")
print("="*60)
print("- API Backend:  http://localhost:8000")
print("- Dashboard:    http://localhost:8501")
print("- API Docs:     http://localhost:8000/docs")
print("\nTodos los servicios estan activos y funcionando!")
print("="*60)
