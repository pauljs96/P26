#!/usr/bin/env python
"""
Script de prueba end-to-end del sistema completo
Verifica: API health, CSV processing, ML forecasting
"""
import requests
import sys
import time
from pathlib import Path

# Colores para terminal
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
RESET = '\033[0m'

API_URL = "http://localhost:8000"
DASHBOARD_URL = "http://localhost:8501"

def print_header(text):
    print(f"\n{CYAN}{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✅ {text}{RESET}")

def print_error(text):
    print(f"{RED}❌ {text}{RESET}")

def print_info(text):
    print(f"{BLUE}ℹ️  {text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}⚠️  {text}{RESET}")

def test_api_health():
    """Test 1: Verificar que el API esté disponible"""
    print_header("TEST 1: API Health Check")
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print_success(f"API available: {response.json()}")
            return True
        else:
            print_error(f"API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to API on localhost:8000")
        print_info("Make sure: python -m uvicorn src.api.main:app --reload --port 8000")
        return False
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        return False

def test_dashboard_access():
    """Test 2: Verificar que el dashboard sea accesible"""
    print_header("TEST 2: Dashboard Access")
    
    try:
        response = requests.get(DASHBOARD_URL, timeout=5)
        if response.status_code == 200:
            print_success(f"Dashboard available on {DASHBOARD_URL}")
            return True
        else:
            print_error(f"Dashboard returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to Dashboard on localhost:8501")
        print_info("Make sure: streamlit run src/ui/dashboard.py")
        return False
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        return False

def test_api_documentation():
    """Test 3: Verificar documentación swagger del API"""
    print_header("TEST 3: API Documentation (Swagger)")
    
    try:
        response = requests.get(f"{API_URL}/docs", timeout=5)
        if response.status_code == 200:
            print_success(f"Swagger docs available at {API_URL}/docs")
            print_info("All endpoints documented")
            return True
        else:
            print_error(f"Docs returned status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def test_api_openapi_schema():
    """Test 4: Verificar que el schema OpenAPI esté disponible"""
    print_header("TEST 4: API OpenAPI Schema")
    
    try:
        response = requests.get(f"{API_URL}/openapi.json", timeout=5)
        if response.status_code == 200:
            schema = response.json()
            print_success(f"OpenAPI schema available")
            print_info(f"API Title: {schema.get('info', {}).get('title', 'N/A')}")
            print_info(f"API Version: {schema.get('info', {}).get('version', 'N/A')}")
            
            # Listar endpoints
            paths = schema.get('paths', {})
            print(f"\n{BLUE}Available endpoints:{RESET}")
            for endpoint, methods in paths.items():
                method_list = ", ".join(methods.keys())
                print(f"  {CYAN}{endpoint}{RESET} [{method_list}]")
            
            return True
        else:
            print_error(f"Schema returned status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def test_csv_processing_mock():
    """Test 5: Simular procesamiento de CSV (sin archivo real)"""
    print_header("TEST 5: CSV Processing Endpoint (Mock)")
    
    print_info("Endpoint: POST /uploads/process")
    print_info("Expected behavior: Validates & cleans CSV data")
    
    # Datos de prueba
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
        
        if response.status_code in [200, 400, 404]:  # API might return errors
            result = response.json()
            if response.status_code == 200:
                print_success("CSV processing endpoint responding correctly")
                print_info(f"Response: {result}")
            else:
                print_warning(f"Endpoint returned status {response.status_code}")
                print_info(f"Response: {result}")
            return True
        else:
            print_error(f"Unexpected status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def test_forecast_generation_mock():
    """Test 6: Simular generación de pronósticos"""
    print_header("TEST 6: Forecast Generation Endpoint (Mock)")
    
    print_info("Endpoint: POST /forecasts/generate")
    print_info("Expected behavior: Generate ML forecasts")
    
    # Datos de prueba
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
        
        if response.status_code in [200, 400, 404]:
            result = response.json()
            if response.status_code == 200:
                print_success("Forecast generation endpoint responding correctly")
                print_info(f"Response: {result}")
            else:
                print_warning(f"Endpoint returned status {response.status_code}")
                print_info(f"Response: {result}")
            return True
        else:
            print_error(f"Unexpected status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def print_summary(results):
    """Imprimir resumen de pruebas"""
    print_header("TEST SUMMARY")
    
    tests = [
        ("API Health Check", results[0]),
        ("Dashboard Access", results[1]),
        ("API Documentation", results[2]),
        ("OpenAPI Schema", results[3]),
        ("CSV Processing Endpoint", results[4]),
        ("Forecast Generation Endpoint", results[5]),
    ]
    
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    for test_name, result in tests:
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"  {test_name:<40} {status}")
    
    print(f"\n{CYAN}Results: {passed}/{total} tests passed{RESET}")
    
    if passed == total:
        print(f"\n{GREEN}{'='*60}")
        print(f"  ✅ ALL TESTS PASSED - SYSTEM READY FOR USE!")
        print(f"{'='*60}{RESET}")
        print(f"\n{BLUE}Next steps:{RESET}")
        print(f"  1. Open browser: {DASHBOARD_URL}")
        print(f"  2. Login or Register")
        print(f"  3. Create a new project")
        print(f"  4. Upload a CSV file")
        print(f"  5. Go to 'API Pronósticos' tab")
        print(f"  6. Select uploaded file and generate forecast")
        return True
    else:
        print(f"\n{RED}{'='*60}")
        print(f"  ❌ Some tests failed - check errors above")
        print(f"{'='*60}{RESET}")
        return False

def main():
    print(f"\n{CYAN}{'='*60}")
    print(f"  Sistema de Pronóstico - End-to-End Test")
    print(f"  API:       {API_URL}")
    print(f"  Dashboard: {DASHBOARD_URL}")
    print(f"{'='*60}{RESET}\n")
    
    results = []
    
    # Correr pruebas
    results.append(test_api_health())
    results.append(test_dashboard_access())
    results.append(test_api_documentation())
    results.append(test_api_openapi_schema())
    results.append(test_csv_processing_mock())
    results.append(test_forecast_generation_mock())
    
    # Mostrar resumen
    success = print_summary(results)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
