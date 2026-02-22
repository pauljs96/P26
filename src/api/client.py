"""
Cliente HTTP para comunicarse con el FastAPI Backend
Usado por el Dashboard (Streamlit)
"""
import requests
from typing import Dict, Any, Optional
import json

class APIClient:
    """Cliente para llamar endpoints del backend"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.timeout = 30
    
    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Hace una request HTTP al API
        
        Returns:
            {"success": bool, "data": response_data, "error": error_msg}
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, params=params, timeout=self.timeout)
            elif method.upper() == "POST":
                response = requests.post(url, json=data, timeout=self.timeout)
            else:
                raise ValueError(f"Método HTTP no soportado: {method}")
            
            response.raise_for_status()
            return {
                "success": True,
                "data": response.json(),
                "status_code": response.status_code
            }
        
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Timeout: El servidor tardó demasiado",
                "status_code": None
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": f"No se puede conectar al API en {self.base_url}\nVerifica que run_both.py está ejecutándose",
                "status_code": None
            }
        except requests.exceptions.HTTPError as e:
            try:
                error_detail = response.json().get("detail", str(e))
            except:
                error_detail = str(e)
            
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {error_detail}",
                "status_code": response.status_code
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error: {str(e)}",
                "status_code": None
            }
    
    # ==================== HEALTH ====================
    
    def health_check(self) -> bool:
        """Verifica que el API está online"""
        result = self._make_request("GET", "/health")
        return result.get("success", False)
    
    # ==================== UPLOADS ====================
    
    def process_upload(
        self,
        upload_id: str,
        user_id: str,
        project_id: str,
        filename: str,
        s3_path: str
    ) -> Dict[str, Any]:
        """
        Procesa un CSV subido a S3
        (validación, limpieza, extracción)
        """
        data = {
            "upload_id": upload_id,
            "user_id": user_id,
            "project_id": project_id,
            "filename": filename,
            "s3_path": s3_path
        }
        return self._make_request("POST", "/uploads/process", data=data)
    
    def get_upload_status(self, upload_id: str) -> Dict[str, Any]:
        """Obtiene el status de procesamiento de un upload"""
        return self._make_request("GET", f"/uploads/{upload_id}/status")
    
    # ==================== FORECASTS ====================
    
    def generate_forecast(
        self,
        upload_id: str,
        product: str,
        model_type: str = "ets",
        forecast_periods: int = 12
    ) -> Dict[str, Any]:
        """
        Genera pronóstico para un producto
        
        model_type: "ets", "rf", o "best"
        """
        data = {
            "upload_id": upload_id,
            "product": product,
            "model_type": model_type,
            "forecast_periods": forecast_periods
        }
        return self._make_request("POST", "/forecasts/generate", data=data)
    
    def get_forecast(
        self, 
        upload_id: str, 
        product: str
    ) -> Dict[str, Any]:
        """Obtiene un pronóstico guardado"""
        return self._make_request("GET", f"/forecasts/{upload_id}/{product}")


# Cliente global (singleton)
_client_instance: Optional[APIClient] = None

def get_api_client(base_url: str = "http://localhost:8000") -> APIClient:
    """Retorna una instancia global del cliente API"""
    global _client_instance
    if _client_instance is None:
        _client_instance = APIClient(base_url)
    return _client_instance
