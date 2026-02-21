"""
Módulo de integración con Supabase (PostgreSQL cloud).

Maneja:
- Autenticación de usuarios
- Persistencia de proyectos/uploads
- Histórico de backtests y recomendaciones
"""

from __future__ import annotations
import os
from typing import Optional, Dict, List, Any
import json

try:
    import supabase
    from supabase import create_client, Client
except ImportError:
    raise ImportError("Instala supabase: pip install supabase")


class SupabaseDB:
    """Cliente Supabase para operaciones CRUD"""

    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        """
        Inicializa cliente Supabase.
        
        Si url/key no se proporcionan, lee de .env
        """
        self.url = url or os.getenv("SUPABASE_URL")
        self.key = key or os.getenv("SUPABASE_KEY")
        
        if not self.url or not self.key:
            raise ValueError(
                "Supabase credenciales no encontradas. "
                "Configura SUPABASE_URL y SUPABASE_KEY en .env"
            )
        
        self.client: Client = create_client(self.url, self.key)

    # ==================== USUARIOS ====================
    
    def register_user(self, email: str, password: str, company_name: str) -> Dict[str, Any]:
        """Registra nuevo usuario/empresa en Supabase Auth"""
        try:
            response = self.client.auth.sign_up({
                "email": email,
                "password": password,
            })
            # Guardar empresa en tabla users
            user_id = response.user.id
            self.client.table("users").insert({
                "id": user_id,
                "email": email,
                "company_name": company_name,
                "created_at": "now()"
            }).execute()
            return {"success": True, "user_id": user_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def login_user(self, email: str, password: str) -> Dict[str, Any]:
        """Autentica usuario"""
        try:
            response = self.client.auth.sign_in_with_password({
                "email": email,
                "password": password,
            })
            return {
                "success": True,
                "user_id": response.user.id,
                "email": response.user.email,
                "session": response.session.access_token
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_user(self, user_id: str) -> Optional[Dict]:
        """Obtiene info de usuario"""
        try:
            response = self.client.table("users").select("*").eq("id", user_id).execute()
            return response.data[0] if response.data else None
        except Exception:
            return None

    # ==================== PROYECTOS ====================
    
    def create_project(self, user_id: str, project_name: str, description: str = "") -> Dict[str, Any]:
        """Crea nuevo proyecto para usuario"""
        try:
            response = self.client.table("projects").insert({
                "user_id": user_id,
                "name": project_name,
                "description": description,
                "created_at": "now()"
            }).execute()
            return {"success": True, "project_id": response.data[0]["id"]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_projects(self, user_id: str) -> List[Dict]:
        """Lista proyectos del usuario"""
        try:
            response = self.client.table("projects").select("*").eq("user_id", user_id).execute()
            return response.data or []
        except Exception:
            return []

    # ==================== UPLOADS (CSV) ====================
    
    def save_upload(
        self, 
        user_id: str, 
        project_id: str, 
        filename: str, 
        s3_path: str = None,
        file_size: int = 0
    ) -> Dict[str, Any]:
        """
        Registra upload de CSV con metadata de S3.
        
        Args:
            user_id: ID del usuario
            project_id: ID del proyecto
            filename: Nombre del archivo
            s3_path: Ruta en S3 (ej: users/123/projects/456/file.csv)
            file_size: Tamaño del archivo en bytes
        
        Returns:
            {"success": bool, "upload_id": str, "error": str (si falla)}
        """
        try:
            print(f"[DEBUG] Insertando en uploads: user_id={user_id}, project_id={project_id}, filename={filename}, s3_path={s3_path}")
            # Build insert dict con las columnas correctas según schema de Supabase
            insert_dict = {
                "user_id": user_id,
                "project_id": project_id,
                "filename": filename,
                "s3_path": s3_path,
                "file_size": file_size,
                "status": "pending"
            }
            response = self.client.table("uploads").insert(insert_dict).execute()
            print(f"[DEBUG] Insert exitoso: {response.data}")
            return {"success": True, "upload_id": response.data[0]["id"]}
        except Exception as e:
            print(f"[DEBUG ERROR] save_upload falló: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def get_uploads(self, project_id: str) -> List[Dict]:
        """Lista uploads de un proyecto"""
        try:
            response = self.client.table("uploads").select("*").eq("project_id", project_id).execute()
            return response.data or []
        except Exception:
            return []

    # ==================== BACKTESTS ====================
    
    def save_backtest(
        self, 
        project_id: str, 
        product_code: str, 
        model_name: str, 
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Guarda resultados de backtest"""
        try:
            response = self.client.table("backtests").insert({
                "project_id": project_id,
                "product_code": product_code,
                "model_name": model_name,
                "metrics": json.dumps(metrics),  # Guardar como JSON
                "created_at": "now()"
            }).execute()
            return {"success": True, "backtest_id": response.data[0]["id"]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_backtests(self, project_id: str, product_code: str = None) -> List[Dict]:
        """Obtiene backtests guardados"""
        try:
            query = self.client.table("backtests").select("*").eq("project_id", project_id)
            if product_code:
                query = query.eq("product_code", product_code)
            response = query.execute()
            return response.data or []
        except Exception:
            return []

    # ==================== RECOMENDACIONES ====================
    
    def save_recommendation(
        self,
        project_id: str,
        product_code: str,
        forecast: float,
        stock_safety: float,
        stock_actual: float,
        quantity_recommended: int,
        model_used: str
    ) -> Dict[str, Any]:
        """Guarda recomendación de producción"""
        try:
            response = self.client.table("recommendations").insert({
                "project_id": project_id,
                "product_code": product_code,
                "forecast": forecast,
                "stock_safety": stock_safety,
                "stock_actual": stock_actual,
                "quantity_recommended": quantity_recommended,
                "model_used": model_used,
                "created_at": "now()"
            }).execute()
            return {"success": True, "rec_id": response.data[0]["id"]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_recommendations(self, project_id: str) -> List[Dict]:
        """Lista recomendaciones de proyecto"""
        try:
            response = self.client.table("recommendations").select("*").eq("project_id", project_id).order("created_at", desc=True).execute()
            return response.data or []
        except Exception:
            return []


# Singleton global (opcional, para evitar re-crear cliente constantemente)
_db_instance: Optional[SupabaseDB] = None


def get_db() -> SupabaseDB:
    """Retorna instancia global de SupabaseDB (lazy initialization)"""
    global _db_instance
    if _db_instance is None:
        _db_instance = SupabaseDB()
    return _db_instance
