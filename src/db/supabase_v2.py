"""
Supabase Database Service - Multi-Tenant RBAC
==============================================
Maneja acceso a Supabase con soporte para multi-tenant.

Tablas:
- organizations: Orgs con datos aislados
- users: Metadata de usuarios autenticados
- user_org_assignments: RBAC (who belongs to org + role)
- data_uploads: Auditoría de uploads
- analysis_results: Histórico de análisis
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Dict, List, Any
from uuid import UUID
from datetime import datetime, timedelta

try:
    from supabase import create_client, Client
except ImportError:
    raise ImportError("Instala supabase: pip install supabase -U")

logger = logging.getLogger(__name__)


class SupabaseDB:
    """Cliente Supabase con soporte multi-tenant."""
    
    def __init__(self, url: str | None = None, key: str | None = None):
        """
        Inicializa cliente Supabase.
        
        Lee de .env si no se proporcionan:
        - SUPABASE_URL
        - SUPABASE_KEY (service_role key)
        """
        self.url = url or os.getenv("SUPABASE_URL")
        self.key = key or os.getenv("SUPABASE_KEY")
        
        if not self.url or not self.key:
            raise ValueError(
                "Supabase no configurado. "
                "Establece SUPABASE_URL y SUPABASE_KEY en .env"
            )
        
        self.client: Client = create_client(self.url, self.key)
        logger.info(f"✅ Supabase conectado: {self.url}")
    
    # ==================== ORGANIZATIONS ====================
    
    def create_organization(
        self,
        name: str,
        description: str = "",
        s3_folder: str | None = None,
    ) -> Dict[str, Any]:
        """
        Crea nueva organización.
        
        Args:
            name: Nombre de la org (UNIQUE)
            description: Descripción
            s3_folder: Carpeta S3 (default: auto-generated)
        
        Returns:
            {
                "success": bool,
                "org_id": str (UUID),
                "error": str
            }
        """
        try:
            org_data = {
                "name": name,
                "description": description,
                "s3_folder": s3_folder or f"data/{name.lower().replace(' ', '-')}",
                "is_active": True,
            }
            
            response = self.client.table("organizations").insert(org_data).execute()
            
            if response.data:
                org_id = response.data[0]["id"]
                logger.info(f"✅ Organización creada: {name} ({org_id})")
                return {"success": True, "org_id": org_id}
            else:
                return {"success": False, "error": "No data returned"}
        
        except Exception as e:
            error = str(e)
            logger.error(f"❌ Error creando org: {error}")
            return {"success": False, "error": error}
    
    def get_organization(self, org_id: str) -> Dict[str, Any]:
        """Obtiene detalles de una organización."""
        try:
            response = self.client.table("organizations").select("*").eq("id", org_id).execute()
            
            if response.data:
                return {"success": True, "organization": response.data[0]}
            else:
                return {"success": False, "error": "Organización no encontrada"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def list_organizations(self, filter_active: bool = True) -> Dict[str, Any]:
        """Lista todas las organizaciones."""
        try:
            query = self.client.table("organizations").select("*")
            
            if filter_active:
                query = query.eq("is_active", True)
            
            response = query.execute()
            
            return {
                "success": True,
                "organizations": response.data,
                "count": len(response.data),
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ==================== USERS ====================
    
    def register_user(
        self,
        email: str,
        password: str,
        full_name: str = "",
        is_master_admin: bool = False,
    ) -> Dict[str, Any]:
        """
        Registra nuevo usuario via Supabase Auth.
        
        Args:
            email: Email del usuario
            password: Contraseña
            full_name: Nombre completo
            is_master_admin: Si es admin master (solo staff)
        
        Returns:
            {
                "success": bool,
                "user_id": str (UUID),
                "error": str
            }
        """
        try:
            # 1. Crear en Auth
            auth_response = self.client.auth.sign_up({
                "email": email,
                "password": password,
            })
            
            user_id = auth_response.user.id
            
            # 2. Guardar metadata en tabla users
            self.client.table("users").insert({
                "id": user_id,
                "email": email,
                "full_name": full_name,
                "is_master_admin": is_master_admin,
            }).execute()
            
            logger.info(f"✅ Usuario registrado: {email} ({user_id})")
            
            return {"success": True, "user_id": user_id}
        
        except Exception as e:
            error = str(e)
            logger.error(f"❌ Error registrando usuario: {error}")
            return {"success": False, "error": error}
    
    def get_user(self, user_id: str) -> Dict[str, Any]:
        """Obtiene info de usuario."""
        try:
            response = self.client.table("users").select("*").eq("id", user_id).execute()
            
            if response.data:
                return {"success": True, "user": response.data[0]}
            else:
                return {"success": False, "error": "Usuario no encontrado"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_user_by_email(self, email: str) -> Dict[str, Any]:
        """Obtiene usuario por email."""
        try:
            response = self.client.table("users").select("*").eq("email", email).execute()
            
            if response.data:
                return {"success": True, "user": response.data[0]}
            else:
                return {"success": False, "error": "Usuario no encontrado"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ==================== RBAC ====================
    
    def assign_user_to_org(
        self,
        user_id: str,
        org_id: str,
        role: str = "viewer",
    ) -> Dict[str, Any]:
        """
        Asigna usuario a organización con rol.
        
        Args:
            user_id: ID del usuario
            org_id: ID de la org
            role: "master_admin", "org_admin", "viewer"
        
        Returns:
            {"success": bool, "error": str}
        """
        if role not in ["master_admin", "org_admin", "viewer"]:
            return {"success": False, "error": f"Rol inválido: {role}"}
        
        try:
            # Obtener role ID
            role_response = self.client.table("roles").select("id").eq("name", role).execute()
            
            if not role_response.data:
                return {"success": False, "error": f"Rol no encontrado: {role}"}
            
            role_id = role_response.data[0]["id"]
            
            # Asignar usuario a org
            assignment = {
                "user_id": user_id,
                "org_id": org_id,
                "role_id": role_id,
            }
            
            response = self.client.table("user_org_assignments").insert(assignment).execute()
            
            if response.data:
                logger.info(f"✅ Usuario asignado a org como {role}")
                return {"success": True}
            else:
                return {"success": False, "error": "No data returned"}
        
        except Exception as e:
            error = str(e)
            if "duplicate" in error.lower():
                logger.warning(f"⚠️ Usuario ya asignado a org")
                return {"success": True}  # Idempotent
            logger.error(f"❌ Error asignando usuario: {error}")
            return {"success": False, "error": error}
    
    def remove_user_from_org(
        self,
        user_id: str,
        org_id: str,
    ) -> Dict[str, Any]:
        """Remueve usuario de org."""
        try:
            response = self.client.table("user_org_assignments").delete().eq(
                "user_id", user_id
            ).eq("org_id", org_id).execute()
            
            logger.info(f"✅ Usuario removido de org")
            return {"success": True}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_user_organizations(self, user_id: str) -> Dict[str, Any]:
        """Obtiene orgs a las que pertenece un usuario."""
        try:
            response = self.client.table("user_org_assignments").select(
                "organizations!inner(id, name, description), roles!inner(name)"
            ).eq("user_id", user_id).execute()
            
            organizations = []
            for item in response.data:
                org = item.get("organizations", {})
                role = item.get("roles", {}).get("name", "viewer")
                organizations.append({
                    "org_id": org.get("id"),
                    "org_name": org.get("name"),
                    "role": role,
                })
            
            return {
                "success": True,
                "organizations": organizations,
                "count": len(organizations),
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_user_role_in_org(
        self,
        user_id: str,
        org_id: str,
    ) -> Dict[str, Any]:
        """Obtiene rol de usuario en una org específica."""
        try:
            response = self.client.table("user_org_assignments").select(
                "roles!inner(name)"
            ).eq("user_id", user_id).eq("org_id", org_id).execute()
            
            if response.data:
                role = response.data[0].get("roles", {}).get("name", "viewer")
                return {"success": True, "role": role}
            else:
                return {"success": False, "error": "Usuario no asignado a esta org"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_org_members(self, org_id: str) -> Dict[str, Any]:
        """Obtiene miembros de una org."""
        try:
            response = self.client.table("user_org_assignments").select(
                "users!inner(id, email, full_name), roles!inner(name)"
            ).eq("org_id", org_id).execute()
            
            members = []
            for item in response.data:
                user = item.get("users", {})
                role = item.get("roles", {}).get("name", "viewer")
                members.append({
                    "user_id": user.get("id"),
                    "email": user.get("email"),
                    "full_name": user.get("full_name"),
                    "role": role,
                })
            
            return {
                "success": True,
                "members": members,
                "count": len(members),
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ==================== DATA UPLOADS ====================
    
    def log_upload(
        self,
        org_id: str,
        uploaded_by: str,
        file_name: str,
        s3_path: str,
        year: str | None = None,
        file_size_mb: float | None = None,
        rows_processed: int | None = None,
    ) -> Dict[str, Any]:
        """
        Registra upload de archivo (auditoría).
        
        Args:
            org_id: ID de org
            uploaded_by: User ID quien subió
            file_name: Nombre del archivo
            s3_path: Path en S3
            year: Año del dato
            file_size_mb: Tamaño en MB
            rows_processed: Filas procesadas
        
        Returns:
            {"success": bool, "upload_id": str}
        """
        try:
            upload_data = {
                "org_id": org_id,
                "uploaded_by": uploaded_by,
                "file_name": file_name,
                "s3_path": s3_path,
                "year": year,
                "file_size_mb": file_size_mb,
                "rows_processed": rows_processed,
                "status": "success",
            }
            
            response = self.client.table("data_uploads").insert(upload_data).execute()
            
            if response.data:
                logger.info(f"✅ Upload registrado: {file_name}")
                return {"success": True, "upload_id": response.data[0]["id"]}
            else:
                return {"success": False, "error": "No data returned"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_org_uploads(self, org_id: str) -> Dict[str, Any]:
        """Obtiene histórico de uploads de una org."""
        try:
            response = self.client.table("data_uploads").select("*").eq(
                "org_id", org_id
            ).order("created_at", desc=True).execute()
            
            return {
                "success": True,
                "uploads": response.data,
                "count": len(response.data),
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ==================== ANALYSIS RESULTS ====================
    
    def save_analysis_result(
        self,
        org_id: str,
        performed_by: str,
        analysis_type: str,
        results_summary: Dict[str, Any],
        s3_results_path: str | None = None,
    ) -> Dict[str, Any]:
        """
        Guarda resultado de análisis en base de datos.
        
        Args:
            org_id: ID de org
            performed_by: User ID
            analysis_type: Tipo de análisis (ej: "demand_forecast", "ets_model")
            results_summary: Resumen de resultados (JSON)
            s3_results_path: Path a resultados detallados en S3
        
        Returns:
            {"success": bool, "result_id": str}
        """
        try:
            result_data = {
                "org_id": org_id,
                "performed_by": performed_by,
                "analysis_type": analysis_type,
                "results_summary": results_summary,
                "s3_results_path": s3_results_path,
            }
            
            response = self.client.table("analysis_results").insert(result_data).execute()
            
            if response.data:
                logger.info(f"✅ Resultado guardado: {analysis_type}")
                return {"success": True, "result_id": response.data[0]["id"]}
            else:
                return {"success": False, "error": "No data returned"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_org_results(
        self,
        org_id: str,
        analysis_type: str | None = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Obtiene análisis guardados de una org."""
        try:
            query = self.client.table("analysis_results").select("*").eq("org_id", org_id)
            
            if analysis_type:
                query = query.eq("analysis_type", analysis_type)
            
            response = query.order("created_at", desc=True).limit(limit).execute()
            
            return {
                "success": True,
                "results": response.data,
                "count": len(response.data),
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ==================== BULK OPERATIONS ====================
    
    def setup_demo_organizations(self) -> Dict[str, Any]:
        """
        Crea 10 organizaciones de demo para presentación.
        
        Returns:
            {"success": bool, "organizations": [{org_id, name}]}
        """
        org_names = [
            "Tech Innovations Inc",
            "Global Retail Corp",
            "Manufacturing Solutions",
            "Energy Systems Ltd",
            "Healthcare Services",
            "Financial Services Group",
            "Logistics Network",
            "Agriculture Systems",
            "Construction Materials",
            "Transportation Solutions",
        ]
        
        created_orgs = []
        
        for name in org_names:
            result = self.create_organization(
                name=name,
                description=f"Demo organization: {name}",
                s3_folder=f"demo/{name.lower().replace(' ', '_')}",
            )
            
            if result["success"]:
                created_orgs.append({
                    "org_id": result["org_id"],
                    "name": name,
                })
        
        logger.info(f"✅ {len(created_orgs)} organizaciones de demo creadas")
        
        return {
            "success": len(created_orgs) == len(org_names),
            "organizations": created_orgs,
            "count": len(created_orgs),
        }


# ==================== SINGLETON ====================

_db_client: SupabaseDB | None = None


def get_supabase_db() -> SupabaseDB:
    """
    Obtiene instancia singleton del cliente Supabase.
    
    Uso:
    ```python
    from src.db.supabase import get_supabase_db
    
    db = get_supabase_db()
    orgs = db.list_organizations()
    ```
    """
    global _db_client
    if _db_client is None:
        _db_client = SupabaseDB()
    return _db_client
