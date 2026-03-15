"""
RBAC Middleware - Control de Acceso Basado en Roles
====================================================

Centraliza validación de permisos para:
- Acceso a datos por org_id
- Operaciones según rol (master_admin, org_admin, viewer)
- Auditoría de acciones
"""

from __future__ import annotations

import logging
from typing import Optional, Callable, Any
from functools import wraps
from datetime import datetime

import streamlit as st

logger = logging.getLogger(__name__)


class RBACMiddleware:
    """
    Middleware para validación centralizada de permisos.
    
    Uso:
    ```python
    @RBACMiddleware.require_role("org_admin", "master_admin")
    @RBACMiddleware.require_org_access
    def upload_data():
        # Solo accessible si user es org_admin o master_admin
        # Y tiene acceso a la org actual
        pass
    ```
    """
    
    # Definir permisos por rol
    ROLE_PERMISSIONS = {
        "master_admin": {
            "view_all_orgs": True,
            "manage_organizations": True,
            "manage_all_users": True,
            "manage_all_data": True,
            "view_audit_logs": True,
            "upload_data": True,
        },
        "org_admin": {
            "manage_org_users": True,
            "upload_data": True,
            "view_org_data": True,
            "manage_org_settings": True,
        },
        "viewer": {
            "view_org_data": True,
        },
    }
    
    @staticmethod
    def get_user_permissions(role: str) -> Dict[str, bool]:
        """Obtiene permisos para un rol."""
        return RBACMiddleware.ROLE_PERMISSIONS.get(role, {})
    
    @staticmethod
    def has_permission(role: str, permission: str) -> bool:
        """Verifica si un rol tiene cierto permiso."""
        perms = RBACMiddleware.get_user_permissions(role)
        return perms.get(permission, False)
    
    @staticmethod
    def require_role(*allowed_roles):
        """
        Decorador: Requiere que el usuario tenga uno de los roles especificados.
        
        Uso:
        ```python
        @RBACMiddleware.require_role("org_admin", "master_admin")
        def page_upload():
            pass
        ```
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                user_role = st.session_state.get("role")
                
                if not user_role:
                    st.error("❌ No estás autenticado")
                    return None
                
                if user_role not in allowed_roles:
                    st.error(
                        f"❌ Acceso denegado. "
                        f"Se requiere rol: {', '.join(allowed_roles)}. "
                        f"Tu rol actual: {user_role}"
                    )
                    logger.warning(
                        f"RBAC DENIED: User {st.session_state.get('email')} "
                        f"tried to access {func.__name__} with role {user_role}"
                    )
                    return None
                
                # Log acceso permitido
                logger.info(
                    f"RBAC ALLOWED: User {st.session_state.get('email')} "
                    f"({user_role}) accessed {func.__name__}"
                )
                
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    @staticmethod
    def require_permission(permission: str):
        """
        Decorador: Requiere que el usuario tenga cierto permiso.
        
        Uso:
        ```python
        @RBACMiddleware.require_permission("upload_data")
        def page_upload():
            pass
        ```
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                user_role = st.session_state.get("role")
                
                if not user_role:
                    st.error("❌ No estás autenticado")
                    return None
                
                if not RBACMiddleware.has_permission(user_role, permission):
                    st.error(
                        f"❌ Acceso denegado. "
                        f"No tienes permiso para: {permission}"
                    )
                    logger.warning(
                        f"RBAC DENIED: User {st.session_state.get('email')} "
                        f"lacks permission {permission}"
                    )
                    return None
                
                logger.info(
                    f"RBAC ALLOWED: User {st.session_state.get('email')} "
                    f"accessed permission {permission}"
                )
                
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    @staticmethod
    def require_org_access(func):
        """
        Decorador: Valida que el usuario tenga acceso a la org actual.
        
        Uso:
        ```python
        @RBACMiddleware.require_org_access
        def load_org_data():
            pass
        ```
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            org_id = st.session_state.get("org_id")
            user_orgs = st.session_state.get("user_orgs", [])
            user_email = st.session_state.get("email", "unknown")
            
            if not org_id:
                st.error("❌ No hay organización seleccionada")
                return None
            
            # Check si org está en user_orgs
            org_access_list = [org["org_id"] for org in user_orgs]
            
            if org_id not in org_access_list:
                st.error("❌ No tienes acceso a esta organización")
                logger.error(
                    f"RBAC DENIED: User {user_email} tried to access "
                    f"org {org_id} without permission"
                )
                return None
            
            logger.info(
                f"RBAC ALLOWED: User {user_email} accessed org {org_id}"
            )
            
            return func(*args, **kwargs)
        return wrapper
    
    @staticmethod
    def validate_org_id(org_id: str) -> bool:
        """
        Valida que el usuario tenga acceso a una org específica.
        
        Uso:
        ```python
        if RBACMiddleware.validate_org_id(org_id):
            # Safe to proceed
        ```
        """
        user_orgs = st.session_state.get("user_orgs", [])
        org_access_list = [org["org_id"] for org in user_orgs]
        
        return org_id in org_access_list
    
    @staticmethod
    def audit_action(
        action: str,
        org_id: str,
        details: Dict[str, Any] | None = None,
        status: str = "success",
    ) -> None:
        """
        Registra acción para auditoría.
        
        Uso:
        ```python
        RBACMiddleware.audit_action(
            action="upload_data",
            org_id=st.session_state.org_id,
            details={"file_name": "data.csv", "rows": 1000},
            status="success"
        )
        ```
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": st.session_state.get("user_id"),
            "user_email": st.session_state.get("email"),
            "org_id": org_id,
            "action": action,
            "details": details or {},
            "status": status,
            "role": st.session_state.get("role"),
        }
        
        logger.info(f"AUDIT: {log_entry}")
        
        # TODO: Guardar en Supabase audit_logs table
    
    @staticmethod
    def check_data_access(org_id: str, user_id: str) -> bool:
        """
        Verifica acceso a datos de una org.
        
        Implementa RLS (Row-Level Security) a nivel de aplicación.
        """
        if not RBACMiddleware.validate_org_id(org_id):
            logger.error(f"Data access denied: {user_id} -> {org_id}")
            return False
        
        return True


class RBACContext:
    """
    Context manager para operaciones bajo RBAC.
    
    Uso:
    ```python
    with RBACContext(st.session_state.org_id):
        # Todas las operaciones aquí son auditadas
        # y validadas para la org
        ...
    ```
    """
    
    def __init__(self, org_id: str):
        self.org_id = org_id
        self.start_time = datetime.now()
    
    def __enter__(self):
        if not RBACMiddleware.validate_org_id(self.org_id):
            raise PermissionError(f"No access to org {self.org_id}")
        
        logger.info(f"RBAC Context: Entered org {self.org_id}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type:
            logger.error(
                f"RBAC Context: Error in org {self.org_id}: {exc_val}"
            )
        else:
            logger.info(
                f"RBAC Context: Exited org {self.org_id} ({elapsed:.2f}s)"
            )


# ============================================================
# SHORTCUTS PARA DECORADORES COMUNES
# ============================================================

def require_master_admin(func):
    """Requiere master_admin role."""
    return RBACMiddleware.require_role("master_admin")(func)


def require_org_admin(func):
    """Requiere org_admin o master_admin role."""
    return RBACMiddleware.require_role("org_admin", "master_admin")(func)


def require_authenticated(func):
    """Requiere estar autenticado."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not st.session_state.get("authenticated"):
            st.error("❌ Debes estar autenticado")
            return None
        return func(*args, **kwargs)
    return wrapper


# ============================================================
# FUNCIONES HELPER
# ============================================================

def get_user_role_display() -> str:
    """Obtiene display string del rol actual."""
    role = st.session_state.get("role", "unknown")
    
    role_display = {
        "master_admin": "👑 Master Admin",
        "org_admin": "🔑 Org Admin",
        "viewer": "👁️ Viewer",
    }
    
    return role_display.get(role, role)


def check_feature_access(feature: str) -> bool:
    """
    Verifica si usuario puede acceder a feature.
    
    Uso:
    ```python
    if check_feature_access("ai_forecasting"):
        # Show AI forecasting feature
    ```
    """
    role = st.session_state.get("role")
    
    feature_access = {
        "upload_data": ["org_admin", "master_admin"],
        "manage_users": ["master_admin", "org_admin"],
        "view_analytics": ["master_admin", "org_admin", "viewer"],
        "export_data": ["org_admin", "master_admin"],
        "ai_forecasting": ["org_admin", "master_admin"],
        "manage_orgs": ["master_admin"],
    }
    
    allowed_roles = feature_access.get(feature, [])
    return role in allowed_roles
