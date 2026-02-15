"""S3 Storage Manager - Upload/Download archivos a AWS S3."""

from __future__ import annotations

import os
import io
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class S3Manager:
    """
    Gestiona upload/download de archivos a AWS S3.
    
    Características:
    - Upload de archivos locales a S3
    - Generación de URLs presignadas (descargas públicas)
    - Listado de archivos por usuario/proyecto
    - Eliminación de archivos
    - Fallback a session storage si S3 no está configurado
    """
    
    def __init__(
        self,
        bucket_name: str | None = None,
        region: str = "us-east-1",
        access_key: str | None = None,
        secret_key: str | None = None,
    ):
        """
        Inicializa S3Manager.
        
        Si no se proporcionan credenciales, intenta leer de env vars:
        - AWS_ACCESS_KEY_ID
        - AWS_SECRET_ACCESS_KEY
        - AWS_S3_BUCKET_NAME
        - AWS_S3_REGION
        """
        self.bucket_name = bucket_name or os.getenv("AWS_S3_BUCKET_NAME")
        self.region = region or os.getenv("AWS_S3_REGION", "us-east-1")
        self.access_key = access_key or os.getenv("AWS_ACCESS_KEY_ID")
        self.secret_key = secret_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        
        self.s3_client = None
        self.is_configured = False
        
        # Intentar conectar a S3
        if all([self.bucket_name, self.access_key, self.secret_key]):
            try:
                self.s3_client = boto3.client(
                    "s3",
                    region_name=self.region,
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key,
                )
                # Test connection
                self.s3_client.head_bucket(Bucket=self.bucket_name)
                self.is_configured = True
                logger.info(f"✅ S3 conectado: {self.bucket_name} ({self.region})")
            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                logger.warning(
                    f"⚠️ No se pudo conectar a S3: {error_code}. "
                    f"Usando fallback (session storage)"
                )
                self.is_configured = False
    
    def upload_file(
        self,
        file_path: str | Path,
        s3_key: str | None = None,
        user_id: str | None = None,
        project_id: str | None = None,
    ) -> Dict[str, Any]:
        """
        Sube un archivo a S3.
        
        Args:
            file_path: Ruta local del archivo
            s3_key: Clave S3 custom (default: users/{user_id}/projects/{project_id}/{filename})
            user_id: ID del usuario (para organizar en S3)
            project_id: ID del proyecto
        
        Returns:
            {
                "success": bool,
                "s3_key": str (clave en S3),
                "s3_url": str (URL del archivo),
                "presigned_url": str (URL para descargar - válida 7 días),
                "error": str (si falla)
            }
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {
                "success": False,
                "error": f"Archivo no existe: {file_path}"
            }
        
        # Default S3 key
        if not s3_key:
            filename = file_path.name
            if user_id and project_id:
                s3_key = f"users/{user_id}/projects/{project_id}/{filename}"
            elif user_id:
                s3_key = f"users/{user_id}/{filename}"
            else:
                s3_key = f"uploads/{filename}"
        
        # Si S3 no está configurado, retornar fallback
        if not self.is_configured:
            return {
                "success": True,
                "s3_key": s3_key,
                "s3_url": f"file://{file_path.absolute()}",
                "presigned_url": None,
                "warning": "⚠️ S3 no configurado - archivo guardado localmente"
            }
        
        try:
            # Cargar archivo a S3
            with open(file_path, "rb") as f:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=f,
                    ContentType=self._get_content_type(file_path),
                    Metadata={
                        "uploaded_at": datetime.now().isoformat(),
                        "user_id": user_id or "unknown",
                    }
                )
            
            # Generar URL presignada (válida 7 días)
            presigned_url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": s3_key},
                ExpiresIn=7 * 24 * 3600,  # 7 días
            )
            
            s3_url = f"s3://{self.bucket_name}/{s3_key}"
            
            logger.info(f"✅ Archivo subido a S3: {s3_url}")
            
            return {
                "success": True,
                "s3_key": s3_key,
                "s3_url": s3_url,
                "presigned_url": presigned_url,
            }
        
        except ClientError as e:
            error = e.response["Error"]["Message"]
            logger.error(f"❌ Error al subir a S3: {error}")
            return {
                "success": False,
                "error": error
            }
    
    def upload_file_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        user_id: str | None = None,
        project_id: str | None = None,
    ) -> Dict[str, Any]:
        """
        Sube archivos desde bytes (útil para archivos en memoria).
        
        Args:
            file_bytes: Contenido del archivo (bytes)
            filename: Nombre del archivo
            user_id: ID del usuario
            project_id: ID del proyecto
        
        Returns:
            Mismo que upload_file()
        """
        s3_key = f"users/{user_id}/projects/{project_id}/{filename}" if user_id else f"uploads/{filename}"
        
        # Si S3 no está configurado
        if not self.is_configured:
            return {
                "success": True,
                "s3_key": s3_key,
                "s3_url": f"memory://{filename}",
                "presigned_url": None,
                "warning": "⚠️ S3 no configurado - archivo en memoria"
            }
        
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=io.BytesIO(file_bytes),
                ContentType=self._get_content_type_from_filename(filename),
                Metadata={
                    "uploaded_at": datetime.now().isoformat(),
                    "user_id": user_id or "unknown",
                }
            )
            
            presigned_url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": s3_key},
                ExpiresIn=7 * 24 * 3600,
            )
            
            s3_url = f"s3://{self.bucket_name}/{s3_key}"
            
            return {
                "success": True,
                "s3_key": s3_key,
                "s3_url": s3_url,
                "presigned_url": presigned_url,
            }
        
        except ClientError as e:
            error = e.response["Error"]["Message"]
            return {"success": False, "error": error}
    
    def delete_file(self, s3_key: str) -> Dict[str, Any]:
        """
        Elimina un archivo de S3.
        
        Args:
            s3_key: Clave S3 del archivo
        
        Returns:
            {"success": bool, "error": str (si falla)}
        """
        if not self.is_configured:
            return {"success": True, "warning": "S3 no configurado"}
        
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"✅ Archivo eliminado de S3: {s3_key}")
            return {"success": True}
        except ClientError as e:
            error = e.response["Error"]["Message"]
            logger.error(f"❌ Error al eliminar de S3: {error}")
            return {"success": False, "error": error}
    
    def list_files(
        self,
        prefix: str | None = None,
        user_id: str | None = None,
        project_id: str | None = None,
    ) -> Dict[str, Any]:
        """
        Lista archivos en S3.
        
        Args:
            prefix: Prefijo S3 (ej: "users/123/")
            user_id: Si se proporciona, lista archivos del usuario
            project_id: Si se proporciona, lista archivos del proyecto
        
        Returns:
            {
                "success": bool,
                "files": list de {"key": str, "size": int, "last_modified": datetime},
                "error": str (si falla)
            }
        """
        if not self.is_configured:
            return {"success": True, "files": []}
        
        if not prefix:
            if user_id and project_id:
                prefix = f"users/{user_id}/projects/{project_id}/"
            elif user_id:
                prefix = f"users/{user_id}/"
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix or ""
            )
            
            files = []
            if "Contents" in response:
                for obj in response["Contents"]:
                    files.append({
                        "key": obj["Key"],
                        "size": obj["Size"],
                        "last_modified": obj["LastModified"].isoformat(),
                    })
            
            return {"success": True, "files": files}
        
        except ClientError as e:
            error = e.response["Error"]["Message"]
            return {"success": False, "error": error}
    
    def get_presigned_url(
        self,
        s3_key: str,
        expires_in_days: int = 7,
    ) -> Dict[str, Any]:
        """
        Genera URL presignada para descargar archivo desde S3.
        
        Args:
            s3_key: Clave S3
            expires_in_days: Días hasta que expire la URL
        
        Returns:
            {
                "success": bool,
                "presigned_url": str,
                "error": str (si falla)
            }
        """
        if not self.is_configured:
            return {
                "success": False,
                "error": "S3 no configurado"
            }
        
        try:
            url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": s3_key},
                ExpiresIn=expires_in_days * 24 * 3600,
            )
            return {"success": True, "presigned_url": url}
        except ClientError as e:
            error = e.response["Error"]["Message"]
            return {"success": False, "error": error}
    
    @staticmethod
    def _get_content_type(file_path: Path) -> str:
        """Determina MIME type basado en extensión."""
        ext = file_path.suffix.lower()
        mime_types = {
            ".csv": "text/csv",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".xls": "application/vnd.ms-excel",
            ".json": "application/json",
            ".pdf": "application/pdf",
            ".txt": "text/plain",
        }
        return mime_types.get(ext, "application/octet-stream")
    
    @staticmethod
    def _get_content_type_from_filename(filename: str) -> str:
        """Determina MIME type desde nombre de archivo."""
        ext = Path(filename).suffix.lower()
        return S3Manager._get_content_type(Path(filename))


# Singleton pattern
_storage_manager: S3Manager | None = None


def get_storage_manager() -> S3Manager:
    """
    Obtiene instancia singleton del S3Manager.
    
    Uso:
    ```python
    from src.storage import get_storage_manager
    
    storage = get_storage_manager()
    result = storage.upload_file("data.csv", user_id="user123", project_id="proj456")
    ```
    """
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = S3Manager()
    return _storage_manager
