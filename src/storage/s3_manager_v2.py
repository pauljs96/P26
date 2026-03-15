"""
S3 Storage Manager - Multi-Tenant Aware
=====================================================
Gestiona upload/download de archivos a AWS S3 con aislamiento por org.

Estructura de directorios:
  s3://bucket/{org_id}/raw/                  # Datos crudos CSV
  s3://bucket/{org_id}/processed/            # Datos procesados (parquet)
  s3://bucket/{org_id}/backups/              # Backups y históricos
"""

from __future__ import annotations

import os
import io
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime, timedelta

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class S3Manager:
    """
    Gestiona S3 con soporte multi-tenant (org-aware).
    
    Características:
    - Upload de CSV/parquet por organización
    - Aislamiento de datos por org (via prefijos S3)
    - URLs presignadas para descargas seguras
    - Listado de archivos de org
    - Versionado de archivos
    """
    
    def __init__(
        self,
        bucket_name: str | None = None,
        region: str = "us-east-1",
        access_key: str | None = None,
        secret_key: str | None = None,
    ):
        """
        Inicializa S3Manager con credenciales AWS.
        
        Si no se proporcionan, lee de variables de entorno:
        - AWS_S3_BUCKET_NAME
        - AWS_S3_REGION
        - AWS_ACCESS_KEY_ID
        - AWS_SECRET_ACCESS_KEY
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
                logger.info(f"✅ S3 conectado: s3://{self.bucket_name} ({self.region})")
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "Unknown")
                logger.warning(
                    f"⚠️ No se pudo conectar a S3: {error_code}. "
                    f"Sistema funcionará sin persistencia S3."
                )
                self.is_configured = False
        else:
            logger.warning("⚠️ Credenciales AWS no configuradas. S3 deshabilitado.")
    
    # ==================== UPLOAD ====================
    
    def upload_file(
        self,
        file_path: str | Path,
        org_id: str,
        data_type: str = "raw",
        year: str | None = None,
        custom_key: str | None = None,
    ) -> Dict[str, Any]:
        """
        Sube archivo a S3 con estructura org-aware.
        
        Estructura generada:
          s3://bucket/{org_id}/{data_type}/data_{year}.csv
          s3://bucket/{org_id}/raw/datos_2025_01_15.csv
        
        Args:
            file_path: Ruta local del archivo
            org_id: ID de la organización (REQUIRED)
            data_type: Tipo - "raw" (default), "processed", "backups"
            year: Año/mes de datos (ej: "2025" o "2025_01")
            custom_key: Override completo de la clave S3
        
        Returns:
            {
                "success": bool,
                "s3_key": str,
                "s3_url": str,
                "presigned_url": str (None si S3 deshabilitado),
                "file_size_mb": float,
                "error": str (si falla)
            }
        """
        file_path = Path(file_path)
        
        # Validaciones básicas
        if not file_path.exists():
            return {"success": False, "error": f"Archivo no existe: {file_path}"}
        
        if not org_id or not isinstance(org_id, str):
            return {"success": False, "error": "org_id inválido"}
        
        # Generar clave S3
        if custom_key:
            s3_key = custom_key
        else:
            filename = file_path.stem  # Sin extensión
            ext = file_path.suffix
            if year:
                s3_key = f"{org_id}/{data_type}/data_{year}{ext}"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                s3_key = f"{org_id}/{data_type}/{filename}_{timestamp}{ext}"
        
        # Si S3 no está configurado, retornar fallback
        if not self.is_configured:
            file_size = file_path.stat().st_size / (1024 * 1024)
            logger.warning(f"⚠️ S3 deshabilitado - archivo no persistido: {s3_key}")
            return {
                "success": True,
                "s3_key": s3_key,
                "s3_url": f"file://{file_path.absolute()}",
                "presigned_url": None,
                "file_size_mb": round(file_size, 2),
                "warning": "S3 deshabilitado - archivo en filesystem"
            }
        
        # Upload a S3
        try:
            file_size = file_path.stat().st_size
            
            # Cargar archivo
            with open(file_path, "rb") as f:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=f,
                    ContentType=self._get_content_type(file_path),
                    Metadata={
                        "uploaded_at": datetime.now().isoformat(),
                        "org_id": org_id,
                        "data_type": data_type,
                    },
                    ServerSideEncryption="AES256"
                )
            
            # Generar URL presignada
            presigned_url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": s3_key},
                ExpiresIn=30 * 24 * 3600,  # 30 días
            )
            
            s3_url = f"s3://{self.bucket_name}/{s3_key}"
            file_size_mb = file_size / (1024 * 1024)
            
            logger.info(f"✅ Archivo subido a S3: {s3_url} ({file_size_mb:.2f} MB)")
            
            return {
                "success": True,
                "s3_key": s3_key,
                "s3_url": s3_url,
                "presigned_url": presigned_url,
                "file_size_mb": round(file_size_mb, 2),
            }
        
        except ClientError as e:
            error = e.response.get("Error", {}).get("Message", str(e))
            logger.error(f"❌ Error al subir a S3: {error}")
            return {"success": False, "error": error}
    
    def upload_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        org_id: str,
        data_type: str = "processed",
    ) -> Dict[str, Any]:
        """
        Sube archivo desde bytes (sin necesidad de archivo local).
        
        Útil para: DataFrames guardados como parquet, JSONs procesados, etc.
        
        Args:
            file_bytes: Contenido del archivo
            filename: Nombre del archivo (incluir extensión)
            org_id: ID de la organización
            data_type: "raw", "processed", etc.
        
        Returns:
            Igual que upload_file()
        """
        if not org_id:
            return {"success": False, "error": "org_id requerido"}
        
        s3_key = f"{org_id}/{data_type}/{filename}"
        
        if not self.is_configured:
            file_size_mb = len(file_bytes) / (1024 * 1024)
            logger.warning(f"⚠️ S3 deshabilitado - {s3_key} ({file_size_mb:.2f} MB) no persistido")
            return {
                "success": True,
                "s3_key": s3_key,
                "s3_url": "memory://",
                "presigned_url": None,
                "file_size_mb": round(file_size_mb, 2),
                "warning": "S3 deshabilitado"
            }
        
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=io.BytesIO(file_bytes),
                ContentType=self._get_content_type_from_filename(filename),
                Metadata={
                    "uploaded_at": datetime.now().isoformat(),
                    "org_id": org_id,
                    "data_type": data_type,
                },
                ServerSideEncryption="AES256"
            )
            
            presigned_url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": s3_key},
                ExpiresIn=30 * 24 * 3600,
            )
            
            s3_url = f"s3://{self.bucket_name}/{s3_key}"
            file_size_mb = len(file_bytes) / (1024 * 1024)
            
            return {
                "success": True,
                "s3_key": s3_key,
                "s3_url": s3_url,
                "presigned_url": presigned_url,
                "file_size_mb": round(file_size_mb, 2),
            }
        
        except ClientError as e:
            error = e.response.get("Error", {}).get("Message", str(e))
            return {"success": False, "error": error}
    
    # ==================== DOWNLOAD / RETRIEVE ====================
    
    def get_presigned_url(
        self,
        s3_key: str,
        org_id: str,
        expires_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Genera URL presignada para descargar archivo (con validación org).
        
        Args:
            s3_key: Clave S3 (debe empezar con org_id/)
            org_id: ID de organización (para validación)
            expires_days: Días hasta expiración
        
        Returns:
            {
                "success": bool,
                "presigned_url": str,
                "error": str (si falla)
            }
        """
        # Validar que la clave pertenece a la org
        if not s3_key.startswith(f"{org_id}/"):
            return {"success": False, "error": "Acceso denegado: org no coincide"}
        
        if not self.is_configured:
            return {"success": False, "error": "S3 deshabilitado"}
        
        try:
            url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": s3_key},
                ExpiresIn=expires_days * 24 * 3600,
            )
            return {"success": True, "presigned_url": url}
        except ClientError as e:
            error = e.response.get("Error", {}).get("Message", str(e))
            return {"success": False, "error": error}
    
    def download_file(
        self,
        s3_key: str,
        org_id: str,
        save_path: str | Path | None = None,
    ) -> Dict[str, Any]:
        """
        Descarga archivo de S3 (con aislamiento org).
        
        Args:
            s3_key: Clave S3
            org_id: ID de organización
            save_path: Ruta local donde guardar (default: temp)
        
        Returns:
            {
                "success": bool,
                "file_path": str,
                "file_bytes": bytes (si save_path es None),
                "error": str (si falla)
            }
        """
        if not s3_key.startswith(f"{org_id}/"):
            return {"success": False, "error": "Acceso denegado: org no coincide"}
        
        if not self.is_configured:
            return {"success": False, "error": "S3 deshabilitado"}
        
        try:
            if save_path is None:
                # Descargar a bytes
                response = self.s3_client.get_object(
                    Bucket=self.bucket_name,
                    Key=s3_key
                )
                file_bytes = response["Body"].read()
                return {
                    "success": True,
                    "file_bytes": file_bytes,
                    "size_mb": len(file_bytes) / (1024 * 1024)
                }
            else:
                # Descargar a archivo
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                
                self.s3_client.download_file(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Filename=str(save_path)
                )
                
                file_size_mb = save_path.stat().st_size / (1024 * 1024)
                return {
                    "success": True,
                    "file_path": str(save_path),
                    "size_mb": round(file_size_mb, 2)
                }
        
        except ClientError as e:
            error = e.response.get("Error", {}).get("Message", str(e))
            return {"success": False, "error": error}
    
    # ==================== LIST / DELETE ====================
    
    def list_org_files(
        self,
        org_id: str,
        data_type: str | None = None,
    ) -> Dict[str, Any]:
        """
        Lista archivos de una organización en S3.
        
        Args:
            org_id: ID de organización
            data_type: "raw", "processed", "backups" (None = todo)
        
        Returns:
            {
                "success": bool,
                "files": [
                    {
                        "key": str,
                        "size_mb": float,
                        "last_modified": str (ISO format),
                        "data_type": str
                    }
                ],
                "total_files": int,
                "total_size_mb": float,
                "error": str (si falla)
            }
        """
        if not self.is_configured:
            return {"success": True, "files": [], "total_files": 0, "total_size_mb": 0}
        
        try:
            # Prefijo: org_id/data_type/ o solo org_id/
            prefix = f"{org_id}/{data_type}/" if data_type else f"{org_id}/"
            
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
            
            files = []
            total_size = 0
            
            for page in pages:
                if "Contents" not in page:
                    continue
                
                for obj in page["Contents"]:
                    # Skip si es solo el prefijo
                    if obj["Key"] == prefix or obj["Key"].endswith("/"):
                        continue
                    
                    size_mb = obj["Size"] / (1024 * 1024)
                    total_size += obj["Size"]
                    
                    # Extraer data_type del path
                    parts = obj["Key"].split("/")
                    file_data_type = parts[1] if len(parts) > 1 else "unknown"
                    
                    files.append({
                        "key": obj["Key"],
                        "filename": parts[-1],
                        "size_mb": round(size_mb, 2),
                        "last_modified": obj["LastModified"].isoformat(),
                        "data_type": file_data_type,
                    })
            
            # Ordenar por fecha (más reciente primero)
            files.sort(key=lambda x: x["last_modified"], reverse=True)
            
            return {
                "success": True,
                "files": files,
                "total_files": len(files),
                "total_size_mb": round(total_size / (1024 * 1024), 2),
            }
        
        except ClientError as e:
            error = e.response.get("Error", {}).get("Message", str(e))
            return {"success": False, "error": error}
    
    def delete_file(
        self,
        s3_key: str,
        org_id: str,
    ) -> Dict[str, Any]:
        """
        Elimina archivo de S3 (con validación org).
        
        Args:
            s3_key: Clave S3
            org_id: ID de organización
        
        Returns:
            {"success": bool, "error": str (si falla)}
        """
        if not s3_key.startswith(f"{org_id}/"):
            return {"success": False, "error": "Acceso denegado: org no coincide"}
        
        if not self.is_configured:
            logger.warning(f"⚠️ S3 deshabilitado - no se puede eliminar {s3_key}")
            return {"success": True}
        
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"✅ Archivo eliminado: {s3_key}")
            return {"success": True}
        except ClientError as e:
            error = e.response.get("Error", {}).get("Message", str(e))
            logger.error(f"❌ Error al eliminar: {error}")
            return {"success": False, "error": error}
    
    def delete_org_folder(self, org_id: str) -> Dict[str, Any]:
        """
        Elimina TODOS los archivos de una organización (CUIDADO).
        
        Args:
            org_id: ID de organización
        
        Returns:
            {"success": bool, "deleted_count": int, "error": str}
        """
        if not self.is_configured:
            logger.warning(f"⚠️ S3 deshabilitado - no se puede eliminar carpeta {org_id}/")
            return {"success": True, "deleted_count": 0}
        
        try:
            prefix = f"{org_id}/"
            deleted_count = 0
            
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
            
            for page in pages:
                if "Contents" not in page:
                    continue
                
                for obj in page["Contents"]:
                    if obj["Key"] != prefix:
                        self.s3_client.delete_object(Bucket=self.bucket_name, Key=obj["Key"])
                        deleted_count += 1
            
            logger.warning(f"🗑️  Carpeta {org_id}/ eliminada ({deleted_count} archivos)")
            return {"success": True, "deleted_count": deleted_count}
        
        except ClientError as e:
            error = e.response.get("Error", {}).get("Message", str(e))
            return {"success": False, "error": error, "deleted_count": 0}
    
    # ==================== UTILITIES ====================
    
    @staticmethod
    def _get_content_type(file_path: Path) -> str:
        """Determina MIME type desde la extensión."""
        ext = file_path.suffix.lower()
        mime_types = {
            ".csv": "text/csv",
            ".parquet": "application/octet-stream",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".json": "application/json",
            ".pdf": "application/pdf",
            ".txt": "text/plain",
            ".gz": "application/gzip",
        }
        return mime_types.get(ext, "application/octet-stream")
    
    @staticmethod
    def _get_content_type_from_filename(filename: str) -> str:
        """Determina MIME type desde nombre de archivo."""
        return S3Manager._get_content_type(Path(filename))


# ==================== SINGLETON ====================

_s3_manager: S3Manager | None = None


def get_s3_manager() -> S3Manager:
    """
    Obtiene instancia singleton del S3Manager.
    
    Uso:
    ```python
    from src.storage import get_s3_manager
    
    s3 = get_s3_manager()
    result = s3.upload_file("data.csv", org_id="org-123", data_type="raw", year="2025")
    ```
    """
    global _s3_manager
    if _s3_manager is None:
        _s3_manager = S3Manager()
    return _s3_manager
