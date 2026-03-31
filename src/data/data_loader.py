"""Loader maestro con auto-detección de formato.

Actúa como router inteligente que:
1. Intenta cargar como formato NUEVO (Data.csv)
2. Si falla, intenta como formato LEGACY (v4)
3. Normaliza el resultado automáticamente

Mantiene interfaz compatible: load_files(uploaded_files) → pd.DataFrame

Beneficio: pipeline.py y dashboard.py NO requieren cambios.
"""

from __future__ import annotations
import pandas as pd
from typing import List, Optional, Tuple
import logging

from src.data.data_loader_new import DataLoaderNew
from src.data.data_loader_legacy import DataLoaderLegacy

logger = logging.getLogger(__name__)


class DataLoader:
    """Router maestro para carga de múltiples formatos de CSV."""
    
    def __init__(self):
        self.loader_new = DataLoaderNew()
        self.loader_legacy = DataLoaderLegacy()
        self.detected_format = None  # "new" o "legacy"
    
    def load_files(self, uploaded_files: List) -> pd.DataFrame:
        """
        Carga archivos detectando automáticamente el formato.
        
        Estrategia:
        1. Intentar cargar como formato NUEVO (Data.csv)
        2. Si falla, intentar como formato LEGACY (v4)
        3. Normalizar el resultado
        
        Args:
            uploaded_files: Lista de archivos subidos (Streamlit UploadedFile)
        
        Returns:
            pd.DataFrame normalizado listo para pipeline
        """
        logger.info("=== DataLoader: Auto-detección de formato ===")
        
        # Intentar nuevo formato primero
        logger.info("[1/2] Intentando cargar como formato NUEVO (Data.csv)...")
        df_new = self.loader_new.load_files(uploaded_files)
        
        if not df_new.empty:
            logger.info("✓ Detectado como NUEVO formato")
            self.detected_format = "new"
            
            # Normalizar
            df_normalized = self.loader_new.normalize(df_new)
            return df_normalized
        
        # Si falla, intentar formato legacy
        logger.info("[2/2] Intentando cargar como formato LEGACY (v4)...")
        df_legacy = self.loader_legacy.load_files(uploaded_files)
        
        if not df_legacy.empty:
            logger.info("✓ Detectado como LEGACY formato (v4)")
            self.detected_format = "legacy"
            
            # Normalizar
            df_normalized = self.loader_legacy.normalize(df_legacy)
            return df_normalized
        
        # Si llega aquí, ambos fallaron
        logger.error("❌ No se pudo detectar formato válido")
        logger.error("Verifica que:")
        logger.error("  - NUEVO: Separa con ';' y tiene 'Producto_codigo'")
        logger.error("  - LEGACY: Separa con ',' y tiene 'Producto_id'")
        return pd.DataFrame()
    
    def get_detected_format(self) -> Optional[str]:
        """Retorna el formato detectado: 'new', 'legacy', o None."""
        return self.detected_format
    
    def is_new_format(self) -> bool:
        """True si se detectó el formato nuevo."""
        return self.detected_format == "new"
    
    def is_legacy_format(self) -> bool:
        """True si se detectó el formato legacy."""
        return self.detected_format == "legacy"
