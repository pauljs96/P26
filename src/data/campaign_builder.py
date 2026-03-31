"""Análisis de campaña: demanda, valor, descuentos por campaña.

Requiere columnas de Data.csv:
- Campana, Fecha, Cantidad, Valor_total, Descuento_pct, Tipo_movimiento
"""

from __future__ import annotations
import pandas as pd
import logging

logger = logging.getLogger(__name__)

from src.utils.config import MOVEMENT_SALE


class CampaignBuilder:
    def build_monthly_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resumen mensual por campaña: cantidad, valor, descuentos.
        
        Entrada: DataFrame con Campana, Fecha, Cantidad, Valor_total, Descuento_pct, Tipo_movimiento
        Salida: DataFrame con (Campana, Año, Mes, Cantidad_total, Valor_total, Desc_total, ROI)
        """
        if df is None or df.empty or "Campana" not in df.columns:
            logger.warning("CampaignBuilder: No hay columna 'Campana'")
            return pd.DataFrame()

        d = df.copy()
        d = d[d["Tipo_movimiento"] == MOVEMENT_SALE].copy()
        
        if d.empty:
            return pd.DataFrame()

        d["Año"] = d["Fecha"].dt.year
        d["Mes"] = d["Fecha"].dt.month

        # Asegurar columnas numéricas
        d["Cantidad"] = pd.to_numeric(d["Cantidad"], errors="coerce").fillna(0)
        
        if "Valor_total" in d.columns:
            d["Valor_total"] = pd.to_numeric(d["Valor_total"], errors="coerce").fillna(0)
        else:
            d["Valor_total"] = 0

        if "Descuento_pct" in d.columns:
            d["Descuento_pct"] = pd.to_numeric(d["Descuento_pct"], errors="coerce").fillna(0)
        else:
            d["Descuento_pct"] = 0

        # Agregar por (Campana, Año, Mes)
        campaign = d.groupby(["Campana", "Año", "Mes"], as_index=False).agg({
            "Cantidad": "sum",
            "Valor_total": "sum",
            "Descuento_pct": "mean"  # Promedio de % descuento
        }).rename(columns={"Cantidad": "Cantidad_total", "Descuento_pct": "Desc_pct_promedio"})

        # Cálculo de descuento total (aproximado)
        campaign["Desc_total_aprox"] = (
            campaign["Valor_total"] * campaign["Desc_pct_promedio"] / 100
        )

        # ROI simplificado: (Valor - Descuento) / Valor
        campaign["ROI_pct"] = (
            ((campaign["Valor_total"] - campaign["Desc_total_aprox"]) / 
             (campaign["Valor_total"] + 0.001)) * 100
        ).round(2)

        logger.info(
            f"CampaignBuilder: {campaign['Campana'].nunique()} campañas, "
            f"{len(campaign)} registros mensuales"
        )

        return campaign.sort_values(["Campana", "Año", "Mes"]).reset_index(drop=True)

    def build_top_campaigns(self, df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """Top N campañas por valor total de ventas.
        
        Entrada: DataFrame con Campana, Valor_total, Tipo_movimiento
        Salida: DataFrame con (Campana, Cantidad_total, Valor_total, ROI_pct)
        """
        if df is None or df.empty or "Campana" not in df.columns:
            return pd.DataFrame()

        d = df.copy()
        d = d[d["Tipo_movimiento"] == MOVEMENT_SALE].copy()
        
        if d.empty:
            return pd.DataFrame()

        d["Cantidad"] = pd.to_numeric(d["Cantidad"], errors="coerce").fillna(0)
        
        if "Valor_total" not in d.columns:
            return pd.DataFrame()
        
        d["Valor_total"] = pd.to_numeric(d["Valor_total"], errors="coerce").fillna(0)
        
        if "Descuento_pct" in d.columns:
            d["Descuento_pct"] = pd.to_numeric(d["Descuento_pct"], errors="coerce").fillna(0)
        else:
            d["Descuento_pct"] = 0

        # Agregar por campaña
        agg = d.groupby("Campana", as_index=False).agg({
            "Cantidad": "sum",
            "Valor_total": "sum",
            "Descuento_pct": "mean"
        }).rename(columns={"Cantidad": "Cantidad_total", "Descuento_pct": "Desc_pct_promedio"})

        agg["Desc_total_aprox"] = agg["Valor_total"] * agg["Desc_pct_promedio"] / 100
        agg["ROI_pct"] = (
            ((agg["Valor_total"] - agg["Desc_total_aprox"]) / 
             (agg["Valor_total"] + 0.001)) * 100
        ).round(2)

        return agg.nlargest(top_n, "Valor_total").reset_index(drop=True)
