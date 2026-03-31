"""Análisis por canal de venta: rendimiento y distribución de demanda.

Requiere columnas de Data.csv:
- Canal_venta, Fecha, Cantidad, Valor_total, Tipo_movimiento
"""

from __future__ import annotations
import pandas as pd
import logging

logger = logging.getLogger(__name__)

from src.utils.config import MOVEMENT_SALE


class ChannelBuilder:
    def build_monthly_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resumen mensual por canal: cantidad, valor, participación.
        
        Entrada: DataFrame con Canal_venta, Fecha, Cantidad, Valor_total, Tipo_movimiento
        Salida: DataFrame con (Canal_venta, Año, Mes, Cantidad_total, Valor_total, Pct_participacion)
        """
        if df is None or df.empty or "Canal_venta" not in df.columns:
            logger.warning("ChannelBuilder: No hay columna 'Canal_venta'")
            return pd.DataFrame()

        d = df.copy()
        d = d[d["Tipo_movimiento"] == MOVEMENT_SALE].copy()
        
        if d.empty:
            return pd.DataFrame()

        d["Año"] = d["Fecha"].dt.year
        d["Mes"] = d["Fecha"].dt.month
        d["Cantidad"] = pd.to_numeric(d["Cantidad"], errors="coerce").fillna(0)
        
        if "Valor_total" in d.columns:
            d["Valor_total"] = pd.to_numeric(d["Valor_total"], errors="coerce").fillna(0)
        else:
            d["Valor_total"] = 0

        # Agregar por (Canal_venta, Año, Mes)
        channel = d.groupby(["Canal_venta", "Año", "Mes"], as_index=False).agg({
            "Cantidad": "sum",
            "Valor_total": "sum"
        }).rename(columns={"Cantidad": "Cantidad_total"})

        # Calcular participación (% sobre total del mes)
        channel["Mes_key"] = channel["Año"].astype(str) + "-" + channel["Mes"].astype(str).str.zfill(2)
        monthly_totals = channel.groupby("Mes_key")["Cantidad_total"].transform("sum")
        channel["Pct_participacion"] = (channel["Cantidad_total"] / (monthly_totals + 0.001) * 100).round(2)

        channel = channel.drop("Mes_key", axis=1)

        logger.info(
            f"ChannelBuilder: {channel['Canal_venta'].nunique()} canales, "
            f"{len(channel)} registros mensuales"
        )

        return channel.sort_values(["Canal_venta", "Año", "Mes"]).reset_index(drop=True)

    def build_channel_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Performance total por canal: cantidad, valor, ticket promedio.
        
        Entrada: DataFrame con Canal_venta, Cantidad, Valor_total, Tipo_movimiento
        Salida: DataFrame con (Canal_venta, Cantidad_total, Valor_total, Ticket_promedio)
        """
        if df is None or df.empty or "Canal_venta" not in df.columns:
            return pd.DataFrame()

        d = df.copy()
        d = d[d["Tipo_movimiento"] == MOVEMENT_SALE].copy()
        
        if d.empty:
            return pd.DataFrame()

        d["Cantidad"] = pd.to_numeric(d["Cantidad"], errors="coerce").fillna(0)
        
        if "Valor_total" not in d.columns:
            d["Valor_total"] = 0
        else:
            d["Valor_total"] = pd.to_numeric(d["Valor_total"], errors="coerce").fillna(0)

        # Contar transacciones por canal
        perf = d.groupby("Canal_venta", as_index=False).agg({
            "Cantidad": "sum",
            "Valor_total": "sum",
            "Fecha": "count"  # Número de transacciones
        }).rename(columns={"Cantidad": "Cantidad_total", "Fecha": "Num_transacciones"})

        # Ticket promedio = Valor_total / Num_transacciones
        perf["Ticket_promedio"] = (
            perf["Valor_total"] / (perf["Num_transacciones"] + 0.001)
        ).round(2)

        # Unidades promedio por transacción
        perf["Unidades_por_transaccion"] = (
            perf["Cantidad_total"] / (perf["Num_transacciones"] + 0.001)
        ).round(2)

        logger.info(f"ChannelBuilder.build_channel_performance: {len(perf)} canales")

        return perf.sort_values("Valor_total", ascending=False).reset_index(drop=True)
