"""Construcción de demanda mensual para Dataset v4 y Data.csv.

Demanda mensual se calcula como la suma de unidades vendidas (Tipo_movimiento='Venta')
agrupadas por (Producto_id, Año-Mes).

Columnas de entrada requeridas:
- Fecha, Producto_id, Tipo_movimiento, Cantidad

Columnas de entrada opcionales (Data.csv):
- Campana, Canal_venta, Empresa_cliente, Valor_total, Descuento_pct

Columnas de salida:
- Básico: Producto_id, Año, Mes, Cantidad_total (demanda)
- Por campaña: Producto_id, Año, Mes, Campana, Cantidad_total, Valor_total
- Por canal: Canal_venta, Año, Mes, Cantidad_total, Valor_total
- Por cliente: Empresa_cliente, Año, Mes, Cantidad_total, Valor_total
"""

from __future__ import annotations
import pandas as pd
import logging

logger = logging.getLogger(__name__)

from src.utils.config import MOVEMENT_SALE


class DemandBuilder:
    def build_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construye tabla de demanda mensual desde transacciones diarias.
        
        Entrada: DataFrame v4 con Fecha, Producto_id, Tipo_movimiento, Cantidad
        Salida: DataFrame con (Producto_id, Año, Mes, Cantidad_total)
        """
        if df is None or df.empty:
            logger.warning("DemandBuilder: DataFrame vacío")
            return pd.DataFrame(columns=["Producto_id", "Año", "Mes", "Cantidad_total"])

        d = df.copy()

        # Filtrar solo ventas (demanda)
        d = d[d["Tipo_movimiento"] == MOVEMENT_SALE].copy()
        
        if d.empty:
            logger.warning("DemandBuilder: No hay movimientos de tipo 'Venta'")
            return pd.DataFrame(columns=["Producto_id", "Año", "Mes", "Cantidad_total"])

        # Extraer año y mes
        d["Año"] = d["Fecha"].dt.year
        d["Mes"] = d["Fecha"].dt.month

        # Normalizar columnas de texto
        d["Producto_id"] = d["Producto_id"].astype(str).str.strip()

        # Agregar por (Producto_id, Año, Mes)
        monthly = (
            d.groupby(["Producto_id", "Año", "Mes"], as_index=False)["Cantidad"]
             .sum()
             .rename(columns={"Cantidad": "Cantidad_total"})
        )

        # Asegurar que Cantidad_total sea siempre positivo (es demanda)
        monthly["Cantidad_total"] = monthly["Cantidad_total"].abs()

        logger.info(
            f"DemandBuilder: {len(monthly)} filas mensuales generadas "
            f"({monthly['Producto_id'].nunique()} productos)"
        )

        return monthly.sort_values(["Producto_id", "Año", "Mes"]).reset_index(drop=True)

    def build_by_campaign(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construye demanda mensual por campaña (si columna existe).
        
        Entrada: DataFrame con Campana (opcional)
        Salida: DataFrame con (Campana, Año, Mes, Cantidad_total, Valor_total)
        """
        if df is None or df.empty or "Campana" not in df.columns:
            return pd.DataFrame()

        d = df.copy()
        d = d[d["Tipo_movimiento"] == MOVEMENT_SALE].copy()
        
        if d.empty:
            return pd.DataFrame()

        d["Año"] = d["Fecha"].dt.year
        d["Mes"] = d["Fecha"].dt.month

        group_cols = ["Campana", "Año", "Mes"]
        agg_dict = {"Cantidad": "sum"}
        
        # Si existe Valor_total, lo incluimos
        if "Valor_total" in d.columns:
            agg_dict["Valor_total"] = "sum"

        campaign = d.groupby(group_cols, as_index=False).agg(agg_dict)
        campaign["Cantidad"] = campaign["Cantidad"].abs()
        campaign = campaign.rename(columns={"Cantidad": "Cantidad_total"})

        logger.info(f"DemandBuilder.build_by_campaign: {len(campaign)} registros, {campaign['Campana'].nunique()} campañas")
        
        return campaign.sort_values(["Campana", "Año", "Mes"]).reset_index(drop=True)

    def build_by_channel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construye demanda mensual por canal de venta (si columna existe).
        
        Entrada: DataFrame con Canal_venta (opcional)
        Salida: DataFrame con (Canal_venta, Año, Mes, Cantidad_total, Valor_total)
        """
        if df is None or df.empty or "Canal_venta" not in df.columns:
            return pd.DataFrame()

        d = df.copy()
        d = d[d["Tipo_movimiento"] == MOVEMENT_SALE].copy()
        
        if d.empty:
            return pd.DataFrame()

        d["Año"] = d["Fecha"].dt.year
        d["Mes"] = d["Fecha"].dt.month

        group_cols = ["Canal_venta", "Año", "Mes"]
        agg_dict = {"Cantidad": "sum"}
        
        if "Valor_total" in d.columns:
            agg_dict["Valor_total"] = "sum"

        channel = d.groupby(group_cols, as_index=False).agg(agg_dict)
        channel["Cantidad"] = channel["Cantidad"].abs()
        channel = channel.rename(columns={"Cantidad": "Cantidad_total"})

        logger.info(f"DemandBuilder.build_by_channel: {len(channel)} registros, {channel['Canal_venta'].nunique()} canales")
        
        return channel.sort_values(["Canal_venta", "Año", "Mes"]).reset_index(drop=True)

    def build_by_client(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construye demanda mensual por cliente (si columna existe).
        
        Entrada: DataFrame con Empresa_cliente (opcional)
        Salida: DataFrame con (Empresa_cliente, Año, Mes, Cantidad_total, Valor_total)
        """
        if df is None or df.empty or "Empresa_cliente" not in df.columns:
            return pd.DataFrame()

        d = df.copy()
        d = d[d["Tipo_movimiento"] == MOVEMENT_SALE].copy()
        
        if d.empty:
            return pd.DataFrame()

        d["Año"] = d["Fecha"].dt.year
        d["Mes"] = d["Fecha"].dt.month

        group_cols = ["Empresa_cliente", "Año", "Mes"]
        agg_dict = {"Cantidad": "sum"}
        
        if "Valor_total" in d.columns:
            agg_dict["Valor_total"] = "sum"

        client = d.groupby(group_cols, as_index=False).agg(agg_dict)
        client["Cantidad"] = client["Cantidad"].abs()
        client = client.rename(columns={"Cantidad": "Cantidad_total"})

        logger.info(f"DemandBuilder.build_by_client: {len(client)} registros, {client['Empresa_cliente'].nunique()} clientes")
        
        return client.sort_values(["Empresa_cliente", "Año", "Mes"]).reset_index(drop=True)
