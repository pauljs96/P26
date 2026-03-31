"""Análisis por cliente: segmentación y comportamiento de compra.

Requiere columnas de Data.csv:
- Empresa_cliente, Departamento_cliente, Fecha, Cantidad, Valor_total, Tipo_movimiento
"""

from __future__ import annotations
import pandas as pd
import logging

logger = logging.getLogger(__name__)

from src.utils.config import MOVEMENT_SALE


class ClientBuilder:
    def build_client_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Demanda mensual por cliente.
        
        Entrada: DataFrame con Empresa_cliente, Fecha, Cantidad, Valor_total, Tipo_movimiento
        Salida: DataFrame con (Empresa_cliente, Año, Mes, Cantidad_total, Valor_total)
        """
        if df is None or df.empty or "Empresa_cliente" not in df.columns:
            logger.warning("ClientBuilder: No hay columna 'Empresa_cliente'")
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

        # Agregar por (Empresa_cliente, Año, Mes)
        client = d.groupby(["Empresa_cliente", "Año", "Mes"], as_index=False).agg({
            "Cantidad": "sum",
            "Valor_total": "sum"
        }).rename(columns={"Cantidad": "Cantidad_total"})

        logger.info(
            f"ClientBuilder: {client['Empresa_cliente'].nunique()} clientes, "
            f"{len(client)} registros mensuales"
        )

        return client.sort_values(["Empresa_cliente", "Año", "Mes"]).reset_index(drop=True)

    def build_client_segmentation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Segmentación de clientes por valor total de compras.
        
        Categorías: Top24% (Premium), 25-75% (Regular), Bottom25% (Casual)
        
        Entrada: DataFrame con Empresa_cliente, Cantidad, Valor_total, Tipo_movimiento
        Salida: DataFrame con (Empresa_cliente, Cantidad_total, Valor_total, Categoria, Pct_valor)
        """
        if df is None or df.empty or "Empresa_cliente" not in df.columns:
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

        # Agregar por cliente
        seg = d.groupby("Empresa_cliente", as_index=False).agg({
            "Cantidad": "sum",
            "Valor_total": "sum"
        }).rename(columns={"Cantidad": "Cantidad_total"})

        # Calcular percentil de valor para categorización
        p75 = seg["Valor_total"].quantile(0.75)
        p25 = seg["Valor_total"].quantile(0.25)

        seg["Categoria"] = "Regular"
        seg.loc[seg["Valor_total"] >= p75, "Categoria"] = "Premium"
        seg.loc[seg["Valor_total"] < p25, "Categoria"] = "Casual"

        # Participación en valor
        total_valor = seg["Valor_total"].sum()
        seg["Pct_valor"] = (seg["Valor_total"] / (total_valor + 0.001) * 100).round(2)

        logger.info(
            f"ClientBuilder.build_client_segmentation: "
            f"Premium={len(seg[seg['Categoria']=='Premium'])}, "
            f"Regular={len(seg[seg['Categoria']=='Regular'])}, "
            f"Casual={len(seg[seg['Categoria']=='Casual'])}"
        )

        return seg.sort_values("Valor_total", ascending=False).reset_index(drop=True)

    def build_department_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Análisis por departamento del cliente.
        
        Entrada: DataFrame con Departamento_cliente, Cantidad, Valor_total, Tipo_movimiento
        Salida: DataFrame con (Departamento_cliente, Cantidad_total, Valor_total, Num_clientes)
        """
        if df is None or df.empty or "Departamento_cliente" not in df.columns:
            logger.warning("ClientBuilder: No hay columna 'Departamento_cliente'")
            return pd.DataFrame()

        d = df.copy()
        d = d[d["Tipo_movimiento"] == MOVEMENT_SALE].copy()
        
        if d.empty:
            return pd.DataFrame()

        d["Cantidad"] = pd.to_numeric(d["Cantidad"], errors="coerce").fillna(0)
        
        if "Valor_total" in d.columns:
            d["Valor_total"] = pd.to_numeric(d["Valor_total"], errors="coerce").fillna(0)
        else:
            d["Valor_total"] = 0

        if "Empresa_cliente" not in d.columns:
            d["Empresa_cliente"] = "Unknown"

        # Agregar por departamento
        dept = d.groupby("Departamento_cliente", as_index=False).agg({
            "Cantidad": "sum",
            "Valor_total": "sum",
            "Empresa_cliente": "nunique"
        }).rename(columns={
            "Cantidad": "Cantidad_total",
            "Empresa_cliente": "Num_clientes"
        })

        logger.info(f"ClientBuilder.build_department_analysis: {len(dept)} departamentos")

        return dept.sort_values("Valor_total", ascending=False).reset_index(drop=True)
