"""Herramienta de comparación: Global vs Por Producto.

Ejecuta backtests en ambas estrategias para mostrar el impacto
de predecir por producto vs agregado global.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from src.ml.backtest_rf import backtest_rf_1step
from src.ml.backtest_ets import backtest_ets_1step
from src.ml.backtest_rf_by_product import backtest_rf_by_product_1step
from src.ml.backtest_ets_by_product import backtest_ets_by_product_1step


def compare_ml_strategies(
    pipeline_result,
    test_months: int = 12,
) -> dict:
    """
    Compara estrategias de ML:
    1. Global (univariante): agrega todo a 1 serie
    2. Por Producto (multivariante): predice cada producto separado
    
    Args:
        pipeline_result: Resultado del pipeline con demand_monthly
        test_months: Meses para backtest
    
    Returns:
        dict con resultados de ambas estrategias
    """
    
    if not hasattr(pipeline_result, 'demand_monthly') or pipeline_result.demand_monthly is None:
        return {"error": "No hay demand_monthly en pipeline_result"}
    
    demand_df = pipeline_result.demand_monthly.copy()
    
    print("=" * 80)
    print("COMPARACION: ML GLOBAL vs ML POR PRODUCTO")
    print("=" * 80)
    
    # 1. ESTRATEGIA GLOBAL (Univariante)
    print("\n[1] ESTRATEGIA GLOBAL (Univariante)")
    print("-" * 80)
    print("Agrega demanda de todos los productos a 1 serie temporal")
    print("Entrena 1 modelo para toda la demanda")
    
    try:
        # Agregar todo
        global_demand = demand_df.groupby('Mes').agg({
            'Demanda_Unid': 'sum',
            'Valor_total': 'sum'
        }).reset_index()
        global_demand.rename(columns={'Demanda_Unid': 'Demanda_Unid'}, inplace=True)
        
        bt_rf_global = backtest_rf_1step(global_demand, test_months=test_months)
        bt_ets_global = backtest_ets_1step(global_demand, test_months=test_months)
        
        print(f"\nRF Global MAE: {bt_rf_global.metrics['MAE'].iloc[0]:.2f}")
        print(f"ETS Global MAE: {bt_ets_global.metrics['MAE'].iloc[0]:.2f}")
        
        global_results = {
            "strategy": "GLOBAL",
            "observation_count": len(global_demand),
            "rf_metrics": bt_rf_global.metrics.to_dict('records')[0],
            "ets_metrics": bt_ets_global.metrics.to_dict('records')[0],
            "rf_predictions": bt_rf_global.predictions,
            "ets_predictions": bt_ets_global.predictions,
        }
    except Exception as e:
        print(f"ERROR en estrategia global: {str(e)[:100]}")
        global_results = {"error": str(e)[:100]}
    
    # 2. ESTRATEGIA POR PRODUCTO (Multivariante con contexto)
    print("\n[2] ESTRATEGIA POR PRODUCTO (Multivariante)")
    print("-" * 80)
    print("Agrupa por Producto_id (si existe)")
    print("Entrena modelos separados para cada producto")
    print("Combina predicciones")
    
    try:
        # Solo si hay Producto_id
        if 'Producto_id' in demand_df.columns:
            bt_rf_product = backtest_rf_by_product_1step(demand_df, test_months=test_months, use_contextual=True)
            bt_ets_product = backtest_ets_by_product_1step(demand_df, test_months=test_months)
            
            # Obtener métrica global
            rf_global_metric = bt_rf_product.metrics[bt_rf_product.metrics['Producto_id'] == 'GLOBAL'].iloc[0] if len(bt_rf_product.metrics) > 0 else None
            ets_global_metric = bt_ets_product.metrics[bt_ets_product.metrics['Producto_id'] == 'GLOBAL'].iloc[0] if len(bt_ets_product.metrics) > 0 else None
            
            if rf_global_metric is not None:
                print(f"\nRF Por Producto MAE: {rf_global_metric['MAE']:.2f}")
                print(f"ETS Por Producto MAE: {ets_global_metric['MAE']:.2f}")
            
            product_results = {
                "strategy": "BY_PRODUCT",
                "n_products": demand_df['Producto_id'].nunique(),
                "observation_count": len(bt_rf_product.aggregated_pred),
                "rf_metrics": rf_global_metric.to_dict() if rf_global_metric is not None else {},
                "ets_metrics": ets_global_metric.to_dict() if ets_global_metric is not None else {},
                "rf_predictions": bt_rf_product.aggregated_pred,
                "ets_predictions": bt_ets_product.aggregated_pred,
                "metrics_detail": bt_rf_product.metrics,
            }
        else:
            print("Columna 'Producto_id' no disponible, saltando estrategia por producto")
            product_results = {"skipped": "No Producto_id column"}
    
    except Exception as e:
        print(f"ERROR en estrategia por producto: {str(e)[:100]}")
        product_results = {"error": str(e)[:100]}
    
    # 3. COMPARACION Y RECOMENDACION
    print("\n" + "=" * 80)
    print("COMPARACION DE RESULTADOS")
    print("=" * 80)
    
    comparison = {
        "global": global_results,
        "by_product": product_results,
    }
    
    # Análisis
    if "error" not in global_results and "error" not in product_results and "skipped" not in product_results:
        mae_global_rf = global_results["rf_metrics"]["MAE"]
        mae_product_rf = product_results["rf_metrics"].get("MAE", float('inf'))
        
        mae_global_ets = global_results["ets_metrics"]["MAE"]
        mae_product_ets = product_results["ets_metrics"].get("MAE", float('inf'))
        
        print(f"\nRF Global MAE: {mae_global_rf:.2f}")
        print(f"RF Por Producto MAE: {mae_product_rf:.2f}")
        print(f"Mejora RF: {((mae_global_rf - mae_product_rf) / mae_global_rf * 100):.1f}%")
        
        print(f"\nETS Global MAE: {mae_global_ets:.2f}")
        print(f"ETS Por Producto MAE: {mae_product_ets:.2f}")
        print(f"Mejora ETS: {((mae_global_ets - mae_product_ets) / mae_global_ets * 100):.1f}%")
        
        print(f"\n[RECOMENDACION]")
        if mae_product_rf < mae_global_rf and mae_product_rf < mae_global_ets:
            print("✓ RF Por Producto es MEJOR - Usar esta estrategia")
        elif mae_product_ets < mae_global_rf:
            print("✓ ETS Por Producto es competitivo - Considerar ensemble")
        else:
            print("• Global sigue siendo mejor - Revisar features contextuales")
    
    print("=" * 80)
    
    return comparison
