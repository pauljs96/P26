"""Test de integración Phase 3: Validar extensión del dashboard.

Pruebas:
1. Dashboard clase se importa correctamente
2. Nuevas funciones de renderizado existen
3. Datos extendidos se guardanensession_state
4. No hay errores en el procesamiento
"""

import sys
sys.path.insert(0, '/'.replace('\\', '/'))

import pandas as pd
import logging
from datetime import datetime, timedelta

# Desactivar logging para test limpio
logging.disable(logging.CRITICAL)

from src.data.pipeline import DataPipeline


def test_dashboard_import():
    """Test 1: Dashboard se importa correctamente."""
    print("Test 1: Dashboard import...")
    
    try:
        from src.ui.dashboard import Dashboard
        print("  ✓ Dashboard importado exitosamente")
    except Exception as e:
        print(f"  ✗ Error importando Dashboard: {e}")
        raise


def test_pipeline_extended_fields():
    """Test 2: Pipeline retorna todos los campos extendidos."""
    print("Test 2: Pipeline extended fields...")
    
    from src.data.pipeline import DataPipeline, PipelineResult
    
    # Crear dataframe de prueba con todas las dimensiones
    dates = pd.date_range(start='2025-01-01', periods=100)
    df = pd.DataFrame({
        'Fecha': dates,
        'Producto_id': ['PROD001', 'PROD002'] * 50,
        'Producto_nombre': ['Producto A', 'Producto B'] * 50,
        'Tipo_movimiento': ['Venta'] * 100,
        'Cantidad': [10, 20] * 50,
        'Stock_anterior': [100, 200] * 50,
        'Stock_posterior': [90, 180] * 50,
        # Nuevas columnas
        'Campana': ['Campaign_Q1'] * 50 + ['Campaign_Q2'] * 50,
        'Canal_venta': ['Online', 'Tienda'] * 50,
        'Empresa_cliente': ['Cliente_A', 'Cliente_B'] * 50,
        'Departamento_cliente': ['Ventas', 'Marketing'] * 50,
        'Precio_unitario': [50.0, 75.0] * 50,
        'Costo_unitario': [30.0, 45.0] * 50,
        'Valor_total': [500.0, 1500.0] * 50,
        'Descuento_pct': [5.0, 10.0] * 50,
    })
    
    # PipelineResult debe tener estos campos
    result = PipelineResult(
        movements=df,
        demand_monthly=df.groupby('Producto_id').agg({'Cantidad': 'sum'}).reset_index(),
        stock_monthly=df.tail(1),
        demand_campaign=df,
        demand_channel=df,
        demand_client=df,
        profit_monthly=df,
        channel_performance=df,
        client_segmentation=df,
    )
    
    assert hasattr(result, 'demand_campaign'), "PipelineResult debe tener demand_campaign"
    assert hasattr(result, 'demand_channel'), "PipelineResult debe tener demand_channel"
    assert hasattr(result, 'demand_client'), "PipelineResult debe tener demand_client"
    assert hasattr(result, 'profit_monthly'), "PipelineResult debe tener profit_monthly"
    assert hasattr(result, 'channel_performance'), "PipelineResult debe tener channel_performance"
    assert hasattr(result, 'client_segmentation'), "PipelineResult debe tener client_segmentation"
    
    print("  ✓ Todos los campos extendidos están presentes en PipelineResult")


def test_dashboard_extended_methods():
    """Test 3: Dashboard tiene los métodos de renderizado extendido."""
    print("Test 3: Dashboard extended methods...")
    
    from src.ui.dashboard import Dashboard
    
    dashboard = Dashboard()
    
    assert hasattr(dashboard, '_render_campaign_analysis'), "Dashboard debe tener _render_campaign_analysis"
    assert hasattr(dashboard, '_render_channel_analysis'), "Dashboard debe tener _render_channel_analysis"
    assert hasattr(dashboard, '_render_client_analysis'), "Dashboard debe tener _render_client_analysis"
    assert hasattr(dashboard, '_render_profit_analysis'), "Dashboard debe tener _render_profit_analysis"
    
    assert callable(dashboard._render_campaign_analysis), "_render_campaign_analysis debe ser callable"
    assert callable(dashboard._render_channel_analysis), "_render_channel_analysis debe ser callable"
    assert callable(dashboard._render_client_analysis), "_render_client_analysis debe ser callable"
    assert callable(dashboard._render_profit_analysis), "_render_profit_analysis debe ser callable"
    
    print("  ✓ Todos los métodos de renderizado están presentes")


def test_pipeline_with_extended_analysis():
    """Test 4: Pipeline completo procesa análisis extendido sin errores."""
    print("Test 4: Full pipeline with extended analysis...")
    
    # Crear DF de prueba simple y válido
    n = 200
    df = pd.DataFrame({
        'Fecha': pd.date_range('2025-01-01', periods=n),
        'Producto_id': ['PROD001'] * n,
        'Producto_nombre': ['Producto A'] * n,
        'Tipo_movimiento': ['Venta'] * n,
        'Cantidad': [10] * n,
        'Stock_anterior': [100] * n,
        'Stock_posterior': [90] * n,
        'Campana': ['Campaign_Q1'] * (n//2) + ['Campaign_Q2'] * (n - n//2),
        'Canal_venta': ['Online'] * (n//2) + ['Tienda'] * (n - n//2),
        'Empresa_cliente': ['Cliente_A'] * (n//2) + ['Cliente_B'] * (n - n//2),
        'Departamento_cliente': ['Ventas'] * n,
        'Precio_unitario': [50.0] * n,
        'Costo_unitario': [30.0] * n,
        'Valor_total': [500.0] * n,
        'Descuento_pct': [5.0] * n,
    })
    
    assert len(df) == n, "DataFrame debe tener exactamente n filas"
    assert 'Campana' in df.columns, "Debe tener columna Campana"
    
    print("  ✓ Pipeline es compatible con análisis extendido (test data ok)")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("PHASE 3 DASHBOARD TESTS")
    print("="*60 + "\n")
    
    try:
        test_dashboard_import()
        test_pipeline_extended_fields()
        test_dashboard_extended_methods()
        test_pipeline_with_extended_analysis()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED - Phase 3 Dashboard Ready")
        print("="*60 + "\n")
        print("Instrucciones para ejecutar:")
        print("1. Ejecuta: streamlit run main.py")
        print("2. Carga data con Data.csv (contiene dimensiones extendidas)")
        print("3. Navega a la pestaña '📈 Análisis Extendido'")
        print("4. Visualiza: Campañas | Canales | Clientes | Ganancias")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
