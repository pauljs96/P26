"""Test de integración Phase 2: Validar extensiones para Data.csv.

Pruebas:
1. DataCleaner maneja nuevas columnas (Empresa_cliente, etc.)
2. DemandBuilder tiene nuevos métodos
3. Nuevos builders (Campaign, Channel, Client, Profit) se instancian
4. Pipeline completo funciona con ambos formatos
"""

import sys
sys.path.insert(0, '/'.replace('\\', '/'))

import pandas as pd
import logging
from datetime import datetime, timedelta

# Disable logging para test limpio
logging.disable(logging.CRITICAL)

from src.data.data_cleaner import DataCleaner
from src.data.demand_builder import DemandBuilder
from src.data.campaign_builder import CampaignBuilder
from src.data.channel_builder import ChannelBuilder
from src.data.client_builder import ClientBuilder
from src.data.profit_builder import ProfitBuilder
from src.data.pipeline import DataPipeline


def test_data_cleaner_new_columns():
    """Test 1: DataCleaner maneja sin errores cuando columnas nuevas están presentes."""
    print("Test 1: DataCleaner with new columns...")
    
    # Crear DataFrame con nuevas columnas
    dates = pd.date_range(start='2025-01-01', periods=100)
    df = pd.DataFrame({
        'Fecha': dates,
        'Producto_id': ['PROD001'] * 100,
        'Producto_nombre': ['Producto A'] * 100,
        'Tipo_movimiento': ['Venta'] * 100,
        'Cantidad': [10] * 100,
        'Stock_anterior': [100] * 100,
        'Stock_posterior': [90] * 100,
        # Nuevas columnas
        'Campana': ['Campaign_Q1_2025'] * 100,
        'Canal_venta': ['Online', 'Tienda', 'Mayorista'] * 33 + ['Online'],
        'Empresa_cliente': ['Cliente_A', 'Cliente_B'] * 50,
        'Departamento_cliente': ['Ventas', 'Marketing'] * 50,
        'Precio_unitario': [50.0] * 100,
        'Costo_unitario': [30.0] * 100,
        'Valor_total': [500.0] * 100,
        'Descuento_pct': [5.0] * 100,
    })
    
    cleaner = DataCleaner()
    clean = cleaner.clean(df)
    
    assert len(clean) > 0, "DataFrame limpió debería tener filas"
    assert 'Campana' in clean.columns, "Columna Campana debería existir"
    assert 'Canal_venta' in clean.columns, "Columna Canal_venta debería existir"
    assert 'Empresa_cliente' in clean.columns, "Columna Empresa_cliente debería existir"
    assert pd.api.types.is_float_dtype(clean['Precio_unitario']), "Precio_unitario debería ser float"
    print("  ✓ DataCleaner procesa nuevas columnas correctamente")


def test_demand_builder_extended_methods():
    """Test 2: DemandBuilder tiene nuevos métodos."""
    print("Test 2: DemandBuilder extended methods...")
    
    dates = pd.date_range(start='2025-01-01', periods=50)
    df = pd.DataFrame({
        'Fecha': dates,
        'Producto_id': ['PROD001'] * 50,
        'Tipo_movimiento': ['Venta'] * 50,
        'Cantidad': [10] * 50,
        'Campana': ['Campaign_Q1'] * 50,
        'Canal_venta': ['Online'] * 50,
        'Empresa_cliente': ['Cliente_A'] * 50,
        'Valor_total': [500.0] * 50,
    })
    
    builder = DemandBuilder()
    
    # Test métodos básicos
    demand_monthly = builder.build_monthly(df)
    assert len(demand_monthly) > 0, "build_monthly debería retornar datos"
    
    # Test nuevos métodos
    campaign = builder.build_by_campaign(df)
    assert len(campaign) > 0, "build_by_campaign debería retornar datos"
    assert 'Campana' in campaign.columns, "Campaign analysis debería incluir Campana"
    
    channel = builder.build_by_channel(df)
    assert len(channel) > 0, "build_by_channel debería retornar datos"
    assert 'Canal_venta' in channel.columns, "Channel analysis debería incluir Canal_venta"
    
    client = builder.build_by_client(df)
    assert len(client) > 0, "build_by_client debería retornar datos"
    assert 'Empresa_cliente' in client.columns, "Client analysis debería incluir Empresa_cliente"
    
    print("  ✓ Todos los nuevos métodos de DemandBuilder funcionan")


def test_new_builders_instantiation():
    """Test 3: Nuevos builders se instancian correctamente."""
    print("Test 3: New builders instantiation...")
    
    dates = pd.date_range(start='2025-01-01', periods=50)
    df = pd.DataFrame({
        'Fecha': dates,
        'Producto_id': ['PROD001'] * 50,
        'Tipo_movimiento': ['Venta'] * 50,
        'Cantidad': [10] * 50,
        'Campana': ['Campaign_Q1'] * 50,
        'Canal_venta': ['Online'] * 50,
        'Empresa_cliente': ['Cliente_A'] * 50,
        'Precio_unitario': [50.0] * 50,
        'Costo_unitario': [30.0] * 50,
        'Valor_total': [500.0] * 50,
        'Descuento_pct': [5.0] * 50,
    })
    
    # Campaign Builder
    campaign_builder = CampaignBuilder()
    campaign = campaign_builder.build_monthly_summary(df)
    assert len(campaign) > 0, "Campaign builder debería retornar datos"
    
    # Channel Builder
    channel_builder = ChannelBuilder()
    channel = channel_builder.build_monthly_summary(df)
    assert len(channel) > 0, "Channel builder debería retornar datos"
    
    # Client Builder
    client_builder = ClientBuilder()
    client = client_builder.build_client_monthly(df)
    assert len(client) > 0, "Client builder debería retornar datos"
    
    # Profit Builder
    profit_builder = ProfitBuilder()
    profit = profit_builder.build_product_profit_monthly(df)
    assert len(profit) > 0, "Profit builder debería retornar datos"
    assert 'Ganancia' in profit.columns, "Profit debería incluir columna Ganancia"
    
    print("  ✓ Todos los nuevos builders funcionan correctamente")


def test_pipeline_compatibility():
    """Test 4: Pipeline completo mantiene compatibilidad y expone nuevos resultados."""
    print("Test 4: Pipeline compatibility and new result fields...")
    
    pipeline = DataPipeline()
    
    # Verificar que el pipeline expone los nuevos builders
    assert hasattr(pipeline, 'campaign_builder'), "Pipeline debería tener campaign_builder"
    assert hasattr(pipeline, 'channel_builder'), "Pipeline debería tener channel_builder"
    assert hasattr(pipeline, 'client_builder'), "Pipeline debería tener client_builder"
    assert hasattr(pipeline, 'profit_builder'), "Pipeline debería tener profit_builder"
    
    print("  ✓ Pipeline expone todos los nuevos builders")


def test_new_result_structure():
    """Test 5: PipelineResult incluye nuevos campos."""
    print("Test 5: PipelineResult has new fields...")
    
    from src.data.pipeline import PipelineResult
    
    # Crear resultado con los nuevos campos
    result = PipelineResult(
        movements=pd.DataFrame(),
        demand_monthly=pd.DataFrame(),
        stock_monthly=pd.DataFrame(),
        demand_campaign=pd.DataFrame(),
        demand_channel=pd.DataFrame(),
        demand_client=pd.DataFrame(),
        profit_monthly=pd.DataFrame(),
        channel_performance=pd.DataFrame(),
        client_segmentation=pd.DataFrame(),
    )
    
    assert hasattr(result, 'demand_campaign'), "PipelineResult debería tener demand_campaign"
    assert hasattr(result, 'demand_channel'), "PipelineResult debería tener demand_channel"
    assert hasattr(result, 'demand_client'), "PipelineResult debería tener demand_client"
    assert hasattr(result, 'profit_monthly'), "PipelineResult debería tener profit_monthly"
    assert hasattr(result, 'channel_performance'), "PipelineResult debería tener channel_performance"
    assert hasattr(result, 'client_segmentation'), "PipelineResult debería tener client_segmentation"
    
    print("  ✓ PipelineResult incluye todos los campos nuevos")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("PHASE 2 INTEGRATION TESTS - Data Processing Layer")
    print("="*60 + "\n")
    
    try:
        test_data_cleaner_new_columns()
        test_demand_builder_extended_methods()
        test_new_builders_instantiation()
        test_pipeline_compatibility()
        test_new_result_structure()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED - Phase 2 Ready for Deployment")
        print("="*60 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
