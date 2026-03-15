"""
API Backend para procesar datos en la nube.
Permite que el dashboard envíe múltiples archivos y los procese correctamente.

Deployment: 
  - Local: python api_backend.py
  - Cloud: Heroku, Render, Railway, etc.
"""

from flask import Flask, request, jsonify
from werkzeug.datastructures import FileStorage
import pandas as pd
import io
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Importar componentes del pipeline
from src.data.data_loader import DataLoader
from src.data.data_cleaner import DataCleaner
from src.data.guide_reconciliation import GuideReconciler
from src.data.demand_builder import DemandBuilder
from src.data.stock_builder import StockBuilder
from src.db import get_db


@app.route('/health', methods=['GET'])
def health():
    """Validar que el API está activo."""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'service': 'PREDICAST Data Processing API'
    })


@app.route('/process-files', methods=['POST'])
def process_files():
    """
    Procesa múltiples archivos CSV subidos por el dashboard.
    
    Esperado (multipart/form-data):
    - files: Array de archivos CSV
    - org_id: Organización (string)
    - user_id: Usuario que sube (string)
    
    Retorna:
    {
        'success': bool,
        'data': {
            'total_raw_records': int,
            'total_clean_records': int,
            'total_demand_records': int,
            'total_stock_records': int,
            'movements': [...],  # JSON para guardar
            'demand': [...],
            'stock': [...]
        },
        'message': str
    }
    """
    
    try:
        # Validar que hay archivos
        if 'files' not in request.files or len(request.files.getlist('files')) == 0:
            return jsonify({
                'success': False,
                'message': 'No files provided'
            }), 400
        
        org_id = request.form.get('org_id', 'UNKNOWN')
        user_id = request.form.get('user_id', 'anonymous')
        
        print(f"\n📥 Procesando {len(request.files.getlist('files'))} archivos para {org_id}")
        
        # Procesar con el pipeline
        loader = DataLoader()
        cleaner = DataCleaner()
        reconciler = GuideReconciler()
        demand_builder = DemandBuilder()
        stock_builder = StockBuilder()
        
        # Cargar archivos
        files = request.files.getlist('files')
        df_raw = loader.load_files(files)
        
        if df_raw.empty:
            return jsonify({
                'success': False,
                'message': 'No data loaded from files'
            }), 400
        
        print(f"✓ Cargados: {len(df_raw):,} filas")
        
        # Limpiar
        df_clean = cleaner.clean(df_raw)
        if df_clean.empty:
            return jsonify({
                'success': False,
                'message': 'Data is empty after cleaning. Column mismatch?'
            }), 400
        
        print(f"✓ Limpiados: {len(df_clean):,} filas")
        
        # Reconciliar
        df_rec = reconciler.reconcile(df_clean)
        print(f"✓ Reconciliados: {len(df_rec):,} filas")
        
        # Construir demanda
        demand = demand_builder.build_monthly(df_rec)
        print(f"✓ Demanda: {len(demand):,} registros")
        
        # Construir stock
        stock = stock_builder.build_monthly(df_rec)
        print(f"✓ Stock: {len(stock):,} registros")
        
        # Guardar en Supabase (opcional, depende de si el dashboard lo hace)
        db = get_db()
        if db:
            try:
                # Guardar metadata de proceso
                upload_meta = {
                    'organization_id': org_id,
                    'uploaded_by': user_id,
                    'uploaded_at': datetime.now().isoformat(),
                    'total_raw_records': len(df_raw),
                    'total_clean_records': len(df_clean),
                    'total_demand_records': len(demand),
                    'total_stock_records': len(stock),
                    'file_count': len(files),
                    'status': 'processed'
                }
                
                result = db.client.table('uploads').insert(upload_meta).execute()
                print(f"✓ Metadata guardada en Supabase")
            except Exception as e:
                print(f"⚠️ No se guardó metadata (pero datos procesados correctamente): {e}")
        
        # Retornar datos procesados
        return jsonify({
            'success': True,
            'message': f'Processed {len(files)} files successfully',
            'data': {
                'total_raw_records': len(df_raw),
                'total_clean_records': len(df_clean),
                'total_demand_records': len(demand),
                'total_stock_records': len(stock),
                'files_processed': len(files),
                'movements': df_rec.to_dict(orient='records'),  # Para Supabase
                'demand': demand.to_dict(orient='records'),
                'stock': stock.to_dict(orient='records')
            }
        }), 200
    
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'message': f'Error processing files: {str(e)}'
        }), 500


@app.route('/process-single-year', methods=['POST'])
def process_single_year():
    """
    Procesa datos de un solo año (útil para cargas incrementales).
    
    Esperado (JSON):
    {
        'year': 2026,
        'org_id': 'SVMPREMIUM',
        'user_id': 'admin@example.com'
    }
    """
    try:
        data = request.get_json()
        year = data.get('year')
        org_id = data.get('org_id', 'UNKNOWN')
        user_id = data.get('user_id', 'anonymous')
        
        print(f"\n📅 Procesando año {year} para {org_id}")
        
        # Buscar archivo local (esto fallará en producción, necesitas S3 o BD)
        archivo = f'D_{year}.csv'
        
        if not os.path.exists(archivo):
            return jsonify({
                'success': False,
                'message': f'File {archivo} not found'
            }), 404
        
        # Procesar
        df = pd.read_csv(archivo, sep=';', decimal=',')
        
        cleaner = DataCleaner()
        reconciler = GuideReconciler()
        demand_builder = DemandBuilder()
        stock_builder = StockBuilder()
        
        df_clean = cleaner.clean(df)
        df_rec = reconciler.reconcile(df_clean)
        demand = demand_builder.build_monthly(df_rec)
        stock = stock_builder.build_monthly(df_rec)
        
        print(f"✓ {year}: {len(demand)} demanda, {len(stock)} stock")
        
        return jsonify({
            'success': True,
            'message': f'Year {year} processed successfully',
            'data': {
                'year': year,
                'total_clean_records': len(df_clean),
                'total_demand_records': len(demand),
                'total_stock_records': len(stock),
                'demand': demand.to_dict(orient='records'),
                'stock': stock.to_dict(orient='records')
            }
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    print("="*70)
    print("PREDICAST Data Processing API")
    print("="*70)
    print(f"Corriendo en puerto {port}")
    print(f"Debug: {debug}")
    print("\nEndpoints:")
    print("  GET  /health")
    print("  POST /process-files")
    print("  POST /process-single-year")
    print("="*70)
    
    app.run(host='0.0.0.0', port=port, debug=debug)

