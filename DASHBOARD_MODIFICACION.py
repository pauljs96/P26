"""
MODIFICACIÓN DEL DASHBOARD - Usa el API backend en lugar de procesar localmente.

Este fragmento reemplaza la sección de carga de archivos en dashboard.py (líneas ~1647-1720)

Esta solución es correcta para la nube porque:
✓ El dashboard solo sube archivos (responsabilidad: UI)
✓ El backend los procesa (responsabilidad: lógica)
✓ Los datos se guardan en Supabase (responsabilidad: persistencia)
✓ Fácil de escalar
"""

import requests
import streamlit as st

# CONFIGURACION
API_URL = os.getenv('API_BACKEND_URL', 'http://localhost:5000')

# ==================== SECCION DE CARGA (dashboard.py línea ~1635) ====================

# Admin puede subir
st.sidebar.header("📤 Subir Datos")
files = st.sidebar.file_uploader(
    "Sube CSV (2021–2025)",
    type=["csv"],
    accept_multiple_files=True
)

if not files:
    st.info("👆 Admin: Sube los CSV para procesar")
    st.stop()

# ==================== PROCESAR CON API (NUEVA FORMA - CORRECTA PARA LA NUBE) ====================

st.info(f"📡 Enviando {len(files)} archivo(s) al backend para procesar...")

with st.spinner("⏳ Procesando en el backend..."):
    try:
        # Preparar multipart/form-data
        api_files = []
        for file in files:
            api_files.append(('files', (file.name, file, 'text/csv')))
        
        # Enviar al backend
        response = requests.post(
            f'{API_URL}/process-files',
            files=api_files,
            data={
                'org_id': org_id,
                'user_id': user_id
            },
            timeout=300  # 5 minutos
        )
        
        if response.status_code != 200:
            error_msg = response.json().get('message', 'Unknown error')
            st.error(f"❌ Error en backend: {error_msg}")
            st.stop()
        
        result = response.json()
        
        if not result.get('success'):
            st.error(f"❌ {result.get('message', 'Processing failed')}")
            st.stop()
        
        # Datos procesados del API
        processed_data = result.get('data', {})
        
        st.success(f"✅ Procesados {processed_data.get('files_processed')} archivos")
        
        # Mostrar estadísticas
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Filas", f"{processed_data.get('total_raw_records', 0):,}")
        with col2:
            st.metric("Limpias", f"{processed_data.get('total_clean_records', 0):,}")
        with col3:
            st.metric("Demanda", f"{processed_data.get('total_demand_records', 0):,}")
        with col4:
            st.metric("Stock", f"{processed_data.get('total_stock_records', 0):,}")
        
        # Guardar en cache de Supabase
        st.info("💾 Guardando en Supabase...")
        if db:
            try:
                # Convertir DataFrames desde JSON
                movements_df = pd.DataFrame(processed_data.get('movements', []))
                demand_df = pd.DataFrame(processed_data.get('demand', []))
                stock_df = pd.DataFrame(processed_data.get('stock', []))
                
                cache_saved = save_org_cache(
                    db=db,
                    org_id=org_id,
                    movements=movements_df,
                    demand_monthly=demand_df,
                    stock_monthly=stock_df,
                    processed_by=user_id,
                    csv_files_count=len(files)
                )
                
                if cache_saved:
                    st.success("✅ Datos guardados en Supabase")
                    st.balloons()
                else:
                    st.warning("⚠️ Error en cache (pero datos procesados)")
            
            except Exception as db_error:
                st.warning(f"⚠️ Error Supabase: {db_error}")
        
        # La lógica del resto del dashboard continúa igual
        res_movements = movements_df
        # ... resto del código original
    
    except requests.exceptions.ConnectionError:
        st.error(f"❌ No puedo conectar al backend en {API_URL}")
        st.info("Verifica que el backend está corriendo")
        st.stop()
    
    except requests.exceptions.Timeout:
        st.error("⏱️ El backend tardó demasiado - archivos muy grandes?")
        st.stop()
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        st.stop()

