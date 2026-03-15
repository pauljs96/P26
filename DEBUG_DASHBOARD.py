"""
Componente de DEBUGGING para el dashboard - Añadir a dashboard.py

Este fragmento se agregar después de que falle the carga, para diagnosticar problema.
"""

import streamlit as st
import pandas as pd

def debug_file_loading(uploaded_files):
    """
    Muestra información detallada de los archivos subidos.
    Útil para diagnosticar por qué falla la carga.
    """
    
    st.markdown("---")
    st.markdown("### 🔧 DEBUGGING - Información de Archivos")
    
    # Mostrar detalles de cada archivo
    for i, file in enumerate(uploaded_files, 1):
        with st.expander(f"📄 {file.name}"):
            file_content = file.read()
            file.seek(0)  # Reset para lectura siguiente
            
            # Info del archivo
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tamaño", f"{len(file_content) / 1024:.1f} KB")
            with col2:
                st.metric("Nombre", file.name)
            with col3:
                # Detectar encoding
                import chardet
                result = chardet.detect(file_content)
                st.metric("Encoding", result.get('encoding', 'unknown'))
            
            # Primeras líneas
            st.write("**Primeras 5 líneas:**")
            try:
                lines = file_content.decode('utf-8').split('\n')[:5]
                for j, line in enumerate(lines):
                    st.code(line[:200], language='text')
            except:
                st.write("(No se puede leer como UTF-8)")
            
            # Intentar cargar y mostrar estructura
            st.write("**Estructura del CSV:**")
            try:
                test_df = pd.read_csv(
                    pd.io.common.BytesIO(file_content),
                    sep=';',
                    encoding='utf-8',
                    nrows=1,
                    on_bad_lines='skip'
                )
                st.write(f"Filas: (verificar después de carga completa)")
                st.write(f"Columnas ({len(test_df.columns)}):")
                for col in test_df.columns:
                    st.write(f"  - `{col}`")
            except Exception as e:
                st.error(f"Error leyendo archivo: {e}")


# ==================== CÓMO USAR EN DASHBOARD ====================
# Después de que falla st.error("No se detectaron columnas..."), agregar:

if __name__ == '__main__':
    # Ejemplo de uso
    st.title("Debugging de carga de archivos")
    
    files = st.file_uploader("Sube archivos", type=['csv'], accept_multiple_files=True)
    
    if files:
        debug_file_loading(files)
        
        # Intentar cargar
        from src.data.data_loader import DataLoader
        loader = DataLoader()
        
        try:
            df = loader.load_files(files)
            st.success(f"Cargados: {len(df)} filas")
            
            st.write("**Columnas cargadas:**")
            for col in df.columns:
                st.write(f"  - `{col}`")
        except Exception as e:
            st.error(f"Error en DataLoader: {e}")
            import traceback
            st.error(traceback.format_exc())

