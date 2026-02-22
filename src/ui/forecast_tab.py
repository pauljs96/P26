"""
Componente Streamlit para generar pronósticos usando el API Backend
Integración con FastAPI para ETS y Random Forest
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from src.api.client import get_api_client


def render_api_forecasts_tab(db):
    """
    Renderiza la tab para generar pronósticos con el API
    """
    st.header("🤖 Pronósticos con ML (API Backend)")
    
    # Verificar que el API esté disponible
    api_client = get_api_client()
    if not api_client.health_check():
        st.error("""
        ❌ API Backend no está disponible
        
        Por favor ejecuta:
        ```bash
        python run_both.py
        ```
        """)
        return
    
    st.success("✅ API Backend disponible")
    
    # ==================== SELECCIÓN DE UPLOAD ====================
    
    st.subheader("1. Selecciona un archivo para procesar")
    
    # Obtener uploads de Supabase
    project_id = st.session_state.get("current_project_id", "default")
    user_id = st.session_state.get("user_id", "demo")
    
    if not db:
        st.error("No se puede conectar a la base de datos")
        return
    
    uploads = db.get_uploads(project_id)
    
    if not uploads:
        st.info("No hay archivos cargados en este proyecto")
        return
    
    # Mostrar uploads disponibles
    upload_options = {f"{u['filename']} ({u['status']})": u for u in uploads}
    selected_upload_name = st.selectbox(
        "Elige un archivo",
        options=list(upload_options.keys())
    )
    
    selected_upload = upload_options[selected_upload_name]
    upload_id = selected_upload["id"]
    filename = selected_upload["filename"]
    
    st.write(f"**Upload ID:** {upload_id}")
    st.write(f"**Status:** {selected_upload.get('status', 'unknown')}")
    st.write(f"**S3 Path:** {selected_upload.get('s3_path', 'N/A')}")
    
    st.divider()
    
    # ==================== PARÁMETROS DEL PRONÓSTICO ====================
    
    st.subheader("2. Configura la predicción")
    
    col1, col2 = st.columns(2)
    
    with col1:
        product_name = st.text_input(
            "Nombre del producto (o parte del nombre)",
            placeholder="Ej: 'Producto A' o 'PROD'"
        )
    
    with col2:
        periods = st.number_input(
            "Períodos a pronosticar",
            min_value=1,
            max_value=24,
            value=12,
            step=1
        )
    
    model_type = st.selectbox(
        "Modelo ML",
        options=["ets", "rf", "best"],
        format_func=lambda x: {
            "ets": "ETS (Holt-Winters)",
            "rf": "Random Forest",
            "best": "Automático (mejor de ambos)"
        }.get(x)
    )
    
    st.divider()
    
    # ==================== GENERAR PRONÓSTICO ====================
    
    if st.button("🚀 Generar Pronóstico", type="primary", use_container_width=True):
        if not product_name:
            st.error("Por favor ingresa el nombre del producto")
            return
        
        with st.spinner(f"Generando pronóstico con {model_type.upper()}..."):
            result = api_client.generate_forecast(
                upload_id=upload_id,
                product=product_name,
                model_type=model_type,
                forecast_periods=periods
            )
        
        if not result.get("success"):
            st.error(f"❌ Error: {result.get('error', 'Error desconocido')}")
            return
        
        forecast_data = result.get("data", {})
        
        st.success(f"✅ Pronóstico generado exitosamente")
        
        # ==================== MOSTRAR RESULTADOS ====================
        
        st.subheader("Resultados")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Producto", forecast_data.get("product", "N/A"))
        with col2:
            st.metric("Modelo", forecast_data.get("model_type", "N/A").upper())
        with col3:
            mape = forecast_data.get("mape")
            if mape is not None:
                st.metric("MAPE (Error %)", f"{mape:.2f}%")
            else:
                st.metric("MAPE", "N/A")
        
        # ==================== GRÁFICO DE PRONÓSTICO ====================
        
        forecast_values = forecast_data.get("forecast_values", [])
        
        if forecast_values:
            # Crear DataFrame para el gráfico
            df_forecast = pd.DataFrame({
                "Período": [f"T+{i+1}" for i in range(len(forecast_values))],
                "Demanda Pronosticada": forecast_values
            })
            
            # Crear gráfico
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_forecast["Período"],
                y=df_forecast["Demanda Pronosticada"],
                mode='lines+markers',
                name='Pronóstico',
                line=dict(color='#FF6B6B', width=2),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title=f"Pronóstico - {forecast_data.get('product', 'Producto')}",
                xaxis_title="Período",
                yaxis_title="Demanda (Unidades)",
                hovermode='x unified',
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar valores en tabla
            st.subheader("Valores de Pronóstico")
            st.dataframe(df_forecast, use_container_width=True)
            
            # Estadísticas
            st.subheader("Estadísticas")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Promedio", f"{pd.Series(forecast_values).mean():.2f}")
            with col2:
                st.metric("Mínimo", f"{pd.Series(forecast_values).min():.2f}")
            with col3:
                st.metric("Máximo", f"{pd.Series(forecast_values).max():.2f}")
            with col4:
                st.metric("Desv. Est.", f"{pd.Series(forecast_values).std():.2f}")
        
        else:
            st.warning("No se obtuvieron valores de pronóstico")
