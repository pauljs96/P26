"""
Dashboard Multi-Tenant - Sistema Tesis
=======================================

Características:
- Autenticación multi-tenant (Supabase JWT)
- RBAC (Master Admin → Org Admin → Viewer)
- Carga datos con DataService (DuckDB + S3)
- Validación org_id en cada operación
- UI adaptada según rol del usuario

Roles:
- Master Admin: Gestionar todas las orgs, usuarios, ver todo
- Org Admin: Gestionar su org, subir datos, ver análisis
- Viewer: Ver datos de su org (lectura)
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging
from datetime import datetime
from typing import Optional, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import servicios multi-tenant
from src.db.supabase_v2 import get_supabase_db
from src.services.data_service import create_data_service
from src.storage.s3_manager_v2 import get_s3_manager


# ========== PAGE CONFIG ==========

st.set_page_config(
    page_title="Sistema Tesis - Multi-Tenant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========== ESTILOS CSS ==========

st.markdown("""
<style>
    /* Header profesional */
    .header-title {
        font-size: 2.5em;
        font-weight: 700;
        color: #1976D2;
        margin-bottom: 0.5em;
    }
    
    /* Tabla de datos */
    .data-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.95em;
    }
    
    /* Métricas destacadas */
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 8px;
        padding: 1.5em;
        color: white;
        text-align: center;
    }
    
    /* Alertas */
    .alert-success {
        background-color: #E8F5E9;
        border-left: 4px solid #4CAF50;
        padding: 1em;
        border-radius: 4px;
    }
    
    .alert-warning {
        background-color: #FFF3E0;
        border-left: 4px solid #FF9800;
        padding: 1em;
        border-radius: 4px;
    }
    
    .alert-error {
        background-color: #FFEBEE;
        border-left: 4px solid #F44336;
        padding: 1em;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ========== SESSION STATE INITIALIZATION ==========

def init_session_state():
    """Inicializa variables de sesión."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "email" not in st.session_state:
        st.session_state.email = None
    if "role" not in st.session_state:
        st.session_state.role = None
    if "org_id" not in st.session_state:
        st.session_state.org_id = None
    if "org_name" not in st.session_state:
        st.session_state.org_name = None
    if "user_orgs" not in st.session_state:
        st.session_state.user_orgs = []


# ========== AUTHENTICATION ==========

def show_login_page():
    """Pantalla de login multi-tenant."""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; margin-top: 2em;'>
            <h1 style='color: #1976D2; font-size: 3em;'>📊 Sistema Tesis</h1>
            <p style='color: #666; font-size: 1.2em;'>Demanda & Pronóstico Multi-Tenant</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Formulario de login
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input(
                "📧 Email",
                placeholder="usuario@empresa.com",
                key="login_email"
            )
            password = st.text_input(
                "🔐 Contraseña",
                type="password",
                placeholder="Tu contraseña",
                key="login_password"
            )
            
            submitted = st.form_submit_button(
                "🚀 Iniciar Sesión",
                use_container_width=True,
                type="primary"
            )
        
        if submitted:
            if not email or not password:
                st.error("❌ Por favor completa todos los campos")
                return
            
            try:
                db = get_supabase_db()
                
                # Intentar login
                login_result = db.client.auth.sign_in_with_password({
                    "email": email,
                    "password": password,
                })
                
                if login_result.user:
                    # Obtener info del usuario
                    user_info = db.get_user(login_result.user.id)
                    
                    if user_info["success"]:
                        user_data = user_info["user"]
                        
                        # Obtener orgs del usuario
                        orgs_result = db.get_user_organizations(login_result.user.id)
                        
                        if orgs_result["success"]:
                            st.session_state.authenticated = True
                            st.session_state.user_id = login_result.user.id
                            st.session_state.email = email
                            st.session_state.is_master_admin = user_data.get("is_master_admin", False)
                            st.session_state.user_orgs = orgs_result["organizations"]
                            
                            # Seleccionar primera org por defecto
                            if orgs_result["organizations"]:
                                first_org = orgs_result["organizations"][0]
                                st.session_state.org_id = first_org["org_id"]
                                st.session_state.org_name = first_org["org_name"]
                                st.session_state.role = first_org["role"]
                            
                            # Limpiar caché
                            st.cache_data.clear()
                            st.cache_resource.clear()
                            
                            st.success(f"✅ Bienvenido {email}!")
                            st.rerun()
                        else:
                            st.error("❌ No tienes acceso a ninguna organización")
                    else:
                        st.error("❌ Error obteniendo info del usuario")
                else:
                    st.error("❌ Credenciales inválidas")
            
            except Exception as e:
                logger.error(f"Login error: {e}")
                st.error(f"❌ Error: {str(e)}")
        
        st.divider()
        st.info(
            """
            💡 **Demo recomendado:**
            - Email: `admin@techinnovations.local`
            - Contraseña: `OrgAdmin@123456`
            
            Si no tienes cuenta, contacta al administrador.
            """
        )


# ========== HEADER & SIDEBAR ==========

def render_header():
    """Renderiza header con info del usuario."""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"""
        <div style='padding: 1em; background: linear-gradient(90deg, #1976D2, #1565C0); 
                    border-radius: 8px; color: white;'>
            <h2 style='margin: 0;'>📊 Sistema Tesis</h2>
            <p style='margin: 0; font-size: 0.9em; opacity: 0.9;'>
                Org: <b>{st.session_state.org_name}</b> | Role: <b>{st.session_state.role}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_id = None
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()


def render_sidebar():
    """Renderiza sidebar con navegación y opciones."""
    with st.sidebar:
        st.title("🎯 Navegación")
        
        # Info del usuario
        st.write(f"**Usuario:** {st.session_state.email}")
        st.write(f"**Rol:** {st.session_state.role}")
        
        # Selector de org (solo si master admin o múltiples orgs)
        if st.session_state.is_master_admin or len(st.session_state.user_orgs) > 1:
            st.divider()
            st.subheader("🏢 Cambiar Org")
            
            org_options = {org["org_name"]: org for org in st.session_state.user_orgs}
            selected_org_name = st.selectbox(
                "Selecciona organización:",
                options=org_options.keys(),
                key="org_selector"
            )
            
            if selected_org_name:
                selected_org = org_options[selected_org_name]
                st.session_state.org_id = selected_org["org_id"]
                st.session_state.org_name = selected_org["org_name"]
                st.session_state.role = selected_org["role"]
                st.cache_data.clear()
                st.rerun()
        
        st.divider()
        st.subheader("📑 Secciones")
        
        # Menú principal
        page = st.radio(
            "Selecciona sección:",
            options=[
                "🏠 Dashboard",
                "📊 Datos",
                "🔍 Análisis",
                "📤 Uploads" if st.session_state.role == "org_admin" else None,
                "👥 Usuarios" if st.session_state.is_master_admin else None,
                "⚙️ Admin" if st.session_state.is_master_admin else None,
            ],
            key="page_selector"
        )
        
        return page


# ========== RBAC VALIDATION ==========

def require_role(*allowed_roles):
    """Decorador para requerir roles específicos."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if st.session_state.role not in allowed_roles:
                st.error(f"❌ Acceso denegado. Se requiere uno de: {', '.join(allowed_roles)}")
                return None
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_org_access():
    """Valida que el usuario tenga acceso a la org actual."""
    if not st.session_state.org_id:
        st.error("❌ No hay org seleccionada")
        return False
    
    valid_orgs = [org["org_id"] for org in st.session_state.user_orgs]
    if st.session_state.org_id not in valid_orgs:
        st.error("❌ No tienes acceso a esta organización")
        return False
    
    return True


# ========== DATA LOADING ==========

@st.cache_resource
def get_data_service():
    """Obtiene DataService cacheado para la org actual."""
    try:
        return create_data_service(st.session_state.org_id)
    except Exception as e:
        logger.error(f"Error creating DataService: {e}")
        st.error(f"❌ Error: {e}")
        return None


@st.cache_data(ttl=300)  # Cache 5 minutos
def load_org_data(years: list[str]) -> Dict[str, Any]:
    """Carga datos de la org desde S3 usando DataService."""
    try:
        ds = get_data_service()
        if not ds:
            return {"success": False, "error": "No DataService available"}
        
        # Cargar múltiples años
        result = ds.load_multiple_csvs(years, prefix="raw")
        
        if result["success"]:
            logger.info(f"✅ Datos cargados: {result['total_rows']} filas")
            return result
        else:
            logger.error(f"Error loading data: {result['errors']}")
            return result
    
    except Exception as e:
        logger.error(f"Error in load_org_data: {e}")
        return {"success": False, "error": str(e)}


# ========== DASHBOARD PAGES ==========

def page_dashboard():
    """Página principal del dashboard."""
    render_header()
    
    st.title("📊 Dashboard Principal")
    st.write(f"Bienvenido **{st.session_state.email}** a **{st.session_state.org_name}**")
    
    if not require_org_access():
        return
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📦 Orgs", len(st.session_state.user_orgs))
    
    with col2:
        st.metric("👤 Rol", st.session_state.role.upper())
    
    with col3:
        st.metric("🔒 Estado", "✅ Autenticado")
    
    with col4:
        st.metric("⏰ Hora", datetime.now().strftime("%H:%M:%S"))
    
    st.divider()
    
    # Cargar datos disponibles
    st.subheader("📥 Datos Disponibles")
    
    years = st.multiselect(
        "Selecciona años de datos:",
        options=["2020", "2021", "2022", "2023", "2024", "2025"],
        default=["2024", "2025"]
    )
    
    if st.button("⬇️ Cargar Datos", use_container_width=True, type="primary"):
        with st.spinner("Cargando datos desde S3..."):
            result = load_org_data(years)
            
            if result["success"]:
                st.success(f"✅ {result['total_rows']:,} filas cargadas en {result['total_size_mb']:.2f} MB")
                
                # Mostrar tabla de archivos cargados
                st.json(result["loaded_tables"])
            else:
                st.error(f"❌ Error: {result['error']}")


def page_datos():
    """Página de exploración de datos."""
    render_header()
    
    st.title("📊 Exploración de Datos")
    
    if not require_org_access():
        return
    
    # Cargar datos
    years = st.multiselect(
        "Años a cargar:",
        ["2020", "2021", "2022", "2023", "2024", "2025"],
        default=["2025"]
    )
    
    if not years:
        st.warning("⚠️ Selecciona al menos un año")
        return
    
    with st.spinner("Cargando datos..."):
        result = load_org_data(years)
    
    if not result["success"]:
        st.error(f"❌ Error: {result['error']}")
        return
    
    st.success(f"✅ {result['total_rows']:,} filas cargadas")
    
    # Tabs para diferentes vistas
    tab1, tab2, tab3 = st.tabs(["📋 Preview", "📈 Estadísticas", "🔍 Query"])
    
    with tab1:
        st.subheader("Vista previa de datos")
        
        # Obtener sample de datos
        try:
            ds = get_data_service()
            sample_result = ds.query(f"SELECT * FROM data_2025 LIMIT 100")
            
            if sample_result["success"]:
                st.dataframe(sample_result["data"], use_container_width=True)
            else:
                st.error(f"Error: {sample_result['error']}")
        
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
    with tab2:
        st.subheader("Estadísticas descriptivas")
        
        try:
            ds = get_data_service()
            stats = ds.get_stats("data_2025")
            
            if stats["success"]:
                st.json(stats["stats"])
            else:
                st.error(f"Error: {stats['error']}")
        
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
    with tab3:
        st.subheader("Ejecutar query SQL")
        
        query = st.text_area(
            "Escribe tu query SQL:",
            value="SELECT * FROM data_2025 LIMIT 10",
            height=150
        )
        
        if st.button("🔍 Ejecutar Query", type="primary", use_container_width=True):
            try:
                ds = get_data_service()
                result = ds.query(query)
                
                if result["success"]:
                    st.dataframe(result["data"], use_container_width=True)
                    st.caption(f"✅ {result['rows']} filas retornadas")
                else:
                    st.error(f"❌ Error: {result['error']}")
            
            except Exception as e:
                st.error(f"❌ Error: {e}")


def page_analisis():
    """Página de análisis."""
    render_header()
    
    st.title("🔍 Análisis")
    st.info("💡 Módulo de análisis en desarrollo...")


def page_uploads():
    """Página de upload de datos (solo org_admin)."""
    render_header()
    
    if st.session_state.role != "org_admin":
        st.error("❌ Solo org_admin puede subir datos")
        return
    
    st.title("📤 Subir Datos")
    st.write(f"Subir datos para **{st.session_state.org_name}**")
    
    # Selector de año
    year = st.selectbox("Año de datos:", ["2020", "2021", "2022", "2023", "2024", "2025"])
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Selecciona archivo CSV",
        type=["csv"],
        accept_multiple_files=False
    )
    
    if uploaded_file and st.button("📤 Subir a S3", type="primary", use_container_width=True):
        with st.spinner("Subiendo archivo..."):
            try:
                s3 = get_s3_manager()
                db = get_supabase_db()
                
                # Leer archivo
                df = pd.read_csv(uploaded_file)
                csv_bytes = df.to_csv(index=False).encode()
                
                # Subir a S3
                result = s3.upload_bytes(
                    file_bytes=csv_bytes,
                    filename=uploaded_file.name,
                    org_id=st.session_state.org_id,
                    data_type="raw"
                )
                
                if result["success"]:
                    # Registrar en DB
                    db.log_upload(
                        org_id=st.session_state.org_id,
                        uploaded_by=st.session_state.user_id,
                        file_name=uploaded_file.name,
                        s3_path=result["s3_key"],
                        year=year,
                        file_size_mb=result["file_size_mb"],
                        rows_processed=len(df)
                    )
                    
                    st.success(f"✅ Archivo subido: {result['s3_key']}")
                    st.info(f"📊 {len(df):,} filas procesadas")
                else:
                    st.error(f"❌ Error: {result['error']}")
            
            except Exception as e:
                st.error(f"❌ Error: {e}")


def page_usuarios():
    """Página de gestión de usuarios (solo master_admin)."""
    render_header()
    
    if not st.session_state.is_master_admin:
        st.error("❌ Solo master_admin puede gestionar usuarios")
        return
    
    st.title("👥 Gestión de Usuarios")
    
    # Obtener todos los usuarios de la org
    try:
        db = get_supabase_db()
        members = db.get_org_members(st.session_state.org_id)
        
        if members["success"]:
            st.dataframe(
                pd.DataFrame(members["members"]),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.error(f"Error: {members['error']}")
    
    except Exception as e:
        st.error(f"❌ Error: {e}")


def page_admin():
    """Página de administración (solo master_admin)."""
    render_header()
    
    if not st.session_state.is_master_admin:
        st.error("❌ Solo master_admin puede acceder")
        return
    
    st.title("⚙️ Panel de Administración")
    
    tab1, tab2, tab3 = st.tabs(["🏢 Organizaciones", "👥 Usuarios", "📊 Auditoría"])
    
    with tab1:
        st.subheader("Organizaciones")
        try:
            db = get_supabase_db()
            orgs = db.list_organizations()
            
            if orgs["success"]:
                st.dataframe(
                    pd.DataFrame(orgs["organizations"]),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.error(f"Error: {orgs['error']}")
        
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
    with tab2:
        st.subheader("Usuarios de la org")
        
        try:
            db = get_supabase_db()
            members = db.get_org_members(st.session_state.org_id)
            
            if members["success"]:
                st.dataframe(
                    pd.DataFrame(members["members"]),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.error(f"Error: {members['error']}")
        
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
    with tab3:
        st.subheader("Histórico de uploads")
        
        try:
            db = get_supabase_db()
            uploads = db.get_org_uploads(st.session_state.org_id)
            
            if uploads["success"]:
                st.dataframe(
                    pd.DataFrame(uploads["uploads"]),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.error(f"Error: {uploads['error']}")
        
        except Exception as e:
            st.error(f"❌ Error: {e}")


# ========== MAIN APP ==========

def main():
    """Función principal de la app."""
    init_session_state()
    
    # Check autenticación
    if not st.session_state.authenticated:
        show_login_page()
        return
    
    # Render sidebar y get página seleccionada
    page = render_sidebar()
    
    # Route a página
    if "Dashboard" in page:
        page_dashboard()
    elif "Datos" in page:
        page_datos()
    elif "Análisis" in page:
        page_analisis()
    elif "Uploads" in page:
        page_uploads()
    elif "Usuarios" in page:
        page_usuarios()
    elif "Admin" in page:
        page_admin()


if __name__ == "__main__":
    main()
