"""
Admin Panel Module
Contiene lógica para administradores: crear usuarios, configurar CSV schema, etc.
"""

import streamlit as st
import pandas as pd
from typing import Optional, Dict, Any
from src.db.supabase import SupabaseDB
import json


class AdminPanel:
    """Gestiona operaciones administrativas en Supabase"""
    
    def __init__(self, db: SupabaseDB):
        self.db = db
    
    def render(self):
        """Renderiza el panel administrativo completo"""
        st.title("⚙️ Panel Administrativo")
        
        # Verificar que el usuario es admin
        if not st.session_state.get("is_admin"):
            st.error("❌ Acceso denegado. Solo administradores pueden acceder aquí.")
            return
        
        org_id = st.session_state.get("organization_id")
        org_name = st.session_state.get("organization_name")
        
        st.info(f"🏢 Organización: **{org_name}** (ID: {org_id[:8]}...)")
        
        # Sub-tabs del admin panel
        admin_tabs = st.tabs([
            "👥 Gestionar Usuarios",
            "⚙️ Configurar CSV",
            "📊 Ver Datos Cacheados",
            "🔄 Refrescar Data"
        ])
        
        with admin_tabs[0]:
            self._render_user_management(org_id)
        
        with admin_tabs[1]:
            self._render_csv_config(org_id)
        
        with admin_tabs[2]:
            self._render_cache_view(org_id)
        
        with admin_tabs[3]:
            self._render_data_refresh(org_id)
    
    # ==================== USER MANAGEMENT ====================
    
    def _render_user_management(self, org_id: str):
        """TAB 1: Crear y listar usuarios de la organización"""
        st.subheader("Gestionar Usuarios")
        
        # Listar usuarios actuales
        st.write("**Usuarios actuales en tu organización:**")
        users = self.db.get_organization_users(org_id)
        
        if users:
            # Convertir a DataFrame para mostrar nices
            users_df = pd.DataFrame([
                {
                    "Email": u.get("email"),
                    "Rol": "👑 Admin" if u.get("is_admin") else "👤 Viewer",
                    "Estado": u.get("status", "active"),
                    "Creado": u.get("created_at", "N/A")[:10]
                }
                for u in users
            ])
            st.dataframe(users_df, use_container_width=True)
        else:
            st.info("📭 No hay usuarios en esta organización.")
        
        st.divider()
        
        # Formulario para crear nuevo usuario
        st.subheader("➕ Crear Nuevo Usuario")
        
        with st.form("create_user_form"):
            email = st.text_input(
                "Email del nuevo usuario",
                placeholder="usuario@empresa.com"
            )
            
            password = st.text_input(
                "Contraseña temporal",
                type="password",
                placeholder="Mínimo 8 caracteres"
            )
            
            is_admin = st.checkbox("✅ Hacer administrador")
            
            submitted = st.form_submit_button("Crear Usuario", type="primary")
            
            if submitted:
                self._create_new_user(org_id, email, password, is_admin)
    
    def _create_new_user(self, org_id: str, email: str, password: str, is_admin: bool):
        """Crear nuevo usuario en la organización"""
        
        # Validaciones
        if not email or "@" not in email:
            st.error("❌ Email inválido")
            return
        
        if len(password) < 8:
            st.error("❌ Contraseña debe tener mínimo 8 caracteres")
            return
        
        # Intentar crear
        current_user_id = st.session_state.get("user_id")
        result = self.db.create_user_in_organization(
            org_id=org_id,
            email=email,
            password=password,
            is_admin=is_admin,
            created_by=current_user_id
        )
        
        if result.get("success"):
            st.success(f"✅ Usuario creado: {email}")
            st.info(f"""
            📧 El usuario puede loguearse con:
            - **Email:** {email}
            - **Contraseña temporal:** (la proporcionada)
            
            ⚠️ **Recomendación:** Pedirle que cambie contraseña en su primer login.
            """)
            st.rerun()
        else:
            st.error(f"❌ Error: {result.get('error', 'Desconocido')}")
    
    # ==================== CSV CONFIG ====================
    
    def _render_csv_config(self, org_id: str):
        """TAB 2: Configurar formato esperado de CSVs"""
        st.subheader("Configurar Formato de CSV")
        
        # Cargar config actual (si existe)
        current_schema = self.db.get_csv_schema(org_id)
        
        if current_schema:
            st.success("✅ Ya tienes una configuración guardada")
            st.json(current_schema)
        else:
            st.info("📋 No hay configuración de CSV. Crea una usando el formulario abajo.")
        
        st.divider()
        
        st.write("""
        **¿Cómo funciona?**
        
        1. Define el **separador** que usa tu CSV (`,`, `;`, `|`, etc)
        2. Define la **codificación** (UTF-8, Latin-1, etc)
        3. Mapea tus **columnas** a los campos esperados:
           - `product`: Código del producto
           - `date`: Fecha del movimiento
           - `quantity`: Cantidad
           - `company`: Empresa/sucursal
        """)
        
        with st.form("csv_config_form"):
            separator = st.selectbox(
                "Separador de CSV",
                options=[",", ";", "|", "\t"],
                index=0,
                help="¿Qué carácter separa las columnas?"
            )
            
            encoding = st.selectbox(
                "Encoding/Codificación",
                options=["utf-8", "latin-1", "cp1252", "iso-8859-1"],
                index=0,
                help="¿Qué encoding usa tu CSV?"
            )
            
            st.write("**Mapeo de Columnas** (¿cómo se llaman en tu CSV?)")
            
            col1, col2 = st.columns(2)
            with col1:
                product_col = st.text_input(
                    "Nombre de columna: Producto",
                    value=current_schema.get("column_mapping", {}).get("product", "producto") if current_schema else "producto"
                )
                quantity_col = st.text_input(
                    "Nombre de columna: Cantidad",
                    value=current_schema.get("column_mapping", {}).get("quantity", "cantidad") if current_schema else "cantidad"
                )
            
            with col2:
                date_col = st.text_input(
                    "Nombre de columna: Fecha",
                    value=current_schema.get("column_mapping", {}).get("date", "fecha") if current_schema else "fecha"
                )
                company_col = st.text_input(
                    "Nombre de columna: Empresa",
                    value=current_schema.get("column_mapping", {}).get("company", "empresa") if current_schema else "empresa"
                )
            
            submitted = st.form_submit_button("💾 Guardar Configuración", type="primary")
            
            if submitted:
                self._save_csv_config(
                    org_id=org_id,
                    separator=separator,
                    encoding=encoding,
                    column_mapping={
                        "product": product_col,
                        "date": date_col,
                        "quantity": quantity_col,
                        "company": company_col
                    }
                )
    
    def _save_csv_config(
        self, 
        org_id: str, 
        separator: str, 
        encoding: str, 
        column_mapping: Dict[str, str]
    ):
        """Guardar configuración de CSV en Supabase"""
        current_user_id = st.session_state.get("user_id")
        result = self.db.save_csv_schema(
            org_id=org_id,
            separator=separator,
            encoding=encoding,
            column_mapping=column_mapping,
            updated_by=current_user_id
        )
        
        if result.get("success"):
            st.success("✅ Configuración guardada correctamente")
            st.json({
                "separator": separator,
                "encoding": encoding,
                "column_mapping": column_mapping
            })
            st.rerun()
        else:
            st.error(f"❌ Error al guardar: {result.get('error')}")
    
    # ==================== CACHE VIEW ====================
    
    def _render_cache_view(self, org_id: str):
        """TAB 3: Ver datos cacheados"""
        st.subheader("Ver Datos Cacheados")
        
        is_loaded = self.db.is_data_loaded(org_id)
        
        if is_loaded:
            st.success("✅ Data cacheada disponible")
            
            # Cargar y mostrar cache
            cache_data = self.db.load_org_data(org_id)
            
            if cache_data.get("success"):
                st.write(f"**Última actualización:** {cache_data.get('updated_at', 'N/A')}")
                st.write(f"**CSVs procesados:** {cache_data.get('csv_files_count', 0)}")
                
                # Preview de demand_monthly
                if cache_data.get("demand_monthly"):
                    st.write("**Demanda Mensual (primeras 5 filas):**")
                    from src.utils.cache_helpers import json_to_dataframe, decompress_json
                    demand_compressed = cache_data.get("demand_monthly")
                    demand_json = decompress_json(demand_compressed) if demand_compressed else None
                    if demand_json:
                        demand_df = json_to_dataframe(demand_json)
                        st.dataframe(demand_df.head(), use_container_width=True)
                
                # Preview de stock_monthly
                if cache_data.get("stock_monthly"):
                    st.write("**Stock Mensual (primeras 5 filas):**")
                    from src.utils.cache_helpers import json_to_dataframe, decompress_json
                    stock_compressed = cache_data.get("stock_monthly")
                    stock_json = decompress_json(stock_compressed) if stock_compressed else None
                    if stock_json:
                        stock_df = json_to_dataframe(stock_json)
                        st.dataframe(stock_df.head(), use_container_width=True)
            else:
                st.error("❌ Error al cargar cache: " + cache_data.get("error", ""))
        
        else:
            st.warning("⚠️ No hay data cacheada. Los usuarios deberán subir CSVs primero.")
    
    # ==================== DATA REFRESH ====================
    
    def _render_data_refresh(self, org_id: str):
        """TAB 4: Refrescar/reprocessar data"""
        st.subheader("Refrescar Datos Procesados")
        
        st.info("""
        Esta sección permite volver a procesar los CSVs y actualizar el cache.
        Útil cuando tienes nuevos datos o cambios en el pipeline.
        """)
        
        st.warning("⚠️ En WEEK 3 agregaremos la UI para subir nuevos CSVs aquí")
        
        is_loaded = self.db.is_data_loaded(org_id)
        if is_loaded:
            st.write("Data actual cacheada: ✅ Sí")
            
            if st.button("🔄 Limpiar Cache (requiere re-subir CSVs)", type="secondary"):
                # Marcar como no cargado
                self.db.client.table("organizations").update({
                    "data_loaded": False
                }).eq("id", org_id).execute()
                
                st.success("✅ Cache limpiado. Solicita a los usuarios que re-suban los CSVs.")
                st.rerun()
        else:
            st.write("Data actual cacheada: ❌ No")
