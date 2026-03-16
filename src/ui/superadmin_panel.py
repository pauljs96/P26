"""
Panel de Administración para Superadmins (Creadores del Sistema)

Permitir a superadmins:
- Crear nuevas organizaciones
- Crear admins de organizaciones
- Crear usuarios y asignarlos a orgs
- Eliminar usuarios de orgs
- Eliminar usuarios completamente
- Ver auditoría de cambios
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any

from src.db.supabase import SupabaseDB


class SuperAdminPanel:
    """Panel para gestión superadmin del sistema"""
    
    def __init__(self, db: Optional[SupabaseDB] = None):
        """
        Inicializa panel superadmin
        
        Args:
            db: Instancia de SupabaseDB
        """
        self.db = db
        if not self.db:
            from src.db.supabase import get_db
            self.db = get_db()
    
    def _is_superadmin(self) -> bool:
        """Verifica si usuario actual es superadmin"""
        import os
        superadmin_emails = os.getenv("SUPERADMIN_EMAILS", "").split(",")
        superadmin_emails = [e.strip().lower() for e in superadmin_emails if e.strip()]
        
        current_email = st.session_state.get("email", "").lower()
        return current_email in superadmin_emails
    
    def render(self):
        """Renderiza panel superadmin con múltiples tabs"""
        
        # Verificar permisos
        if not self._is_superadmin():
            st.error("❌ Acceso denegado. Solo superadmins pueden acceder a este panel.")
            return
        
        # Header
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2em; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 2.5em; border-radius: 12px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);'>
            <h1 style='color: #FFD700; font-size: 2.2em; margin: 0; font-weight: bold;'>⚙️ Panel de Superadministrador</h1>
            <p style='color: rgba(255, 255, 255, 0.8); font-size: 1em; margin-top: 0.8em; margin-bottom: 0;'>Gestión completa del sistema: organizaciones, usuarios y permisos</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Información de sesión
        with st.expander("📋 Información de Sesión", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("👤 Usuario", st.session_state.get("email", "N/A"))
            with col2:
                st.metric("🔐 Rol", "🏆 SUPERADMIN")
            with col3:
                st.metric("🔧 Sistema", "Predicast")
        
        st.divider()
        
        # Tabs principales
        tab_orgs, tab_users, tab_ops = st.tabs([
            "🏢 Organizaciones",
            "👥 Usuarios",
            "⚙️ Operaciones"
        ])
        
        # ==================== TAB 1: ORGANIZACIONES ====================
        with tab_orgs:
            self._render_organizations_tab()
        
        # ==================== TAB 2: USUARIOS ====================
        with tab_users:
            self._render_users_tab()
        
        # ==================== TAB 3: OPERACIONES ====================
        with tab_ops:
            self._render_operations_tab()
    
    def _render_organizations_tab(self):
        """Tab para gestionar organizaciones"""
        st.subheader("🏢 Administración de Organizaciones")
        
        tab_crear, tab_listar = st.tabs([
            "➕ Crear Nueva Org",
            "📊 Listar Todas"
        ])
        
        # ===== CREAR ORG =====
        with tab_crear:
            st.markdown("""
            <div style='background: #e3f2fd; padding: 14px; border-left: 4px solid #1976d2; border-radius: 8px; margin-bottom: 1.5em;'>
                <p style='margin: 0; color: #333; font-size: 0.9em;'>
                    <strong>📝 Crear nueva organización:</strong> Ingresa los detalles y asigna un administrador inicial.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.form("crear_org_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    org_nombre = st.text_input(
                        "Nombre de la Organización",
                        placeholder="Ej: Tech Solutions Inc."
                    )
                
                with col2:
                    org_description = st.text_input(
                        "Descripción (opcional)",
                        placeholder="Ej: Empresa de soluciones tecnológicas"
                    )
                
                # Obtener usuarios existentes para asignar admin
                all_users = self.db.get_all_users()
                users_without_org = [u for u in all_users if not u.get("organization_id")]
                
                if users_without_org:
                    admin_emails = [u["email"] for u in users_without_org]
                    selected_admin_email = st.selectbox(
                        "Admin de la Organización",
                        options=admin_emails,
                        help="Usuario que será administrador de esta org"
                    )
                    admin_user_id = next(u["id"] for u in users_without_org if u["email"] == selected_admin_email)
                else:
                    st.warning("⚠️ No hay usuarios disponibles sin organización. Crea usuarios primero.")
                    admin_user_id = None
                
                st.divider()
                
                submit = st.form_submit_button("✅ Crear Organización", type="primary", use_container_width=True)
                
                if submit:
                    if not org_nombre:
                        st.error("❌ El nombre de la organización es requerido")
                    elif not admin_user_id:
                        st.error("❌ Selecciona un administrador")
                    else:
                        result = self.db.create_organization(
                            nombre=org_nombre,
                            admin_user_id=admin_user_id,
                            description=org_description
                        )
                        
                        if result["success"]:
                            st.success(f"✅ Organización '{org_nombre}' creada exitosamente")
                            st.balloons()
                        else:
                            st.error(f"❌ Error al crear organización: {result.get('error')}")
        
        # ===== LISTAR ORGS =====
        with tab_listar:
            st.markdown("""
            <div style='background: #f3e5f5; padding: 14px; border-left: 4px solid #9c27b0; border-radius: 8px; margin-bottom: 1.5em;'>
                <p style='margin: 0; color: #333; font-size: 0.9em;'>
                    <strong>📊 Vista de todas las organizaciones:</strong> Datos y estado de cada una.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            orgs = self.db.list_all_organizations()
            
            if not orgs:
                st.info("📭 No hay organizaciones creadas aún")
            else:
                # Preparar dataframe
                orgs_df = []
                for org in orgs:
                    org_users = self.db.get_organization_users(org["id"])
                    orgs_df.append({
                        "ID": org["id"][:8] + "...",
                        "Nombre": org.get("name", org.get("nombre", "N/A")),
                        "Descripción": org.get("description", "N/A"),
                        "Usuarios": len(org_users),
                        "Data Cargada": "✅" if org.get("data_loaded") else "❌",
                        "Creada": org.get("created_at", "N/A")[:10]
                    })
                
                df = pd.DataFrame(orgs_df)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Expandir para detalles
                with st.expander("🔍 Detalles de Organizaciones", expanded=False):
                    for org in orgs:
                        org_name = org.get("name", org.get("nombre", "N/A"))
                        with st.expander(f"📍 {org_name}", expanded=False):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("ID", org["id"][:12] + "...")
                            with col2:
                                st.metric("Usuarios", len(self.db.get_organization_users(org["id"])))
                            with col3:
                                st.metric("Data Cargada", "Sí" if org.get("data_loaded") else "No")
                            
                            st.write(f"**Descripción:** {org.get('description', 'N/A')}")
                            st.write(f"**Admin ID:** {org.get('admin_user_id')}")
    
    def _render_users_tab(self):
        """Tab para gestionar usuarios"""
        st.subheader("👥 Administración de Usuarios")
        
        tab_crear_user, tab_listar_user, tab_manage = st.tabs([
            "➕ Crear Usuario",
            "📊 Listar Todos",
            "✏️ Modificar"
        ])
        
        # ===== CREAR USUARIO =====
        with tab_crear_user:
            st.markdown("""
            <div style='background: #e8f5e9; padding: 14px; border-left: 4px solid #4caf50; border-radius: 8px; margin-bottom: 1.5em;'>
                <p style='margin: 0; color: #333; font-size: 0.9em;'>
                    <strong>👤 Crear nuevo usuario:</strong> El usuario recibirá credenciales para acceder.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.form("crear_user_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    user_email = st.text_input("Email del Usuario", placeholder="user@company.com")
                
                with col2:
                    user_password = st.text_input("Contraseña Inicial", placeholder="Min 8 caracteres", type="password")
                
                # Seleccionar organización
                orgs = self.db.list_all_organizations()
                if orgs:
                    org_names = [o.get("name", o.get("nombre", "N/A")) for o in orgs]
                    selected_org_name = st.selectbox("Asignar a Organización", options=org_names)
                    selected_org = next(o for o in orgs if o.get("name", o.get("nombre")) == selected_org_name)
                    org_id = selected_org["id"]
                else:
                    org_id = None
                    st.info("⚠️ No hay organizaciones. Crea una primero.")
                
                # Rol
                is_admin = st.checkbox("¿Es administrador de esta org?", value=False)
                
                st.divider()
                
                submit = st.form_submit_button("✅ Crear Usuario", type="primary", use_container_width=True)
                
                if submit:
                    if not org_id:
                        st.error("❌ Debes crear una organización primero")
                    elif not user_email or not user_password:
                        st.error("❌ Todos los campos son requeridos")
                    elif len(user_password) < 8:
                        st.error("❌ La contraseña debe tener al menos 8 caracteres")
                    else:
                        result = self.db.create_user_in_organization(
                            org_id=org_id,
                            email=user_email,
                            password=user_password,
                            is_admin=is_admin,
                            created_by=st.session_state.get("user_id")
                        )
                        
                        if result["success"]:
                            st.cache_data.clear()  # Limpiar cache para refrescar listado
                            st.success(f"✅ Usuario '{user_email}' creado exitosamente")
                            st.info(f"🔑 ID del usuario: {result['user_id'][:12]}...")
                        else:
                            st.error(f"❌ Error al crear usuario: {result.get('error')}")
        
        # ===== LISTAR USUARIOS =====
        with tab_listar_user:
            st.markdown("""
            <div style='background: #fff3e0; padding: 14px; border-left: 4px solid #ff9800; border-radius: 8px; margin-bottom: 1.5em;'>
                <p style='margin: 0; color: #333; font-size: 0.9em;'>
                    <strong>📊 Listado de todos los usuarios del sistema:</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Botón de refresh
            if st.button("🔄 Refrescar Listado", key="refresh_users", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
            
            all_users = self.db.get_all_users()
            
            if not all_users:
                st.info("📭 No hay usuarios creados aún")
            else:
                # Filtro por organización
                orgs = self.db.list_all_organizations()
                org_options = ["Todas"] + [o.get("name", o.get("nombre", "N/A")) for o in orgs]
                selected_org_filter = st.selectbox("Filtrar por Organización", options=org_options)
                
                # Preparar dataframe
                users_df = []
                for user in all_users:
                    org_id = user.get("organization_id")
                    org_name = "Sin Org"
                    
                    if org_id:
                        org = self.db.get_organization(org_id)
                        org_name = org.get("name", org.get("nombre", "Desconocida")) if org else "Desconocida"
                    
                    # Aplicar filtro
                    if selected_org_filter != "Todas" and org_name != selected_org_filter:
                        continue
                    
                    users_df.append({
                        "Email": user["email"],
                        "Organización": org_name,
                        "Rol": "👑 Admin" if user.get("is_admin") else "👤 Viewer",
                        "Estado": user.get("status", "invited"),
                        "Creado": user.get("created_at", "N/A")[:10]
                    })
                
                if users_df:
                    df = pd.DataFrame(users_df)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("No hay usuarios en esa organización")
        
        # ===== MODIFICAR USUARIOS =====
        with tab_manage:
            st.markdown("""
            <div style='background: #fce4ec; padding: 14px; border-left: 4px solid #e91e63; border-radius: 8px; margin-bottom: 1.5em;'>
                <p style='margin: 0; color: #333; font-size: 0.9em;'>
                    <strong>✏️ Modificar permisos y asignaciones de usuarios:</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            all_users = self.db.get_all_users()
            
            if not all_users:
                st.info("📭 No hay usuarios en el sistema")
            else:
                user_emails = [u["email"] for u in all_users]
                selected_user_email = st.selectbox("Seleccionar Usuario", options=user_emails)
                selected_user = next(u for u in all_users if u["email"] == selected_user_email)
                
                st.divider()
                
                # Mostrar info actual
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Email", selected_user["email"])
                with col2:
                    st.metric("Rol", "👑 Admin" if selected_user.get("is_admin") else "👤 Viewer")
                with col3:
                    org_id = selected_user.get("organization_id")
                    if org_id:
                        org = self.db.get_organization(org_id)
                        org_name = org.get("name", org.get("nombre", "N/A")) if org else "N/A"
                    else:
                        org_name = "Sin Org"
                    st.metric("Organización", org_name)
                
                st.divider()
                
                # Operaciones
                col_op1, col_op2 = st.columns(2)
                
                with col_op1:
                    st.subheader("🔄 Cambiar Rol")
                    new_is_admin = st.toggle("¿Hacerlo Administrador?", value=selected_user.get("is_admin"))
                    if st.button("💾 Guardar Rol", key="save_role"):
                        result = self.db.update_user_role(selected_user["id"], new_is_admin)
                        if result["success"]:
                            st.success(f"✅ Rol actualizado: {'👑 Admin' if new_is_admin else '👤 Viewer'}")
                        else:
                            st.error(f"❌ Error: {result.get('error')}")
                
                with col_op2:
                    st.subheader("🏢 Cambiar Org")
                    orgs = self.db.list_all_organizations()
                    if orgs:
                        org_names = ["Sin Org"] + [o.get("name", o.get("nombre", "N/A")) for o in orgs]
                        selected_new_org = st.selectbox("Nueva Organización", options=org_names, key="new_org_select")
                        
                        if st.button("💾 Guardar Organización", key="save_org"):
                            if selected_new_org == "Sin Org":
                                # Remover de org
                                result = self.db.delete_user_from_organization(
                                    selected_user["id"], 
                                    selected_user.get("organization_id")
                                )
                            else:
                                new_org_id = next(o["id"] for o in orgs if o.get("name", o.get("nombre")) == selected_new_org)
                                result = self.db.assign_user_to_organization(selected_user["id"], new_org_id)
                            
                            if result["success"]:
                                st.success(f"✅ Organización actualizada a '{selected_new_org}'")
                            else:
                                st.error(f"❌ Error: {result.get('error')}")
                    else:
                        st.warning("No hay organizaciones disponibles")
                
                st.divider()
                
                st.subheader("🗑️ Eliminar Usuario")
                with st.form("delete_user_form"):
                    col_del1, col_del2 = st.columns(2)
                    
                    with col_del1:
                        action = st.radio(
                            "¿Qué quieres hacer?",
                            options=[
                                ("Remover de su organización (conservar en sistema)", "remove_from_org"),
                                ("Eliminar completamente del sistema", "delete_complete")
                            ]
                        )
                    
                    with col_del2:
                        confirm = st.checkbox("✅ Confirmo la operación (no se puede deshacer)")
                    
                    st.divider()
                    
                    submit_delete = st.form_submit_button("🗑️ Ejecutar", type="secondary", use_container_width=True)
                    
                    if submit_delete:
                        if not confirm:
                            st.error("❌ Debes confirmar la operación")
                        else:
                            if action == "remove_from_org":
                                if selected_user.get("organization_id"):
                                    result = self.db.delete_user_from_organization(
                                        selected_user["id"],
                                        selected_user["organization_id"]
                                    )
                                    if result["success"]:
                                        st.success(f"✅ Usuario removido de su organización (aún existe en el sistema)")
                                    else:
                                        st.error(f"❌ Error: {result.get('error')}")
                                else:
                                    st.warning("El usuario ya no tiene organización")
                            
                            else:  # delete_complete
                                result = self.db.delete_user_completely(selected_user["id"])
                                if result["success"]:
                                    st.success(f"✅ Usuario eliminado completamente del sistema")
                                else:
                                    st.error(f"❌ Error: {result.get('error')}")
    
    def _render_operations_tab(self):
        """Tab para operaciones y auditoría"""
        st.subheader("⚙️ Operaciones del Sistema")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_orgs = len(self.db.list_all_organizations())
            st.metric("🏢 Organizaciones", num_orgs)
        
        with col2:
            num_users = len(self.db.get_all_users())
            st.metric("👥 Usuarios", num_users)
        
        with col3:
            # Contar usuarios sin org
            all_users = self.db.get_all_users()
            users_without_org = len([u for u in all_users if not u.get("organization_id")])
            st.metric("⚠️ Sin Org", users_without_org)
        
        st.divider()
        
        st.markdown("""
        <div style='background: #e0f2f1; padding: 14px; border-left: 4px solid #009688; border-radius: 8px;'>
            <p style='margin: 0; color: #333; font-size: 0.9em;'>
                <strong>🔧 Mantenimiento:</strong><br>
                Este panel permite gestionar completamente el sistema. Los cambios se aplican inmediatamente en Supabase.
            </p>
        </div>
        """, unsafe_allow_html=True)
