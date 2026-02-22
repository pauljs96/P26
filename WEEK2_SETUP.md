# WEEK 2: Admin Panel & User Management

## 🎯 Objetivo
Implementar panel administrativo donde admins pueden:
- ✅ Crear usuarios en su organización
- ✅ Configurar formato de CSV esperado
- ✅ Ver datos cacheados
- ✅ Refrescar/limpiar cache

## 📋 Lo que se implementó

### 1. **Módulo AdminPanel** (`src/ui/admin_panel.py`)

Clase `AdminPanel` con 4 sub-tabs:

#### ✅ TAB 1: Gestionar Usuarios
- Listar usuarios actuales en la organización
- Formulario para crear nuevo usuario
  - Email, contraseña temporal, rol (admin/viewer)
  - Validación de email y contraseña
  - Llamada a `db.create_user_in_organization()`

#### ✅ TAB 2: Configurar CSV
- Formulario para definir formato esperado
  - Separador (`,`, `;`, `|`, `\t`)
  - Encoding (utf-8, latin-1, cp1252, iso-8859-1)
  - Mapeo de columnas (producto, fecha, cantidad, empresa)
- Cargar config existente si ya existe
- Guardar a `org_csv_schema` table

#### ✅ TAB 3: Ver Datos Cacheados
- Mostrar si data está cacheada o no
- Preview de primeras 5 filas de demand_monthly y stock_monthly
- Mostrar timestamp de última actualización
- Mostrar cantidad de CSVs procesados

#### ✅ TAB 4: Refrescar Data
- Botón para limpiar cache (marcar data_loaded=FALSE)
- Requiere re-upload de CSVs

### 2. **Dashboard Updates** (`src/ui/dashboard.py`)

#### ✅ Login Mejorado
- Captura `organization_id` del usuario desde BD
- Captura `is_admin` flag
- Obtiene nombre de la organización
- Demo mode fallback mantiene compatibilidad

#### ✅ Conditional Admin Tab
- Si usuario es admin: muestra tab "⚙️ Panel Admin" al inicio
- Si usuario es viewer: no muestra tab de admin
- Import dinámico de AdminPanel

#### ✅ Sidebar Mejorado
- Muestra email del usuario
- Muestra organización a la que pertenece
- Muestra rol (Admin/Viewer)
- Limpia session_state al logout

## 🧪 Tests Pasados

```
✅ admin_panel.py - Sintaxis OK
✅ dashboard.py - Sintaxis OK  
✅ AdminPanel imports OK
✅ SupabaseDB imports OK
```

## 📊 Arquitectura

```
Dashboard (Streamlit)
│
├─ Login Screen (Multi-tenant)
│  ├─ Capture: user_id, email, organization_id, is_admin, org_name
│  └─ Session State initialized
│
├─ Main Dashboard
│  ├─ Sidebar: User Info + Organization + Role
│  │
│  └─ Tabs:
│     ├─ [IF ADMIN] ⚙️ Admin Panel
│     │  ├─ 👥 Gestionar Usuarios
│     │  ├─ ⚙️ Configurar CSV
│     │  ├─ 📊 Ver Cache
│     │  └─ 🔄 Refrescar
│     │
│     ├─ 🧩 Demanda (para todos)
│     ├─ 🔮 Baselines (para todos)
│     ├─ 📈 ETS (para todos)
│     ├─ 🤖 RF (para todos)
│     └─ ... (resto de tabs)
```

## 🔐 Flujo de Seguridad

```
1. Usuario intenta login
2. Supabase autentica credenciales
3. Dashboard obtiene user record de BD:
   - organization_id
   - is_admin flag
   - organization name
4. Session state se llena:
   - st.session_state.user_id
   - st.session_state.organization_id
   - st.session_state.is_admin
   - st.session_state.organization_name
5. Si is_admin=TRUE → admin panel disponible
6. Si is_admin=FALSE → solo tabs de analysis
7. RLS policies en Supabase (org_cache) aseguran que:
   - Users solo ven cache de su org
   - Admin only puede INSERT en org_cache
```

## 🎮 Cómo Probar WEEK 2

### Prerequisito: Tener WEEK 1 completado
- ✅ SQL ejecutado en Supabase
- ✅ Tablas creadas (organizations, org_cache, org_csv_schema)

### Paso 1: Crear una Organización
En Supabase Console → SQL Editor:
```sql
INSERT INTO organizations (nombre, description) 
VALUES ('Mi Empresa Test', 'Test organization');
```
Guarda el `id` (ej: `550e8400-e29b-41d4-a716-446655440000`)

### Paso 2: Crear un Usuario Admin (Manual)
En Supabase → Authentication → Users:
- Click "Add user"
- Email: `admin@miemp.com`
- Password: `TestPassword123!`

Luego en SQL Editor:
```sql
UPDATE users 
SET organization_id = '550e8400-e29b-41d4-a716-446655440000',  -- ID de la org
    is_admin = TRUE
WHERE email = 'admin@miemp.com';
```

### Paso 3: Ejecutar Dashboard
```powershell
streamlit run main.py
```

En URL: `http://localhost:8501`

### Paso 4: Login & Test

1. **Login** como admin@miemp.com / TestPassword123!
2. **Verificar sidebar** muestra:
   - Email: admin@miemp.com
   - Org: Mi Empresa Test
   - Rol: Admin

3. **Ir a tab "⚙️ Panel Admin"** (debe aparecer primero)

4. **Test Crear Usuario:**
   - Form: email `viewer@miemp.com`, password `ViewerPw123!`, role=Viewer
   - Click "Crear Usuario"
   - Debe ver: ✅ Usuario creado

5. **Test Configurar CSV:**
   - Separador: `,`
   - Encoding: UTF-8
   - Mapeo:
     - Producto: `codigo`
     - Fecha: `fecha`
     - Cantidad: `cantidad`
     - Empresa: `empresa`
   - Click "Guardar Configuración"
   - Debe ver: ✅ Configuración guardada

6. **Test Ver Cache:**
   - Debe mostrar: "⚠️ No hay data cacheada" (porque no subimos CSVs aún)

### Paso 5: Login como Viewer
1. Logout (click botón sidebar)
2. Login como `viewer@miemp.com` / `ViewerPw123!`
3. **Verificar:**
   - NO tiene tab "⚙️ Panel Admin" (debe estar oculto)
   - El sidebar muestra "Rol: Viewer"

## 📝 Próximos Pasos (WEEK 3)

Una vez WEEK 2 esté listo:
- [ ] Integrar caching en upload de CSVs
- [ ] Crear tablas de datos en admin panel (histórico de uploads)
- [ ] Vista de usuarios por aplicación de políticas RLS
- [ ] Demo con 2 orgs teniendo datos aislados

## 🆘 Troubleshooting

### Error: "organization_id null after login"
- Verificar que el usuario tiene organization_id en la tabla users
- En SQL: `SELECT id, email, organization_id FROM users;`

### Error: "AdminPanel not found"
```powershell
python -c "from src.ui.admin_panel import AdminPanel; print('OK')"
```

### Error: "Org name not showing in sidebar"
- Verificar que `db.get_organization(org_id)` retorna data
- En SQL: `SELECT * FROM organizations;`

### Error al crear usuario "column already exists"
- Verificar que la tabla users tiene las columnas (created_by, status, etc)
- En SQL: `SELECT column_name FROM information_schema.columns WHERE table_name='users';`

## ✅ Checklist WEEK 2

- [ ] SQL de WEEK 1 ejecutado en Supabase
- [ ] AdminPanel renderiza correctamente
- [ ] Login captura organization_id e is_admin
- [ ] Sidebar muestra org y rol
- [ ] Admin ve tab "⚙️ Panel Admin"
- [ ] Non-admin NO ve tab de admin
- [ ] Crear usuario funciona
- [ ] Guardar CSV schema funciona
- [ ] Ver cache muestra preview
- [ ] Logout limpia session state

---

## 📚 Archivos Principales

| Archivo | Purpose | Status |
|---------|---------|--------|
| `src/ui/admin_panel.py` | Admin panel class | ✅ New |
| `src/ui/dashboard.py` | Main dashboard (updated) | ✅ Modified |
| `src/db/supabase.py` | DB client (updated Week 1) | ✅ Ready |
| `WEEK1_SETUP.md` | Database setup guide | ✅ Reference |

---

🔗 **Ver también:** [WEEK1_SETUP.md](WEEK1_SETUP.md) | [db_migrations/001_multi_tenant_schema.sql](db_migrations/001_multi_tenant_schema.sql)
