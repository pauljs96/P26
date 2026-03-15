# 🚀 Multi-Tenant SaaS Architecture - Sistema Tesis

## Descripción General

Refactorización completa del proyecto a arquitectura **multi-tenant** con:
- ✅ **10+ organizaciones** con datos aislados en S3
- ✅ **RBAC** (Master Admin → Org Admin → Viewer)  
- ✅ **Supabase** para metadata + autenticación
- ✅ **AWS S3** para almacenamiento de datos
- ✅ **DuckDB** para procesamiento rápido sin cargar en memoria
- ✅ **Zero downtime** en Streamlit Cloud

---

## 📋 Arquitetura

```
┌─── USUARIOS ───┐
│  Master Admin  │──► Manage Orgs, Users
│  Org Admins    │──► Upload Data, Manage Org
│  Viewers       │──► Read-Only Access
└────────────────┘

┌─── SUPABASE ───┐
│ Organizations  │
│ Users + Auth   │
│ User-Org RBAC  │
│ Audit Logs     │
└────────────────┘

┌─── AWS S3 ───┐
│ {org_id}/    │
│ ├── raw/     │ ← CSV inputs
│ ├── processed│ ← Parquet outputs
│ └── backups/ │ ← Historical
└──────────────┘

┌─── APP ────┐
│ Streamlit  │ ← Multi-tenant UI
│ DuckDB     │ ← Fast queries
│ Polars     │ ← DataFrame processing
└────────────┘
```

---

## 🔧 Setup Paso a Paso

### PASO 1: Preparar Variables de Entorno

Crear archivo `.env` en raíz del proyecto:

```bash
# Supabase (RLS para datos separados por org)
SUPABASE_URL=https://YOUR_PROJECT.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# AWS S3
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_S3_BUCKET_NAME=sistema-tesis-data
AWS_S3_REGION=us-east-1

# Logging
LOG_LEVEL=INFO
```

### PASO 2: Ejecutar Schema SQL en Supabase

1. Abrir **Supabase Dashboard** → SQL Editor
2. Copiar contenido de `SETUP_MULTITENANT_SCHEMA.sql`
3. Ejecutar SQL completo

Verifica que las tablas existan:
```sql
SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';
```

Debe listar:
- `organizations`
- `users`
- `roles`
- `user_org_assignments`
- `data_uploads`
- ... (más tablas)

### PASO 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

Verifica instalación:
```bash
python -c "import duckdb, polars; print('✅ OK')"
```

### PASO 4: Inicializar Datos Demo

```bash
python scripts/init_multitenant.py
```

Este script:
- ✅ Crea Master Admin: `admin@sistematesis.com`
- ✅ Crea 10 organizaciones (Tech, Retail, Manufacturing, etc.)
- ✅ Crea 10 usuarios por org (1 admin + 9 viewers)
- ✅ Configura RBAC para cada usuario

**Credenciales Demo** (cambiar en producción):
```
Master Admin:  admin@sistematesis.com / Admin@123456
Org Admin:     admin@{org}.local / OrgAdmin@123456
Viewers:       user{1-10}@{org}.local / User@123456
```

### PASO 5: Preparar Datos en S3

El script espera estructura:
```
s3://sistema-tesis-data/
└── {org_id}/
    └── raw/
        ├── data_2020.csv
        ├── data_2021.csv
        ├── data_2022.csv
        ├── data_2023.csv
        ├── data_2024.csv
        └── data_2025.csv
```

**Cada CSV debe tener columnas estándar:**
```
Mes, Año, Demanda, Precio, Cantidad, Stock, ...
```

**Subir manualmente a S3:**
```bash
# Instalar AWS CLI si no lo tienes
aws s3 cp data_2020.csv s3://sistema-tesis-data/{org_id}/raw/data_2020.csv

# O usar Supabase Storage / AWS Console
```

---

## 📁 Estructura de Código

### Nuevos Archivos Creados:

```
src/
├── storage/
│   └── s3_manager_v2.py          ← S3 org-aware (replazará el viejo)
├── services/
│   └── data_service.py           ← DuckDB + Polars (NUEVO)
├── db/
│   └── supabase_v2.py            ← Supabase con RBAC (NUEVO)
└── ui/
    └── dashboard.py              ← Refactor (próximamente)

scripts/
└── init_multitenant.py           ← Setup script (NUEVO)
```

### Archivos Antiguo a Reemplazar:

```
src/storage/s3_manager.py         → Usar s3_manager_v2.py
src/db/supabase.py                → Usar supabase_v2.py
src/services/cache_service.py     → ❌ ELIMINAR (Supabase ya no almacena DataFrames)
```

---

## 💻 Uso en Aplicación

### Cargar Datos desde S3 (ejemplo):

```python
from src.services.data_service import create_data_service
from src.db.supabase_v2 import get_supabase_db

# 1. Obtener org del usuario autenticado
org_id = "org-uuid-123"

# 2. Crear servicio de datos
ds = create_data_service(org_id)

# 3. Cargar múltiples años
result = ds.load_multiple_csvs(
    year_list=["2020", "2021", "2022", "2023", "2024", "2025"],
    prefix="raw"
)

if result["success"]:
    print(f"✅ {len(result['loaded_tables'])} años cargados")
    print(f"📊 Total: {result['total_rows']} filas")
    
    # 4. Ejecutar queries rápidas con DuckDB
    demand_by_month = ds.query("""
        SELECT Mes, Año, SUM(Demanda) as total_demand
        FROM data_2024
        GROUP BY Mes, Año
        ORDER BY Año, Mes
    """)
```

### RBAC en Dashboard:

```python
import streamlit as st
from src.db.supabase_v2 import get_supabase_db

# Obtener rol del usuario en org
db = get_supabase_db()
user_id = st.session_state.user_id
org_id = st.session_state.org_id

role_result = db.get_user_role_in_org(user_id, org_id)

if role_result["success"]:
    role = role_result["role"]
    
    if role == "org_admin":
        # Mostrar button para upload CSV
        st.file_uploader("Upload CSV")
    elif role == "viewer":
        # Solo lectura
        st.read_only(True)
```

---

## ✅ Checklist de Implementación

- [ ] Variables de entorno (.env) configuradas
- [ ] SQL schema ejecutado en Supabase
- [ ] Dependencias instaladas (requirements.txt)
- [ ] Script de inicialización ejecutado: `python scripts/init_multitenant.py`
- [ ] Datos CSV subidos a S3 para cada org
- [ ] Dashboard refactorizado para multi-tenant
- [ ] RBAC implementado en dashboard
- [ ] Tests en development
- [ ] Deploy a Streamlit Cloud

---

## 🧪 Testing Local

```bash
# 1. Setup ambiente
python -m venv venv
source venv/Scripts/activate  # Windows
source venv/bin/activate      # macOS/Linux

# 2. Instalar deps
pip install -r requirements.txt

# 3. Ejecutar scripts
python scripts/init_multitenant.py

# 4. Iniciar app
streamlit run main.py
```

**URLs de prueba:**
- Master Admin: http://localhost:8501 → admin@sistematesis.com
- Org Admin: http://localhost:8501 → admin@techinnovations.local
- Viewer: http://localhost:8501 → user1@techinnovations.local

---

## 🚨 Problemas Comunes

### Error: "Supabase credentials not found"
```bash
# Verificar .env
cat .env | grep SUPABASE

# O usar variables de entorno
export SUPABASE_URL=...
export SUPABASE_KEY=...
```

### Error: "AWS S3 not configured"
```bash
# Verificar credenciales AWS
aws s3 ls

# O configurar en shell
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

### Error: "No tables in DuckDB"
```python
# Verify S3 files exist
from src.storage.s3_manager_v2 import get_s3_manager
s3 = get_s3_manager()
files = s3.list_org_files("org-id-123")
print(files)
```

### Datos no cargan en Streamlit
```python
# Agregar @st.cache_resource para DataService
import streamlit as st
from src.services.data_service import create_data_service

@st.cache_resource
def get_data_service():
    return create_data_service(st.session_state.org_id)

ds = get_data_service()
```

---

## 📊 Monitoreo en Producción

### Logs de Supabase
Dashboard → Logs → Auth, Database, Storage

### Logs de S3
AWS CloudWatch → S3 metrics

### Streamlit Cloud
https://share.streamlit.io → Manage app → Logs

---

## 🔐 Seguridad

✅ **Implementado:**
- Row-Level Security (RLS) en Supabase  
- Org_id validation en S3  
- Role-based access control  
- Audit logging de acciones

⚠️ **TODO en Producción:**
- Cambiar credenciales demo
- Habilitar 2FA en Supabase  
- Encriptar datos en S3
- Implementar rate limiting
- Setup alertas en CloudWatch

---

## 📞 Soporte

Para issues o preguntas, revisar:
1. Logs en Streamlit
2. Supabase dashboard
3. AWS CloudWatch

---

**Última actualización:** 2025-01-15
**Versión:** 2.0 (Multi-Tenant SaaS)
