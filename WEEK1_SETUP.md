# WEEK 1: Multi-Tenant Database Schema Setup

## 🎯 Objetivo
Configurar la base de datos Supabase para soporte multi-tenant con caching de datos procesados.

## 📋 Lo que haremos
1. ✅ Crear tablas nuevas: `organizations`, `org_cache`, `org_csv_schema`
2. ✅ Alterar tablas existentes: `users`, `uploads`
3. ✅ Crear índices y políticas RLS
4. ✅ Actualizar código Python (supabase.py con métodos nuevos)
5. ✅ Crear helpers para serialización de DataFrames

## 🚀 Pasos a Seguir

### PASO 1: Ejecutar SQL en Supabase Console

1. Ve a **Supabase Dashboard** → Tu proyecto
2. Abre **SQL Editor** (o ve a **SQL**)
3. Copia TODO el contenido del archivo:
   ```
   db_migrations/001_multi_tenant_schema.sql
   ```
4. Pega en el editor SQL
5. Haz click en **Run** (botón azul abajo a la derecha)

**Qué hace este SQL:**
- ✅ Crea tabla `organizations` (nombres únicos, admin_user_id)
- ✅ Crea tabla `org_cache` (demanda + stock como JSONB)
- ✅ Crea tabla `org_csv_schema` (mapeo de columnas por org)
- ✅ Agrega columnas a `users`: organization_id, is_admin, created_by, status
- ✅ Agrega columna a `uploads`: organization_id
- ✅ Crea índices para queries rápidas
- ✅ Habilita Row-Level Security (RLS) en org_cache

**Salida esperada:**
```
✓ organizations table created
✓ org_cache table created  
✓ org_csv_schema table created
✓ users table altered
✓ uploads table altered
✓ Indexes created
✓ RLS policies enabled
```

---

### PASO 2: Verificar las Tablas Creadas

En Supabase Console, ve a **Table Editor**:
- [ ] ¿Ves tabla `organizations`?
- [ ] ¿Ves tabla `org_cache`?
- [ ] ¿Ves tabla `org_csv_schema`?
- [ ] ¿Tiene `users` las columnas nuevas (organization_id, is_admin, etc)?
- [ ] ¿Tiene `uploads` la columna organization_id?

Si todas = ✅, continúa.

---

### PASO 3: Crear Organizaciones de Prueba (Opcional)

En SQL Editor, ejecuta:
```sql
-- Org 1: Tech Company
INSERT INTO organizations (nombre, description) 
VALUES ('Tech Company S.A.', 'Primera organización de prueba');

-- Org 2: Retail Company
INSERT INTO organizations (nombre, description) 
VALUES ('Retail Company Ltd.', 'Segunda organización de prueba');
```

Luego en **Table Editor**, ve a `organizations` y verifica que existan.

---

### PASO 4: Crear Usuarios Admin para las Orgs de Prueba

**Opción A: Vía Dashboard (Cuando esté listo)**
Habrá un nuevo botón "Admin Panel" que permitirá crear usuarios.

**Opción B: Manualmente en Supabase Auth**

1. Ve a **Authentication** → **Users**
2. Click **Add user**
3. Email: `admin1@techcompany.com`, Password: `TempPass123!`
4. Luego en **SQL Editor**, ejecuta:
   ```sql
   UPDATE users 
   SET organization_id = (SELECT id FROM organizations WHERE nombre = 'Tech Company S.A.'),
       is_admin = TRUE,
       email = 'admin1@techcompany.com'
   WHERE email = 'admin1@techcompany.com';
   ```

---

### PASO 5: Validar el Código Python

En tu terminal, ejecuta:
```powershell
# Desde la carpeta del proyecto
python -c "from src.db.supabase import SupabaseDB; db = SupabaseDB(); print('✅ SupabaseDB imports OK')"
```

**Salida esperada:**
```
✅ SupabaseDB imports OK
```

---

### PASO 6: Probar Métodos Nuevos (Optional)

Crea un archivo `test_week1.py`:
```python
from src.db.supabase import SupabaseDB
from src.utils.cache_helpers import dataframe_to_json, json_to_dataframe
import pandas as pd

db = SupabaseDB()

# Test 1: Obtener org
org = db.get_organization("<ORG_ID_AQUI>")
print(f"✅ Org: {org['nombre']}")

# Test 2: Verificar si data está loaded
is_loaded = db.is_data_loaded("<ORG_ID_AQUI>")
print(f"✅ Data loaded: {is_loaded}")

# Test 3: Serializar DataFrame
test_df = pd.DataFrame({
    "producto": ["A", "B", "C"],
    "cantidad": [10, 20, 30]
})
json_str = dataframe_to_json(test_df)
recovered_df = json_to_dataframe(json_str)
print(f"✅ Serialization OK: {len(recovered_df)} rows")
```

Ejecuta:
```powershell
python test_week1.py
```

---

## 📊 Resumen de Cambios

### Nuevas Tablas
| Tabla | Purpose | Campos principales |
|-------|---------|-------------------|
| `organizations` | Contiene orgs | nombre, admin_user_id, data_loaded |
| `org_cache` | Results cache | demand_monthly, stock_monthly, updated_at |
| `org_csv_schema` | CSV config | separator, encoding, column_mapping |

### Columnas Nuevas en Tablas Existentes
| Tabla | Columnas Nuevas | Tipo |
|-------|-----------------|------|
| `users` | organization_id | UUID FK |
| `users` | is_admin | BOOLEAN |
| `users` | created_by | UUID FK |
| `users` | status | VARCHAR |
| `uploads` | organization_id | UUID FK |

### Métodos Nuevos en `supabase.py`
```python
# Organizations
db.create_organization(nombre, admin_user_id, description)
db.get_organization(org_id)
db.get_user_organization(user_id)
db.create_user_in_organization(org_id, email, password, is_admin)
db.get_organization_users(org_id)

# Caching
db.save_org_data(org_id, demand_json, stock_json, movements_json, processed_by)
db.load_org_data(org_id)  # Retorna cached data
db.is_data_loaded(org_id)  # Boolean

# CSV Schema
db.save_csv_schema(org_id, separator, encoding, column_mapping)
db.get_csv_schema(org_id)
```

### Nuevos Helpers en `cache_helpers.py`
```python
dataframe_to_json(df) -> str
json_to_dataframe(json_str) -> DataFrame
validate_dataframe(df, required_columns) -> bool
serialize_pipeline_result(movements, demand, stock) -> tuple
deserialize_pipeline_result(movements_json, demand_json, stock_json) -> tuple
```

---

## ✅ Checklist de Verificación

- [ ] SQL ejecutado sin errores en Supabase
- [ ] Tablas nuevas visibles en Table Editor
- [ ] Columnas nuevas en `users` y `uploads`
- [ ] Índices creados (visible en Supabase Indexes)
- [ ] `python -c "from src.db.supabase import SupabaseDB"` sin errores
- [ ] `cache_helpers.py` existe y importa correctamente
- [ ] (Opcional) Orgs de prueba creadas
- [ ] (Opcional) test_week1.py ejecuta sin errores

---

## 🆘 Troubleshooting

### Error: "column already exists"
- Significa el SQL ya se ejecutó antes. Es safe ignorar.

### Error: "table already exists"
- Mismo caso. Si quieres un reset:
  ```sql
  DROP TABLE IF EXISTS org_cache;
  DROP TABLE IF EXISTS org_csv_schema;
  DROP TABLE IF EXISTS organizations;
  ```
  Luego ejecuta el SQL de nuevo.

### Error: "organization_id violates foreign key constraint"
- Los `users` necesitan tener `organization_id` válido.
- Primero crea una org, luego crea users con esa org_id.

### Importa error: "No module named 'supabase'"
```powershell
pip install -q supabase
```

---

## 📝 Próximos Pasos (WEEK 2)

Una vez WEEK 1 esté completo:
- Agregar Admin Panel al dashboard
- Implementar "Crear Usuario" form
- Implementar validación de CSV schema

🔗 Ver: [WEEK2_PLAN.md](../WEEK2_PLAN.md)
