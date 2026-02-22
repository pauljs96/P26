# WEEK 3: Cache Integration & Multi-Org Data Isolation

## 🎯 Objetivo
Implementar el sistema de caching para que:
- ✅ Cuando admin sube CSVs → se procesan → se cachean en Supabase
- ✅ Siguientes usuarios leen desde cache (instant load)
- ✅ Datos están aislados por organización
- ✅ Si data_loaded=TRUE → no mostrar upload form (solo admins pueden refrescar)

## 📋 Lo que se implementó

### 1. **Módulo cache_service** (`src/services/cache_service.py`)

Dos funciones principales:

#### `check_and_load_org_cache(db, org_id)`
```python
has_cache, data_dict = check_and_load_org_cache(db, org_id)
# Returns:
# - has_cache: bool (¿hay cache?)
# - data_dict: {
#     demand_monthly: DataFrame,
#     stock_monthly: DataFrame,
#     movements: DataFrame,
#     csv_files_count: int,
#     updated_at: timestamp
#   }
```

**Qué hace:**
1. Verifica si `org.data_loaded = TRUE`
2. Si sí, carga JSON de `org_cache`
3. Deserializa DataFrames (json → pandas)
4. Retorna datos listos para usar

#### `save_org_cache(db, org_id, movements, demand_monthly, stock_monthly, processed_by, csv_files_count)`
**Qué hace:**
1. Serializa DataFrames a JSON
2. Guarda en `org_cache` (INSERT o UPDATE)
3. Marca `org.data_loaded = TRUE`

### 2. **Dashboard Updates** (`src/ui/dashboard.py`)

#### Nuevo flujo de data loading:
```
Login completado
    ↓
Check: ¿hay cache para esta org?
    ↓
SI: ✅ Cargar desde cache (instant)
    ├─ Mostrar: "✅ Datos Cacheados"
    ├─ Mostrar: última actualización
    └─ Continuar al análisis
    ↓
NO: ❓ ¿Es admin?
    ├─ SI: Mostrar upload form
    │   ├─ Upload CSVs a S3
    │   ├─ Procesar con DataPipeline
    │   ├─ Guardar en org_cache
    │   └─ Continuar al análisis
    │
    └─ NO: Mostrar "⏳ Esperando..."
        └─ Return (sin datos)
```

#### Cambios específicos:
1. **Import nuevo:**
   ```python
   from src.services.cache_service import check_and_load_org_cache, save_org_cache
   ```

2. **Check de cache (después del login):**
   ```python
   has_cache, cached_data = check_and_load_org_cache(db, org_id)
   
   if has_cache and cached_data:
       # Cargar desde cache
       res_movements = cached_data.get("movements")
       res_demand = cached_data.get("demand_monthly")
       res_stock = cached_data.get("stock_monthly")
   ```

3. **Condicional por rol:**
   - Si admin: mostrar upload form
   - Si viewer: mostrar "esperando..."

4. **Guardar en cache después de procesar:**
   ```python
   cache_saved = save_org_cache(
       db=db, org_id=org_id,
       movements=res.movements,
       demand_monthly=res.demand_monthly,
       stock_monthly=res.stock_monthly,
       processed_by=user_id,
       csv_files_count=len(saved_files)
   )
   ```

5. **Referenciar data cacheada en tabs:**
   - Antes: `res.demand_monthly`, `res.movements`, `res.stock_monthly`
   - Ahora: `res_demand`, `res_movements`, `res_stock`

## 🧪 Tests Pasados

```
✅ syntax validation - dashboard.py OK
✅ cache_service imports OK
✅ All res.* references fixed
```

## 📊 Arquitectura de Datos (WEEK 3)

```
org_cache table (Supabase):
  organization_id (PK)
  demand_monthly (JSONB)  ← Serialized DataFrame
  stock_monthly (JSONB)   ← Serialized DataFrame
  movements (JSONB)       ← Serialized DataFrame
  updated_at (timestamp)
  processed_by (user_id)
  csv_files_count (int)

Flujo:
[Admin upload] → [Pipeline.run()] → [serialize] → [save org_cache] → [data_loaded=TRUE]
                                                        ↓
                                    [Next user login] → [check_and_load] → [instant load]
```

## 🎮 Cómo Probar WEEK 3

### Prerequisito: Tener WEEK 1+2 completado
✅ SQL de WEEK 1 ejecutado
✅ Org y usuarios creados en WEEK 2

### Test Setup

En Supabase SQL Editor:
```sql
-- Crear org de prueba
INSERT INTO organizations (nombre, description) 
VALUES ('Test Cache Org', 'Testing cache system');

-- Copiar el ID (ej: 550e8400-...)
-- Actualizar usuario admin para esta org
UPDATE users 
SET organization_id = '550e8400-...',
    is_admin = TRUE
WHERE email = 'admin@test.com';
```

### Paso 1: Admin sube datos
1. Login como admin@test.com
2. Ir a tab "📤 Subir Datos"
3. Subir CSVs (usa el sample_data.csv si tienes)
4. Ver en sidebar: "💾 Guardando datos en cache..."
5. Ver: "✅ Datos guardados en cache"
6. Los datos deberían aparecer en las tabs de análisis

### Paso 2: Verificar que se guardó en cache
En Supabase Console → Table Editor → org_cache:
- ✅ Debe haber un row con tu org_id
- ✅ demand_monthly y stock_monthly deben tener JSON
- ✅ updated_at debe ser reciente

### Paso 3: Login como nuevo usuario (viewer)
1. Logout
2. Request que admin cree otro usuario: viewer2@test.com (sin admin role)
3. Login como viewer2@test.com
4. **IMPORTANTE:** Sidebar debe mostrar: "✅ Datos Cacheados"
5. **NO debería haber upload form**
6. Los datos en las tabs deben ser **idénticos** a los que vio el admin

### Paso 4: Verificar aislamiento de datos (BONUS)
Si tienes 2 orgs:
1. Org A (admin1): sube datos A
2. Org B (admin2): sube datos B
3. user_a login (en org A): ve datos A
4. user_b login (en org B): ve datos B
5. **Verificar:** Datos son totalmente aislados

## 📝 Próximos Pasos (WEEK 4)

Una vez WEEK 3 funcione:
- [ ] Deploy a Streamlit Cloud
- [ ] Configurar environment variables en cloud
- [ ] Test multi-org en producción
- [ ] Performance testing

## 🔍 Debugging

### Si sidebar muestra "⚠️ Sin Datos Cacheados" pero ya subiste:
1. Verificar org_cache table en Supabase:
   ```sql
   SELECT organization_id, csv_files_count, updated_at FROM org_cache;
   ```
2. Verificar que org.data_loaded = TRUE:
   ```sql
   SELECT id, nombre, data_loaded FROM organizations;
   ```

### Si viewer ve upload form (no debería):
- Verificar is_admin de usuario:
  ```sql
  SELECT email, is_admin, organization_id FROM users;
  ```

### Si datos son diferentes entre admin y viewer:
- Puede ser que cache no se cargó bien
- Verificar que deserialize está funcionando
- Ver logs de check_and_load_org_cache()

## ✅ Checklist WEEK 3

- [ ] cache_service.py creado y funciona
- [ ] Dashboard importa cache_service correctamente
- [ ] Admin sube data → se cachea → se guarda en org_cache
- [ ] Sidebar muestra "Datos Cacheados" después de admin upload
- [ ] Viewer ve datos desde cache (sin upload form)
- [ ] Datos correctamente aislados por org
- [ ] All res.* → res_* references fixed
- [ ] Syntax validation passes

## 📚 Archivos Principales

| Archivo | Purpose | Status |
|---------|---------|--------|
| `src/services/cache_service.py` | Cache load/save | ✅ New |
| `src/ui/dashboard.py` | Main dashboard (updated) | ✅ Modified |
| `src/utils/cache_helpers.py` | JSON serialization | ✅ Ready |
| `src/db/supabase.py` | Org & cache DB ops | ✅ Ready |

---

## 🔗 Referencia

**Volver a:**
- [WEEK1_SETUP.md](WEEK1_SETUP.md) - Database schema
- [WEEK2_SETUP.md](WEEK2_SETUP.md) - Admin panel

**Próximo:**
- [WEEK4_DEPLOYMENT.md](WEEK4_DEPLOYMENT.md) - Cloud deployment (TBD)
