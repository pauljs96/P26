# Dataset Samples & Upload Configuration

## Archivos Disponibles

### 1. **Inventario_ML_Completo_v4.csv** (1.1GB - PRODUCCIÓN)
- 1,921,610 transacciones completas
- 200 productos, período 2022-2025
- **Uso:** Análisis completo, modelos, reportes finales
- **Limitación:** Requiere límite de upload aumentado a 2GB

### 2. **Inventario_v4_SAMPLE_200K.csv** (50MB - DESARROLLO)
- 200,000 transacciones (primeras filas)
- Same estructura y columnas que v4 completo
- **Uso:** Testing del dashboard, desarrollo local
- **Ventaja:** Entra en límite default de 200MB, procesa rápido (~10 seg)

---

## Configuración Streamlit

### Archivo: `.streamlit/config.toml`

```ini
[client]
maxUploadSize = 2000  # 2GB - permite archivo v4 completo
```

**Sin esta config:** Límite de 200MB (solo entra sample)  
**Con esta config:** Límite de 2GB (entra archivo completo)

---

## Flujo de Uso Recomendado

### **Desarrollo Local:**
1. Subir `Inventario_v4_SAMPLE_200K.csv` al dashboard
2. Verificar que procesa correctamente
3. Testear vistas y análisis
4. Tiempo procesamiento: ~10 seconds

### **Testing Completo:**
1. Subir `Inventario_ML_Completo_v4.csv` al dashboard
2. Verificar desempeño con 1.9M registros
3. Validar cache en Supabase
4. Tiempo procesamiento: ~90 seconds

### **Producción:**
1. Usar siempre `Inventario_ML_Completo_v4.csv`
2. Confiar en cache para renders posteriores
3. Monitorear performance en cloud

---

## Tamaños de Archivo

| Archivo | Filas | Tamaño | Procesamiento |
|---------|-------|--------|------|
| v4 Completo | 1,921,610 | 1.1GB | ~90 seg |
| v4 Sample | 200,000 | 50MB | ~10 seg |

---

## Notas

- Sample tiene misma estructura/columnas que completo
- Proporcionalmente menor volume de datos
- Perfecto para dev/testing, no para análisis exploratorio final
- Ambos procesados por mismo pipeline v4

---

**Última actualización:** 2025
