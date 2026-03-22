# 📊 Dataset v4 Dividido por Años

## Archivos Disponibles

### **Para Streamlit Cloud (RECOMENDADO):**
```
├── Inventario_v4_2022.csv (76MB)
├── Inventario_v4_2023.csv (74MB)
├── Inventario_v4_2024.csv (74MB)
└── Inventario_v4_2025.csv (74MB)
│
Total: 1,921,610 transacciones (2022-2025)
```

**Ventajas:**
- ✅ Cada archivo < 200MB (entra en límite Streamlit Cloud)
- ✅ Puedes subir todos en una sola sesión
- ✅ Pipeline los procesa como si fuera uno solo
- ✅ Estructura idéntica al v4 completo

---

## Cómo Usarlo en el Dashboard

### **En Streamlit Cloud:**
1. **Abre tu dashboard** en streamlit.io
2. **"📤 Subir Datos"** en sidebar
3. **Selecciona los 4 archivos:**
   - Inventario_v4_2022.csv
   - Inventario_v4_2023.csv
   - Inventario_v4_2024.csv
   - Inventario_v4_2025.csv
4. Click en **upload**
5. El dashboard procesa todos juntos → resultado = dataset completo v4

### **En Local (si ejecutas streamlit localmente):**
```bash
streamlit run src/ui/dashboard.py
# Sube los 4 archivos
# El pipeline los concatena automáticamente
```

---

## Comparativa de Archivos

| Archivo | Filas | Tamaño | Año |
|---------|-------|--------|-----|
| **v4_2022.csv** | 494,952 | 76MB | 2022 |
| **v4_2023.csv** | 475,715 | 74MB | 2023 |
| **v4_2024.csv** | 476,354 | 74MB | 2024 |
| **v4_2025.csv** | 474,589 | 74MB | 2025 |
| **TOTAL** | **1,921,610** | **298MB** | 2022-2025 ✅ |
| **Completo (v4.csv)** | 1,921,610 | 1.1GB | 2022-2025 |

---

## Otros Archivos Disponibles (Para Referencia)

```
Inventario_ML_Completo_v4.csv (1.1GB)
└─ Archivo completo (no entra en Streamlit Cloud)
   └─ Usa este localmente o para análisis Backend

Inventario_v4_SAMPLE_200K.csv (50MB)
└─ Sample de 200K filas
   └─ Usa para testing rápido (10 seg de procesamiento)
```

---

## Generación de Estos Archivos

```bash
# Script que creó esta división:
python split_v4_by_year.py

# Crea automáticamente los 4 archivos anuales
```

---

## Notas Técnicas

- **Estructura idéntica:** 19 columnas, mismo orden
- **Tipo movimiento:** Venta + Producción en cada año
- **Validación:** Suma de filas = v4 completo ✅
- **Pipeline:** Ya soporta múltiples archivos (`accept_multiple_files=True`)
- **Concatenación:** Automática en DataLoader

---

## Recomendación

**AHORA EN STREAMLIT CLOUD:**
```
Sube todos los 4 archivos juntos:
✓ Inventario_v4_2022.csv
✓ Inventario_v4_2023.csv
✓ Inventario_v4_2024.csv
✓ Inventario_v4_2025.csv

Resultado: Dataset completo v4 procesado
No hay límite de 200MB ✅
```

---

**Última actualización:** 2025
