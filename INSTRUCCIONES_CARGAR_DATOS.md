# INSTRUCCIONES PARA CARGAR DATOS EN STREAMLIT CLOUD

## 📊 Archivos disponibles

Se han creado 4 archivos CSV con datos de **20 productos principales** (cajas de pase, tableros eléctricos):

- `Inventario_v4_20PRODUCTOS_2022.csv` (8.7 MB) - 58,370 registros
- `Inventario_v4_20PRODUCTOS_2023.csv` (8.6 MB) - 57,202 registros
- `Inventario_v4_20PRODUCTOS_2024.csv` (8.7 MB) - 57,533 registros
- `Inventario_v4_20PRODUCTOS_2025.csv` (8.6 MB) - 57,141 registros

**Total: 230,246 registros, 4 años de datos** ✅

## 🔧 Pasos para cargar en Streamlit Cloud

### 1. Descargar el archivo limpio
```
Descarga los 4 archivos CSV desde este directorio
```

### 2. Esperar a que la aplicación cargue (después del login)
Después de ingresar con tus credenciales, la app mostrará:
- Opción de cargar archivos
- Panel lateral con opciones de admin

### 3. Seleccionar los 4 archivos
```
Haz clic en "Sube CSV (2021-2025)"
Selecciona los 4 archivos simultáneamente (CTRL+A después de hacer clic en el primero)
Streamlit cargará y procesará automáticamente
```

### 4. Verificar datos cargados
Una vez cargados, deberías ver:
- Banner "✅ Dataset v4 detectado"
- Selector de productos con 20 opciones
- Gráficos y análisis de demanda, stock, etc.

## 🏢 Usuarios disponibles para probar

```
Usuarios configurados en Supabase (con credenciales de producción):
  • luna@gmail.com       (Empresa1 - org_admin)
  • paulmaster@gmail.com (Sin Org - viewer)
  • brisa10@gmail.com    (Empresa2 - org_admin)
  • chachi@gmail.com     (Empresa3 - org_admin)
  • zu@gmail.com         (Empresa4 - org_admin) ← RECOMENDADO
```

## 🛠️ Si algo falla

### Error: "Oh no" después de ingresar
**Causa**: Datos no cargados o tabla de cache vacía

**Solución**:
1. Reboot la app: ⋮ → "Reboot app"
2. Ingresa nuevamente
3. Inmediatamente carga los 4 archivos CSV
4. Espera a que aparezca "✅ Procesados X archivos"

### Error: "KeyError" o columnas no encontradas
**Causa**: Normalmente resuelto con los últimos commits

**Solución**:
- La app debería estar en versión actualizada
- Si persiste, avisa

### Datos muy lentos
**Causa**: 230K registros se procesan con pipeline

**Esperado**: Primera carga tarda 2-3 minutos, luego cacheado

## 📋 Productos incluidos

```
1. MATRIZ DE EMBUTIDO CAJA OCTOGONAL
2. MATRIZ PUNZONADORA Y DOBLADORA
3-20. Varios tipos de CAJAS DE PASE en diferentes medidas
   - 100X100X69MM 3/4
   - 150X150X100MM 1/2
   - 200X200X75MM 1 1/4
   - 200X200X70MM 3/4
   - 350X350X150MM
   - 500X300X300MM CIEGA
   - 500X500X200MM CIEGA
   - 650X350X150MM
   - 650X350X150MM TIPO C
   - 650X500X150MM C/H
   - 700X700X250MM CIEGA
   - 800X500X150MM CIEGA
   - 1100X800X200MM
   - ELECTRICA 3990X245X200MM
```

## ✅ Checklist antes de cargar

- [ ] Archivos descargados en tu computadora
- [ ] Ingresaste con zu@gmail.com (o tu usuario)
- [ ] Dashboard mostró la pantalla de carga
- [ ] Tienes los 4 archivos seleccionados
- [ ] Internet estable durante la carga

## 📞 Próximos pasos

Una vez los datos estén cargados:
1. Prueba seleccionar diferentes productos
2. Explora las 7 tabs (Demanda, Stock, Comparación, etc.)
3. Verifica que los gráficos carguen
4. Si todo funciona: ¡Done! 🎉

---

**Última actualización**: Marzo 22, 2026
**Estado**: Listo para cargar
**Archivos listos**: ✅ Todos
