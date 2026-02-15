# GitHub Actions CI/CD - Sistema_Tesis

## Workflows

### 1. CI Workflow (`.github/workflows/ci.yml`)

**Qué hace:**
- Corre en: cada `push` a `main` o `develop`, y en `pull_request`
- Python versions: 3.10, 3.11, 3.12 (matrix testing)
- Steps:
  1. ✅ Installinstalla dependencias
  2. 🔍 **Linting con flake8** - valida sintaxis PEP8
  3. 🔍 **Pylint** - análisis estático básico
  4. ✅ **Syntax check** - compila archivos clave (.py compile)
  5. ✅ **Import test** - verifica que módulos principales se importan
  6. ⚠️ **File size check** - detecta archivos > 500KB (refactor needed)
  7. ✅ **requirements.txt validation** - verifica formato y completitud

**Salida esperada:**
```
✓ flake8 OK
✓ Syntax check OK
✓ Dashboard OK
✓ Supabase DB OK
✓ S3 Manager OK
✓ ML Services OK
✓ requirements.txt OK (11 packages)
✅ Pre-deployment checks passed
```

**Errores que detecta:**
- Syntax errors en Python
- Missing imports
- Large files (> 500KB)
- Invalid requirements.txt format
- Dependency conflicts

### 2. Pre-commit Workflow (`.github/workflows/pre-commit.yml`)

**Qué hace:**
- Corre en: `pull_request`
- Steps:
  1. 📏 **File size limit** - Max 1000KB por archivo
  2. 🔐 **Secret scanning** - detecta SUPABASE_KEY hardcodeadas
  3. 📁 **.env check** - verifica que .env no está committeado
  4. ✅ **requirements.txt** - verifica todos los packages necesarios

**Protege contra:**
- Accidental credential commits
- Large files (bad for Git)
- Missing dependencies

---

## Cómo activar CI/CD

### Para repositorio GitHub existente:

1. Push a GitHub:
```bash
git remote add origin https://github.com/tu-usuario/Sistema_Tesis.git
git branch -M main
git push -u origin main
```

2. **GitHub Actions se activa automáticamente**
   - Ve a: `repo → Actions → Workflows`
   - Verás "CI - Linting & Tests" ejecutándose
   - Otros cambios: badge ✅ o ❌ en el README

### Para desarrollo local (pre-commit hooks):

1. Instala pre-commit:
```bash
pip install pre-commit
pre-commit install
```

2. Crea `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: local
    hooks:
      - id: python-syntax
        name: Python Syntax
        entry: python -m py_compile
        language: system
        types: [python]
```

---

## Interpretando resultados

### ✅ Workflow passed
```
All jobs completed successfully
✅ Pre-deployment checks passed
```
→ OK para merge a `main`

### ❌ Workflow failed
Ejemplos:
```
flake8: E302 expected 2 blank lines
→ Agrega espacios en blanco entre funciones

ImportError: No module named 'boto3'
→ Instala: pip install boto3

❌ File too large: src/some_file.py (1500 KB)
→ Refactor en múltiples archivos
```

---

## Mejoras futuras

### Phase 2:
- [ ] Unit tests (pytest)
- [ ] Coverage reports (pytest-cov)
- [ ] Code quality metrics (SonarQube)
- [ ] Deploy preview con Streamlit Cloud

### Phase 3:
- [ ] Docker image build
- [ ] Security scanning (Bandit)
- [ ] Performance regression tests
- [ ] Auto-deploy a Cloud Run

---

## Referencias

- [GitHub Actions Docs](https://docs.github.com/en/actions)
- [Flake8 Rules](https://flake8.pycqa.org/)
- [Pre-commit Framework](https://pre-commit.com/)
