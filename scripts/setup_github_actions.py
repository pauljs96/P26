#!/usr/bin/env python3
"""
GitHub Actions CI/CD Setup Guide
Sistema Tesis Multi-Tenant
"""

import os
from pathlib import Path

print("""
╔═══════════════════════════════════════════════════════════╗
║   GITHUB ACTIONS CI/CD SETUP                             ║
║   Sistema Tesis Multi-Tenant                             ║
╚═══════════════════════════════════════════════════════════╝

Este script crea los archivos necesarios para:
✅ Ejecutar tests automáticos en cada push
✅ Crear releases automáticas
✅ Deployar a Streamlit Cloud

Presiona ENTER para continuar...
""")
input()

github_dir = Path(".github/workflows")
github_dir.mkdir(parents=True, exist_ok=True)

# Create test workflow
test_workflow = github_dir / "tests.yml"
test_workflow.write_text("""name: Run Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run linting
      run: |
        pip install pylint
        pylint src/ --disable=all --enable=E,F
      continue-on-error: true
    
    - name: Run unit tests
      run: pytest tests/test_services.py -v
    
    - name: Run dashboard tests
      run: pytest tests/test_dashboard.py -v
    
    - name: Generate coverage report
      run: pytest tests/ --cov=src --cov-report=xml
      continue-on-error: true

""")

# Create deploy workflow
deploy_workflow = github_dir / "deploy.yml"
deploy_workflow.write_text("""name: Deploy to Streamlit Cloud

on:
  push:
    branches: [ main ]
  workflow_run:
    workflows: ["Run Tests"]
    types:
      - completed

jobs:
  deploy:
    runs-on: ubuntu-latest
    if: github.event.workflow_run.conclusion == 'success' || github.event_name == 'push'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Check Python syntax
      run: python -m py_compile main.py src/**/*.py
    
    - name: Validate requirements
      run: |
        pip install -r requirements.txt --dry-run
    
    - name: Verify Streamlit config
      run: |
        test -f .streamlit/config.toml
        echo "✅ Streamlit config valid"
    
    - name: Verify secrets template
      run: |
        test -f .streamlit/secrets.toml.example
        echo "✅ Secrets template exists"
    
    - name: Verify gitignore
      run: |
        grep ".streamlit/secrets.toml" .gitignore
        echo "✅ Secrets excluded from git"
    
    - name: Streamlit deployment notice
      run: |
        echo "✅ Code is ready for Streamlit Cloud deployment"
        echo "Go to: https://share.streamlit.io to deploy manually"

""")

# Create release workflow
release_workflow = github_dir / "release.yml"
release_workflow.write_text("""name: Create Release

on:
  push:
    tags:
      - 'v*'

jobs:
  create-release:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Create Release Notes
      run: |
        echo "## Release ${{ github.ref }}" > RELEASE_NOTES.md
        echo "" >> RELEASE_NOTES.md
        echo "**Commit:** \`${{ github.sha }}\`" >> RELEASE_NOTES.md
        echo "" >> RELEASE_NOTES.md
        git log --oneline HEAD~10..HEAD >> RELEASE_NOTES.md
    
    - name: Create Release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: ${{ github.ref }}
        release_name: Release ${{ github.ref }}
        body_path: RELEASE_NOTES.md
        draft: false
        prerelease: false

""")

print(f"""
✅ GitHub Actions workflows creados:

  {test_workflow.relative_to('.')}
  {deploy_workflow.relative_to('.')}
  {release_workflow.relative_to('.')}

📋 Workflows:

1. tests.yml
   ✅ Ejecuta en cada push
   ✅ Corre pytest
   ✅ Genera coverage report
   ✅ Reporta resultados

2. deploy.yml
   ✅ Ejecuta después de tests exitosos
   ✅ Valida Python syntax
   ✅ Verifica dependencias
   ✅ Prepara para Streamlit Cloud

3. release.yml
   ✅ Ejecuta al crear un tag (v1.0, v1.1, etc)
   ✅ Crea release notes automáticas
   ✅ Publica en GitHub Releases

🚀 Cómo usar:

1. Commit los cambios:
   git add .github/
   git commit -m "Add GitHub Actions CI/CD"
   git push origin main

2. Ver workflows ejecutándose:
   GitHub → Actions tab

3. Crear release:
   git tag v1.0.0
   git push origin v1.0.0

4. Ver results:
   GitHub → Releases

📊 Status badges (agregar a README.md):

[![Run Tests](https://github.com/pauljs96/P26/actions/workflows/tests.yml/badge.svg)](https://github.com/pauljs96/P26/actions/workflows/tests.yml)
[![Deploy to Streamlit](https://github.com/pauljs96/P26/actions/workflows/deploy.yml/badge.svg)](https://github.com/pauljs96/P26/actions/workflows/deploy.yml)

""")
