#!/usr/bin/env python
"""
Script para ejecutar Streamlit correctamente con PYTHONPATH configurado
"""
import subprocess
import sys
import os

if __name__ == "__main__":
    # Configurar PYTHONPATH para que encuentre 'src'
    env = os.environ.copy()
    project_root = os.path.dirname(os.path.abspath(__file__))
    env['PYTHONPATH'] = project_root
    
    # Ejecutar streamlit
    cmd = [
        sys.executable, 
        "-m", 
        "streamlit", 
        "run", 
        "src/ui/dashboard.py",
        "--server.runOnSave=true",
        "--client.showErrorDetails=true"
    ]
    
    print(f"[*] Ejecutando: {' '.join(cmd)}")
    print(f"[*] PYTHONPATH: {env['PYTHONPATH']}")
    print(f"[*] CWD: {project_root}")
    print()
    
    subprocess.run(cmd, cwd=project_root, env=env)
