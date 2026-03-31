#!/usr/bin/env bash

# Script para configurar entornos virtuales con Python 3.13
# chmod +x setup_crystal.sh para convertir este fichero en ejecutable

# Esto genera una carpeta llamada .venv con su propio intérprete
# Para activarlo: source .venv/bin/activate
# Para desactivarlo: deactivate

# Una vez activo el entorno virtual, se pueden instalar librerias con pip
# python -m pip install --upgrade pip
# pip install numpy scipy matplotlib jupyterlab

python3.13 -m venv crystal_env
source crystal_env/bin/activate
python -m pip install --upgrade pip
pip install numpy scipy matplotlib jupyterlab ipykernel
python -m ipykernel install --user --name=crystal_env --display-name "Python 3.13 (crystal_env)"
echo "Entorno 'crystal_env' listo y kernel registrado para Jupyter."