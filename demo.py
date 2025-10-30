"""
Demo rápido - Red de Hamming para diagnóstico de enfermedades

Este script demuestra el uso básico de la red de Hamming con el dataset Mendeley.
Para evaluaciones completas, ver los scripts en evaluaciones/
"""

import sys
sys.path.append("src")

from Nnhamming import Nnhamming
import pandas as pd
import random

print("DEMO - RED DE HAMMING")

# Cargar dataset
print("1. Cargando dataset Mendeley...")
df = pd.read_csv('datasets/mendeley_enfermedades/gold/mendeley_dataset.csv')
print(f"Dataset: {len(df)} muestras, {len(df['prognosis'].unique())} enfermedades")

# Entrenar red
print("\n2. Entrenando red...")
red = Nnhamming()
red.fit_from_df(df)
print(f"Red entrenada con {len(red.prototipos)} prototipos")

# Demostración 1: Predecir con un caso real del dataset
print("\n3. Demostraciones:")
print("\nDEMO A: Predicción con caso real del dataset")

patron_real = df.drop('prognosis', axis=1).iloc[0].values.tolist()
enfermedad_real = df['prognosis'].iloc[0]

print(f"Patrón de entrada: {patron_real[:10]}... (20 bits)")
print(f"Enfermedad real: {enfermedad_real}")

predicciones = red.predict(patron_real, k=3)

print("\nTop 3 predicciones:")
for i, (enfermedad, confianza) in enumerate(predicciones, 1):
    marca = "[CORRECTO]" if enfermedad == enfermedad_real else ""
    print(f"  {i}. {enfermedad:30s} (confianza: {confianza:.2f}) {marca}")

# Demostración 2: Predecir con vector aleatorio
print()
print("DEMO B: Predicción con vector aleatorio")

vector_aleatorio = [random.randint(0, 1) for _ in range(20)]
print(f"Patrón aleatorio: {vector_aleatorio[:10]}... (20 bits)")

predicciones_aleatorio = red.predict(vector_aleatorio, k=3)

print("\nTop 3 predicciones:")
for i, (enfermedad, confianza) in enumerate(predicciones_aleatorio, 1):
    print(f"  {i}. {enfermedad:30s} (confianza: {confianza:.2f})")

print()
print("*"*70)
print("DEMO COMPLETADO")
print("*"*70)
print("\nPara evaluaciones completas, ejecutar:")
print("  cd evaluaciones")
print("  python3 01_matriz_confusion.py")
print("  python3 02_optimizacion_k.py")
print("  (ver README.md para más opciones)")
print("*"*70)

