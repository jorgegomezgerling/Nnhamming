"""
Evaluación 2: Optimizar el K.
Determinar el valor óptimo de K para maximizar la accuracy. (Es la idea, al menos).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
sys.path.append("../src")
from Nnhamming import Nnhamming

# 1. Carga y preparación de datos

df = pd.read_csv('../dataset/gold/kaggle_dataset.csv')
X = df.drop('prognosis', axis=1)
Y = df['prognosis']


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    stratify=Y,
    random_state=42
)

# 2. ENTRENAMIENTO

n_enfermedades = df['prognosis'].nunique()
print(f"\nDataset: {len(df)} muestras, {n_enfermedades} enfermedades")
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

train_df = X_train.copy()
train_df['prognosis'] = Y_train.values

print(f"\nEntrenando red...")
red = Nnhamming()
red.fit_from_df(train_df)
print(f"Red entrenada con {len(red.prototipos)} prototipos")


# 3. PREDICCIÓN CON K=5

print(f"\nPrediciendo con k=5...")

y_real = []
y_pred_top1 = []  # Solo el mejor
y_pred_top3 = []  # Los 3 mejores
y_pred_top5 = []  # Los 5 mejores

for i in range(len(X_test)):
    if i % 100 == 0:
        print(f"  Progreso: {i}/{len(X_test)}")
    
    vector = X_test.iloc[i].values.tolist()
    real = Y_test.iloc[i]
    
    # Predecir con k=5
    predicciones = red.predict(vector, k=5)
    
    # Extraer solo los nombres (sin confianza)
    top5_nombres = [pred[0] for pred in predicciones]
    
    # Guardar
    y_real.append(real)
    y_pred_top1.append(top5_nombres[0])                    # Top 1
    y_pred_top3.append(top5_nombres[:3])                   # Top 3
    y_pred_top5.append(top5_nombres[:5])                   # Top 5

print(f"Predicciones completadas")

# 4. CALCULAR ACCURACIES

print(f"\n{'='*70}")
print("RESULTADOS POR K")
print(f"{'='*70}")

accuracy_k1 = sum(real == pred for real, pred in zip(y_real, y_pred_top1)) / len(y_real)

accuracy_k3 = sum(real in pred for real, pred in zip(y_real, y_pred_top3)) / len(y_real)

accuracy_k5 = sum(real in pred for real, pred in zip(y_real, y_pred_top5)) / len(y_real)

print(f"Resultados:")
print(f"Accuracy1 (k=1): {accuracy_k1*100:.2f}%")
print(f"Accuracy3 (k=3): {accuracy_k3*100:.2f}%")
print(f"Accuracy5 (k=5): {accuracy_k5*100:.2f}%")

print(f"Mejora:")
print(f"k=3 vs k=1: +{(accuracy_k3 - accuracy_k1)*100:.2f}%")
print(f"k=5 vs k=1: +{(accuracy_k5 - accuracy_k1)*100:.2f}%")

import os
import matplotlib.pyplot as plt

os.makedirs('../resultados', exist_ok=True)

# Guardar métricas
with open('../resultados/optimizacion_k.txt', 'w') as f:
    f.write("OPTIMIZACIÓN DE K\n")

    f.write(f"Accuracy 1 (k=1): {accuracy_k1*100:.2f}%\n")
    f.write(f"Accuracy 3 (k=3): {accuracy_k3*100:.2f}%\n")
    f.write(f"Accuracy 5 (k=5): {accuracy_k5*100:.2f}%\n")
    f.write(f"\nMejora k=3 vs k=1: +{(accuracy_k3 - accuracy_k1)*100:.2f}%\n")
    f.write(f"Mejora k=5 vs k=1: +{(accuracy_k5 - accuracy_k1)*100:.2f}%\n")

print(f"Resultados guardados: resultados/optimizacion_k.txt")

# Visualización
k_values = [1, 3, 5]
accuracies = [accuracy_k1*100, accuracy_k3*100, accuracy_k5*100]

plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linewidth=2, markersize=10, color='steelblue')
plt.fill_between(k_values, accuracies, alpha=0.3, color='steelblue')

for k, acc in zip(k_values, accuracies):
    plt.text(k, acc + 2, f'{acc:.2f}%', ha='center', fontsize=11, fontweight='bold')

plt.xlabel('Valor de K', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Optimización de K: Accuracy vs Número de Candidatos', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.ylim(0, 70)
plt.xticks(k_values)

plt.tight_layout()
plt.savefig('../resultados/optimizacion_k.png', dpi=150, bbox_inches='tight')
print(f"Gráfico guardado: resultados/optimizacion_k.png")

plt.close()

print(f"\n{'='*70}")
print("Evaluación completada")
print(f"{'='*70}\n")