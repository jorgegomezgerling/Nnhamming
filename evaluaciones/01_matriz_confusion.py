"""
Evaluación 1: Matriz de Confusión
Evalúa el rendimiento de la red de Hamming usando train/test split.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append("../src")
from Nnhamming import Nnhamming



# 1. CARGA Y PREPARACIÓN DE DATOS

print("="*70)
print("EVALUACIÓN: MATRIZ DE CONFUSIÓN")
print("="*70)

df = pd.read_csv('../dataset/gold/kaggle_dataset.csv')
X = df.drop('prognosis', axis=1)
Y = df['prognosis']

n_enfermedades = df['prognosis'].nunique()
print(f"\nDataset cargado:")
print(f"  Filas: {len(df)}")
print(f"  Features: {X.shape[1]}")
print(f"  Enfermedades únicas: {n_enfermedades}")


# 2. DIVISIÓN TRAIN/TEST

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    stratify=Y,
    random_state=42
)

print(f"\nDivisión Train/Test (80/20):")
print(f"  Train: {X_train.shape[0]} muestras")
print(f"  Test:  {X_test.shape[0]} muestras")


# 3. ENTRENAMIENTO

train_df = X_train.copy()
train_df['prognosis'] = Y_train.values

print(f"\nEntrenando red de Hamming...")
red = Nnhamming()
red.fit_from_df(train_df)
print(f"Red entrenada con {len(red.prototipos)} prototipos")


# 4. PREDICCIÓN SOBRE TEST

print(f"\nPrediciendo sobre test...")

y_real = []
y_pred = []

for i in range(len(X_test)):
    
    vector = X_test.iloc[i].values.tolist()
    real = Y_test.iloc[i]
    
    prediccion = red.predict(vector, k=1)
    predicho = prediccion[0][0]
    
    y_real.append(real)
    y_pred.append(predicho)

print(f"Predicciones completadas: {len(y_pred)} casos")


# 5. CONSTRUCCIÓN DE MATRIZ Y MÉTRICAS

print(f"\n{'='*70}")
print("RESULTADOS")
print(f"{'='*70}")

matriz = confusion_matrix(y_real, y_pred)
accuracy = accuracy_score(y_real, y_pred)

correctas = sum(r == p for r, p in zip(y_real, y_pred))
incorrectas = len(y_real) - correctas

print(f" Métricas Generales:")
print(f"  Accuracy: {accuracy*100:.2f}%")
print(f"  Correctas: {correctas}/{len(y_real)}")
print(f"  Incorrectas: {incorrectas}/{len(y_real)}")
print(f"  Matriz: {matriz.shape}")

import os
os.makedirs('../resultados', exist_ok=True)

enfermedades = sorted(df['prognosis'].unique())
matriz_df = pd.DataFrame(matriz, index=enfermedades, columns=enfermedades)
matriz_df.to_csv('../resultados/matriz_confusion_completa.csv', index=False)
print(f"\nMatriz completa guardada: resultados/matriz_confusion_completa.csv")

# Guardar métricas
with open('../resultados/metricas_generales.txt', 'w') as f:
    f.write("MÉTRICAS DE EVALUACIÓN - MATRIZ DE CONFUSIÓN 01\n")

    f.write(f"Accuracy: {accuracy*100:.2f}%\n")
    f.write(f"Correctas: {correctas}/{len(y_real)}\n")
    f.write(f"Incorrectas: {incorrectas}/{len(y_real)}\n")
    f.write(f"\nEnfermedades: {n_enfermedades}\n")
    f.write(f"Train: {len(X_train)} muestras\n")
    f.write(f"Test: {len(X_test)} muestras\n")

print(f"\n{'='*70}")
print("GENERANDO VISUALIZACIÓN")
print(f"{'='*70}")

# Identificar las 20 enfermedades más comunes en test
top_20_enfermedades = Y_test.value_counts().head(20).index.tolist()

# Obtener índices de esas enfermedades en la lista completa
indices_top20 = [i for i, enf in enumerate(enfermedades) if enf in top_20_enfermedades]

# Crear sub-matriz solo con esas 20
matriz_top20 = matriz[np.ix_(indices_top20, indices_top20)]
nombres_top20 = [enfermedades[i] for i in indices_top20]

# Crear heatmap
plt.figure(figsize=(16, 14))
sns.heatmap(
    matriz_top20,
    annot=True,           # Mostrar números
    fmt='d',              # Formato entero
    cmap='YlOrRd',        # Colores
    xticklabels=nombres_top20,
    yticklabels=nombres_top20,
    cbar_kws={'label': 'Cantidad de predicciones'}
)

plt.title(f'Matriz de Confusión - Top 20 Enfermedades\nAccuracy: {accuracy*100:.2f}%', 
          fontsize=14, fontweight='bold')
plt.xlabel('Predicción', fontsize=12)
plt.ylabel('Real', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()

# Guardar
plt.savefig('../resultados/matriz_confusion_top20.png', dpi=150, bbox_inches='tight')
print(f"Heatmap guardado: resultados/matriz_confusion_top20.png")

plt.close()