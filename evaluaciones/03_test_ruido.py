"""
Evaluación 3: Robustez ante Ruido
Evaluamos el accuracy cuando se agrega ruido a los datos del test.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import matplotlib.pyplot as plt
import sys

sys.path.append("../src")
from Nnhamming import Nnhamming

def agregar_ruido(vector, porcentaje_ruido):
    """
    Agrega ruido a un vector binario cambiando bits al azar.

    Args:
        vector (list): vector binario original.
        porcentaje_ruido (float): porcentaje de bits a cambiar.
    
    Returns:
        list: vector con ruido
    """
    vector_ruidoso = vector.copy()
    n_bits = len(vector)
    n_cambios = int(n_bits * porcentaje_ruido)

    posiciones = np.random.choice(n_bits, n_cambios, replace=False) # Replace=False para que no repita.

    for pos in posiciones:
        vector_ruidoso[pos] = 1 - vector_ruidoso[pos] # 1 - 0 = 1; 1 - 1 = 0
    
    return vector_ruidoso

# vector_test = [1, 0, 1, 1, 0]
# print(f"Vector original: {vector_test}")

# ruidoso = agregar_ruido(vector_test, 0.2)
# print(f"Con ruido: {ruidoso}")

# Carga y preparación de datos: 

df = pd.read_csv('../dataset/gold/kaggle_dataset.csv')
X = df.drop('prognosis', axis=1)
Y = df['prognosis']


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    stratify=Y,
    random_state=42
)

# Entrenar: 

train_df = X_train.copy()
train_df['prognosis'] = Y_train.values

red = Nnhamming()
red.fit_from_df(train_df)

# Evaluación con diferentes niveles de RUIDO

print("Prueba con diferentes niveles de ruido: ")

niveles_de_ruido = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
resultados = []

for nivel in niveles_de_ruido:

    aciertos = 0

    for i in range(len(X_test)):
        vector = X_test.iloc[i].values.tolist()

        if nivel > 0:
            vector_ruidoso = agregar_ruido(vector, nivel)
        else:
            vector_ruidoso = vector
    
        prediccion = red.predict(vector_ruidoso)
        predicho = prediccion[0][0]
        real = Y_test.iloc[i]

        if predicho == real:
            aciertos += 1

    accuracy = aciertos / len(X_test)

    resultados.append({
        'ruido': nivel * 100,
        'accuracy': accuracy * 100
    })

resultados_df = pd.DataFrame(resultados)

# Visualización:

import os

os.makedirs('../resultados', exist_ok=True)

resultados_df.to_csv('../resultados/test_ruido.txt', index=False)

with open('../resultados/test_ruido.csv', 'w') as f:
    f.write("Robustez ante ruido: ")

    for _, row in resultados_df.iterrows():
        f.write(f"Ruido {row['ruido']:5.1f}% → Accuracy: {row['accuracy']:5.2f}%\n")

    degradacion = resultados_df.iloc[0]['accuracy'] - resultados_df.iloc[-1]['accuracy']
    f.write(f"\nDegradación total (0% → 30%): -{degradacion:.2f}%\n")

plt.figure(figsize=(10, 6))
plt.plot(resultados_df['ruido'], resultados_df['accuracy'], 
         marker='o', linewidth=2, markersize=8, color='crimson', label='Accuracy')
plt.fill_between(resultados_df['ruido'], resultados_df['accuracy'], 
                 alpha=0.3, color='crimson')

# sin ruido
baseline = resultados_df.iloc[0]['accuracy']
plt.axhline(y=baseline, color='green', linestyle='--', 
            linewidth=1.5, alpha=0.7, label=f'Baseline (0% ruido)')

# Anotar puntos clave
for _, row in resultados_df.iterrows():
    plt.text(row['ruido'], row['accuracy'] + 1, f"{row['accuracy']:.1f}%", 
             ha='center', fontsize=9)

plt.xlabel('Nivel de Ruido (%)', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Robustez ante Ruido: Degradación del Accuracy', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')
plt.ylim(0, 30)

plt.tight_layout()
plt.savefig('../resultados/test_ruido.png', dpi=150, bbox_inches='tight')