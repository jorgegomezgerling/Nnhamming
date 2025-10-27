"""

Evaluación: Test de Ruido
Evalúa la robustez de la red agregando ruido a los patrones de test

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
import os

sys.path.append("../src")
from Nnhamming import Nnhamming
from config import get_dataset_config

config = get_dataset_config()
dataset_id = config['id']
dataset_nombre = config['nombre']

os.makedirs(f'../resultados/{dataset_id}/graficos', exist_ok=True)
os.makedirs(f'../resultados/{dataset_id}/metricas', exist_ok=True)

def agregar_ruido(vector, porcentaje_ruido):
    """
    Invierte bits aleatorios en un vector binario
    
    Args:
        vector: Lista de valores binarios (0 o 1)
        porcentaje_ruido: Proporción de bits a invertir (0.0 a 1.0)
    
    Returns:
        Lista con bits invertidos
    """
    vector_ruidoso = vector.copy()
    n_bits = len(vector)
    n_cambios = int(n_bits * porcentaje_ruido)
    
    if n_cambios > 0:
        posiciones = np.random.choice(n_bits, n_cambios, replace=False)
        for pos in posiciones:
            vector_ruidoso[pos] = 1 - vector_ruidoso[pos]
    
    return vector_ruidoso

df = pd.read_csv(config['path'])
X = df.drop(config['target'], axis=1)
Y = df[config['target']]

n_features = X.shape[1]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    stratify=Y,
    random_state=42
)

train_df = X_train.copy()
train_df[config['target']] = Y_train.values

red = Nnhamming()
red.fit_from_df(train_df)

niveles_ruido = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
resultados = []

np.random.seed(42)

for nivel in niveles_ruido:
    aciertos = 0
    
    for i in range(len(X_test)):
        vector = X_test.iloc[i].values.tolist()
        
        if nivel > 0:
            vector_ruidoso = agregar_ruido(vector, nivel)
        else:
            vector_ruidoso = vector
        
        prediccion = red.predict(vector_ruidoso, k=1)
        predicho = prediccion[0][0]
        real = Y_test.iloc[i]
        
        if predicho == real:
            aciertos += 1
    
    accuracy = aciertos / len(X_test) * 100
    bits_afectados = int(n_features * nivel)
    
    resultados.append({
        'nivel_ruido': nivel * 100,
        'bits_afectados': bits_afectados,
        'aciertos': aciertos,
        'total': len(X_test),
        'accuracy': accuracy
    })

df_resultados = pd.DataFrame(resultados)

accuracy_base = df_resultados.iloc[0]['accuracy']
accuracy_30 = df_resultados.iloc[-1]['accuracy']
degradacion_total = accuracy_base - accuracy_30

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(df_resultados['nivel_ruido'], df_resultados['accuracy'], 
         marker='o', linewidth=2.5, markersize=10, color='crimson', label='Accuracy')
ax1.fill_between(df_resultados['nivel_ruido'], df_resultados['accuracy'], 
                  alpha=0.2, color='crimson')

ax1.axhline(y=accuracy_base, color='green', linestyle='--', linewidth=1.5, 
            alpha=0.6, label=f'Baseline (0% ruido): {accuracy_base:.1f}%')

for _, row in df_resultados.iterrows():
    ax1.text(row['nivel_ruido'], row['accuracy'] + 1.2, f"{row['accuracy']:.1f}%", 
             ha='center', fontsize=9, fontweight='bold')

ax1.set_xlabel('Nivel de Ruido (%)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax1.set_title('Degradación del Accuracy con Ruido', fontweight='bold', fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim(0, max(df_resultados['accuracy']) + 5)

degradaciones = []
for i in range(1, len(df_resultados)):
    deg = df_resultados.iloc[0]['accuracy'] - df_resultados.iloc[i]['accuracy']
    degradaciones.append(deg)

ax2.bar(df_resultados['nivel_ruido'][1:], degradaciones, 
        color='crimson', alpha=0.7, edgecolor='black')

for nivel, deg in zip(df_resultados['nivel_ruido'][1:], degradaciones):
    ax2.text(nivel, deg + 0.3, f'-{deg:.1f}%', 
             ha='center', fontsize=9, fontweight='bold')

ax2.set_xlabel('Nivel de Ruido (%)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Pérdida de Accuracy (%)', fontsize=11, fontweight='bold')
ax2.set_title('Pérdida Acumulada vs Baseline', fontweight='bold', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')

plt.suptitle(f'{dataset_nombre} | Test de Ruido | {n_features} bits | Test: {len(X_test)} muestras', 
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'../resultados/{dataset_id}/graficos/08_test_ruido.png', dpi=200, bbox_inches='tight')
plt.close()

with open(f'../resultados/{dataset_id}/metricas/06_test_ruido.txt', 'w', encoding='utf-8') as f:
    f.write(f"DATASET: {dataset_nombre}\n\n")
    
    f.write("TEST DE RUIDO: ROBUSTEZ DE LA RED\n\n")
    
    f.write("CONFIGURACIÓN\n")
    f.write(f"  Features:          {n_features} bits\n")
    f.write(f"  Train:             {len(X_train)} muestras\n")
    f.write(f"  Test:              {len(X_test)} muestras\n")
    f.write(f"  Niveles de ruido:  0%, 5%, 10%, 15%, 20%, 25%, 30%\n\n")
    
    f.write("RESULTADOS\n")
    f.write(f"  {'Ruido':>7s}  {'Bits':>6s}  {'Aciertos':>11s}  {'Accuracy':>10s}  {'Pérdida':>10s}\n")
    f.write(f"  {'-'*7}  {'-'*6}  {'-'*11}  {'-'*10}  {'-'*10}\n")
    
    for _, row in df_resultados.iterrows():
        ruido = row['nivel_ruido']
        bits = int(row['bits_afectados'])
        aciertos = int(row['aciertos'])
        total = int(row['total'])
        accuracy = row['accuracy']
        perdida = accuracy_base - accuracy
        
        f.write(f"  {ruido:5.0f}%  {bits:6d}  {aciertos:4d}/{total:4d}  {accuracy:9.2f}%  "
                f"{'-' if ruido == 0 else f'-{perdida:.2f}%':>10s}\n")
    
    f.write(f"\n  Degradación total (0% → 30%): -{degradacion_total:.2f}%\n\n")
    
    f.write("CONCLUSIÓN\n")
    f.write(f"  Accuracy base: {accuracy_base:.2f}%\n")
    f.write(f"  Accuracy con 30% ruido: {accuracy_30:.2f}%\n")
    f.write(f"  Pérdida total: {degradacion_total:.2f}%\n\n")
    
    if degradacion_total < 10:
        f.write(f"  La red muestra robustez razonable ante perturbaciones.\n")
    else:
        f.write(f"  La red es sensible al ruido (esperado con distancia de Hamming).\n")

df_resultados.to_csv(f'../resultados/{dataset_id}/metricas/07_test_ruido_detalle.csv', index=False)