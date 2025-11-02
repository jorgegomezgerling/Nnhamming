"""

Evaluación: Optimización del parámetro K
Determina el valor óptimo de K (número de candidatos) para la predicción

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

df = pd.read_csv(config['path'])
X = df.drop(config['target'], axis=1)
Y = df[config['target']]

n_enfermedades = Y.nunique()

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

# Recolectamos top 10 predicciones para cada muestra de test
k_max = 10
y_real = []  # Diagnósticos reales
predicciones_por_muestra = []  # Top 10 para cada paciente

for i in range(len(X_test)):
    vector = X_test.iloc[i].values.tolist()
    real = Y_test.iloc[i]
    
    # Obtenemos top 10 diagnósticos (una sola vez por eficiencia)
    predicciones = red.predict(vector, k=k_max)
    top_k = [pred[0] for pred in predicciones]  # Solo nombres, sin confianza
    
    y_real.append(real)
    predicciones_por_muestra.append(top_k)

# Evaluamos accuracy para diferentes valores de K
k_values = [1, 2, 3, 4, 5, 7, 10]
resultados = []

for k in k_values:
    # Contamos: si el real está en el top K
    aciertos = sum(real in preds[:k] for real, preds in zip(y_real, predicciones_por_muestra))
    accuracy = aciertos / len(y_real) * 100
    
    resultados.append({
        'k': k,
        'aciertos': aciertos,
        'total': len(y_real),
        'accuracy': accuracy
    })

df_resultados = pd.DataFrame(resultados)

# Identificamos el mejor K y el baseline (K=1)
mejor_k = df_resultados.loc[df_resultados['accuracy'].idxmax()]
accuracy_k1 = df_resultados[df_resultados['k'] == 1]['accuracy'].values[0]
mejora_total = mejor_k['accuracy'] - accuracy_k1

# Crear figura con un solo gráfico más grande
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(df_resultados['k'], df_resultados['accuracy'], 
        marker='o', linewidth=3, markersize=12, color='steelblue', label='Accuracy')
ax.fill_between(df_resultados['k'], df_resultados['accuracy'], 
                 alpha=0.2, color='steelblue')

# Anotaciones con accuracy y mejora
for _, row in df_resultados.iterrows():
    if row['k'] == 1:
        # K=1: solo mostrar accuracy
        texto = f"{row['accuracy']:.1f}%"
    else:
        # K>1: mostrar accuracy y mejora
        mejora = row['accuracy'] - accuracy_k1
        texto = f"{row['accuracy']:.1f}%\n(+{mejora:.1f}%)"
    
    ax.text(row['k'], row['accuracy'] + 2, texto, 
            ha='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax.axhline(y=accuracy_k1, color='red', linestyle='--', linewidth=2, 
           alpha=0.6, label=f'Baseline (K=1): {accuracy_k1:.1f}%')

ax.scatter([mejor_k['k']], [mejor_k['accuracy']], 
           color='gold', s=300, zorder=5, marker='*', edgecolors='green', linewidths=2,
           label=f'Mejor K={int(mejor_k["k"])} (+{mejora_total:.1f}%)')

ax.set_xlabel('Valor de K (número de candidatos)', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title(f'{dataset_nombre} | Optimización del Parámetro K | Test: {len(X_test)} muestras\n' +
             f'Mejora máxima: +{mejora_total:.1f}% (K=1 → K={int(mejor_k["k"])})',
             fontweight='bold', fontsize=13)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=11, loc='lower right')
ax.set_xticks(k_values)
ax.set_ylim(min(df_resultados['accuracy']) - 5, max(df_resultados['accuracy']) + 8)

plt.tight_layout()
plt.savefig(f'../resultados/{dataset_id}/graficos/07_optimizacion_k.png', dpi=200, bbox_inches='tight')
plt.close()

with open(f'../resultados/{dataset_id}/metricas/04_optimizacion_k.txt', 'w', encoding='utf-8') as f:
    f.write(f"DATASET: {dataset_nombre}\n\n")
    
    f.write("OPTIMIZACIÓN DEL PARÁMETRO K\n\n")
    
    f.write("CONFIGURACIÓN\n")
    f.write(f"  Dataset:       {len(df)} muestras, {n_enfermedades} enfermedades\n")
    f.write(f"  Train:         {len(X_train)} muestras\n")
    f.write(f"  Test:          {len(X_test)} muestras\n")
    f.write(f"  K evaluados:   {k_values}\n\n")
    
    f.write("RESULTADOS POR K\n")
    f.write(f"  {'K':>3s}  {'Aciertos':>10s}  {'Accuracy':>10s}  {'Mejora vs k=1':>15s}\n")
    f.write(f"  {'-'*3}  {'-'*10}  {'-'*10}  {'-'*15}\n")
    
    for _, row in df_resultados.iterrows():
        k = int(row['k'])
        aciertos = int(row['aciertos'])
        total = int(row['total'])
        accuracy = row['accuracy']
        mejora = accuracy - accuracy_k1
        
        marca = " *" if k == int(mejor_k['k']) else "  "
        f.write(f"  {k:3d}  {aciertos:4d}/{total:4d}  {accuracy:9.2f}%  {mejora:+14.2f}%{marca}\n")
    
    f.write(f"\n  * Mejor K: {int(mejor_k['k'])}\n\n")
    
    f.write("RESUMEN\n")
    f.write(f"  Baseline (K=1): {accuracy_k1:.2f}%\n")
    f.write(f"  Mejor (K={int(mejor_k['k'])}): {mejor_k['accuracy']:.2f}%\n")
    f.write(f"  Mejora: +{mejor_k['accuracy'] - accuracy_k1:.2f}%\n\n")
    
df_resultados.to_csv(f'../resultados/{dataset_id}/metricas/05_optimizacion_k_detalle.csv', index=False)