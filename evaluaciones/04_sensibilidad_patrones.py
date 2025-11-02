"""

Evaluación: Sensibilidad a Cantidad de Patrones
Evalúa cómo el tamaño del conjunto de entrenamiento afecta el accuracy

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

X_train_full, X_test, Y_train_full, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    stratify=Y,
    random_state=42
)

porcentajes = [0.25, 0.50, 0.75, 1.0]
resultados = []

for idx, porcentaje in enumerate(porcentajes):
    n_muestras_train = int(len(X_train_full) * porcentaje)
    
    if porcentaje < 1.0:
        X_train_sub, _, Y_train_sub, _ = train_test_split(
            X_train_full, Y_train_full,
            train_size=porcentaje,
            stratify=Y_train_full,
            random_state=42
        )
    else:
        X_train_sub = X_train_full
        Y_train_sub = Y_train_full
    
    train_df = X_train_sub.copy()
    train_df[config['target']] = Y_train_sub.values
    
    red = Nnhamming()
    red.fit_from_df(train_df)
    
    aciertos = 0
    for i in range(len(X_test)):
        vector = X_test.iloc[i].values.tolist()
        real = Y_test.iloc[i]
        
        prediccion = red.predict(vector, k=1)
        predicho = prediccion[0][0]
        
        if predicho == real:
            aciertos += 1
    
    accuracy = aciertos / len(X_test) * 100
    
    resultados.append({
        'porcentaje': porcentaje * 100,
        'n_muestras': n_muestras_train,
        'n_prototipos': len(red.prototipos),
        'aciertos': aciertos,
        'total': len(X_test),
        'accuracy': accuracy
    })

df_resultados = pd.DataFrame(resultados)

mejor_idx = df_resultados['accuracy'].idxmax()
mejor = df_resultados.iloc[mejor_idx]

# Crear figura con un solo gráfico más grande
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(df_resultados['porcentaje'], df_resultados['accuracy'], 
        marker='o', linewidth=3, markersize=12, color='darkblue', label='Accuracy')
ax.fill_between(df_resultados['porcentaje'], df_resultados['accuracy'], 
                 alpha=0.2, color='darkblue')

ax.scatter([mejor['porcentaje']], [mejor['accuracy']], 
           s=250, color='gold', edgecolors='red', linewidths=3, zorder=5,
           label=f'Mejor: {mejor["porcentaje"]:.0f}% ({mejor["accuracy"]:.2f}%)')

# Anotaciones mejoradas con accuracy y número de prototipos
for _, row in df_resultados.iterrows():
    ax.text(row['porcentaje'], row['accuracy'] + 1.2, 
            f"{row['accuracy']:.2f}%\n({int(row['n_prototipos'])} prototipos)", 
            ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

mejora = df_resultados.iloc[-1]['accuracy'] - df_resultados.iloc[0]['accuracy']

ax.set_xlabel('Porcentaje de Train (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title(f'{dataset_nombre} | Sensibilidad a Cantidad de Patrones | Test: {len(X_test)} muestras\n' +
             f'Mejora total: {"+" if mejora >= 0 else ""}{mejora:.2f}% (25% → 100%)', 
             fontweight='bold', fontsize=13)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xticks(df_resultados['porcentaje'])
ax.set_ylim(min(df_resultados['accuracy']) - 5, max(df_resultados['accuracy']) + 5)

plt.tight_layout()
plt.savefig(f'../resultados/{dataset_id}/graficos/09_sensibilidad_patrones.png', dpi=200, bbox_inches='tight')
plt.close()

with open(f'../resultados/{dataset_id}/metricas/08_sensibilidad_patrones.txt', 'w', encoding='utf-8') as f:
    f.write(f"DATASET: {dataset_nombre}\n\n")
    
    f.write("SENSIBILIDAD A CANTIDAD DE PATRONES\n\n")
    
    f.write("CONFIGURACIÓN\n")
    f.write(f"  Train completo:    {len(X_train_full)}\n")
    f.write(f"  Test (fijo):       {len(X_test)}\n")
    f.write(f"  Porcentajes:       25%, 50%, 75%, 100%\n\n")
    
    f.write("RESULTADOS\n")
    f.write(f"  {'Train':>6s}  {'Muestras':>9s}  {'Prototipos':>11s}  {'Accuracy':>10s}\n")
    f.write(f"  {'-'*6}  {'-'*9}  {'-'*11}  {'-'*10}\n")
    
    for _, row in df_resultados.iterrows():
        porc = row['porcentaje']
        muestras = int(row['n_muestras'])
        prototipos = int(row['n_prototipos'])
        accuracy = row['accuracy']
        
        f.write(f"  {porc:5.0f}%  {muestras:9d}  {prototipos:11d}  {accuracy:9.2f}%\n")
    
    mejora = df_resultados.iloc[-1]['accuracy'] - df_resultados.iloc[0]['accuracy']
    
    f.write(f"\nRESUMEN\n")
    f.write(f"  Accuracy con 25%:   {df_resultados.iloc[0]['accuracy']:.2f}%\n")
    f.write(f"  Accuracy con 100%:  {df_resultados.iloc[-1]['accuracy']:.2f}%\n")
    f.write(f"  Diferencia:         {'+' if mejora >= 0 else ''}{mejora:.2f}%\n")

df_resultados.to_csv(f'../resultados/{dataset_id}/metricas/09_sensibilidad_patrones_detalle.csv', index=False)