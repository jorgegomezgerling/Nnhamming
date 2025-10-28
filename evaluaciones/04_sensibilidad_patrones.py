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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(df_resultados['porcentaje'], df_resultados['accuracy'], 
         marker='o', linewidth=2.5, markersize=10, color='darkblue', label='Accuracy')
ax1.fill_between(df_resultados['porcentaje'], df_resultados['accuracy'], 
                  alpha=0.2, color='darkblue')

ax1.scatter([mejor['porcentaje']], [mejor['accuracy']], 
            s=200, color='gold', edgecolors='red', linewidths=2, zorder=5,
            label=f'Mejor: {mejor["porcentaje"]:.0f}%')

for _, row in df_resultados.iterrows():
    ax1.text(row['porcentaje'], row['accuracy'] + 0.8, 
             f"{row['accuracy']:.2f}%", 
             ha='center', fontsize=9, fontweight='bold')

ax1.set_xlabel('Porcentaje de Train (%)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax1.set_title('Accuracy vs Tamaño de Train', fontweight='bold', fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xticks(df_resultados['porcentaje'])

ax2.bar(df_resultados['porcentaje'], df_resultados['n_prototipos'], 
        color='steelblue', alpha=0.7, edgecolor='black')

for _, row in df_resultados.iterrows():
    ax2.text(row['porcentaje'], row['n_prototipos'] + 5, 
             f"{int(row['n_prototipos'])}", 
             ha='center', fontsize=9, fontweight='bold')

ax2.set_xlabel('Porcentaje de Train (%)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Número de Prototipos', fontsize=11, fontweight='bold')
ax2.set_title('Prototipos Generados', fontweight='bold', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_xticks(df_resultados['porcentaje'])

plt.suptitle(f'{dataset_nombre} | Sensibilidad a Patrones | Test: {len(X_test)} muestras', 
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
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