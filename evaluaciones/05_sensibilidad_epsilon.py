"""

Evaluación: Sensibilidad del Parámetro Epsilon
Evalúa cómo el factor de inhibición epsilon afecta el accuracy y convergencia

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

epsilon_factors = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
resultados = []

for idx, epsilon_factor in enumerate(epsilon_factors):
    
    aciertos = 0
    total_iteraciones = 0
    
    for i in range(len(X_test)):
        vector = X_test.iloc[i].values.tolist()
        real = Y_test.iloc[i]
        
        predicciones, iteraciones = red.predict(vector, k=1, epsilon_factor=epsilon_factor, return_iterations=True)
        predicho = predicciones[0][0]
        
        total_iteraciones += iteraciones
        
        if predicho == real:
            aciertos += 1
    
    accuracy = aciertos / len(X_test) * 100
    promedio_iteraciones = total_iteraciones / len(X_test)
    
    resultados.append({
        'epsilon_factor': epsilon_factor,
        'epsilon_real': epsilon_factor / (len(red.prototipos) + 1),
        'aciertos': aciertos,
        'total': len(X_test),
        'accuracy': accuracy,
        'promedio_iteraciones': promedio_iteraciones
    })

df_resultados = pd.DataFrame(resultados)

mejor_idx = df_resultados['accuracy'].idxmax()
mejor = df_resultados.iloc[mejor_idx]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(df_resultados['epsilon_factor'], df_resultados['accuracy'], 
         marker='o', linewidth=2.5, markersize=10, color='darkgreen', label='Accuracy')
ax1.fill_between(df_resultados['epsilon_factor'], df_resultados['accuracy'], 
                  alpha=0.2, color='darkgreen')

ax1.scatter([mejor['epsilon_factor']], [mejor['accuracy']], 
            s=200, color='gold', edgecolors='red', linewidths=2, zorder=5,
            label=f'Mejor: ε={mejor["epsilon_factor"]:.1f}')

for _, row in df_resultados.iterrows():
    ax1.text(row['epsilon_factor'], row['accuracy'] + 0.8, 
             f"{row['accuracy']:.2f}%", 
             ha='center', fontsize=9, fontweight='bold')

ax1.set_xlabel('Factor de Epsilon', fontsize=11, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax1.set_title('Accuracy vs Factor de Inhibición', fontweight='bold', fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, linestyle='--')

ax2.plot(df_resultados['epsilon_factor'], df_resultados['promedio_iteraciones'], 
         marker='s', linewidth=2.5, markersize=10, color='crimson', label='Iteraciones')
ax2.fill_between(df_resultados['epsilon_factor'], df_resultados['promedio_iteraciones'], 
                  alpha=0.2, color='crimson')

for _, row in df_resultados.iterrows():
    ax2.text(row['epsilon_factor'], row['promedio_iteraciones'] + 0.3, 
             f"{row['promedio_iteraciones']:.1f}", 
             ha='center', fontsize=9, fontweight='bold')

ax2.set_xlabel('Factor de Epsilon', fontsize=11, fontweight='bold')
ax2.set_ylabel('Iteraciones Promedio', fontsize=11, fontweight='bold')
ax2.set_title('Convergencia vs Factor de Inhibición', fontweight='bold', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, linestyle='--')

plt.suptitle(f'{dataset_nombre} | Sensibilidad Epsilon | Test: {len(X_test)} muestras', 
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'../resultados/{dataset_id}/graficos/10_sensibilidad_epsilon.png', dpi=200, bbox_inches='tight')
plt.close()

with open(f'../resultados/{dataset_id}/metricas/10_sensibilidad_epsilon.txt', 'w', encoding='utf-8') as f:
    f.write(f"DATASET: {dataset_nombre}\n\n")
    
    f.write("SENSIBILIDAD DEL PARÁMETRO EPSILON\n\n")
    
    f.write("CONFIGURACIÓN\n")
    f.write(f"  Train:             {len(X_train)}\n")
    f.write(f"  Test:              {len(X_test)}\n")
    f.write(f"  Prototipos:        {len(red.prototipos)}\n")
    f.write(f"  Epsilon factors:   {epsilon_factors}\n")
    f.write(f"  Fórmula:           ε = factor / (M + 1), M = prototipos\n\n")
    
    f.write("RESULTADOS\n")
    f.write(f"  {'ε factor':>9s}  {'ε real':>10s}  {'Accuracy':>10s}  {'Iter. prom.':>12s}\n")
    f.write(f"  {'-'*9}  {'-'*10}  {'-'*10}  {'-'*12}\n")
    
    for _, row in df_resultados.iterrows():
        eps_factor = row['epsilon_factor']
        eps_real = row['epsilon_real']
        accuracy = row['accuracy']
        iter_prom = row['promedio_iteraciones']
        
        marca = " *" if eps_factor == mejor['epsilon_factor'] else "  "
        f.write(f"  {eps_factor:9.1f}  {eps_real:10.6f}  {accuracy:9.2f}%  {iter_prom:12.1f}{marca}\n")
    
    f.write(f"\n  * Mejor accuracy\n\n")
    
    f.write("RESUMEN\n")
    f.write(f"  Mejor ε factor:         {mejor['epsilon_factor']:.1f}\n")
    f.write(f"  Accuracy máximo:        {mejor['accuracy']:.2f}%\n")
    f.write(f"  Iteraciones promedio:   {mejor['promedio_iteraciones']:.1f}\n")

df_resultados.to_csv(f'../resultados/{dataset_id}/metricas/11_sensibilidad_epsilon_detalle.csv', index=False)