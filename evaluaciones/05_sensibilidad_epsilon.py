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

epsilon_factors = [0.1, 1.0, 2.0, 5.0, 10.0]
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

# Calcular rango de variación de iteraciones
iter_min = df_resultados['promedio_iteraciones'].min()
iter_max = df_resultados['promedio_iteraciones'].max()
iter_variacion = iter_max - iter_min

# Crear figura con un solo gráfico más grande
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(df_resultados['epsilon_factor'], df_resultados['accuracy'], 
        marker='o', linewidth=3, markersize=12, color='darkgreen', label='Accuracy')
ax.fill_between(df_resultados['epsilon_factor'], df_resultados['accuracy'], 
                 alpha=0.2, color='darkgreen')

ax.scatter([mejor['epsilon_factor']], [mejor['accuracy']], 
           s=250, color='gold', edgecolors='red', linewidths=3, zorder=5,
           label=f'Mejor: ε={mejor["epsilon_factor"]:.1f} ({mejor["accuracy"]:.2f}%)')

# Anotaciones con offsets verticales para evitar solapamiento
offsets = [3.5, 3.5, 2.5, 2, 2]  # Offsets verticales personalizados
for idx, (_, row) in enumerate(df_resultados.iterrows()):
    offset = offsets[idx]
    
    # Solo mostrar iteraciones si hay variación significativa (>1)
    if iter_variacion > 1.0:
        texto = f"{row['accuracy']:.2f}%\n({row['promedio_iteraciones']:.1f} iter.)"
    else:
        # Si las iteraciones son casi constantes, solo mostrar accuracy
        texto = f"{row['accuracy']:.2f}%"
    
    ax.text(row['epsilon_factor'], row['accuracy'] + offset, 
            texto, 
            ha='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='gray', linewidth=0.5))

# Subtítulo con info de convergencia
if iter_variacion > 1.0:
    convergencia_info = f'Convergencia: {iter_min:.1f}-{iter_max:.1f} iteraciones (variación: {iter_variacion:.1f})'
else:
    convergencia_info = f'Convergencia estable: ~{iter_min:.1f} iteraciones'

ax.set_xlabel('Factor de Epsilon', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title(f'{dataset_nombre} | Sensibilidad del Parámetro Epsilon | Test: {len(X_test)} muestras\n{convergencia_info}', 
             fontweight='bold', fontsize=13)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(0, max(df_resultados['accuracy']) + 10)

plt.tight_layout()
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
    
    f.write("ANÁLISIS DE CONVERGENCIA\n")
    f.write(f"  Iteraciones mínimas:    {iter_min:.1f}\n")
    f.write(f"  Iteraciones máximas:    {iter_max:.1f}\n")
    f.write(f"  Variación:              {iter_variacion:.1f}\n")
    if iter_variacion < 1.0:
        f.write(f"  Conclusión:             Epsilon NO afecta significativamente la convergencia\n\n")
    else:
        f.write(f"  Conclusión:             Epsilon afecta la velocidad de convergencia\n\n")
    
    f.write("RESUMEN\n")
    f.write(f"  Mejor ε factor:         {mejor['epsilon_factor']:.1f}\n")
    f.write(f"  Accuracy máximo:        {mejor['accuracy']:.2f}%\n")
    f.write(f"  Iteraciones promedio:   {mejor['promedio_iteraciones']:.1f}\n")

df_resultados.to_csv(f'../resultados/{dataset_id}/metricas/11_sensibilidad_epsilon_detalle.csv', index=False)