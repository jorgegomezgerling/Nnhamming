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

os.makedirs('../resultados/graficos', exist_ok=True)
os.makedirs('../resultados/metricas', exist_ok=True)

df = pd.read_csv('../dataset/gold/kaggle_dataset.csv')
X = df.drop('prognosis', axis=1)
Y = df['prognosis']

n_enfermedades = Y.nunique()

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    stratify=Y,
    random_state=42
)

train_df = X_train.copy()
train_df['prognosis'] = Y_train.values

red = Nnhamming()
red.fit_from_df(train_df)

k_max = 10
y_real = []
predicciones_por_muestra = []

for i in range(len(X_test)):
    vector = X_test.iloc[i].values.tolist()
    real = Y_test.iloc[i]
    
    predicciones = red.predict(vector, k=k_max)
    top_k = [pred[0] for pred in predicciones]
    
    y_real.append(real)
    predicciones_por_muestra.append(top_k)

k_values = [1, 2, 3, 4, 5, 7, 10]
resultados = []

for k in k_values:
    aciertos = sum(real in preds[:k] for real, preds in zip(y_real, predicciones_por_muestra))
    accuracy = aciertos / len(y_real) * 100
    
    resultados.append({
        'k': k,
        'aciertos': aciertos,
        'total': len(y_real),
        'accuracy': accuracy
    })

df_resultados = pd.DataFrame(resultados)

mejor_k = df_resultados.loc[df_resultados['accuracy'].idxmax()]
accuracy_k1 = df_resultados[df_resultados['k'] == 1]['accuracy'].values[0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(df_resultados['k'], df_resultados['accuracy'], 
         marker='o', linewidth=2.5, markersize=10, color='steelblue', label='Accuracy')
ax1.fill_between(df_resultados['k'], df_resultados['accuracy'], 
                  alpha=0.2, color='steelblue')

for _, row in df_resultados.iterrows():
    ax1.text(row['k'], row['accuracy'] + 1.5, f"{row['accuracy']:.1f}%", 
             ha='center', fontsize=9, fontweight='bold')

ax1.axhline(y=accuracy_k1, color='red', linestyle='--', linewidth=1.5, 
            alpha=0.6, label=f'Baseline (k=1): {accuracy_k1:.1f}%')

ax1.scatter([mejor_k['k']], [mejor_k['accuracy']], 
            color='green', s=200, zorder=5, marker='*', 
            label=f'Mejor k={int(mejor_k["k"])}')

ax1.set_xlabel('Valor de K', fontsize=11, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax1.set_title('Accuracy vs K: Top-K Candidatos', fontweight='bold', fontsize=12)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=9)
ax1.set_xticks(k_values)

mejoras = []
for k in k_values[1:]:
    acc_k = df_resultados[df_resultados['k'] == k]['accuracy'].values[0]
    mejora = acc_k - accuracy_k1
    mejoras.append(mejora)

ax2.bar(k_values[1:], mejoras, color='green', alpha=0.7, edgecolor='black')
ax2.axhline(y=0, color='red', linestyle='-', linewidth=1)

for k, mejora in zip(k_values[1:], mejoras):
    ax2.text(k, mejora + 0.5, f'+{mejora:.1f}%', 
             ha='center', fontsize=9, fontweight='bold')

ax2.set_xlabel('Valor de K', fontsize=11, fontweight='bold')
ax2.set_ylabel('Mejora vs k=1 (%)', fontsize=11, fontweight='bold')
ax2.set_title('Mejora Relativa por K', fontweight='bold', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_xticks(k_values[1:])

plt.suptitle(f'Optimización del Parámetro K | Test: {len(X_test)} muestras', 
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('../resultados/graficos/07_optimizacion_k.png', dpi=200, bbox_inches='tight')
plt.close()

with open('../resultados/metricas/04_optimizacion_k.txt', 'w', encoding='utf-8') as f:

    f.write("OPTIMIZACIÓN DEL PARÁMETRO K\n")

    
    f.write("CONFIGURACIÓN:\n")
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
    
    f.write("\n")
    f.write(f"  * Mejor K: {int(mejor_k['k'])}\n\n")  

    f.write("INTERPRETACIÓN\n")
    
    f.write(f"  BASELINE (K=1):\n")
    f.write(f"    Accuracy: {accuracy_k1:.2f}%\n")
    f.write(f"    La red selecciona solo el candidato más cercano.\n\n")
    
    f.write(f"  MEJOR RESULTADO (K={int(mejor_k['k'])}):\n")
    f.write(f"    Accuracy: {mejor_k['accuracy']:.2f}%\n")
    f.write(f"    Mejora: +{mejor_k['accuracy'] - accuracy_k1:.2f}% puntos\n")
    f.write(f"    La respuesta correcta está entre los {int(mejor_k['k'])} candidatos\n")
    f.write(f"    más cercanos en {mejor_k['accuracy']:.1f}% de los casos.\n\n")
    
df_resultados.to_csv('../resultados/metricas/05_optimizacion_k_detalle.csv', index=False)