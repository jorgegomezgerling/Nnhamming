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

os.makedirs('../resultados/graficos', exist_ok=True)
os.makedirs('../resultados/metricas', exist_ok=True)

df = pd.read_csv('../dataset/gold/kaggle_dataset.csv')
X = df.drop('prognosis', axis=1)
Y = df['prognosis']

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
    print(f"[{idx+1}/{len(porcentajes)}] Train: {porcentaje*100:.0f}% ({n_muestras_train} muestras)...", end=" ")
    
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
    train_df['prognosis'] = Y_train_sub.values
    
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
    
    print(f"Accuracy: {accuracy:.2f}%")

df_resultados = pd.DataFrame(resultados)

mejor_idx = df_resultados['accuracy'].idxmax()
mejor = df_resultados.iloc[mejor_idx]
peor_idx = df_resultados['accuracy'].idxmin()
peor = df_resultados.iloc[peor_idx]

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

plt.suptitle(f'Sensibilidad a Cantidad de Patrones | Test fijo: {len(X_test)} muestras', 
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('../resultados/graficos/09_sensibilidad_patrones.png', dpi=200, bbox_inches='tight')
plt.close()

with open('../resultados/metricas/08_sensibilidad_patrones.txt', 'w', encoding='utf-8') as f:

    f.write("SENSIBILIDAD A CANTIDAD DE PATRONES\n")

    
    f.write("CONFIGURACIÓN:\n")
    f.write(f"  Train completo:    {len(X_train_full)} muestras\n")
    f.write(f"  Test (fijo):       {len(X_test)} muestras\n")
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
    f.write(f"\n  Mejora (25% → 100%): {'+' if mejora >= 0 else ''}{mejora:.2f}% puntos\n\n")
    
    f.write("CONCLUSIÓN\n")

    
    f.write(f"  Con 25% del train: {df_resultados.iloc[0]['accuracy']:.2f}%\n")
    f.write(f"  Con 100% del train: {df_resultados.iloc[-1]['accuracy']:.2f}%\n\n")
    
    if abs(mejora) < 2:
        f.write(f"  La cantidad de datos tiene poco impacto en el accuracy.\n")
        f.write(f"  El problema está limitado por la separabilidad, no por datos.\n\n")
    elif mejora > 0:
        f.write(f"  Más datos mejoran el accuracy en {mejora:.2f}% puntos.\n\n")
    else:
        f.write(f"  Más datos empeoran el accuracy en {abs(mejora):.2f}% puntos.\n")
        f.write(f"  Posible presencia de ruido o casos atípicos en el train.\n\n")