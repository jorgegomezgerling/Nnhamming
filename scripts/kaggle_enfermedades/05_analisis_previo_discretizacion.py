"""

Análisis de Discretización: 

"""

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('../../resultados/kaggle_enfermedades/graficos', exist_ok=True)
os.makedirs('../../resultados/kaggle_enfermedades/informes', exist_ok=True)

df = pd.read_csv('../../datasets/kaggle_enfermedades/silver/02_mutual_10comp.csv')
X = df.drop('prognosis', axis=1)
Y = df['prognosis']

discretizador = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
X_discretizado = discretizador.fit_transform(X)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

comp_ejemplo = X.columns[0]
col_idx = 0

valores = X[comp_ejemplo].values
ax1.hist(valores, bins=40, alpha=0.7, color='steelblue', edgecolor='black')

edges = discretizador.bin_edges_[col_idx]

ax1.axvline(edges[1], color='red', linestyle='--', linewidth=2.5, alpha=0.8, label='Límites de bins')
ax1.axvline(edges[2], color='red', linestyle='--', linewidth=2.5, alpha=0.8)

y_max = ax1.get_ylim()[1]
ax1.text(edges[0] + (edges[1] - edges[0])/2, y_max * 0.9, 
         'Bin 0\n(bajo)', ha='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax1.text(edges[1] + (edges[2] - edges[1])/2, y_max * 0.9, 
         'Bin 1\n(medio)', ha='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
ax1.text(edges[2] + (edges[3] - edges[2])/2, y_max * 0.9, 
         'Bin 2\n(alto)', ha='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

ax1.set_xlabel('Valor de la componente', fontsize=11)
ax1.set_ylabel('Frecuencia', fontsize=11)
ax1.set_title(f'División en 3 bins (strategy=quantile)\nEjemplo: {comp_ejemplo}', 
              fontweight='bold', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

balance_por_bin = []
for i in range(X_discretizado.shape[1]):
    conteo = np.bincount(X_discretizado[:, i].astype(int))
    for bin_val, count in enumerate(conteo):
        balance_por_bin.append({
            'bin': bin_val,
            'muestras': count,
            'porcentaje': count / len(X_discretizado) * 100
        })

df_balance = pd.DataFrame(balance_por_bin)
balance_promedio = df_balance.groupby('bin').agg({'muestras': 'mean', 'porcentaje': ['mean', 'std']}).reset_index()

bins = [0, 1, 2]
porcentajes = [balance_promedio[balance_promedio['bin']==b][('porcentaje', 'mean')].values[0] for b in bins]
std_devs = [balance_promedio[balance_promedio['bin']==b][('porcentaje', 'std')].values[0] for b in bins]
muestras = [balance_promedio[balance_promedio['bin']==b][('muestras', 'mean')].values[0] for b in bins]

colores = ['lightgreen', 'lightyellow', 'lightcoral']
bars = ax2.bar(bins, porcentajes, color=colores, alpha=0.8, edgecolor='black', linewidth=1.5)

ax2.axhline(y=33.33, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Ideal (33.33%)')

for i, (bar, pct, count, std) in enumerate(zip(bars, porcentajes, muestras, std_devs)):
    ax2.text(bar.get_x() + bar.get_width()/2., pct + 1,
             f'{pct:.1f}%\n({int(count)} muestras)',
             ha='center', fontsize=10, fontweight='bold')

ax2.set_xlabel('Bin', fontsize=11)
ax2.set_ylabel('Porcentaje de muestras', fontsize=11)
ax2.set_title('Balance de muestras por bin\n(promedio de todas las componentes)', 
              fontweight='bold', fontsize=12)
ax2.set_xticks(bins)
ax2.set_xticklabels(['Bin 0\n(bajo)', 'Bin 1\n(medio)', 'Bin 2\n(alto)'])
ax2.set_ylim(0, 45)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

plt.suptitle('Análisis de Discretización: 3 bins con strategy=quantile', 
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('../../resultados/kaggle_enfermedades/graficos/03_discretizacion_analisis.png', dpi=200, bbox_inches='tight')

informe = f"""INFORME: ANÁLISIS DE DISCRETIZACIÓN

CONFIGURACIÓN:
  n_bins:   3
  encode:   ordinal
  strategy: quantile

DATASET:
  Componentes:  {X.shape[1]}
  Muestras:     {X.shape[0]}

JUSTIFICACIÓN

1. NÚMERO DE BINS (3):
   • Permite representación binaria con 2 bits:
     Bin 0 → 00
     Bin 1 → 01
     Bin 2 → 10

2. STRATEGY (quantile):
   • Divide los datos en percentiles
   • Garantiza balance: ~33.3% de muestras por bin
   • Evita bins vacíos o desbalanceados

3. ENCODE (ordinal):
   • Representa bins como números: 0, 1, 2
   • Mantiene orden: bajo < medio < alto

BALANCE LOGRADO:

Promedio de todas las componentes:

  Bin 0 (bajo):  {porcentajes[0]:.1f}% ({int(muestras[0])} muestras)
  Bin 1 (medio): {porcentajes[1]:.1f}% ({int(muestras[1])} muestras)
  Bin 2 (alto):  {porcentajes[2]:.1f}% ({int(muestras[2])} muestras)

Ideal: 33.33% por bin
Resultado: Balance excelente.

EJEMPLO: LÍMITES PARA {comp_ejemplo}

  Bin 0 (bajo):  valores ≤ {edges[1]:.3f}
  Bin 1 (medio): valores entre {edges[1]:.3f} y {edges[2]:.3f}
  Bin 2 (alto):  valores > {edges[2]:.3f}

RESULTADO:

Transformación exitosa: valores continuos: 3 categorías discretas
Balance uniforme logrado con strategy='quantile'

"""

with open('../../resultados/kaggle_enfermedades/informes/03_informe_discretizacion.txt', 'w', encoding='utf-8') as f:
    f.write(informe)