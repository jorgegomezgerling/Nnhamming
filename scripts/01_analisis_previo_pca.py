"""

Análisis PRE-PCA: Varianza y Componentes.

"""

import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

raw_df = pd.read_csv('../dataset/bronze/kaggle_dataset.csv')

if 'Unnamed: 0' in raw_df.columns:
    raw_df = raw_df.drop('Unnamed: 0', axis=1)

X = raw_df.drop('prognosis', axis=1)
Y = raw_df['prognosis']

resultados = []

for n in [10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 400]:
    pca = PCA(n_components=n)
    pca.fit(X)
    var_capturada = pca.explained_variance_ratio_.sum()
    resultados.append({
        'n_componentes': n,
        'varianza_capturada': var_capturada * 100
    })

df_resultados = pd.DataFrame(resultados)

pca_full = PCA().fit(X)
varianza_acumulada_full = np.cumsum(pca_full.explained_variance_ratio_) * 100

plt.figure(figsize=(10, 6))

colores = ['red' if v < 70 else 'orange' if v < 80 else 'green' 
           for v in df_resultados['varianza_capturada']]

plt.bar(range(len(df_resultados)),
        df_resultados['varianza_capturada'],
        color=colores, alpha=0.7, edgecolor='black', width=0.7)

for i, row in df_resultados.iterrows():
    plt.text(i, row['varianza_capturada'] + 1, 
             f"{row['varianza_capturada']:.1f}%",
             ha='center', fontsize=9)

plt.xticks(range(len(df_resultados)), 
           df_resultados['n_componentes'], 
           rotation=45)
plt.xlabel('Número de componentes', fontsize=12)
plt.ylabel('Varianza capturada (%)', fontsize=12)
plt.title('Análisis PCA: Selección de Número de Componentes', 
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.ylim(0, 105)
plt.tight_layout()

plt.savefig('../resultados/graficos/01_pca_varianza_barras.png', dpi=200, bbox_inches='tight')

# INFORME: 

informe = f"""Informe: Análisis para PCA - SELECCION DE SINTOMAS

DATASET: {X.shape}

Número de sintomas originales: {X.shape[1]}
Número de muestras/pacientes: {X.shape[0]}
Número de enfermedades diferentes: {len(Y.unique())}

VARIANZA CAPTURADA:

"""

for _, row in df_resultados.iterrows():
    n = int(row['n_componentes'])
    var = row['varianza_capturada']
    informe += f"  {n:3d} componentes → {var:6.2f}%\n"

informe += f"""

INTERPRETACION Y DECISION SOBRE ANALISIS:

- Optamos por quedarnos con 100 componentes (síntomas). Los cuales capturan un {df_resultados[df_resultados['n_componentes']==100]['varianza_capturada'].values[0]:.2f}%
- La reducción es de {X.shape[1]} a 100 componentes. Un 75% menos.
"""

with open('../resultados/informes/01_informe_para_pca.txt', 'w') as f:
    f.write(informe)