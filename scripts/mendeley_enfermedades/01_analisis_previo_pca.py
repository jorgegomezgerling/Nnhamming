"""

Análisis PRE-PCA: Varianza y Componentes

"""

import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os

df = pd.read_csv('../../datasets/mendeley_enfermedades/silver/00_clean_dataset.csv')

X = df.drop('prognosis', axis=1)
Y = df['prognosis']

resultados = []

for n in [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70]:
        pca = PCA(n_components=n, random_state=42)
        pca.fit(X)
        var_capturada = pca.explained_variance_ratio_.sum()
        resultados.append({
            'n_componentes': n,
            'varianza_capturada': var_capturada * 100
        })

df_resultados = pd.DataFrame(resultados)

# Gráfico
plt.figure(figsize=(10, 6))

colores = ['red' if v < 70 else 'orange' if v < 85 else 'green'
           for v in df_resultados['varianza_capturada']]

plt.bar(range(len(df_resultados)),
        df_resultados['varianza_capturada'],
        color=colores, alpha=0.7, edgecolor='black', width=0.7)

for i, row in df_resultados.iterrows():
    plt.text(i, row['varianza_capturada'] + 1.5,
             f"{row['varianza_capturada']:.1f}%",
             ha='center', fontsize=9, fontweight='bold')

plt.xticks(range(len(df_resultados)),
           df_resultados['n_componentes'],
           rotation=45)
plt.xlabel('Número de componentes', fontsize=12)
plt.ylabel('Varianza capturada (%)', fontsize=12)
plt.title('MENDELEY | Análisis PCA: Selección de Componentes',
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.ylim(0, 105)
plt.legend()
plt.tight_layout()

os.makedirs('../../resultados/mendeley_enfermedades/graficos', exist_ok=True)
plt.savefig('../../resultados/mendeley_enfermedades/graficos/01_pca_varianza_barras.png', 
            dpi=200, bbox_inches='tight')


# Informe
os.makedirs('../../resultados/mendeley_enfermedades/informes', exist_ok=True)

informe = f""" MENDELEY INFORME - ANALISIS PCA - SELECCION DE COMPONENTES

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

- Optamos por quedarnos con 40 componentes (sintomas). Los cuales capturan un 96.10%. Los cuales capturan un {df_resultados[df_resultados['n_componentes']==40]['varianza_capturada'].values[0]:.2f}%
- La reducción es de {X.shape[1]} a 40 componentes.
  Porcentaje reducción: 45.9%
"""

with open('../../resultados/mendeley_enfermedades/informes/02_analisis_pca.txt', 'w') as f:
     f.write(informe)

