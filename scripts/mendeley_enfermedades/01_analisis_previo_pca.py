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

print("="*70)
print("ANÁLISIS PRE-PCA: Mendeley Dataset")
print("="*70)
print(f"\nDataset: {X.shape}")
print(f"Features: {X.shape[1]}")
print(f"Muestras: {X.shape[0]}")
print(f"Clases: {Y.nunique()}\n")

# Probar diferentes números de componentes
resultados = []

for n in [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70]:
    if n <= X.shape[1]:  # No más componentes que features
        pca = PCA(n_components=n, random_state=42)
        pca.fit(X)
        var_capturada = pca.explained_variance_ratio_.sum()
        resultados.append({
            'n_componentes': n,
            'varianza_capturada': var_capturada * 100
        })
        print(f"{n:3d} componentes → {var_capturada*100:6.2f}%")

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
plt.axhline(y=85, color='orange', linestyle='--', linewidth=1.5, alpha=0.7,
            label='Umbral 85%')
plt.axhline(y=95, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
            label='Umbral 95%')
plt.legend()
plt.tight_layout()

os.makedirs('../../resultados/mendeley_enfermedades/graficos', exist_ok=True)
plt.savefig('../../resultados/mendeley_enfermedades/graficos/02_pca_varianza_barras.png', 
            dpi=200, bbox_inches='tight')

print(f"\nGráfico guardado: resultados/mendeley_enfermedades/graficos/02_pca_varianza_barras.png")

# Decidir cuántos componentes (ejemplo: 95% varianza)
var_objetivo = 95
componentes_elegidos = df_resultados[df_resultados['varianza_capturada'] >= var_objetivo]['n_componentes'].min()
var_elegida = df_resultados[df_resultados['n_componentes']==componentes_elegidos]['varianza_capturada'].values[0]

# Si no se alcanza 95%, tomar el máximo
if pd.isna(componentes_elegidos):
    componentes_elegidos = df_resultados['n_componentes'].max()
    var_elegida = df_resultados['varianza_capturada'].max()

# Informe
os.makedirs('../../resultados/mendeley_enfermedades/informes', exist_ok=True)

with open('../../resultados/mendeley_enfermedades/informes/02_analisis_pca.txt', 'w') as f:
    f.write("ANÁLISIS PRE-PCA: Mendeley Dataset\n\n")
    
    f.write("DATASET (SILVER01)\n")
    f.write(f"  Features: {X.shape[1]}\n")
    f.write(f"  Muestras: {X.shape[0]}\n")
    f.write(f"  Clases: {Y.nunique()}\n\n")
    
    f.write("VARIANZA CAPTURADA POR COMPONENTES\n")
    for _, row in df_resultados.iterrows():
        n = int(row['n_componentes'])
        var = row['varianza_capturada']
        marca = " *" if n == componentes_elegidos else ""
        f.write(f"  {n:3d} componentes → {var:6.2f}%{marca}\n")
    
    f.write(f"\n  * Componentes seleccionados\n\n")
    
    f.write("DECISIÓN\n")
    f.write(f"  Componentes seleccionados: {int(componentes_elegidos)}\n")
    f.write(f"  Varianza capturada: {var_elegida:.2f}%\n")
    f.write(f"  Reducción: {X.shape[1]} → {int(componentes_elegidos)} features\n")
    f.write(f"  Porcentaje reducción: {(1-componentes_elegidos/X.shape[1])*100:.1f}%\n\n")
    
    f.write("INTERPRETACIÓN\n")
    f.write(f"  Con {int(componentes_elegidos)} componentes se retiene el {var_elegida:.1f}%\n")
    f.write(f"  de la varianza original, eliminando información redundante.\n")

print(f"Informe guardado: resultados/mendeley_enfermedades/informes/02_analisis_pca.txt")
print(f"\n{'='*70}")
print(f"DECISIÓN: Usar {int(componentes_elegidos)} componentes ({var_elegida:.1f}% varianza)")
print(f"{'='*70}")

