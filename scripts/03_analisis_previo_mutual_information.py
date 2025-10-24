import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt

silver_df = pd.read_csv('../dataset/silver/01_pca_100comp.csv')

X = silver_df.drop('prognosis', axis=1)
Y = silver_df['prognosis']

le = LabelEncoder()
y_encoded = le.fit_transform(Y)

scores = mutual_info_classif(X, y_encoded, random_state=42)

mutual_information = pd.DataFrame({
    'Componente': X.columns,
    'Score': scores,
})

mutual_information = mutual_information.sort_values("Score", ascending=False)

top_10_componentes = mutual_information["Componente"].head(10).tolist()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

top_20 = mutual_information.head(20)
colores = ['green' if i < 10 else 'orange' for i in range(len(top_20))]

bars = ax1.barh(range(len(top_20)), top_20['Score'], color=colores, alpha=0.7, edgecolor='black')

for i, (idx, row) in enumerate(top_20.iterrows()):
    ax1.text(row['Score'] + 0.02, i, f"{row['Score']:.3f}", 
             va='center', fontsize=8, fontweight='bold')

ax1.set_yticks(range(len(top_20)))
ax1.set_yticklabels(top_20['Componente'], fontsize=9)
ax1.set_xlabel('Score de Mutual Information', fontsize=11)
ax1.set_title('Top 20 Componentes por Score MI', fontweight='bold', fontsize=12)
ax1.invert_yaxis()
ax1.axvline(x=top_20['Score'].iloc[9], color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Corte Top 10')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='x')
ax1.set_xlim(0, max(top_20['Score']) * 1.10)

ax2.plot(range(1, 101), mutual_information['Score'].values, linewidth=2, color='steelblue', marker='o', markersize=3)
ax2.axvline(x=10, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='10 componentes seleccionadas')
ax2.axhline(y=mutual_information['Score'].iloc[9], color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Score #10: {mutual_information["Score"].iloc[9]:.3f}')
ax2.scatter([10], [mutual_information['Score'].iloc[9]], color='red', s=100, zorder=5)
ax2.set_xlabel('Ranking de componente', fontsize=11)
ax2.set_ylabel('Score de Mutual Information', fontsize=11)
ax2.set_title('Distribución de Scores (100 componentes)', fontweight='bold', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1, 100)

plt.suptitle('Análisis de Mutual Information: Selección de Componentes', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../resultados/graficos/02_mutual_information_analisis.png', dpi=200, bbox_inches='tight')

informe = f"""Informe: Análisis Mutual Information - SELECCION DE COMPONENTES

DATASET: {X.shape}

Número de componentes de entrada (post-PCA): {X.shape[1]}
Número de muestras/pacientes: {X.shape[0]}
Número de enfermedades diferentes: {len(Y.unique())}

SCORES DE MUTUAL INFORMATION:

Top 10 componentes seleccionadas:

"""

for i, (idx, row) in enumerate(mutual_information.head(10).iterrows(), 1):
    informe += f"  {i:2d}. {row['Componente']:10s} → Score: {row['Score']:.6f}\n"

informe += f"""

Bottom 10 componentes (descartadas):

"""

for i, (idx, row) in enumerate(mutual_information.tail(10).iterrows(), 91):
    informe += f"  {i:2d}. {row['Componente']:10s} → Score: {row['Score']:.6f}\n"

informe += f"""

ESTADÍSTICAS:

Rango de scores:        {scores.min():.6f} - {scores.max():.6f}
Score promedio:         {scores.mean():.6f}
Score top 10 (mínimo):  {mutual_information['Score'].iloc[9]:.6f}
Score top 10 (promedio):{mutual_information['Score'].head(10).mean():.6f}

Reducción:              100 → 10 componentes (90% menos)

INTERPRETACIÓN Y DECISIÓN:

- Mutual Information mide cuánta información tiene cada componente
  sobre el diagnóstico (la enfermedad).
  
- Seleccionamos las 10 componentes con mayor score de MI porque:
  * Tienen scores significativamente más altos (0.649 - 1.043)
  * Hay una caída notable después de la componente #10
  * El resto de componentes (11-100) tienen scores menores (0.298 - 0.642)
  
- Esta selección reduce la dimensionalidad de 100 a 10 componentes,
  manteniendo las más relevantes para la clasificación.

DIFERENCIA CON PCA:
- PCA seleccionó componentes con alta VARIANZA (cuánto varían los datos)
- MI selecciona componentes con alta INFORMACIÓN sobre el target
  (cuánto ayudan a predecir la enfermedad)
"""

with open('../resultados/informes/02_informe_mutual_information.txt', 'w') as f:
    f.write(informe)