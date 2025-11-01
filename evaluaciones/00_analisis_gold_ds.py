"""

Análisis del Dataset Gold: Caracterización inicial del problema

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from config import get_dataset_config  

config = get_dataset_config()
dataset_id = config['id']
dataset_nombre = config['nombre']
dataset_fuente = config.get('fuente')

df = pd.read_csv(config['path'])
X = df.drop(config['target'], axis=1)
Y = df[config['target']]

os.makedirs(f'../resultados/{dataset_id}/graficos', exist_ok=True)
os.makedirs(f'../resultados/{dataset_id}/metricas', exist_ok=True)

n_muestras = len(df)
n_enfermedades = Y.nunique()
n_features = X.shape[1]

casos_por_enfermedad = Y.value_counts()
enfermedades_raras = (casos_por_enfermedad < 10).sum()

ratio_clases_features = n_enfermedades / n_features

# Calcula bits necesarios: log2 redondea hacia arriba
# Ej: 10 enfermedades → log2(10)=3.32 → ceil=4 bits
bits_necesarios = int(np.ceil(np.log2(n_enfermedades)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(casos_por_enfermedad, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
ax1.axvline(casos_por_enfermedad.mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Promedio: {casos_por_enfermedad.mean():.1f}')
ax1.set_xlabel('Casos por enfermedad', fontsize=10)
ax1.set_ylabel('Frecuencia', fontsize=10)
ax1.set_title('Distribución de Casos', fontweight='bold', fontsize=11)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

ax2.axis('off')
resumen = f"""CARACTERIZACIÓN DEL PROBLEMA

DATASET: {dataset_nombre}
Fuente:  {dataset_fuente}

DIMENSIONES:
  • Muestras:              {n_muestras}
  • Enfermedades:          {n_enfermedades}
  • Features binarias:     {n_features}

COMPLEJIDAD:
  • Bits necesarios (teórico): {bits_necesarios}
  • Bits disponibles:          {n_features}
  • Ratio clases/features:     {ratio_clases_features:.2f}

DISTRIBUCIÓN:
  • Mín casos:             {casos_por_enfermedad.min()}
  • Máx casos:             {casos_por_enfermedad.max()}
  • Promedio:              {casos_por_enfermedad.mean():.1f}
  • Enfermedades <10 casos: {enfermedades_raras}
"""

ax2.text(0.05, 0.95, resumen, transform=ax2.transAxes,
         fontsize=10, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.suptitle(f'{dataset_nombre} | Análisis Gold Dataset', 
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig(f'../resultados/{dataset_id}/graficos/00_analisis_dataset_gold.png', 
            dpi=200, bbox_inches='tight')
plt.close()

with open(f'../resultados/{dataset_id}/metricas/00_caracterizacion_problema.txt', 
          'w', encoding='utf-8') as f:

    f.write(f"DATASET: {dataset_nombre}\n")
    f.write(f"Fuente:  {dataset_fuente}\n\n")

    f.write("DIMENSIONES\n")
    f.write(f"  Muestras:              {n_muestras}\n")
    f.write(f"  Enfermedades:          {n_enfermedades}\n")
    f.write(f"  Features binarias:     {n_features}\n\n")

    f.write("DISTRIBUCIÓN\n")
    f.write(f"  Mínimo:                {casos_por_enfermedad.min()} casos\n")
    f.write(f"  Máximo:                {casos_por_enfermedad.max()} casos\n")
    f.write(f"  Promedio:              {casos_por_enfermedad.mean():.1f} casos\n")
    f.write(f"  Enfermedades <10:      {enfermedades_raras}\n\n")
    
    f.write("COMPLEJIDAD\n")
    f.write(f"  Bits necesarios (log₂): {np.log2(n_enfermedades):.2f} ≈ {bits_necesarios} bits\n")
    f.write(f"  Bits disponibles:       {n_features} bits\n")
    f.write(f"  Ratio clases/features:  {ratio_clases_features:.2f}\n")