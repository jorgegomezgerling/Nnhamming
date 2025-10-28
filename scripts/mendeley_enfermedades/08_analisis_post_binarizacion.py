"""

Análisis de Binarización:

"""

import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs('../../resultados/mendeley_enfermedades/graficos', exist_ok=True)
os.makedirs('../../resultados/mendeley_enfermedades/informes', exist_ok=True)

df_discretizado = pd.read_csv('../../datasets/mendeley_enfermedades/silver/03_discretizado_10comp.csv')
df_gold = pd.read_csv('../../datasets/mendeley_enfermedades/gold/mendeley_dataset.csv')

X_disc = df_discretizado.drop('prognosis', axis=1)
Y = df_discretizado['prognosis']
X_bin = df_gold.drop('prognosis', axis=1)

total_bits = X_bin.size
conteo_unos = (X_bin == 1).sum().sum()
conteo_ceros = (X_bin == 0).sum().sum()
pct_unos = conteo_unos / total_bits * 100
pct_ceros = conteo_ceros / total_bits * 100

fig = plt.figure(figsize=(14, 6))

ax1 = plt.subplot(1, 2, 1)
ax1.axis('off')

codificacion_texto = """
CODIFICACIÓN UTILIZADA
══════════════════════════════════════════

   Valor discreto  →  Representación binaria

        0 (bajo)   →      1    0
        1 (medio)  →      1    1
        2 (alto)   →      0    1


TRANSFORMACIÓN COMPLETA:
════════════════════════════════════════════

   10 componentes discretas (valores 0, 1, 2)
   
        ↓  × 2 bits por componente
        
   20 features binarias (valores 0, 1)


JUSTIFICACIÓN:
════════════════════════════════════════════

   Compatible con Red de Hamming
"""

ax1.text(0.05, 0.95, codificacion_texto, transform=ax1.transAxes,
         fontsize=11, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.4, pad=1))

ax2 = plt.subplot(1, 2, 2)
ax2.axis('off')

ejemplo_idx = 0
ejemplo_disc = X_disc.iloc[ejemplo_idx].values
ejemplo_bin = X_bin.iloc[ejemplo_idx].values

ejemplo_texto = f"""
EJEMPLO: PACIENTE #{ejemplo_idx}
══════════════════════════════════════════

Enfermedad: {Y.iloc[ejemplo_idx]}


TRANSFORMACIÓN:
══════════════════════════════════════════

Componente  │ Discreto │  Binario
────────────┼──────────┼───────────
"""

for i, comp in enumerate(X_disc.columns):
    val_disc = int(ejemplo_disc[i])
    bit1 = int(ejemplo_bin[i*2])
    bit2 = int(ejemplo_bin[i*2+1])
    ejemplo_texto += f"{comp:10s}  │    {val_disc}     │   {bit1}  {bit2}\n"

ejemplo_texto += f"""

RESULTADO FINAL:
══════════════════════════════════════════

   Dataset shape: {df_gold.shape}
   Features:      {X_bin.shape[1]} binarias
   Target:        1 columna (prognosis)
   
   Balance de bits:
   • Ceros: {pct_ceros:.1f}%
   • Unos:  {pct_unos:.1f}%

"""

ax2.text(0.05, 0.95, ejemplo_texto, transform=ax2.transAxes,
         fontsize=10, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4, pad=1))

plt.suptitle('MENDELEY | Binarización: 3 valores discretos → 2 bits binarios', 
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('../../resultados/mendeley_enfermedades/graficos/05_binarizacion_analisis.png', dpi=200, bbox_inches='tight')

informe = f"""
INFORME: ANÁLISIS DE BINARIZACIÓN

DATASET: MENDELEY - Diagnóstico de Enfermedades

TRANSFORMACIÓN:
  Input:   {X_disc.shape[1]} componentes discretas (valores: 0, 1, 2)
  Output:  {X_bin.shape[1]} features binarias (valores: 0, 1)
  Muestras: {X_bin.shape[0]}

CODIFICACIÓN UTILIZADA

   Valor discreto  →  Bits binarios
   ────────────────────────────────
        0 (bajo)   →      1  0
        1 (medio)  →      1  1
        2 (alto)   →      0  1

JUSTIFICACIÓN:
    Compatible con Red de Hamming

VERIFICACIÓN

  Dataset discretizado: {df_discretizado.shape}
  Dataset binarizado:   {df_gold.shape}
  
  Cálculo: {X_disc.shape[1]} componentes × 2 bits = {X_bin.shape[1]} features ✓


BALANCE DE BITS

  Total de bits: {total_bits:,}
  
  Distribución:
    Ceros (0): {conteo_ceros:,} bits ({pct_ceros:.1f}%)
    Unos  (1): {conteo_unos:,} bits ({pct_unos:.1f}%)
  
  NOTA: El desbalance (más unos que ceros) es esperado debido a la
        codificación elegida. El valor 1 genera dos unos (11), mientras
        que los valores 0 y 2 generan un uno y un cero cada uno.


EJEMPLO DE TRANSFORMACIÓN

Paciente #{ejemplo_idx} - Enfermedad: {Y.iloc[ejemplo_idx]}

  Componente    Discreto  →  Binario
  ──────────────────────────────────
"""

for i, comp in enumerate(X_disc.columns):
    val_disc = int(ejemplo_disc[i])
    bit1 = int(ejemplo_bin[i*2])
    bit2 = int(ejemplo_bin[i*2+1])
    informe += f"  {comp:12s}    {val_disc}      →    {bit1} {bit2}\n"

informe += f"""


PIPELINE COMPLETO

  1. Limpieza:         172 features    → 74 features (filtrado)
  2. PCA:              74 features     → 40 componentes (96.1% varianza)
  3. Mutual Info:      40 componentes  → 10 componentes (top MI)
  4. Discretización:   Continuos       → 3 valores discretos (0,1,2)
  5. Binarización:     3 discretos     → 20 bits binarios

RESULTADO: 172 features originales → 20 features binarias
"""

with open('../../resultados/mendeley_enfermedades/informes/04_binarizacion_analisis.txt', 'w', encoding='utf-8') as f:
    f.write(informe)

