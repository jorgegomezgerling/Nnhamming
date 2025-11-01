"""
Evaluación: Matriz de Confusión
Evalúa el rendimiento de la Red de Hamming usando train/test split.
Genera matriz de confusión, métricas y visualizaciones.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
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

n_enfermedades = Y.nunique()
# shape[1] columnas, shape[0] filas.
n_features = X.shape[1]

# Primero creamos conjuntos separados:

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    stratify=Y,
    random_state=42
)

# Creamos una copia de X_train para agregarle los values de Y_train (la red espera sintomas + prognosis)
train_df = X_train.copy()
train_df[config['target']] = Y_train.values

# Instanciamos nuestra red y la entrenamos con el 80% de los datos
red = Nnhamming()
red.fit_from_df(train_df)  # La red guarda estos datos en memoria

y_real = []  # Diagnósticos verdaderos del test (20%)
y_pred = []  # Diagnósticos que la red predice

# Evaluamos con el 20% restante que la red NUNCA vio
for i in range(len(X_test)):
    # Tomamos un paciente del test
    vector = X_test.iloc[i].values.tolist()  # Ej: [1, 0, 1]
    real = Y_test.iloc[i]                     # Ej: "Gastroenteritis"
    
    # Le preguntamos a la red qué opina de este paciente NUEVO
    # La red compara con los prototipos que guardó del train
    prediccion = red.predict(vector, k=1)
    predicho = prediccion[0][0]  # Extraemos solo el nombre
    
    # Guardamos ambos para comparar después
    y_real.append(real)      # Lo que DEBERÍA ser
    y_pred.append(predicho)  # Lo que la red CREE que es

# Comparamos: Cuántas veces la red acertó con datos que nunca vio?
matriz = confusion_matrix(y_real, y_pred)
accuracy = accuracy_score(y_real, y_pred)

# Contamos cuántas predicciones fueron correctas
# Comparamos elemento por elemento: True=1, False=0
correctas = (np.array(y_real) == np.array(y_pred)).sum()
# Ejemplo: [True, True, False, True].sum() = 3

# Las incorrectas son simplemente el total menos las correctas
incorrectas = len(y_real) - correctas

# Obtenemos lista ÚNICA de enfermedades, ordenada alfabéticamente
enfermedades = sorted(Y.unique())
# Ejemplo: ['Alergia', 'Covid', 'Gastroenteritis', 'Gripe']

# Lista vacía donde guardaremos métricas de CADA enfermedad
metricas_por_enf = []

# Calculamos métricas por cada enfermedad individual
for i, enf in enumerate(enfermedades):
    tp = matriz[i, i]                    # True Positives (diagonal)
    total_real = matriz[i, :].sum()      # Total real en test
    acc_enf = tp / total_real if total_real > 0 else 0  # Accuracy individual
    
    metricas_por_enf.append({
        'enfermedad': enf,
        'muestras_test': total_real,
        'correctas': tp,
        'accuracy': acc_enf * 100
    })

# DataFrame ordenado por accuracy (mejor a peor)
df_metricas = pd.DataFrame(metricas_por_enf).sort_values('accuracy', ascending=False)

# Identificamos enfermedades con 0% accuracy (problemáticas)
enf_cero = df_metricas[df_metricas['accuracy'] == 0]
n_enf_cero = len(enf_cero)
muestras_cero = enf_cero['muestras_test'].sum()

# Filtramos enfermedades detectables (accuracy > 0)
enf_no_cero = df_metricas[df_metricas['accuracy'] > 0]
muestras_no_cero = enf_no_cero['muestras_test'].sum()
correctas_no_cero = enf_no_cero.apply(lambda row: int(row['correctas']), axis=1).sum()

# Accuracy excluyendo enfermedades no detectables
accuracy_sin_ceros = correctas_no_cero / muestras_no_cero * 100 if muestras_no_cero > 0 else 0

# Matriz de confusión como DataFrame
matriz_df = pd.DataFrame(matriz, index=enfermedades, columns=enfermedades)
matriz_df.to_csv(f'../resultados/{dataset_id}/metricas/01_matriz_confusion_completa.csv')

df_metricas.to_csv(f'../resultados/{dataset_id}/metricas/03_metricas_por_enfermedad.csv', index=False)

with open(f'../resultados/{dataset_id}/metricas/02_metricas_confusion.txt', 'w', encoding='utf-8') as f:
    f.write(f"DATASET: {dataset_nombre}\n\n")
    
    f.write("CONFIGURACIÓN\n")
    f.write(f"  Train/Test split: 80/20 (stratified)\n")
    f.write(f"  K (predicción):   1\n\n")
    
    f.write("DIMENSIONES\n")
    f.write(f"  Total muestras:   {len(df)}\n")
    f.write(f"  Enfermedades:     {n_enfermedades}\n")
    f.write(f"  Features:         {n_features}\n")
    f.write(f"  Train:            {len(X_train)}\n")
    f.write(f"  Test:             {len(X_test)}\n\n")
    
    f.write("RESULTADOS\n")
    f.write(f"  Accuracy:         {accuracy*100:.2f}%\n")
    f.write(f"  Correctas:        {correctas}/{len(y_real)}\n")
    f.write(f"  Incorrectas:      {incorrectas}/{len(y_real)}\n\n")
    
    f.write("ANÁLISIS POR ACCURACY\n")
    f.write(f"  Enfermedades con 0%:       {n_enf_cero}/{n_enfermedades} ({n_enf_cero/n_enfermedades*100:.1f}%)\n")
    f.write(f"  Muestras afectadas:        {int(muestras_cero)}/{len(Y_test)} ({muestras_cero/len(Y_test)*100:.1f}%)\n")
    f.write(f"  Accuracy excluyendo 0%:    {accuracy_sin_ceros:.1f}%\n")
    f.write(f"  Ratio clases/features:     {n_enfermedades/n_features:.2f}\n\n")
    
    f.write("TOP 10 MEJOR ACCURACY\n")
    for i, row in df_metricas.head(10).iterrows():
        f.write(f"  {row['enfermedad'][:40]:40s}: {row['accuracy']:5.1f}% "
                f"({int(row['correctas'])}/{int(row['muestras_test'])})\n")
    f.write("\n")
    
    f.write("TOP 10 PEOR ACCURACY\n")
    for i, row in df_metricas.tail(10).iterrows():
        f.write(f"  {row['enfermedad'][:40]:40s}: {row['accuracy']:5.1f}% "
                f"({int(row['correctas'])}/{int(row['muestras_test'])})\n")

top_20_enfermedades = Y_test.value_counts().head(20).index.tolist()
indices_top20 = [i for i, enf in enumerate(enfermedades) if enf in top_20_enfermedades]
matriz_top20 = matriz[np.ix_(indices_top20, indices_top20)]
nombres_top20 = [enfermedades[i] for i in indices_top20]

plt.figure(figsize=(16, 14))
sns.heatmap(
    matriz_top20,
    annot=True,
    fmt='d',
    cmap='YlOrRd',
    xticklabels=nombres_top20,
    yticklabels=nombres_top20,
    cbar_kws={'label': 'Cantidad de predicciones'},
    linewidths=0.5,
    linecolor='gray'
)

plt.title(f'{dataset_nombre} | Matriz de Confusión - Top 20\n' +
          f'Accuracy: {accuracy*100:.2f}% | Test: {len(X_test)} muestras | K=1', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Predicción', fontsize=12, fontweight='bold')
plt.ylabel('Real', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()

plt.savefig(f'../resultados/{dataset_id}/graficos/05_matriz_confusion_top20.png', dpi=200, bbox_inches='tight')
plt.close()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

top20 = df_metricas.head(20)
ax1.barh(range(len(top20)), top20['accuracy'], color='green', alpha=0.7, edgecolor='black')
ax1.set_yticks(range(len(top20)))
ax1.set_yticklabels([enf[:35] for enf in top20['enfermedad']], fontsize=8)
ax1.set_xlabel('Accuracy (%)', fontsize=10)
ax1.set_title('Top 20: Enfermedades con Mejor Accuracy', fontweight='bold', fontsize=11)
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3, axis='x')
ax1.axvline(x=accuracy*100, color='red', linestyle='--', linewidth=2, alpha=0.7, 
            label=f'Promedio: {accuracy*100:.1f}%')
ax1.legend()

ax2.hist(df_metricas['accuracy'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
ax2.axvline(x=accuracy*100, color='red', linestyle='--', linewidth=2, 
            label=f'Promedio: {accuracy*100:.1f}%')
ax2.set_xlabel('Accuracy (%)', fontsize=10)
ax2.set_ylabel('Número de enfermedades', fontsize=10)
ax2.set_title('Distribución de Accuracy', fontweight='bold', fontsize=11)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

if n_enf_cero > 0:
    ax2.text(5, ax2.get_ylim()[1]*0.85, f'{n_enf_cero} enfermedades\ncon 0% accuracy',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7),
             fontsize=10, fontweight='bold')

plt.suptitle(f'{dataset_nombre} | Análisis de Accuracy', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'../resultados/{dataset_id}/graficos/06_accuracy_por_enfermedad.png', dpi=200, bbox_inches='tight')
plt.close()