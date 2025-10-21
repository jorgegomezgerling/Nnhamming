"""
Evaluación 4: Sensibilidad a cantidad de patrones.
Evalua como el tamaño del conjunto de entrenamiento afecta el accuracy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys

sys.path.append("../src")
from Nnhamming import Nnhamming

df = pd.read_csv('../dataset/gold/kaggle_dataset.csv')
X = df.drop('prognosis', axis=1)
Y = df['prognosis']

X_train_full, X_test, Y_train_full, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=42
)

print(f"\nTrain completo: {len(X_train_full)} | Test (fijo): {len(X_test)}")

print("ENTRENANDO CON DIFERENTES TAMAÑOS DE TRAIN")

porcentajes = [0.25, 0.50, 0.75, 1.0]
resultados = []

for porcentaje in porcentajes:
    print(f"Entrenando con {porcentaje*100:.0f}% de train ({int(len(X_train_full)*porcentaje)} prototipos)")

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
    # Preparar DataFrame para entrenar
    
    train_df = X_train_sub.copy()
    train_df['prognosis'] = Y_train_sub.values
    
    # Entrenar red
    print(f"  Entrenando red...")
    red = Nnhamming()
    red.fit_from_df(train_df)
    print(f"Red entrenada con {len(red.prototipos)} prototipos")
    
    # Evaluar en test (SIEMPRE el mismo)
    print(f"  Evaluando en test...")
    aciertos = 0
    
    for i in range(len(X_test)):
        if i % 100 == 0:
            print(f"    Progreso: {i}/{len(X_test)}")
        
        vector = X_test.iloc[i].values.tolist()
        real = Y_test.iloc[i]
        
        prediccion = red.predict(vector, k=1)
        predicho = prediccion[0][0]
        
        if predicho == real:
            aciertos += 1
    
    accuracy = aciertos / len(X_test)
    
    resultados.append({
        'porcentaje': porcentaje * 100,
        'n_prototipos': len(red.prototipos),
        'accuracy': accuracy * 100
    })
    
    print(f"Accuracy: {accuracy*100:.2f}%")

resultados_df = pd.DataFrame(resultados)

print(f"\n{'='*70}")
print("RESUMEN DE RESULTADOS")
print(f"{'='*70}")
print(resultados_df.to_string(index=False))

import os
os.makedirs('../resultados', exist_ok=True)

print(f"\n{'='*70}")
print("GUARDANDO RESULTADOS")
print(f"{'='*70}")

# Guardar CSV
resultados_df.to_csv('../resultados/sensibilidad_patrones.csv', index=False)
print(f"Resultados guardados: resultados/sensibilidad_patrones.csv")

# Guardar métricas
with open('../resultados/sensibilidad_patrones.txt', 'w') as f:
    f.write("SENSIBILIDAD A CANTIDAD DE PATRONES\n")
    f.write("="*50 + "\n\n")
    for _, row in resultados_df.iterrows():
        f.write(f"{row['porcentaje']:5.0f}% train ({row['n_prototipos']:4.0f} prototipos) → Accuracy: {row['accuracy']:5.2f}%\n")
    
    mejor_idx = resultados_df['accuracy'].idxmax()
    mejor = resultados_df.iloc[mejor_idx]
    f.write(f"Mejor resultado: {mejor['porcentaje']:.0f}% train → {mejor['accuracy']:.2f}% accuracy\n")

print(f"Métricas guardadas: resultados/sensibilidad_patrones.txt")

# Visualización
plt.figure(figsize=(10, 6))
plt.plot(resultados_df['porcentaje'], resultados_df['accuracy'], 
         marker='o', linewidth=2.5, markersize=10, color='darkblue', label='Accuracy')
plt.fill_between(resultados_df['porcentaje'], resultados_df['accuracy'], 
                 alpha=0.2, color='darkblue')

# Marcar el mejor punto
mejor_idx = resultados_df['accuracy'].idxmax()
mejor = resultados_df.iloc[mejor_idx]
plt.scatter(mejor['porcentaje'], mejor['accuracy'], 
           s=200, color='gold', edgecolors='red', linewidths=2, zorder=5,
           label=f'Óptimo: {mejor["porcentaje"]:.0f}%')

# Anotar valores
for _, row in resultados_df.iterrows():
    plt.text(row['porcentaje'], row['accuracy'] + 0.5, 
             f"{row['accuracy']:.2f}%\n({row['n_prototipos']:.0f})", 
             ha='center', fontsize=9, fontweight='bold')

plt.xlabel('Porcentaje de Datos de Entrenamiento (%)', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Sensibilidad a Cantidad de Patrones de Entrenamiento', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(resultados_df['porcentaje'])
plt.ylim(15, 30)

plt.tight_layout()
plt.savefig('../resultados/sensibilidad_patrones.png', dpi=150, bbox_inches='tight')
print(f"Gráfico guardado: resultados/sensibilidad_patrones.png")

plt.close()

print(f"\n{'='*70}")
print("Evaluación completada")
print(f"{'='*70}\n")