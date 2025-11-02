import pandas as pd

# Cargar dataset ORIGINAL (antes de todo procesamiento)
df = pd.read_csv('../datasets/kaggle_enfermedades/bronze/kaggle_dataset.csv')

# Contar casos por enfermedad
distribucion = df['prognosis'].value_counts().sort_values()

print("DISTRIBUCIÓN DE CASOS - DATASET ORIGINAL KAGGLE (BRONZE)")

print(f"\nTotal de enfermedades: {len(distribucion)}")
print(f"Total de muestras: {len(df)}")

print(f"\nMínimo de casos: {distribucion.min()}")
print(f"Máximo de casos: {distribucion.max()}")
print(f"Promedio: {distribucion.mean():.1f}")
print(f"Mediana: {distribucion.median():.1f}")

print(f"\nEnfermedades con menos de 10 casos: {(distribucion < 10).sum()}")
print(f"Enfermedades con menos de 15 casos: {(distribucion < 15).sum()}")

print("\n10 enfermedades con MENOS casos:")
print(distribucion.head(10))

print("\n10 enfermedades con MÁS casos:")
print(distribucion.tail(10))

print("\nEnfermedades con menos de 10 casos:")
if (distribucion < 10).sum() > 0:
    print(distribucion[distribucion < 10])
else:
    print("  Ninguna enfermedad tiene menos de 10 casos")