import pandas as pd

df = pd.read_csv('../dataset/gold/kaggle_dataset.csv')
casos_por_enfermedad = df['prognosis'].value_counts()

n_enfermedades = df['prognosis'].nunique()
print(f"Enfermedades únicas: {n_enfermedades}")

print(f"Total filas: {len(df)}")
print(f"Top 10 enfermedades más comunes:")
print(casos_por_enfermedad.head(10))
print(f"10 enfermedades menos comunes:")
print(casos_por_enfermedad.tail(10))