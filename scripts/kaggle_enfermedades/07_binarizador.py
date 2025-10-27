"""

Binarización de datos discretos siguiendo el patrón especificado en 
el análisis.

"""

import pandas as pd

df = pd.read_csv('../../datasets/kaggle_enfermedades/silver/03_discretizado_10comp.csv')

X = df.drop('prognosis', axis=1)
Y = df['prognosis']

def binarizar_valor(valor):
    if valor == 0:
        return [1,0]
    elif valor == 1:
        return [1,1]
    elif valor == 2:
        return [0,1]
    else:
        raise ValueError(f"Valor inesperado: {valor}. Se espera 0, 1, o 2.")
    
data = {}

for col in X.columns:

    col_b1 = []
    col_b2 = []

    for valor in X[col]:
        bits = binarizar_valor(valor)
        col_b1.append(bits[0])
        col_b2.append(bits[1])
    
    data[f'{col}_b1'] = col_b1
    data[f'{col}_b2'] = col_b2

print("Binarización completada.")

df = pd.DataFrame(data)

df['prognosis'] = Y.values

df.to_csv('../../datasets/kaggle_enfermedades/gold/kaggle_dataset.csv', index=False)
print(df.shape)