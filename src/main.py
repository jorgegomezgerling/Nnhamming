from Nnhamming import *
import random

df = pd.read_csv('../dataset/gold/kaggle_dataset.csv') 
red = Nnhamming()
red.fit_from_df(df)

def cargar_vector_alteatorio(vector):
    for i in range(0, 20):
        vector.append(random.randint(0,1))
    return vector

def prueba_1(lista):
    for indice, (etiqueta, nivel) in enumerate(lista):
        print()
        print(f'La enfermedad en la posición: {indice + 1} es: {etiqueta} con un nivel de coincidencia normalizado de: {nivel}')

# Primeramente probamos con un vector de síntomas aleatorios:

print("Prueba 1:")

vector = []

prueba_1(red.predict(cargar_vector_alteatorio(vector), k=3))