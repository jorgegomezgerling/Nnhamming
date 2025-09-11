from Nnhamming import *
import random

df = pd.read_csv('kaggle_dataset.csv')
red = Nnhamming()
red.fit_from_df(df)
vector_sintomas = []

for i in range(0, 401):
    vector_sintomas.append(random.randint(0,1))

# print(len(red.prototipos))
# print(len(red.etiquetas))

lista_enfermedades =  red.predict(vector_sintomas, k=3)

for i, (enfermedad, confianza) in enumerate(lista_enfermedades):
    print(f'La enfermedad posible en la {i + 1} posición es: {enfermedad}')
    print(f'Nivel de confianza: {confianza:.2f}')
    print("*"*5)
