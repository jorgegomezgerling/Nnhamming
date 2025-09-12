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

print(lista_enfermedades)
