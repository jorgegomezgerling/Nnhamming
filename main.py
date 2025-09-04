from Nnhamming import *
import random

df = pd.read_csv('kaggle_dataset.csv')
# Aca hay algo mal, no deberia ser dos veces leer el df. 
red = Nnhamming()
red.fit_from_df(df)
vector = []

for i in range(0, 401):
    vector.append(random.randint(0,1))

print(len(red.prototipos))
print(len(red.etiquetas))

print("Prueba 1:", red.predict(vector, k=3))