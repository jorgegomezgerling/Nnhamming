from Nnhamming import *

df = pd.read_csv('prototypes.csv')

red = Nnhamming()
red.fit_from_df(df)

vector_prueba = [1, 1, 1, 1, 0]  # fiebre + tos + dolor garganta + fatiga
print("Prueba 1:", red.predict(vector_prueba, k=3))

vector_prueba2 = [0, 0, 0, 1, 1]  # fatiga + dolor cabeza
print("Prueba 2:", red.predict(vector_prueba2, k=3))

vector_prueba3 = [1, 0, 0, 1, 1]  # fiebre + fatiga + dolor cabeza
print("Prueba 3:", red.predict(vector_prueba3, k=3))