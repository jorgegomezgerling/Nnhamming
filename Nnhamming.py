import pandas as pd
from numpy import argsort
df = pd.read_csv('kaggle_dataset.csv')

class Nnhamming:
    def __init__(self):
        self.prototipos = [] 
        self.etiquetas = []      

    def add_prototype(self, vector, etiqueta):
        self.prototipos.append(vector)
        self.etiquetas.append(etiqueta)

    def fit_from_df(self, df):
        prototypes = df.drop('prognosis', axis=1).apply(pd.to_numeric).values.tolist()
        etiquetas = df['prognosis'].values.tolist()

        for prototype, etiqueta in zip(prototypes, etiquetas):
            prototype_bipolar = [1 if x == 1 else -1 for x in prototype]
            self.add_prototype(prototype_bipolar, etiqueta)
    
    def calculate_distance(self, vector1, vector2):
        """

        Calcula distancia de vectores a través del producto punto de los mismos.
        Cada coincidencia suma +1, cada diferencia suma -1.

        """
        escalar = sum([v1*v2 for v1, v2 in zip(vector1, vector2)])
        n = len(vector1)
        distance = (n - escalar) // 2
        return distance

        
    def predict(self, vector, k=1):
        vector_bipolar = [1 if x == 1 else -1 for x in vector]

        if len(vector) != len(self.prototipos[0]):
            return None

        if k > len(self.prototipos):
            k = len(self.prototipos)

        activaciones_lista = []

        for prototipo, etiqueta in zip(self.prototipos, self.etiquetas):
            distancia = self.calculate_distance(vector_bipolar, prototipo)
            activacion = len(prototipo) - distancia
            activaciones_lista.append(activacion)

        epsilon = 0.1
        max_iter = 100

        for _ in range(max_iter):
            nuevas = activaciones_lista[:]

            for i in range(len(activaciones_lista)):
                inhibicion = epsilon * (sum(activaciones_lista) - activaciones_lista[i])
                nueva_activacion = activaciones_lista[i] - inhibicion
                nuevas[i] = max(0, nueva_activacion)

            if activaciones_lista == nuevas:
                break

            activaciones_lista = nuevas

        indices_ordenados = sorted(range(len(activaciones_lista)), key=lambda i: activaciones_lista[i], reverse=True)

        top_k_indices = indices_ordenados[:k]

        lista_candidatos = []
        
        for j in top_k_indices:
            etiqueta = self.etiquetas[j]
            confianza = activaciones_lista[j] / len(vector)
            lista_candidatos.append((etiqueta, activaciones_lista[j], confianza))

        return lista_candidatos







    


