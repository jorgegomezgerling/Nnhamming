import pandas as pd
df = pd.read_csv('prototypes.csv')

class Nnhamming:
    def __init__(self):
        self.prototipos = [] 
        self.etiquetas = []      

    def add_prototype(self, vector, etiqueta):
        self.prototipos.append(vector)
        self.etiquetas.append(etiqueta)

    def fit_from_df(self, df):
        prototypes = df.drop('enfermedad', axis=1).apply(pd.to_numeric).values.tolist()
        etiquetas = df['enfermedad'].values.tolist()

        for prototype, etiqueta in zip(prototypes, etiquetas):
            self.add_prototype(prototype, etiqueta)
    
    def calculate_distance(self, vector1, vector2):
        distance = 0
        if len(vector1) == len(vector2):
            for i in range(len(vector1)):
                if vector1[i] != vector2[i]:
                    distance += 1
        return distance
    
    def predict(self, vector, k=1):
        enfermedades_cantidatas = []

        if k > len(self.prototipos):
            k = len(self.prototipos)

        if len(vector) != len(self.prototipos[0]):
            return None

        for prototype, etiqueta in zip(self.prototipos, self.etiquetas):
            distance = self.calculate_distance(vector, prototype)
            enfermedades_cantidatas.append((etiqueta, distance))

        return sorted(enfermedades_cantidatas, key=lambda item: item[1])[:k]
        


vector_prueba = [0, 1, 1, 1, 0]
red = Nnhamming()
red.fit_from_df(df)
print(red.predict(vector_prueba))







    


