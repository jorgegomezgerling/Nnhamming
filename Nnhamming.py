import pandas as pd
from numpy import argsort

class Nnhamming:
    """
    Clase principal. Con los métodos iniciales básicos para una red de Hamming.
    """
    def __init__(self):
        """
        Inicializa clase. Con prototipos y etiquetas como listas vacías.
        """
        self.prototipos = [] 
        self.etiquetas = []      

    def add_prototype(self, vector, etiqueta):
        """    
        Añade el vector y la etiqueta a las listas de clase correspondientes.
        """
        self.prototipos.append(vector)
        self.etiquetas.append(etiqueta)

    def fit_from_df(self, df):
        """
        Carga desde el data frame (df).

        Esta funcion toma el dataframe del argumento como input 
        y con pandas crea dos listas. Una para síntomas (prototypes),
        otra para nombre de enfermedades (etiquetas).
        Para cada elemento de las listas crea un vector bipolar y 
        se añade mediante add_prototype a las listas de la clase.

        Args:
            df (DataFrame): DataFrame sin procesar datos crudos.
        """
        prototypes = df.drop('prognosis', axis=1).apply(pd.to_numeric).values.tolist()
        etiquetas = df['prognosis'].values.tolist()

        for prototype, etiqueta in zip(prototypes, etiquetas):
            prototype_bipolar = [1 if x == 1 else -1 for x in prototype]
            self.add_prototype(prototype_bipolar, etiqueta)
    
    def calculate_distance(self, vector1, vector2):
        """
        Calcula distancia de vectores a través del producto punto de los mismos.
        Cada coincidencia suma +1, cada diferencia suma -1.

        Args:
            vector1 (vector): vector bipolar de síntomas.
            vector2 (vector): vector bipolar de síntomas.

        Returns:
            distance (int): distancia de hamming. 
        """
        escalar = sum([v1*v2 for v1, v2 in zip(vector1, vector2)])
        n = len(vector1)
        distance = (n - escalar) // 2
        return distance


    def predict(self, vector, k=1):
        """
        Función que toma un vector y un parametro k.
        Convierte el vector a vector bipolar.
        Calcula cada activación para cada vector.

        Args:
            vector (vector): vector a comparar.
            k (escalar): número de posibles candidatos. Por default establecido en el candidato más fuerte = 1.
        
        Returns:
            lista_candidatos (list): enfemerdades con su nivel de confianza.
        """

        vector_bipolar = [1 if x == 1 else -1 for x in vector]

        if len(vector) != len(self.prototipos[0]):
            return None
        
        # Calcular cada activación para cada vector.
        activaciones = [len(p) - self.calculate_distance(vector_bipolar, p) 
                        for p in self.prototipos]
        
        # Orden de mayor a menor según activación.
        indices_ordenados = sorted(range(len(activaciones)), 
                                key=lambda i: activaciones[i], 
                                reverse=True)
        
        # Se utiliza min para asegurarse de que k no exceda el límite de los índices.
        k = min(k, len(indices_ordenados))
        lista_candidatos = []
        for i in indices_ordenados[:k]: # Los k primeros.
            etiqueta = self.etiquetas[i] 
            confianza = activaciones[i] / len(vector)
            lista_candidatos.append((etiqueta, confianza))

        return lista_candidatos







    


