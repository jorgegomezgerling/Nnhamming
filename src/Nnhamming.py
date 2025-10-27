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
            df (DataFrame): DataFrame (datos)
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


    def predict(self, vector, k=1, epsilon_factor=1.0, return_iterations=False):
        """
        Función que toma un vector y un parametro k.
        Convierte el vector a vector bipolar.
        Calcula cada activación para cada vector.

        Args:
            vector (vector): vector a comparar.
            k (escalar): número de posibles candidatos. Por default establecido en el candidato más fuerte = 1.
            epsilon_factor (float): multiplicador del epsilon para ajustar inhibición (default=1.0)
            return_iterations (bool): si True, devuelve también el número de iteraciones usadas (default=False)
        
        Returns:
            Si return_iterations=False: lista de tuplas [(enfermedad, confianza), ...]
            Si return_iterations=True: tupla (lista, iteraciones_usadas)
        """
        vector_bipolar = [1 if x == 1 else -1 for x in vector]
        if len(vector) != len(self.prototipos[0]):
            return None

        activaciones = [len(p) - self.calculate_distance(vector_bipolar, p)
                        for p in self.prototipos]

        M = len(activaciones)
        epsilon = epsilon_factor / (M + 1)

        iteraciones_usadas = 0
        
        for iteracion in range(20):
            iteraciones_usadas = iteracion + 1
            
            nuevas = activaciones[:]
            for i in range(M):
                inhibicion = epsilon * (sum(activaciones) - activaciones[i])
                nuevas[i] = max(0, activaciones[i] - inhibicion)

            activaciones = nuevas

            if sum(a > 0 for a in activaciones) == 1:
                break
        
        # # Verificar si todos los candidatos fueron eliminados
        # activos = sum(a > 0 for a in activaciones)
        # if activos == 0 and epsilon_factor >= 2.0:  #Solo alertar con epsilon alto
        #     print(f"WARNING: Epsilon {epsilon:.6f} (factor={epsilon_factor}) eliminó todos los candidatos")

        indices_ordenados = sorted(range(M), key=lambda i: activaciones[i], reverse=True)
        k = min(k, len(indices_ordenados))
        
        resultados = [(self.etiquetas[i], activaciones[i] / len(vector)) for i in indices_ordenados[:k]]
        
        if return_iterations:
            return resultados, iteraciones_usadas
        else:
            return resultados







    


