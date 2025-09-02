import pandas as pd, numpy as np

provisional_dataset = 'dataset.csv'

df = pd.read_csv(provisional_dataset)

vector1 = [0,1,1,0,1] 
vector2 = [1,0,0,0,1]
vector3 = [0,1,0,1,0]
vector4 = [0,0,0,0,1]
vector5 = [0,1,1,1,0]

def calculate_distance(vector1, vector2):
        """
        Valida que los vectores sean de igual tamanio,
        posteriormente, compara valor a valor y devuelve la 
        distancia total de bits entre vectores.
        """
        distance = 0
        if len(vector1) == len(vector2):
                for i in range(len(vector1)):
                    if vector1[i] != vector2[i]:
                           distance +=1
                return distance
                
def clasificar(df, vector):
    df_modif = df.drop('enfermedad', axis=1)
    df_numeric = df_modif.apply(pd.to_numeric)
    valores = df_numeric.values.tolist()
    criterio = 999999999
    selected_index = -1

    for index, values in enumerate(valores):
        new_value = calculate_distance(values, vector)
        if new_value < criterio:
            criterio = new_value
            selected_index = index       

    if selected_index != -1:
        enfermedad = df.iloc[selected_index]['enfermedad']
    
    return enfermedad

print(clasificar(df, vector2))
