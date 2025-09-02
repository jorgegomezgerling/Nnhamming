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
        
# 💡 Pistas para pasar de 1-NN a k-NN

             

def clasificar(df, vector):
    df_modif = df.drop('enfermedad', axis=1)
    df_numeric = df_modif.apply(pd.to_numeric)
    valores = df_numeric.values.tolist()
    dic_enf = {}
    largo_df = len(df_modif.columns)

    for index, values in enumerate(valores):
        new_value = calculate_distance(values, vector) # Acordate que aca va Nnhamming despues como clase.
        dic_enf[df.iloc[index]['enfermedad']] = new_value
    
    dic_enf = list(sorted(dic_enf.items(), key=lambda item: item[1]))[:3]

    enf1, prob1 = dic_enf[0][0], (largo_df - dic_enf[0][1])/len(df_modif.columns)
    enf2, prob2 = dic_enf[1][0], (largo_df - dic_enf[1][1])/len(df_modif.columns)
    enf3, prob3 = dic_enf[2][0], (largo_df - dic_enf[2][1])/len(df_modif.columns)

    lista = [(enf1, prob1), (enf2, prob2), (enf3, prob3)]

    return lista


# Revisá tu función actual
# Hoy seguramente calculás la distancia de Hamming de un vector de entrada contra cada fila del dataset.
# Después tomás el mínimo y te quedás con ese índice.
# Cambio conceptual
# En vez de quedarte solo con el mínimo, ahora tenés que:
# Ordenar todas las distancias.
# Tomar los primeros k elementos más chicos.
nuevo_paciente1 = [0, 0, 0, 0, 0]
enfermedades_prob = clasificar(df, nuevo_paciente1)

print(f'Las probabilidades son: {enfermedades_prob[0][0]} un % {enfermedades_prob[0][1]*100}')
print(f'Las probabilidades son: {enfermedades_prob[1][0]} un % {enfermedades_prob[1][1]*100}')
print(f'Las probabilidades son: {enfermedades_prob[2][0]} un % {enfermedades_prob[2][1]*100}')














# Caso 1: muy parecido al primer registro (gripe)
# nuevo_paciente1 = [1, 1, 1, 1, 0] # CHECK!
# # debería dar: gripe

# # Caso 2: parecido al segundo registro (gripe)
# nuevo_paciente2 = [1, 1, 0, 1, 0] # CHECK!
# # debería dar: gripe

# # Caso 3: parecido al tercer registro (resfrio)
# nuevo_paciente3 = [0, 1, 1, 0, 0] # CHECK!
# # debería dar: resfrio

# # Caso 4: parecido al cuarto registro (resfrio)
# nuevo_paciente4 = [0, 1, 0, 0, 0] # CHECK!
# # debería dar: resfrio

# # Caso 5: parecido al quinto registro (migrania)
# nuevo_paciente5 = [0, 0, 0, 0, 1] # CHECK!
# # debería dar: migrania

# # Caso 6: parecido al sexto registro (migrania)
# nuevo_paciente6 = [0, 0, 0, 1, 1] # CHECK!
# # debería dar: migrania

# # Caso 7: ambiguo (mezcla de síntomas de gripe y resfrio)
# nuevo_paciente7 = [1, 1, 1, 0, 0] 
# # probablemente de "gripe" porque está a distancia 1 del primer registro
