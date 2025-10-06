import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from kneed import KneeLocator #Falló la implementación del kneelocator.

raw_df = pd.read_csv('./dataset/bronze/kaggle_dataset.csv')
sintomas_df = raw_df.drop('prognosis', axis=1)
enfermedades_df = raw_df['prognosis']

# print(raw_df.head())

le = LabelEncoder()

y_encoded = le.fit_transform(enfermedades_df)

chi_scores, p_values = chi2(sintomas_df, y_encoded)

chi2_resultados = pd.DataFrame({
    'Sintoma': sintomas_df.columns,
    'Chi2': chi_scores,
    'p_value': p_values
})

chi2_resultados = chi2_resultados.sort_values(by='Chi2', ascending=False)
# print(chi2_resultados.head())

# Visualizamos la pendiente de corte posible:

plt.figure(figsize=(10,5))
plt.plot(range(len(chi2_resultados)), chi2_resultados['Chi2'].values, marker='o', linewidth=1)
plt.title('Distribución de valores chi2 por síntoma ')
plt.xlabel('Sintomas ordenados por importancia')
plt.ylabel('Valor chi2')
plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()

# Calculamos el punto óptimo de corte (punto codo) donde la pendiente cae abruptamente.

valores = chi2_resultados['Chi2'].values

inicio_corte = 15
final_corte = 100

caidas_absolutas = -np.diff(valores)

caidas_busqueda = caidas_absolutas[inicio_corte:final_corte]

indice_local_mayor_caida = np.argmax(caidas_busqueda)

punto_codo_real = inicio_corte + indice_local_mayor_caida + 1


print(f"El punto codo está en el índice: {punto_codo_real}")
print(f"Por lo tanto se deben conservar los primeros: {punto_codo_real} sintomas")


# caidas_busqueda = caidas_absolutas[13:50]
# lista = []

# for i in range(1,100):
#     caida_busqueda = caidas_absolutas[i:100]
#     punto_codo_heuristico = np.argmax(caida_busqueda) + 1
#     lista.append(punto_codo_heuristico.item())

# print(lista)

# Resultado:

# [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 1, 46, 45, 44, 43, 
# 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 
# 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 
# 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 11, 10, 9, 8, 7, 6, 5, 
# 4, 3, 2, 1, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 
# 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    





