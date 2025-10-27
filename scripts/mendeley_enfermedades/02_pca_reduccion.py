"""

PCA: Reducción de Dimensionalidad
Reduce de 74 features a 40 componentes principales

"""

import pandas as pd
from sklearn.decomposition import PCA
import os

df = pd.read_csv('../../datasets/mendeley_enfermedades/silver/00_clean_dataset.csv')

X = df.drop('prognosis', axis=1)
Y = df['prognosis']

N_COMPONENTES = 40

pca = PCA(n_components=N_COMPONENTES, random_state=42)
X_pca = pca.fit_transform(X)

varianza_explicada = pca.explained_variance_ratio_.sum()

columnas_pca = [f'Comp{i+1}' for i in range(N_COMPONENTES)]
df_pca = pd.DataFrame(X_pca, columns=columnas_pca, index=X.index)
df_pca['prognosis'] = Y.values

os.makedirs('../../datasets/mendeley_enfermedades/silver', exist_ok=True)
df_pca.to_csv('../../datasets/mendeley_enfermedades/silver/01_pca_40comp.csv', index=False)
