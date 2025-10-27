import pandas as pd
from sklearn.decomposition import PCA

raw_df = pd.read_csv('../../datasets/kaggle_enfermedades/bronze/kaggle_dataset.csv')

if 'Unnamed: 0' in raw_df.columns:
    raw_df = raw_df.drop('Unnamed: 0', axis=1)

X = raw_df.drop('prognosis', axis=1)
Y = raw_df['prognosis']

N_COMPONENTES = 100

pca_final = PCA(n_components=N_COMPONENTES)
X_pca = pca_final.fit_transform(X)

silver_df = pd.DataFrame(X_pca, 
                         columns=[f'Comp_{i+1}' for i in range(N_COMPONENTES)])

silver_df['prognosis'] = Y.values

silver_df.to_csv('../../datasets/kaggle_enfermedades/silver/01_pca_100comp.csv', index=False)