import pandas as pd
from sklearn.decomposition import PCA

raw_df = pd.read_csv('../dataset/bronze/kaggle_dataset.csv')

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

silver_df.to_csv('../dataset/silver/01_pca_100comp.csv', index=False)

print(f"Dataset SILVER creado correctamente.")
print(f"Shape original: {X.shape}")
print(f"Shape final: {silver_df.shape}")
print(f"Reducción: {X.shape[1]} → {N_COMPONENTES} columnas ({N_COMPONENTES/X.shape[1]*100:.1f}%)")
varianza_capturada = pca_final.explained_variance_ratio_.sum()
print(f"Varianza capturada: {varianza_capturada*100:.2f}%")