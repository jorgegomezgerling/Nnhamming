import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

df = pd.read_csv('../dataset/silver/02_mutual_10comp.csv')

X = df.drop('prognosis', axis=1)
Y = df['prognosis']


discretizador = KBinsDiscretizer(
    n_bins=3,
    encode='ordinal',
    strategy='quantile'
)

X_discretizado = discretizador.fit_transform(X)

df_discretizado = pd.DataFrame(
    X_discretizado,
    columns=X.columns
)

df_discretizado['prognosis'] = Y.values

df_discretizado.to_csv('../dataset/silver/03_discretizado_10comp.csv', index=False)
print(f"Archivo guardado: silver/03_discretizado_10comp.csv")
print(f"Shape: {df_discretizado.shape}")