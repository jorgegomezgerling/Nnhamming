"""

Discretizador: toma datos de silver dataset y los discretiza con
la estrategia seleccionada en el análisis previo.

"""

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

df = pd.read_csv('../dataset/silver/02_mutual_10comp.csv')

X = df.drop('prognosis', axis=1)
Y = df['prognosis']


discretizador = KBinsDiscretizer(
    n_bins=3,
    encode='ordinal', # Solo un numero por bin
    strategy='quantile'
)

X_discretizado = discretizador.fit_transform(X)

df_discretizado = pd.DataFrame(
    X_discretizado,
    columns=X.columns
)

df_discretizado['prognosis'] = Y.values

df_discretizado.to_csv('../dataset/silver/03_discretizado_10comp.csv', index=False)