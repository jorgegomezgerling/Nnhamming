"""

Aplicación de Mutual Information para selección de componentes
Selecciona las top 10 componentes con mayor score de MI

"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif

silver_df = pd.read_csv('../dataset/silver/01_pca_100comp.csv')
X = silver_df.drop('prognosis', axis=1)
Y = silver_df['prognosis']

le = LabelEncoder()
y_encoded = le.fit_transform(Y)

scores = mutual_info_classif(X, y_encoded, random_state=42)

mutual_information = pd.DataFrame({
    'Componente': X.columns,
    'Score': scores,
})

mutual_information = mutual_information.sort_values("Score", ascending=False)

N_COMPONENTES = 10
top_componentes = mutual_information["Componente"].head(N_COMPONENTES).tolist()

df_reducido = silver_df[top_componentes + ['prognosis']]

df_reducido.to_csv('../dataset/silver/02_mutual_10comp.csv', index=False)