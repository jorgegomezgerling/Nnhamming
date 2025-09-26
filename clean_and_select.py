import pandas as pd
from sklearn.feature_selection import chi2

raw_df = pd.read_csv('kaggle_dataset.csv')
sintomas_df = raw_df.drop('prognosis', axis=1)
enfermedades_df = raw_df['prognosis']

