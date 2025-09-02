import pandas as pd
provisional_dataset = 'dataset.csv'
df = pd.read_csv(provisional_dataset)

class Nnhamming:
    @staticmethod
    def calculate_distance(vector1, vector2):
        distance = 0
        if len(vector1) == len(vector2):
            for i in range(len(vector1)):
                if vector1[i] != vector2[i]:
                    distance += 1
        return distance
    
    @staticmethod
    def clasificar(df, vector):
        df_modif = df.drop('enfermedad', axis=1)
        df_numeric = df_modif.apply(pd.to_numeric)
        valores = df_numeric.values.tolist()
        dic_enf = {}
        largo_df = len(df_modif.columns)

        for index, values in enumerate(valores):
            new_value = Nnhamming.calculate_distance(values, vector) 
            dic_enf[df.iloc[index]['enfermedad']] = new_value
        
        lista = list(sorted(dic_enf.items(), key=lambda item: item[1]))[:3]

        return lista


nuevo_paciente7 = [1, 1, 1, 0,1] 
print(Nnhamming.clasificar(df, nuevo_paciente7))


