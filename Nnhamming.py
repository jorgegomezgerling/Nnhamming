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
        criterio = float("inf")
        selected_index = -1

        for index, values in enumerate(valores):
            new_value = Nnhamming.calculate_distance(values, vector)
            if new_value < criterio:
                criterio = new_value
                selected_index = index       

        if selected_index != -1:
            return df.iloc[selected_index]['enfermedad']
        return None


nuevo_paciente = [1, 0, 1, 0, 0]
resultado = Nnhamming.clasificar(df, nuevo_paciente)
print("El paciente tiene:", resultado)
