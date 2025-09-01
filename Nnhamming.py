import pandas as pd

# provisional_dataset = 'dataset.csv'

# df = pd.read_csv(provisional_dataset)

# print(df)

class Nnhamming():

    def calculate_distance(vector1, vector2):
        distance = 0
        for i in vector1:
            for j in vector2:
                if i != j:
                    distance += 1
                    



