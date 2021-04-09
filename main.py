import pandas as pd
from data_analyse import analysis
from transform_data import transform
from train import training
from result_and_plot import results

def reading():
    '''
    Reading the dataset
    '''
    df = pd.read_csv('database.csv', usecols=[*range(0, 10000), *range(400000, 410000), *range(790000, 800003)])
    print(df.head())

    return df

if __name__ == "__main__":
    print("Carregando os dados...")
    df = reading()

    print("\nAnálise da estrutura dos dados: ")
    analysis(df)

    print("\nTransformações necessárias no dataset: ")
    df_train, df_test = transform(df)

    print("\nTreinando o modelo...: ")
    prediction, y_test = training(df_train, df_test)

    print("\nAnálise de resultados: ")
    results(y_test, prediction)
