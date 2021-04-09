
def analysis(df):
    df.info()
    print("Informações sobre a base de dados: ", df.describe())

    target_count = df.target.value_counts()
    print("Número de entradas de cada classe: (Ocorre ou não ocorre descargas parciais)")
    print('Classe Não Ocorre:', target_count[0])
    print('Classe Ocorre:', target_count[1])

    target_count.plot(kind='bar', title='Count (target)');
