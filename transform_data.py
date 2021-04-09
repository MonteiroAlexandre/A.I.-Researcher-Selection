from sklearn.model_selection import train_test_split
import pandas as pd

def transform(df):
    new_df = df.copy()

    labels = ['not occur', 'occur']
    new_df['Desc. Parciais'] = pd.cut(df['target'], bins=2, labels=labels)

    new_df = new_df.drop(columns=['signal_id', 'phase'])

    df_train, df_test = train_test_split(new_df, test_size = 0.2, random_state = 42)

    target_count_split = df_train.target.value_counts()
    print('Classe Não Ocorre:', target_count_split[0])
    print('Classe Ocorre:', target_count_split[1])

    target_count_split.plot(kind='bar', title='Count (target)');

    # Class count
    count_class_notOccur, count_class_occur = df_train.target.value_counts()

    # Divide by class
    df_class_notOccur = df_train[df_train['target'] == 0]
    df_class_occur = df_train[df_train['target'] == 1]

    df_class_occur_over = df_class_occur.sample(count_class_notOccur, replace=True)
    df_train_over = pd.concat([df_class_notOccur, df_class_occur_over], axis=0)

    print('Random over-sampling:')
    print('Classe Não Ocorre:', df_train_over.target.value_counts()[0])
    print('Classe Ocorre:', df_train_over.target.value_counts()[1])

    df_train_over.target.value_counts().plot(kind='bar', title='Count (target)');

    return df_train_over, df_test