from xgboost import XGBClassifier

def training(df_train_over, df_test):

    X_train = df_train_over.drop('Desc. Parciais', axis=1)
    y_train = df_train_over['Desc. Parciais']

    X_test = df_test.drop('Desc. Parciais', axis=1)
    y_test = df_test['Desc. Parciais']

    #Modelo escolhido: XGBoost
    model = XGBClassifier()
    model.fit(X_train, y_train)
    pred_xgboost = model.predict(X_test)

    return pred_xgboost, y_test