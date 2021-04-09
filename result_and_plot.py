from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

def results(y_test, pred_xgboost):
    print('Classification Report XGBoost: \n', classification_report(y_test, pred_xgboost))

    conf_mat = confusion_matrix(y_test, pred_xgboost)
    print('Confusion Matrix: \n', conf_mat)

    '''
    labels = ['Class NotOccur', 'Class Occur']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)   
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('Expected')
    plt.show()
    '''