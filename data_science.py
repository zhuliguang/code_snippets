import pandas as pd
import numpy as np

def get_missing_value(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100*mis_val/len(df)
    mis_table = pd.concat([mis_val,mis_val_percent],axis=1)
    mis_table = mis_table.rename(columns = {0:'Missing Values counts',1:'% Missing'})
    return mis_table.sort(['% Missing'], ascending=[0])

def get_categoricals(df):
    return df.select_dtypes(include='object')

def get_numericals(df):
    return df.select_dtypes(include='number')

def get_categorical_columns(df):
    return list(df.dtypes[df.dtypes == "object"].index)

def get_numerical_columns(df):
    return list(df.dtypes[df.dtypes != "object"].index)

def granularity(df):
    return pd.DataFrame(df[get_categorical_columns(df)].apply(pd.Series.unique, axis=0)).rename(columns = {0:"Granularity"})

def mi_categorical(df, columns):
    from scipy.stats import chi2_contingency
    if len(columns) != 2:
        raise "needs to have two columns"
    ct = pd.crosstab(df[columns[0]], df[columns[1]])
    g,p,dof,expected = chi2_contingency( ct )
    mi = 0.5*g / ct.sum() 
def get_auc(clf, X_test, y_test):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve, auc
    test_prob = clf.predict_proba(X_test)
    print("testing ", roc_auc_score(y_test, test_prob[:,1]))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_test, test_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), test_prob[1].ravel())
        # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr[1], tpr[1], color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
