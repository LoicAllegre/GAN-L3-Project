import pickle 
import pandas as pd
from scipy.stats import randint
import numpy as np
import string
import time
import base64
import sys
import csv 


# librairie affichage
import matplotlib.pyplot as plt
import seaborn as sns


import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets

from scipy.stats import norm

from imblearn.over_sampling import SMOTE



def MyshowAllScores(y_test,y_pred):
  classes= np.unique(y_test)
  print("Accuracy : %0.3f"%(accuracy_score(y_test,y_pred)))
  print("Classification Report")
  print(classification_report(y_test,y_pred,digits=5))    
  cnf_matrix = confusion_matrix(y_test,y_pred)
  plot_confusion_matrix(cnf_matrix, classes)


# chargement du jeu de données à partir de l'URL
url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['SepalLengthCm', 'SepalWidthCm', 
         'PetalLengthCm', 'PetalWidthCm', 
         'Species']

# creation d'un dataframe pour récupérer les données
df_iris = pd.read_csv(url, names=names)


def df_to_nparray(df):
    array = df.values #necessité de convertir le dataframe en numpy
    X = array[:,0:4] 
    y = array[:,4]

    return X, y



def generate_and_train_model(trainSize, testSize, df):

    X, y = df_to_nparray(df)

    model = GaussianNB()
    seed = 10
    X_train,X_test,y_train,y_test=train_test_split(X, 
                                               y, 
                                               train_size=trainSize, 
                                               random_state=seed,
                                               test_size=testSize)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    #MyshowAllScores(y_test, y_pred)

    return model, accuracy_score(y_test,y_pred)


def resample_gaussian(df, N):

    df_trunc = df.values[:150-N]

    classification = 'Iris-virginica'

    mean_sepL, std_sepL = norm.fit([df_trunc[i][0] for i in range(100,150-N)])
    mean_sepW, std_sepW = norm.fit([df_trunc[i][1] for i in range(100,150-N)])
    mean_petL, std_petL = norm.fit([df_trunc[i][2] for i in range(100,150-N)])
    mean_petW, std_petW = norm.fit([df_trunc[i][3] for i in range(100,150-N)])

    data_sepL = np.random.normal(mean_sepL, std_sepL, N)
    data_sepW = np.random.normal(mean_sepW, std_sepW, N)
    data_petL = np.random.normal(mean_petL, std_petL, N)
    data_petW = np.random.normal(mean_petW, std_petW, N)

    new_data = np.array([[data_sepL[i], data_sepW[i], data_petL[i], data_petW[i], classification] for i in range(N)])

    iris = np.concatenate((df_trunc, new_data))
    df_iris = pd.DataFrame(iris, columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'])
    df_iris['SepalLengthCm'] = df_iris['SepalLengthCm'].astype(float)
    df_iris['SepalWidthCm'] = df_iris['SepalWidthCm'].astype(float)
    df_iris['PetalLengthCm'] = df_iris['PetalLengthCm'].astype(float)
    df_iris['PetalWidthCm'] = df_iris['PetalWidthCm'].astype(float)

    return df_iris



def resample_SMOTE(dfIris, N):

    iris = dfIris.values[:150-N]

    smote = SMOTE()
    
    df = pd.DataFrame(iris, columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'])
    df['SepalLengthCm'] = df['SepalLengthCm'].astype(float)
    df['SepalWidthCm'] = df['SepalWidthCm'].astype(float)
    df['PetalLengthCm'] = df['PetalLengthCm'].astype(float)
    df['PetalWidthCm'] = df['PetalWidthCm'].astype(float)
    
    dx = {'SepalLengthCm': df.SepalLengthCm, 'SepalWidthCm': df.SepalWidthCm, 'PetalLengthCm': df.PetalLengthCm, 'PetalWidthCm': df.PetalWidthCm}
    dy = {'Species': df.Species}
    X = pd.DataFrame (data=dx)
    y = pd.DataFrame (data=dy)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return pd.concat([X_resampled, y_resampled], axis = 1)



def pairplot_df(iris, filename):
    fig = plt.figure(figsize=(16,9))
    fig.subplots(2, 2)
    for i, col in enumerate(iris.columns[:4]):
        plt.subplot(2, 2, i+1)
        sns.kdeplot(iris.loc[iris['Species'] == 'Iris-setosa', col], shade=True, label='setosa')
        sns.kdeplot(iris.loc[iris['Species'] == 'Iris-versicolor', col], shade=True, label='versicolor')
        sns.kdeplot(iris.loc[iris['Species'] == 'Iris-virginica', col], shade=True, label='virginica')
        plt.xlabel('cm')
        plt.title(col)
        if i == 1:
            plt.legend(loc='upper right')
        else:
            plt.legend().remove()
    plt.savefig(filename, bbox_inches='tight')



def evaluate(iris_df, model):
    iris_array = iris_df.values #necessité de convertir le dataframe en numpy

    X = iris_array[:,0:4] 
    y = iris_array[:,4]

    y_pred = model.predict(X)
    return classification_report(y,y_pred,digits=5, output_dict=True)


def evaluate_training(train_df, test_df):
    X_train, y_train = df_to_nparray(train_df)
    X_test, y_test = df_to_nparray(test_df)

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    #MyshowAllScores(y_test, y_pred)

    return accuracy_score(y_test,y_pred)



model, accuracy = generate_and_train_model(0.7, 0.3, df_iris)

print(f'Model accuracy : {accuracy}')




original_report = evaluate(df_iris, model)
pairplot_df(df_iris, 'figs/Original.png')

gauss_reports = [{
    'IrisGenerated' : 0,
    'Accuracy' : original_report['accuracy'],
    'Precision' : original_report['macro avg']['precision'],
    'Recall' :  original_report['macro avg']['recall'],
    'TrainedModelAccuracy' : accuracy
}]
smote_reports = [{
    'IrisGenerated' : 0,
    'Accuracy' : original_report['accuracy'],
    'Precision' : original_report['macro avg']['precision'],
    'Recall' :  original_report['macro avg']['recall'],
    'TrainedModelAccuracy' : accuracy
}]


for N in range(1, 41):
    print(f'N = {N}')
    gaussian_df = resample_gaussian(df_iris, N)
    smote_df = resample_SMOTE(df_iris, N)

    pairplot_df(gaussian_df, f'figs/Gauss_{N}.png')
    pairplot_df(smote_df, f'figs/SMOTE_{N}.png')

    gaussian_report = evaluate(gaussian_df, model)
    smote_report = evaluate(smote_df, model)

    gauss_model_accuracy = evaluate_training(gaussian_df, df_iris)
    smote_model_accuracy = evaluate_training(smote_df, df_iris)

    gauss_reports.append({
        'IrisGenerated' : N,
        'Accuracy' : gaussian_report['accuracy'],
        'Precision' : gaussian_report['macro avg']['precision'],
        'Recall' :  gaussian_report['macro avg']['recall'],
        'TrainedModelAccuracy' : gauss_model_accuracy
    })
    smote_reports.append({
        'IrisGenerated' : N,
        'Accuracy' : smote_report['accuracy'],
        'Precision' : smote_report['macro avg']['precision'],
        'Recall' :  smote_report['macro avg']['recall'],
        'TrainedModelAccuracy' : smote_model_accuracy
    })


dialect = csv.excel
dialect.delimiter = ','



csvGaussFile =  open('gauss_reports.csv', 'w', newline='')
gaussWriter = csv.DictWriter(csvGaussFile, fieldnames=gauss_reports[0].keys(), dialect=dialect)
gaussWriter.writeheader()
gaussWriter.writerows(gauss_reports)
csvGaussFile.close()


csvSmoteFile =  open('smote_reports.csv', 'w', newline='')
smoteWriter = csv.DictWriter(csvSmoteFile, fieldnames=smote_reports[0].keys(), dialect=dialect)
smoteWriter.writeheader()
smoteWriter.writerows(smote_reports)
csvSmoteFile.close()

