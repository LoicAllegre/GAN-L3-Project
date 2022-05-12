import pandas as pd
import numpy as np
import string
import base64
import sys
import re
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score
import matplotlib.pyplot as plt

from CleanText import TextNormalizer, MyCleanText

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer 
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

 
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')


input_file = sys.argv[1]


def calculate_scores(y_test,y_pred):
    labels = [
        'Negative',
        'Positive'
    ]

    print("Accuracy : %0.3f"%(accuracy_score(y_test,y_pred)))
    print("Classification Report")
    print(classification_report(y_test,y_pred,digits=5))    
    cnf_matrix = confusion_matrix(y_test,y_pred)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=labels)
    plt.show()



model = pickle.load(open('GeneratedClassifierModel.pkl', 'rb'))


df=pd.read_csv(input_file, names=['sentence','sentiment', 'source'], header='infer',sep='\t', encoding='utf8')
X_test=df.sentence
y_test=df.sentiment

y_pred = model.predict(X_test)

# autres mesures et matrice de confusion
calculate_scores(y_test,y_pred)