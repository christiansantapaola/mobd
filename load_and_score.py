from cmath import nan
import math
from multiprocessing.sharedctypes import Value
from operator import eq
import re
from site import makepath
from pyparsing import col
import scipy
import sklearn
import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
# from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.svm import SVC, LinearSVC
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn import tree
import graphviz
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from pprint import pprint
from imblearn.pipeline import Pipeline, make_pipeline
import pickle

def score_model(model, x, y):
    """score_model(model, x, y) score a model on the testing set x with class y"""
    score = {'Accuracy': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0}
    y_pred = model.predict(x)
    score['Accuracy'] = accuracy_score(y, y_pred)      
    score['Precision'] = precision_score(y, y_pred)
    score['Recall'] = recall_score(y, y_pred)
    score['F1'] = f1_score(y, y_pred)
    return score


def load_and_score(model_path='model.pkl', test_path='test_set.csv'):
    """load_and_score(model_path : str, test_path : str) will load a machine learning model from a pickle file and score it on a test set stored in the csv_path."""
    # open file, load it with pickle, load test set, then predict and score performance.
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
        test_set = pd.read_csv(test_path)
        x = test_set.iloc[:, :-1].to_numpy()
        y = test_set.iloc[:, -1].to_numpy()
        score = score_model(model, x, y)
        return score


if __name__ == '__main__':
    score = load_and_score()
    pprint(score)