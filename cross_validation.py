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


def filterNan(x):
  """ filterNan() will return true if x is not a float or if is a float if is not nan."""
  if isinstance(x, float):
     return not math.isnan(x)
  else:
    return True

def get_category_features(data, category_features_index):
  """ get_category_features() will return a dictionary containing all the info about the categorical features in the dataset given their index."""
  category_features = {f'F{j}': {'feature': f'F{j}', 'len': len(data[f'F{j}'].unique()), 'type': data[f'F{j}'].dtype, 'category': list(filter(lambda x: filterNan(x), data[f'F{j}'].unique())),} for j in category_features_index}
  categories = [[] for _ in range(len(data.columns[:-1]))]

  for i in category_features_index:
    categories[i] = category_features[f'F{i}']['category']
    if category_features[f'F{i}']['type'] == 'int64':
      categories[i].sort()
  return categories

def score_model(model, x, y):
    """score_model(model, x, y) score a model on the testing set x with class y"""
    score = {'Accuracy': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0}
    y_pred = model.predict(x)
    score['Accuracy'] = accuracy_score(y, y_pred)      
    score['Precision'] = precision_score(y, y_pred)
    score['Recall'] = recall_score(y, y_pred)
    score['F1'] = f1_score(y, y_pred)
    return score

def train_and_score(model, x, y, iteration=5):
  """ train_and_score will train a model on the dataset (x,y) using a stratified Kfold strategy and return the model score."""
  skf = StratifiedKFold(n_splits=iteration, random_state = 42, shuffle=True)
  score = {'Accuracy': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0}
  for train_index, test_index in skf.split(x, y):
      X_train, X_test = x[train_index], x[test_index]
      y_train, y_test = y[train_index], y[test_index]
      model.fit(X_train, y_train)
      y_test_pred = model.predict(X_test)
      score['Accuracy'] += accuracy_score(y_test, y_test_pred)      
      score['Precision'] += precision_score(y_test, y_test_pred)
      score['Recall'] += recall_score(y_test, y_test_pred)
      score['F1'] += f1_score(y_test, y_test_pred)
  score['Accuracy'] /= iteration
  score['Precision'] /= iteration
  score['Recall'] /= iteration
  score['F1'] /= iteration
  return score


def test_all(dataset_path):
  """ test_all() will prove most combination of samplers and prediction algorithm and print their results."""
  data = pd.read_csv(dataset_path)
  features = data.columns[:-1]
  category_features_index = [1,3,4,5,6,7,8,9,13]
  categories = get_category_features(data, category_features_index)
  x = data.iloc[:, :-1].to_numpy()
  y = data.iloc[:, -1].to_numpy()
  imputer = make_column_transformer(
      ('passthrough', list(range(0,1))),
      (SimpleImputer(strategy='most_frequent', missing_values=nan), [1]),
      ('passthrough', list(range(2,6))),
      (SimpleImputer(strategy='most_frequent', missing_values=nan), [6]),
      ('passthrough', list(range(7,13))),
      (SimpleImputer(strategy='most_frequent', missing_values=nan), [13]),
      remainder = 'drop'
  )
  featureHandler = make_column_transformer(
    ('passthrough', list(range(0, 1))),
    (OrdinalEncoder(categories=[categories[1]]), [1]),
    ('passthrough', [2]),
    (OrdinalEncoder(categories=[categories[3]]), [3]),
    (OrdinalEncoder(categories=[categories[4]]), [4]),
    (OrdinalEncoder(categories=[categories[5], categories[6], categories[7]]), list(range(5,8))),
    (OneHotEncoder(categories= [categories[8], categories[9]],sparse=False, handle_unknown='ignore'), list(range(8,10))),
    ('passthrough', list(range(10, 13))),
    (OneHotEncoder(categories=[categories[13]],sparse=False, handle_unknown='ignore'), [13]),
    remainder = 'drop'
  )

  samplers = {'RandomUnderSample': RandomUnderSampler(), 'RandomOverSample': RandomOverSampler(), 'SMOTE': SMOTE()}
  models = {
    'SVC(linear)' : SVC(kernel='linear'),
    'SVC': SVC(),
    'RandomForest_gini' : RandomForestClassifier(n_estimators=1000, criterion='gini', n_jobs=6, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, bootstrap=True, oob_score=True, random_state=534, verbose=0, warm_start=False, class_weight={0: 0.50 ,1: 1 - 0.5}),
    'RandomForest_entropy' : RandomForestClassifier(n_estimators=1000, criterion='entropy', n_jobs=6, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, bootstrap=True, oob_score=True, random_state=534, verbose=0, warm_start=False, class_weight={0: 0.50 ,1: 1 - 0.5}),
    'RandomForest_log_loss' : RandomForestClassifier(n_estimators=1000, criterion='log_loss',n_jobs=6, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, bootstrap=True, oob_score=True, random_state=534, verbose=0, warm_start=False, class_weight={0: 0.50 ,1: 1 - 0.5}),
    'AdaBoost': AdaBoostClassifier(),
    'GradientBoosting_log_loss': GradientBoostingClassifier(n_estimators=1000, loss='log_loss'),
    'GradientBoosting_exp': GradientBoostingClassifier(n_estimators=1000, loss='exponential')
  }
  for model_name, model in models.items():
    for samplers_name, sampler in samplers.items():
      pipeline = make_pipeline(imputer, featureHandler, sampler, model)
      print(model_name)
      print(samplers_name)
      score = train_and_score(pipeline, x, y)
      print(score)


if __name__ == '__main__':
  path = 'train.csv'
  test_all(path)  
  