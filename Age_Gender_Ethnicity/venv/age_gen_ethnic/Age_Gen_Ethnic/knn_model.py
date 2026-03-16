import data_preprocessing
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV

data = data_preprocessing.data
age_y = np.array(data_preprocessing.df['age'])
gender_y = np.array(data_preprocessing.df['gender'])
eth_y = np.array(data_preprocessing.df['ethnicity'])
knn_best_parm = []
leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))
p=[1,2]
weights = ['uniform', 'distance']
algo = ['auto','ball_tree','kd_tree','brute']

def age_detection(x,y):
    rs=RandomizedSearchCV(KNeighborsRegressor(),{'n_neighbors':n_neighbors,'leaf_size':leaf_size,'p':p,'weights':weights,'algorithm':algo},cv=10,return_train_score=None)
    rs.fit(x, y)
    knn_best_parm.append({
        'Prediction': 'Age',
        'best_score': rs.best_score_,
        'best_params': rs.best_params_
    })

def gender_detection(x,y):
    rs=RandomizedSearchCV(KNeighborsClassifier(),{'n_neighbors':n_neighbors,'leaf_size':leaf_size,'p':p,'weights':weights,'algorithm':algo},cv=10,return_train_score=None)
    rs.fit(x, y)
    knn_best_parm.append({
        'Prediction': 'Gender',
        'best_score': rs.best_score_,
        'best_params': rs.best_params_
    })

def ethnicity_detection(x,y):
    rs=RandomizedSearchCV(KNeighborsClassifier(),{'n_neighbors':n_neighbors,'leaf_size':leaf_size,'p':p,'weights':weights,'algorithm':algo},cv=10,return_train_score=None)
    rs.fit(x, y)
    knn_best_parm.append({
        'Prediction': 'Ethnicity',
        'best_score': rs.best_score_,
        'best_params': rs.best_params_
    })

age_detection(data,age_y)
gender_detection(data,gender_y)
ethnicity_detection(data,eth_y)

print(pd.DataFrame(knn_best_parm))
