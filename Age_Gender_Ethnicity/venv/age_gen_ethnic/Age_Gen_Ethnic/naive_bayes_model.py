import data_preprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

data = data_preprocessing.data
age_y = np.array(data_preprocessing.df['age'])
gender_y = np.array(data_preprocessing.df['gender'])
eth_y = np.array(data_preprocessing.df['ethnicity'])
NB_best_parm = []

def age_detection(x,y):
    pass

def gender_detection(x,y):
    pass

def ethnicity_detection(x,y):
    pass

age_detection(data,age_y)
gender_detection(data,gender_y)
ethnicity_detection(data,eth_y)

print(pd.DataFrame(NB_best_parm))