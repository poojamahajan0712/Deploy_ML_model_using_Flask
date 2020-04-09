# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 10:14:43 2020

@author: pooja
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os

os.chdir("D:\Pooja\Learning\Deployment_using_flask")

dataset = pd.read_csv('sales.csv')
list(dataset)
dataset.drop(['Unnamed: 4'],axis=1, inplace=True)

dataset['rate'].fillna(0, inplace=True)

dataset['sales_in_first_month'].fillna(dataset['sales_in_first_month'].mean(), inplace=True)

X = dataset.iloc[:, :3]

def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

X['rate'] = X['rate'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X, y)

pickle.dump(regressor, open('model.pkl','wb'))        #wb means write binary

model = pickle.load(open('model.pkl','rb'))          #rb means read binary 
print(model.predict([[4, 300, 500]]))
