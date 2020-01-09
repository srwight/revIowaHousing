#sklearn.linear_model.Ridge
#Linear least squares with l2 regularization.
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

def algoB (x, y) -> str: #x is fatures, y is the predicted variable
    xtrain, xtest, ytrain, ytest = train_test_split(x,y, 
                                                   test_size = .2,
                                                   train_size = .8,
                                                   random_state = None,
                                                   shuffle = True,
                                                   stratify = None) 
    model = Ridge()
    model.fit(xtrain, ytrain)
    score = model.score(xtest, ytest)
    #xtrain, xtest, ytrain, ytest
    return score
