#sklearn.linear_model.Ridge
#Linear least squares with l2 regularization.
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def algoB (x, y) -> str: #x is fatures, y is the predicted variable
    xtrain, xtest, ytrain, ytest = train_test_split(x,y, 
                                                   test_size = .2,
                                                   train_size = .8,
                                                   random_state = None,
                                                   shuffle = True,
                                                   stratify = None) 
    model = Ridge(alpha=1.0,
                  fit_intercept=True,
                  normalize=False,
                  copy_X=True,
                  max_iter=None,
                  tol=1e-3,
                  solver="auto",
                  random_state=None)
    model.fit(xtrain, ytrain)
    score = model.score(xtest, ytest)
    #xtrain, xtest, ytrain, ytest
    return score
