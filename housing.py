#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 23:44:51 2020

@author: srwight
"""
import pandas as pd
import numpy as np
from clean_data import clean
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from joblib import dump, load
from sklearn.linear_model import Ridge, LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split

def main():
    #read our training data
    df_init = pd.read_csv('train.csv')
    y = df_init.SalePrice
    
    #apply feature engineering
    df_ext = clean(df_init)
    
    df_obj = df_ext.loc[:,(df_ext.dtypes=='object')]
    df_num = df_ext.loc[:,~(df_ext.dtypes=='object')]

    #apply OneHotEncoding
    enc = OneHotEncoder(
        sparse = False,
        handle_unknown='ignore'
    )
    OneHotted = pd.concat([df_num, pd.DataFrame(enc.fit_transform(df_obj))], axis=1)
    dump(enc, 'onehot.joblib')

    #apply PCA
    pcamodel = PCA(n_components = 0.95)
    PostPCA = pcamodel.fit_transform(OneHotted)
    dump(pcamodel, 'pca.joblib')

    #train-test Split

    X_train, X_test, y_train, y_test = train_test_split(
        PostPCA,
        y,
        train_size = 0.8,
    )

    #train Ridge
    ridge_model = Ridge(
        alpha=1.0,
        fit_intercept=True,
        normalize=True,
        copy_X=True,
        max_iter=None,
        tol=1e-3,
        solver='auto',
        random_state=None
    )

    ridge_model.fit(X_train,y_train)
    dump(ridge_model,'models_ridge.joblib')

    #train SGD
    SGD_model = SGDRegressor(
        loss='squared_loss',
        penalty='l2',
        alpha=0.0001,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        shuffle=True,
        verbose=0,
        random_state=None,
        learning_rate='invscaling',
        eta0=0.01,
        power_t=0.25,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        warm_start=False,
        average=False
    )

    SGD_model.fit(X_train,y_train)
    dump(SGD_model,'models_SGD.joblib')

    #train OLS
    OLS_model = LinearRegression(
        fit_intercept=True,
        normalize=True,
        copy_X=True,
        n_jobs=None
    )

    OLS_model.fit(X_train,y_train)
    dump(OLS_model,'models_OLS.joblib')

    scores = {
        'Ridge':ridge_model.score(X_test,y_test),
        'SGD':SGD_model.score(X_test,y_test),
        'OLS':OLS_model.score(X_test,y_test)
    }

    scores = pd.Series(scores)

    print(scores)

if __name__ == '__main__':
    main()