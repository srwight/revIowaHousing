#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 23:44:51 2020

@author: srwight
"""
import pandas as pd
import numpy as np
from clean_data import clean
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from joblib import dump, load
from sklearn.linear_model import Ridge, LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import skew

def main():
    #read our training data
    df = pd.read_csv('train.csv')
    y = df.SalePrice
    
    #apply feature engineering
    df_num, df_obj = clean(df)

    #Get Skew info, use it to create a list of column headers so we can log the same columns in our unknown data
    df_num_skew = df_num.apply(skew)
    df_num_skew = df_num_skew.loc[df_num_skew > 0.75]
    skewcols = list(df_num_skew.index)

    # We have to save the information that describes when we applied skew.
    dump(skewcols,'skewcols.joblib')
    df_num[skewcols] = np.log1p(df_num[skewcols])

    #save the means of our numerical values for use in our deployment model
    dump(df_num.mean(), 'num_means.joblib')

    #apply OneHotEncoding
    enc = OneHotEncoder(
        sparse = False,
        handle_unknown='ignore'
    )
    df = pd.concat(
        [
            df_num, 
            pd.DataFrame(
                enc.fit_transform(df_obj)
                )
        ], 
        axis=1
    )
    dump(enc, 'onehot.joblib')

    # apply normalization
    normer = MinMaxScaler()
    df = normer.fit_transform(df)
    dump(normer, 'scaler.joblib')

    # apply PCA
    pcamodel = PCA(n_components = 0.999)
    dfpca = pcamodel.fit_transform(df)
    dump(pcamodel, 'pca.joblib')

    #train-test Split

    rndState = np.random.RandomState()

    X_train, X_test, y_train, y_test = train_test_split(
        df,
        y,
        train_size = 0.9,
        random_state = rndState
    )

    X_trainp, X_testp, y_trainp, y_testp = train_test_split(
        dfpca,
        y,
        train_size=0.9,
        random_state = rndState
    )

    #train Ridge
    ridge_model = Ridge(
        alpha=1.0,
        fit_intercept=True,
        normalize=True,
        copy_X=True,
        max_iter=None,
        tol=1e-2,
        solver='auto',
        random_state=None
    )
    ridge_modelp = Ridge(
        alpha=1.0,
        fit_intercept=True,
        normalize=True,
        copy_X=True,
        max_iter=None,
        tol=1e-2,
        solver='auto',
        random_state=None
    )

    ridge_model.fit(X_train,y_train)
    dump(ridge_model,'models_ridge.joblib')

    ridge_modelp.fit(X_trainp, y_trainp)
    dump(ridge_modelp,'modelspca_ridge.joblib')

    #train OLS
    OLS_model = LinearRegression(
        fit_intercept=True,
        normalize=False,
        copy_X=True,
        n_jobs=-1
    )

    OLS_modelp = LinearRegression(
        fit_intercept=True,
        normalize=False,
        copy_X=True,
        n_jobs=-1
    )

    OLS_model.fit(X_train,y_train)
    dump(OLS_model,'models_OLS.joblib')

    OLS_modelp.fit(X_trainp, y_trainp)
    dump(OLS_modelp,'modelspca_OLS.joblib')

    #train SGD
    SGD_model = SGDRegressor(
        loss='squared_loss',
        penalty='l2',
        alpha=0.0001,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=10000,
        tol=1e-4,
        shuffle=True,
        verbose=0,
        random_state=None,
        learning_rate='invscaling',
        eta0=0.01,
        power_t=0.25,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        warm_start=True,
        average=False
    )

    SGD_modelp = SGDRegressor(
        loss='squared_loss',
        penalty='l2',
        alpha=0.0001,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=10000,
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
        warm_start=True,
        average=False
    )

    SGD_model.fit(X_train,y_train)
    dump(SGD_model,'models_SGD.joblib')

    SGD_modelp.fit(X_trainp, y_trainp)
    dump(SGD_modelp,'modelspca_SGD.joblib')

    scores = {
        'Ridge':ridge_model.score(X_test,y_test),
        'SGD':SGD_model.score(X_test,y_test),
        'OLS':OLS_model.score(X_test,y_test)
    }

    scorespca = {
        'Ridge':ridge_modelp.score(X_testp,y_testp),
        'SGD':SGD_modelp.score(X_testp,y_testp),
        'OLS':OLS_modelp.score(X_testp,y_testp)
    }

    scores = pd.Series(scores).rename('NO PCA')
    scorespca = pd.Series(scorespca).rename('PCA')

    print(pd.concat([scores, scorespca],axis=1))

if __name__ == '__main__':
    main()