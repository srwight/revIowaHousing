#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 08:49:51 2020

@author: michael
"""

# Importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl


traindata = pd.read_csv("/home/michael/Downloads/Dataset(1)/train.csv")
testdata = pd.read_csv("/home/michael/Downloads/Dataset(1)/test.csv")


def feature_extract(traindata):
    #Now we focus on my variables of interest
    mydata = traindata[['MoSold','YrSold','YearBuilt','YearRemodAdd','GarageYrBlt']]
    #GarageYrBlt has total missing values of 81 and percentage of missing values of  0.055479
    #we need to replace NaN values with the mean, 1978, since it results in the least standard deviation
    mydata['GarageYrBlt']=mydata['GarageYrBlt'].fillna(1978.0).apply(int)
    #this gives me a std deviation of 23.994863
    #I filled in NaN values with the mean, 1978
    
    for column in mydata:
        mydata[column] = mydata[column].apply(str)
    
    #Need to use Dummy variables since my data points are in months('MoSold') or in years(YrSold','YearBuilt','YearRemodAdd','GarageYrBlt')
    df_temp=pd.get_dummies(data=mydata,
                           prefix=mydata.columns,
                           prefix_sep= '_',
                           dummy_na=False,)
    return df_temp
feature_extract(traindata)

