#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 08:49:51 2020

@author: michael
"""

# Importing packages
import pandas as pd
import numpy as np
from datetime import datetime


#traindata = pd.read_csv("/home/michael/Downloads/Dataset(1)/train.csv")
#testdata = pd.read_csv("/home/michael/Downloads/Dataset(1)/test.csv")


def feature_extract(traindata):
    #Now we focus on my variables of interest
    mycolnames = ['MoSold','YrSold','YearBuilt','YearRemodAdd','GarageYrBlt']
    mydata = traindata[mycolnames]

    #GarageYrBlt has total missing values of 81 and percentage of missing values of  0.055479
    #we need to replace NaN values with the mean, 1978, since it results in the least standard deviation
    newcol = mydata['GarageYrBlt'].fillna(1978).apply(int)
    mydata['GarageYrBlt'] = newcol

    #this gives me a std deviation of 23.994863
    #I filled in NaN values with the mean, 1978
    # s: I think if we make a new dataframe object, we can do this like so:
    yrmo = mydata[['YrSold','MoSold']]
    yrmo.columns = ['Year','Month']
    yrmo['Day'] = 1

    # s: ok. I found it. But it's no good to have it as a datetime object.
    # s: I'm going to divide it out a few times to get to seconds, then minutes
    # s: then hours, then days. Eventually it's going to be normalized so
    # s: I guess that doesn't matter too much, but hey.
    datecol = pd.to_datetime(yrmo).astype(int)/10**9/60/60/24

    # now to drop those columns
    mydata.drop(['YrSold','MoSold'], axis = 1, inplace = True)
    

    # I don't think any of the rest of this is necessary.

    # for column in mydata:
    #     newcol = mydata[column].apply(str)
    #     mydata[column] = newcol
    
    #Need to use Dummy variables since my data points are in months('MoSold') or in years(YrSold','YearBuilt','YearRemodAdd','GarageYrBlt')
    # df_temp = pd.get_dummies(data=mydata,
    #                        prefix=mydata.columns,
    #                        prefix_sep= '_',
    #                        dummy_na=False,)

    # s: we need to build the names of the columns out.
    
    colnames = list(mydata.columns)
    colnames.append('DateBuilt')
    retdata = pd.concat([mydata, datecol], axis=1)
    retdata.columns = colnames
    return retdata

def main():
    df = pd.read_csv('train.csv')
    print(feature_extract(df))

if __name__ == '__main__':
    main()