# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 15:50:14 2020

@author: eende
"""

import pandas as pd

__doc__ = """

Dependencies: pandas

This module is for feature engineering on the following features:
    BsmtCond
    BsmtExposure
    BsmtFinType1
    BsmtFinType2
    
This module converts the defined features from Objects to Strings so that they can be used
as categorical data.  It then creates a new DataFrame named basement_data and inserts
the new categorical features into it.  

"""

def eendreFeatures(df: pd.DataFrame) -> pd.DataFrame :
    
    """
    
    Parameters:
        df: pd.DataFrame = The entire Housing Prices DataFrame
        
    Returns:
        pd.DataFrame = known as basement_dummies, returns dummy variables of the
        defined features
        
    """
    
    # Converting feature objects into strings
    df['BsmtCond'] = df['BsmtCond'].astype(str)
    df['BsmtExposure'] = df['BsmtExposure'].astype(str)
    df['BsmtFinType1'] = df['BsmtFinType1'].astype(str)
    df['BsmtFinType2'] = df['BsmtFinType2'].astype(str)
    
    basement_data = pd.DataFrame()
    myFeatures = ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    
    # Inserting String features into basement_data DataFrame
    for feature in myFeatures:
        basement_data.insert(loc=0,column=feature,value=df[feature])
    
    # Convert basement_data features into dummy variables
    basement_dummies = pd.get_dummies(basement_data, drop_first = False)
    return basement_dummies

# This section for testing purposes only
def main():
    data = pd.read_csv(r"C:\Users\eende\Desktop\House Price Dataset\train.csv")
    print(eendreFeatures(data))
    
if __name__ == '__main__':
    main()