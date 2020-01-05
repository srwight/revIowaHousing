# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 15:44:42 2020

@author: hanan
"""

__doc__ = """

Dependencies used (subject to change): pandas

This .py file is meant to handle these 4 features/variables in feature engineering:
    GarageArea - the size/area of the garage in square feet
    GarageCars - the car capacity of the garage (number of cars)
    OpenPorchSF - the size/area of the open porch in square feet
    WoodDeckSF - the size/area of the wood deck in square feet
    
The function named feature_extract is where the dataset will be passed in & the 4 features
will be extracted for processing, including handling of missing values & outliers
and possibly normalization/handling of skewed data.
"""

import pandas as pd # import pandas in order to be able to bring in the dataset

def feature_extract(dataset:pd.DataFrame) -> pd.DataFrame:
    """
    A function for the extraction of these 4 variables/features:
        GarageArea
        GarageCars
        OpenPorchSF
        WoodDeckSF
        
    Parameters
    ----------
    dataset : pd.DataFrame
        This is the dataset that will be passed into the function (as a parameter) 
        for processing.

    Returns
    -------
    myFeatures : pd.DataFrame
        This is the DataFrame containing the 4 selected features (mentioned 
        above) that will be returned as output.

    """
    
    # Picking out the assigned columns & putting them into a separate DataFrame
    myFeatures = pd.DataFrame(dataset, columns = ['GarageArea', 'GarageCars', 'OpenPorchSF', 'WoodDeckSF'])
    
    # Handling missing values by filling/imputing them with the median
    for c1 in myFeatures.columns.values:
        myFeatures = myFeatures.fillna(myFeatures[c1].mean())
        
    # Handling outliers by replacing the values of those outliers
    # with the value used to filter the data or with some value close to it
    for ga in myFeatures['GarageArea']: # Garage area in square feet
        if (ga >= 1300):
            myFeatures['GarageArea'].replace(to_replace = ga, value = 1300)
        
    for gc in myFeatures['GarageCars']: # Garage car capacity
        if (gc >= 4):
            myFeatures['GarageCars'].replace(to_replace = gc, value = 3)
            
    for op in myFeatures['OpenPorchSF']: # Open porch area in square feet
        if (op > 450):
            myFeatures['OpenPorchSF'].replace(to_replace = op, value = 440)
            
    for wd in myFeatures['WoodDeckSF']: # Wood deck area in square feet
        if (wd >= 800):
            myFeatures['WoodDeckSF'].replace(to_replace = wd, value = 790)
            
    # Handle skewness and/or normalization (the next few lines of code are placeholders)
    #print("GarageArea skew: %s\n" % myFeatures['GarageArea'].skew())
    #print("GarageCars skew: %s\n" % myFeatures['GarageCars'].skew())
    #print("OpenPorchSF skew: %s\n" % myFeatures['OpenPorchSF'].skew())
    #print("WoodDeckSF skew: %s\n" % myFeatures['WoodDeckSF'].skew())     
            
    # Returning the assigned features (still as a DataFrame) after processing them
    return myFeatures

# Code that's only meant to be used for testing
# This only runs when this .py file is the main file, not when it's imported.           
def main():
    df = pd.read_csv("C:/Users/hanan/Desktop/BatchProject/Dataset/train.csv")
    print(feature_extract(df))
    
if __name__ == '__main__':
    main()
    
