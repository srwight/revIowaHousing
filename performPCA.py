#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ahtouw
"""
__doc__ = """
Dependencies: pandas
              sklearn.preprocessing -> MinMaxScaler
              sklearn.decomposition -> PCA
              
This module is for performing PCA on a dataset
NOTE: DATA GETS SCALED DOWN PRIOR TO PCA
This is done in the function performPCA(), which will receive the original
dataframe as input and return PCA dataframe

NOTE: This is before the dataset is split into test and train. 
PCA can be done after but this function would need to be reconfigured
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def performPCA(df:pd.DataFrame, var = 0.95, printComponents = False) -> pd.DataFrame:
    """
    Performing PCA on the dataset
    
    Parameters
    ----------
    df : pd.DataFrame
        This should be the entire Housing Prices dataframe.
    var : float (OPTIONAL)
        This should be the desired variance to be retained after PCA is performed.
        If not specified, will be set to 0.95 (95% retained)
    printComponents : Boolean (OPTIONAL)
        This should be the value that specifies whether or not to print
        the number of components remaining after PCA
        If not specified, will be set to False (no print)
    Returns
    -------
    pd.DataFrame
        This dataframe will include ONLY remaining components from PCA.
    """
    # Data must be normalized prior to PCA, I chose to use MinMaxScaler
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)
    
    # Perfoming PCA on the dataset using the 'var' parameter
    pca = PCA(n_components = var)
    principalComponents = pca.fit_transform(df)
    
    # Converting principalComponents (numpy array) to a pd.dataframe
    principalDf = pd.DataFrame(data = principalComponents)
    
    # Print number of components remaining after PCA if requested
    if printComponents:
        print("PCA with variance of %.f%%"%(var*100))
        print("Number of components: %d"%len(pca.components_))
    
    return principalDf
  
if __name__ == '__main__':
    df=pd.read_csv('fixed_No.csv')
    
    # ONLY THE FIRST PARAMETER IS REQUIRED
    df=performPCA(df,var=.95,printComponents=True)

