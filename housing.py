#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 23:44:51 2020

@author: srwight
"""
import pandas as pd

import sdiq, ende, hend, buck, wigh, gril
import clou, wine, davi, nyam, sene, ahto
import jack, hill, gade, avet, pais, kwok

def main():
    #read our training data
    df_init = pd.read_csv('train.csv')
    
    #apply feature engineering
    df_ext = sdiq.feature_extract(df_init)
    df_ext = pd.concat(df_ext, ende.feature_extract(df_init))
    df_ext = pd.concat(df_ext, hend.feature_extract(df_init))
    df_ext = pd.concat(df_ext, buck.feature_extract(df_init))
    df_ext = pd.concat(df_ext, wigh.feature_extract(df_init))
    df_ext = pd.concat(df_ext, gril.feature_extract(df_init))
    df_ext = pd.concat(df_ext, clou.feature_extract(df_init))
    df_ext = pd.concat(df_ext, wine.feature_extract(df_init))
    df_ext = pd.concat(df_ext, davi.feature_extract(df_init))
    df_ext = pd.concat(df_ext, nyam.feature_extract(df_init))
    df_ext = pd.concat(df_ext, sene.feature_extract(df_init))
    df_ext = pd.concat(df_ext, ahto.feature_extract(df_init))
    df_ext = pd.concat(df_ext, jack.feature_extract(df_init))
    df_ext = pd.concat(df_ext, hill.feature_extract(df_init))
    df_ext = pd.concat(df_ext, gade.feature_extract(df_init))
    df_ext = pd.concat(df_ext, avet.feature_extract(df_init))
    df_ext = pd.concat(df_ext, pais.feature_extract(df_init))
    df_ext = pd.concat(df_ext, kwok.feature_extract(df_init))
    
    #see what we did
    print(df_ext)
    
if __name__ == '__main__':
    main()