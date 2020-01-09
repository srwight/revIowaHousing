#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 23:44:51 2020

@author: srwight
"""
import pandas as pd
import os, importlib
from algorithm import algoB

modules = []
for module in os.listdir(os.getcwd()):
    if '.py' in module:
        modules.append(importlib.import_module(module.replace('.py', '')))

def main():
    #read our training data
    df_init = pd.read_csv('train.csv')
    
    #apply feature engineering
    df_ext = modules[0].feature_extract(df_init)
    for module in modules:
        if module == modules[0] or module.__name__ == 'housing':
            continue
        df_ext = pd.concat([df_ext, module.feature_extract(df_init)], axis=1)
    
    y = df_init.SalePrice

    sum = 0
    total = 0
    min = 1
    max = 0
    for i in range(10):
        result = algoB.algoB(df_ext, y)
        print(result)
        sum += result
        total = i+1
        if result < min:
            min = result
        if result > max:
            max = result
    
    print('\n\n\nMean:\t',sum/total)
    print('Min:\t',min)
    print('Max:\t',max)
if __name__ == '__main__':
    main()