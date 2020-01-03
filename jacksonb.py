#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 14:05:15 2020

@author: staxx
"""

def jacksonb(a:int)->str:
    """
    Function has Parameter
    ------
    a: int
        int either 1 or 2
    Returns
    -------
    String str
            If a is 1, it returns "What a nice day in the neighborhood"
            If a is 2, it returns "It is cold"
            If a i anything else, it returns "I am ready to go home"

    """
    
    if a == 1: 
        return "What a nice day in the neighborhood"
    
    elif a == 2:
        return "It is cold"
    else:
        return "I am ready to go home"