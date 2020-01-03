# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 13:19:54 2020

@author: hanan
"""

def kwokh (a:int) -> str:
    
    """Parameters: a should be a 1 or a 2"""
    
    if (a == 1):
        return "This is the 1st string."
    elif (a == 2):
        return "This is the 2nd string."
    else:
        return "You passed some other number %d." % a
