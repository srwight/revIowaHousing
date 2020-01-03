#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 13:02:22 2020

@author: srwight
"""

def wights(a:int) -> str:
    """

    Parameters
    ----------
    a : int
        1 or 2.

    Returns
    -------
    str
        If a is 1, it returns "You passed a 1."
        If a is 2, it returns "You passed a 2."
        If a is anything else, it returns "What have you done?!"

    """
    if a==1:
        return 'You passed a 1'
    elif a==2:
        return 'You passed a 2'
    else:
        return 'What have you done?!'
    
