#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:09:59 2017

@author: SaintlyVi
"""
import pandas as pd

def uncertaintystats(submodel):
    """
    Creates a dict with statistics for observed hourly profiles for a given year. 
    Use evaluation.evalhelpers.observedHourlyProfiles() to generate the input dataframe.
    """
    allstats = list()
    
    for c in submodel['class'].unique():
        stats = submodel[submodel['class']==c].describe()
        stats['customer_class'] = c
        stats.reset_index(inplace=True)
        stats.set_index(['customer_class','index'], inplace=True)
        allstats.append(stats)
        
    df = pd.concat(allstats)
    
    return df[['AnswerID_count','valid_obs_ratio']]

