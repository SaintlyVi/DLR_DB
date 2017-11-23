#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:09:59 2017

@author: SaintlyVi
"""
import pandas as pd

import evaluation.evalhelpers as eh
import features.ts as ts

def generateDataModel(year, experiment_dir):
    
    pp = ts.getProfilePower(year)
    aggpp = ts.aggProfilePower(pp, 'M')
    md = ts.maxDemand(pp)
    amd = ts.annualIntervalDemand(aggpp)
    adtd = ts.aggDaytypeDemand(pp)
    ods = eh.observedDemandSummary(amd, year, experiment_dir)
    ohp = eh.observedHourlyProfiles(adtd, year, experiment_dir)
    
    return ods, ohp, md

def uncertaintyStats(submodel):
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

def validModel(submodels, min_answerid, min_obsratio):
    """
    This function returns the slice of submodels that meet the specified minimum uncertainty requirements. Submodels must form part of the same experiment (eg demand summary and hourly profiles.)
    """
    if isinstance(submodels, list):
        models = submodels
    else:
        models = [submodels]

    validmodels = pd.DataFrame(columns = ['submodel_name','valid_data', 'uncertainty_index'])
    
    for m in models:
        name = m.name
        valid_data = m[(m.AnswerID_count>=min_answerid) & (m.valid_obs_ratio>=min_obsratio)]
        uix = len(valid_data) / len(m)
    
        validmodels = validmodels.append({'submodel_name':name, 'valid_data':valid_data, 'uncertainty_index':uix}, ignore_index=True)        
        
    return validmodels

def modelSimilarity(old_submodel, old_ts, new_submodel, new_ts, index_cols):
    
    old_submodel.set_index(index_cols, inplace=True)
    new_submodel.set_index(index_cols, inplace=True)
    
    slice_old_sub = old_submodel[new_submodel] #CHECK!!
    sim = slice_old_sub[old_ts] - new_submodel[new_ts]
    #CALCULATE MINKOWSKI DISTANCE
    
    return sim