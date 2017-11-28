#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:09:59 2017

@author: SaintlyVi
"""
import pandas as pd
import numpy as np
import datetime as dt
import os

import evaluation.evalhelpers as eh
import features.ts as ts
import expertmod.excore as expert
from support import eval_dir

def generateDataModel(year, experiment_dir):
    
    pp = ts.getProfilePower(year)
    aggpp = ts.aggProfilePower(pp, 'M')
#    md = ts.maxDemand(pp)
    amd = ts.annualIntervalDemand(aggpp)
    adtd = ts.aggDaytypeDemand(pp)
    ods = eh.observedDemandSummary(amd, year, experiment_dir)
    ohp = eh.observedHourlyProfiles(adtd, year, experiment_dir)
    
    return ods, ohp, adtd, amd, aggpp, pp

def uncertaintyStats(submodel):
    #TODO Visualise!
    #plot amd (answerIDcount for each year electrified + valid obs ratio) & adtd
    #do stacked bar chart for all classes, yaxis=AnswerIDcount, xaxis=yearsElectrified, colour=glass, opacity=validObservations
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

def dataUncertainty(submodels, min_answerid, min_obsratio):
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
        
    validmodels.set_index('submodel_name', drop=True, inplace=True)
        
    return validmodels

def modelSimilarity(old_submodel, old_ts, new_submodel, new_ts, index_cols):
    
    old_submodel.set_index(index_cols, inplace=True)
    new_submodel.set_index(index_cols, inplace=True)
    
    slice_old_sub = old_submodel[old_submodel.index.isin(new_submodel.index)]
    simvec = new_submodel[new_ts] - slice_old_sub[old_ts]
    simvec.dropna(inplace=True)
    simveccount = len(simvec)
    
    eucliddist = np.sqrt(sum(simvec**2))
    
    return eucliddist, simveccount

def logCalibration(year, experiment_dir, min_answerid = 2, min_obsratio = 0.85):

    #Generate data model
    dm = generateDataModel(year, experiment_dir)
    ods = dm[0]
    ohp = dm[1]
    
    ods.name = 'demand_summary'
    ohp.name = 'hourly_profiles'
    data_uncert = dataUncertainty([ods, ohp], min_answerid, min_obsratio)
    
    exhp = expert.expertHourlyProfiles()
    exds = expert.expertDemandSummary()
    
    indexhp = ['class','YearsElectrified','month','daytype','hour']
    indexds = ['class','YearsElectrified']
    
    old_dsts = 'Energy [kWh]'
    old_hpts = 'Mean [kVA]'
    new_dsts = 'M_kw_mean'
    new_hpts = 'kva_mean'

    euclid_ds, count_ds = modelSimilarity(exds, old_dsts, ods, new_dsts, indexds)
    euclid_hp, count_hp = modelSimilarity(exhp, old_hpts, ohp, new_hpts, indexhp)

    logrowds = [dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), year, experiment_dir, ods.name, 
                min_answerid, min_obsratio, data_uncert.at['demand_summary','uncertainty_index'],
                euclid_ds, count_ds]
    logrowhp = [dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), year, experiment_dir, ohp.name, 
                min_answerid, min_obsratio, data_uncert.at['hourly_profiles','uncertainty_index'],
                euclid_hp, count_hp]
    
    log = pd.DataFrame([logrowds, logrowhp], columns = ['timestamp', 'year', 'experiment',
                       'submodel', 'min_answerid_count', 'min_valid_obsratio', 'uncertainty_ix',
                       'sim_eucliddist', 'sim_count'])
    
    log_dir = os.path.join(eval_dir, 'out')
    os.makedirs(log_dir , exist_ok=True)
    logpath = os.path.join(log_dir, 'log_calibrate.csv')
    
    if os.path.isfile(logpath):
        log.to_csv(logpath, mode='a', header=False, columns = log.columns, index=False)
    else:
        log.to_csv(logpath, mode='w', columns = log.columns, index=False)
    
    return