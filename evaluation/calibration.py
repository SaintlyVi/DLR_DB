#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:09:59 2017

@author: SaintlyVi
"""
import pandas as pd
import numpy as np

from support import writeLog

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

def dataIntegrity(submodels, min_answerid, min_obsratio):
    """
    This function returns the slice of submodels that meet the specified minimum uncertainty requirements. Submodels must form part of the same experiment (eg demand summary and hourly profiles).
    """
    if isinstance(submodels, list):
        models = submodels
    else:
        models = [submodels]

    validmodels = pd.DataFrame(columns = ['submodel_name','valid_data','uncertainty_index', 
                                          'valid_unit_count', 'unit'])
    
    for m in models:
        name = m.name
        valid_data = m[(m.AnswerID_count>=min_answerid) & (m.valid_obs_ratio>=min_obsratio)]
        uix = len(valid_data) / len(m)
        try:
            valid_unit_count = valid_data['valid_hours'].sum()
            unit = 'total_valid_hours' 
        except:
            valid_unit_count = valid_data['AnswerID_count'].sum()
            unit = 'valid_AnswerID_count' 
            
        validmodels = validmodels.append({'submodel_name':name, 
                                          'valid_data':valid_data, 
                                          'uncertainty_index':uix, 
                                          'valid_unit_count':valid_unit_count, 
                                          'unit':unit}, ignore_index=True) 
        
    validmodels.set_index('submodel_name', drop=True, inplace=True)

    return validmodels

def modelSimilarity(ex_submodel, ex_ts, valid_new_submodel, new_ts, submod_type):
    """
    This function calcualtes the evaluation measure for the run.
    
    ex_submodel = (DataFrame) either existing/expert demand_summary or hourly_profiles submodel
    valid_new_submodel = (DataFrame) output from dataIntegrity function 
                            -> only want to compare valid data
    submod_type = (str) one of [ds, hp] 
                            -> ds=demand_summary, hp=hourly_profiles
    """
    
    if submod_type == 'ds':
        index_cols = ['class','YearsElectrified']
    elif submod_type == 'hp':
        index_cols = ['class','YearsElectrified','month','daytype','hour']
    else:
        return(print('Valid submod_type is one of [ds, hp] -> ds=demand_summary, hp=hourly_profiles.'))
    
    merged_sub = ex_submodel.merge(valid_new_submodel, on=index_cols)
    simvec = merged_sub[new_ts] - merged_sub[ex_ts]
    simvec.dropna(inplace=True)
    simveccount = len(simvec)
    
    eucliddist = np.sqrt(sum(simvec**2))
    
    return eucliddist, simveccount, merged_sub

def logCalibration(bm_model, year, exp_model, min_answerid = 2, min_obsratio = 0.85):
    """
    This function logs the evaluation results of the run.
    
    ex_model = [demand_summary, hourly_profiles, ds_val_col_name, hp_val_col_name]
    """
    #Generate data model
    ods = pd.read_csv('data/experimental_model/'+exp_model+'/demand_summary_'+year+'.csv')
    ohp = pd.read_csv('data/experimental_model/'+exp_model+'/hourly_profiles_'+year+'.csv')
    
    #Check data integrity
    ods.name = 'demand_summary'
    ohp.name = 'hourly_profiles'
    validmodels = dataIntegrity([ods, ohp], min_answerid, min_obsratio)
    valid_new_ds = validmodels.at['demand_summary','valid_data']
    valid_new_hp = validmodels.at['hourly_profiles','valid_data']
    new_dsts = 'M_kw_mean'
    new_hpts = 'kva_mean'
    
    #Fetch benchmark model
    bm_ds = bm_model[0]
    bm_hp = bm_model[1]
    bm_dsts = bm_model[2]
    bm_hpts = bm_model[3]
    
    #Calculate model similarity
    euclid_ds, count_ds, slice_ex_ds = modelSimilarity(bm_ds, bm_dsts, valid_new_ds, new_dsts, 'ds')
    euclid_hp, count_hp, sliced_ex_hp = modelSimilarity(bm_hp, bm_hpts, valid_new_hp, new_hpts, 'hp')
    
    #Prepare and write logs
    ds_uix = validmodels.at['demand_summary','uncertainty_index']
    ds_vuc = validmodels.at['demand_summary','valid_unit_count']
    ds_unit = validmodels.at['demand_summary','unit']
    hp_uix = validmodels.at['hourly_profiles','uncertainty_index']
    hp_vuc = validmodels.at['hourly_profiles','valid_unit_count']
    hp_unit = validmodels.at['hourly_profiles','unit']
    
    loglineds = [year, exp_model, ods.name, min_answerid, min_obsratio, ds_uix, ds_vuc,
                 ds_unit, euclid_ds, count_ds]
    loglinehp = [year, exp_model, ohp.name, min_answerid, min_obsratio, hp_uix, hp_vuc, 
                 hp_unit, euclid_hp, count_hp]
    
    log_lines = pd.DataFrame([loglineds, loglinehp], columns = ['year','experiment',
                       'submodel','min_answerid_count','min_valid_obsratio',
                       'uncertainty_ix','valid_unit_count','unit','sim_eucliddist','sim_count'])
    
    writeLog(log_lines,'log_calibration')
