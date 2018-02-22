#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 09:49:43 2017

NB 2013 had no surveys taken
NB 2014 AnswerIDs have not been matched to ProfileIDs

@author: SaintlyVi
"""

import pandas as pd  
import os

from support import cdata_dir, emdata_dir, writeLog
import features.feature_socios as socios
import features.feature_ts as ts
    
def readClasses(year, dir_name):
    """
    This function gets the inferred class for each AnswerID from 'DLR_DB/classmod/out/experiment_dir'.
    """    
    try:
        dir_path = os.path.join(cdata_dir, dir_name)
        file_name = [s for s in os.listdir(dir_path) if str(year) in s][0]
        classes = pd.read_csv(os.path.join(dir_path, file_name), header=0, index_col=0)
        return classes
    
    except IndexError:
        print('No classes inferred for '+ str(year))
        raise
        
def selectClasses(year, dir_name, threshold='max'):
    """
    This function sets the inferred class for each AnswerID.
    """    
    df = readClasses(year, dir_name)
    if threshold == 'max':
        inferredclass = df.idxmax(axis=1) #USER MUST BE ABLE TO CHANGE THIS
    
    inferredclass = inferredclass.reset_index()
    inferredclass.rename(columns={0:'class'}, inplace=True)
    
    return inferredclass

def yearsElectrified(year):
    """
    This function gets the number of years since electrification for each AnswerID.
    """
    
    try:
        if 1994 <= year <= 1999:
            data = socios.buildFeatureFrame(['years'], year)[0]
        elif 2000 <= year:
            data = socios.buildFeatureFrame(['electricity'], year)[0]
        
        data.columns = ['AnswerID','YearsElectrified']
        cats = [0] + list(range(2, 16)) + [100]
        data.YearsElectrified = pd.cut(data.YearsElectrified, cats, right=False, labels = list(range(1, 16)), include_lowest=False)
        data.YearsElectrified = data.YearsElectrified.astype('int', copy=False)    
    except:
        return print('Could not retreive valid data for the given year.')
    
    return data

def observedMaxDemand(profilepowerdata, year, classes_dir):
    """
    This function selects the maximum demand in kVA for each Answer ID in a year and returns it with its time of occurence.    
    """
    if year <= 2009:
        power_col = 'kw_calculated'
    else:
        power_col = 'Unitsread_kva'
    
    maxdemand = profilepowerdata.iloc[profilepowerdata.reset_index().groupby(['AnswerID'])[power_col].idxmax()].reset_index(drop=True)
        
    maxdemand['month'] = maxdemand['Datefield'].dt.month
    maxdemand['daytype'] = maxdemand['Datefield'].dt.dayofweek
    maxdemand['hour'] = maxdemand['Datefield'].dt.hour                                 
    md = maxdemand[['AnswerID','RecorderID',power_col ,'month','daytype','hour']] 

    classes = selectClasses(year, classes_dir)
    yearselect = yearsElectrified(year)        
    meta = pd.merge(classes, yearselect, on='AnswerID')
    profiles = pd.merge(md, meta, on='AnswerID')  
        
    return profiles

def observedDemandSummary(annual_monthly_demand_data, year, classes_dir):
    """
        This function generates a demand summary model based on a year of data.
        The model contains aggregate hourly kW readings for the factors:
        Customer Class
        Years Electrified
    """

    interval = annual_monthly_demand_data.interval[0]

    classes = selectClasses(year, classes_dir)
    yearselect = yearsElectrified(year)
    
    meta = pd.merge(classes, yearselect, on='AnswerID')
    
    richprofiles = pd.merge(annual_monthly_demand_data, meta, on='AnswerID')
    
    profiles = richprofiles.groupby(['class','YearsElectrified']).agg({
            interval+'_kw_mean':['mean','std'],
            interval+'_kw_std':['mean','std'], 
#            interval+'_kva_mean':['mean','std'],
#            interval+'_kva_std':['mean','std'],
            'valid_hours':'sum', 
            'interval_hours_sum':'sum', 
            'AnswerID':'count'})
    
    profiles.columns = ['_'.join(col).strip() for col in profiles.columns.values]
    profiles.rename(columns={interval+'_kw_mean_mean':interval+'_kw_mean',
                             interval+'_kw_mean_std':interval+'_kw_mean_diversity', 
                             interval+'_kw_std_mean':interval+'_kw_std',
                             interval+'_kw_std_std':interval+'_kw_std_diversity',
#                             interval+'_kva_mean_mean':interval+'_kva_mean',
#                             interval+'_kva_mean_std':interval+'_kva_mean_diversity', 
#                             interval+'_kva_std_mean':interval+'_kva_std',
#                             interval+'_kva_std_std':interval+'_kva_std_diversity', 
                             'valid_hours_sum':'valid_hours',
                             'interval_hours_sum_sum': 'interval_hours'}, inplace=True)
    
    profiles['valid_obs_ratio'] = profiles['valid_hours'] / profiles['interval_hours']
    profiles.drop(columns=['valid_hours', 'interval_hours'], inplace=True)    

    return profiles.reset_index()

def observedHourlyProfiles(aggdaytype_demand_data, year, classes_dir):
    """
    This function generates an hourly load profile model based on a year of data. 
    The model contains aggregate hourly kVA readings for the factors:
        Customer Class
        Month
        Daytype [Weekday, Sunday, Monday]
        Hour
        Years Electrified
    """
    
    classes = selectClasses(year, classes_dir)
    yearselect = yearsElectrified(year)
    
    meta = pd.merge(classes, yearselect, on='AnswerID')
    
    richprofiles = pd.merge(aggdaytype_demand_data, meta, on='AnswerID')
    
    profiles = richprofiles.groupby(['class','YearsElectrified','month','daytype','hour']).agg({
            'kw_mean':['mean','std'],
            'kw_std':['mean','std'], 
#           'kva_mean':['mean','std'],
#           'kva_std':['mean','std'],
            'valid_hours':'sum', 
            'AnswerID':'count', 
            'total_hours_sum':'sum'})
    
    profiles.columns = ['_'.join(col).strip() for col in profiles.columns.values]
    profiles.rename(columns={
                             'kw_mean_mean':'kw_mean',
                             'kw_mean_std':'kw_mean_diversity', 
                             'kw_std_mean':'kw_std',
                             'kw_std_std':'kw_std_diversity',                            
#                             'kva_mean_mean':'kva_mean',
#                             'kva_mean_std':'kva_mean_diversity', 
#                             'kva_std_mean':'kva_std',
#                             'kva_std_std':'kva_std_diversity', 
                              'valid_hours_sum':'valid_hours',
                              'total_hours_sum_sum': 'total_hours'}, inplace=True)
    
    profiles['valid_obs_ratio'] = profiles['valid_hours'] / profiles['total_hours']        

    return profiles.reset_index()

def generateRun(year, experiment, algorithm, run):
    """
    This function generates the experimental model from observations.
    """
    
    classes_dir = experiment+'_'+algorithm+'_'+str(run)

    try:
        pp = ts.getProfilePower(year)
        aggpp = ts.aggProfilePower(pp, 'M')
        amd = ts.annualIntervalDemand(aggpp)
        adtd = ts.aggDaytypeDemand(pp)
        md = observedMaxDemand(pp, year, classes_dir)
        ods = observedDemandSummary(amd, year, classes_dir)
        ohp = observedHourlyProfiles(adtd, year, classes_dir)

    except Exception as e:
        print(e)
        raise
    
    return ods, ohp, md, adtd, amd, aggpp, pp

def saveExpModel(year, experiment, algorithm, run):
    """
    This function generates the experimental model from observations.
    """
    
    ods, ohp, md, adtd, amd, aggpp, pp = generateRun(year, experiment, algorithm, run)

    run_dir = experiment+'_'+algorithm+'_'+str(run)
    dir_path = os.path.join(emdata_dir, run_dir)
    os.makedirs(dir_path , exist_ok=True)   
    
    loglines = []
    
    for k, v in {'demand_summary':ods, 'hourly_profiles':ohp}.items():
        try:
            file_path = os.path.join(dir_path, k + '_'+ str(year) + '.csv')          
            v.to_csv(file_path, index=False)          
            status = 1
            message = 'Success!'
            print('Successfully saved to' + file_path)
        except Exception as e:
            pass
            status = 0 
            message = e
            print('Could not save '+ k)    
    
    l = [k, year, experiment, algorithm, run, status, message]    
    loglines.append(l)
    logs = pd.DataFrame(loglines, columns = ['submodel_type', 'year', 'experiment', 'algorithm',
                                             'run', 'status','message'])
    writeLog(logs,'log_modelRun')
    
    return