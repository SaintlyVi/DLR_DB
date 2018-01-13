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

from support import cdata_dir
import features.feature_socios as socios
import features.feature_ts as ts
    
def readClasses(year, dir_name):
    """
    This function gets the inferred class for each AnswerID from 'DLR_DB/classmod/out/experiment_dir'.
    """    
    try:
        dir_path = os.path.join(cdata_dir, dir_name)
        file_name = [s for s in os.listdir(dir_path) if str(year) in s][0]
        classes = pd.read_csv(os.path.join(dir_path, file_name), header=None, names=['AnswerID','class'])        
        return classes
    
    except:
        print('No classes inferred for '+ str(year))

def yearsElectrified(year):
    """
    This function gets the number of years since electrification for each AnswerID.
    """
    try:
        if 1994 <= year <= 1999:
            data = socios.buildFeatureFrame(['years'], year)[0]
        elif 2000 <= year:
            data = socios.buildFeatureFrame(['electricity'], year)[0]
        else:
            return print('Please enter a year after 1994')
        
        data.columns = ['AnswerID','YearsElectrified']
        cats = [0] + list(range(2, 16)) + [100]
        data.YearsElectrified = pd.cut(data.YearsElectrified, cats, right=False, labels = list(range(1, 16)), include_lowest=False)
        data.YearsElectrified = data.YearsElectrified.astype('int', copy=False)    
    except:
        return print('Could not retreive valid data for the given year.')
    
    return data

def observedMaxDemand(profilepowerdata, year, class_dir):
    """
    This function selects the maximum demand in kVA for each Answer ID in a year and returns it with its time of occurence.    
    """
    try:
        try:
            maxdemand = profilepowerdata.iloc[profilepowerdata.reset_index().groupby(['AnswerID'])['Unitsread_kva'].idxmax()].reset_index(drop=True)
            
            maxdemand['month'] = maxdemand['Datefield'].dt.month
            maxdemand['daytype'] = maxdemand['Datefield'].dt.dayofweek
            maxdemand['hour'] = maxdemand['Datefield'].dt.hour                                 
            md = maxdemand[['AnswerID','RecorderID','Unitsread_kva','month','daytype','hour']] 

        except: 
            print('Check if year is in range 2009 - 2014.')

        classes = readClasses(year, class_dir)
        yearselect = yearsElectrified(year)        
        meta = pd.merge(classes, yearselect, on='AnswerID')
        profiles = pd.merge(md, meta, on='AnswerID')          
        return profiles
    
    except:
        print('No classes inferred for '+ str(year))

def observedDemandSummary(annual_monthly_demand_data, year, class_dir):
    """
        This function generates a demand summary model based on a year of data.
        The model contains aggregate hourly kW readings for the factors:
        Customer Class
        Years Electrified
    """
    interval = annual_monthly_demand_data.interval[0]
    
    try:
        classes = readClasses(year, class_dir)
        yearselect = yearsElectrified(year)
        
        meta = pd.merge(classes, yearselect, on='AnswerID')
        
        richprofiles = pd.merge(annual_monthly_demand_data, meta, on='AnswerID')
        
        profiles = richprofiles.groupby(['class','YearsElectrified']).agg({
                interval+'_kw_mean':['mean','std'],
                interval+'_kw_std':['mean','std'], 
                interval+'_kva_mean':['mean','std'],
                interval+'_kva_std':['mean','std'],
                'valid_hours':'sum', 
                'interval_hours_sum':'sum', 
                'AnswerID':'count'})
        
        profiles.columns = ['_'.join(col).strip() for col in profiles.columns.values]
        profiles.rename(columns={interval+'_kw_mean_mean':interval+'_kw_mean',
                                 interval+'_kw_mean_std':interval+'_kw_mean_diversity', 
                                 interval+'_kw_std_mean':interval+'_kw_std',
                                 interval+'_kw_std_std':interval+'_kw_std_diversity', 
                                 'valid_hours_sum':'valid_hours',
                                 'interval_hours_sum_sum': 'interval_hours'}, inplace=True)
        
        profiles['valid_obs_ratio'] = profiles['valid_hours'] / profiles['interval_hours']
        profiles.drop(columns=['valid_hours', 'interval_hours'], inplace=True)    
        return profiles.reset_index()
    
    except:
        print('No classes inferred for '+ str(year))

def observedHourlyProfiles(aggdaytype_demand_data, year, class_dir):
    """
    This function generates an hourly load profile model based on a year of data. 
    The model contains aggregate hourly kVA readings for the factors:
        Customer Class
        Month
        Daytype [Weekday, Sunday, Monday]
        Hour
        Years Electrified
    """
    
    try:
        classes = readClasses(year, class_dir)
        yearselect = yearsElectrified(year)
        
        meta = pd.merge(classes, yearselect, on='AnswerID')
        
        richprofiles = pd.merge(aggdaytype_demand_data, meta, on='AnswerID')
        
        profiles = richprofiles.groupby(['class','YearsElectrified','month','daytype','hour']).agg({
                'kva_mean':['mean','std'],
                'kva_std':['mean','std'], 
                'valid_hours':'sum', 
                'AnswerID':'count', 
                'total_hours_sum':'sum'})
        
        profiles.columns = ['_'.join(col).strip() for col in profiles.columns.values]
        profiles.rename(columns={'kva_mean_mean':'kva_mean',
                                 'kva_mean_std':'kva_mean_diversity', 
                                 'kva_std_mean':'kva_std',
                                 'kva_std_std':'kva_std_diversity', 
                                 'valid_hours_sum':'valid_hours',
                                 'total_hours_sum_sum': 'total_hours'}, inplace=True)
        
        profiles['valid_obs_ratio'] = profiles['valid_hours'] / profiles['total_hours']        
        return profiles.reset_index()
    
    except:
        print('No classes inferred for '+ str(year))


def experimentalModel(year, class_dir):
    """
    This function generates the experimental model from observations.
    """
    pp = ts.getProfilePower(year)
    aggpp = ts.aggProfilePower(pp, 'M')
    amd = ts.annualIntervalDemand(aggpp)
    adtd = ts.aggDaytypeDemand(pp)
    md = observedMaxDemand(pp, year, class_dir)
    ods = observedDemandSummary(amd, year, class_dir)
    ohp = observedHourlyProfiles(adtd, year, class_dir)
    
    return ods, ohp, md, adtd, amd, aggpp, pp