#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 09:49:43 2017

NB 2013 had no surveys taken
NB 2014 AnswerIDs have not been matched to ProfileIDs

@author: SaintlyVi
"""

import pandas as pd    
import numpy as np
import os
from math import ceil

import plotly.offline as offline
import plotly.graph_objs as go
import plotly as py
offline.init_notebook_mode(connected=True) #set for plotly offline plotting

from support import classout_dir
import features.socios as socios
import expertmod.excore as expert
    
def inferredClasses(year, experiment_dir):
    """
    This function gets the inferred class for each AnswerID from 'DLR_DB/classmod/out/experiment_dir'.
    """
    try:
        dirpath = os.path.join(classout_dir, experiment_dir)
        filename = 'classes_' + str(year) + '.csv'
        classes = pd.read_csv(os.path.join(dirpath, filename), header=None, names=['AnswerID','class'])
        
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
    
    except:
        return print('No valid data exists for the supplied year')
    
    return data

def observedDemandSummary(annualmonthlydemanddata, year, experiment_dir):

    interval = annualmonthlydemanddata.interval[0]
    
    try:
        classes = inferredClasses(year, experiment_dir)
        yearselect = yearsElectrified(year)
        
        meta = pd.merge(classes, yearselect, on='AnswerID')
        
        richprofiles = pd.merge(annualmonthlydemanddata, meta, on='AnswerID')
        
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

def observedHourlyProfiles(aggdaytypedemanddata, year, experiment_dir):
    """
    This function generates an hourly load profile model based on a year of data. 
    The model contains aggregate hourly kw readings for the parameters:
        Customer Class
        Month
        Daytype [Weekday, Sunday, Monday]
        Hour
    """
    
    try:
        classes = inferredClasses(year, experiment_dir)
        yearselect = yearsElectrified(year)
        
        meta = pd.merge(classes, yearselect, on='AnswerID')
        
        richprofiles = pd.merge(aggdaytypedemanddata, meta, on='AnswerID')
        
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

def plotAnnualHourlyProfiles(customer_class, daytype='Weekday', years_electrified=7, 
                             model_dir=None, **kwargs):
    """
    This function plots the hourly load profile for a subset of a customer class for a specified year since electrification. The function requires either a data model or an expert model as input.
    
    **kwargs must specify:
        model = 'data' OR model = 'expert'
    if model = data, then **kwargs must also contain: 
        year(int) of observations, 
        experiment_dir(str),
        data(ts.aggDaytypeDemand(ts.getProfilePower(year)))
    """
    
    try:
        model = kwargs.get('model')
    except:
        return print('You must specify model = "expert" or model = "data" within your kwargs')
    
    if model == 'expert':
        if model_dir is None:        
            df = expert.expertHourlyProfiles()
        else:
            df = expert.expertHourlyProfiles(model_dir)
        df.columns = ['YearsElectrified', 'mean_monthly_kw', 'month', 'daytype', 'hour', 
                      'kva_mean', 'kva_std', 'class']
    elif model == 'data':
        try:
            year = kwargs.get('year')
            experiment_dir = kwargs.get('experiment_dir')
            adtd = kwargs.get('data')
        except:
            return print('You must specify year, experiment_dir and data arguements within your kwargs')
        df = observedHourlyProfiles(adtd, year, experiment_dir)
        df = df[['class', 'YearsElectrified', 'month', 'daytype', 
                         'hour', 'kva_mean', 'kva_std']]
        
    df = df[(df['daytype']==daytype) & (df['YearsElectrified']==years_electrified) & (df['class']==customer_class)]
    maxdemand = df['kva_mean'].max()
    
    #generate plot data
    traces = []
    y_raw = df.loc[:, 'hour']
    y_raw = y_raw.reset_index(drop=True)
    months = np.flipud(df['month'].unique())
    count = 0
    for m in months:
        z_raw = df.loc[df['month'] == m, 'kva_mean']
        z_raw = z_raw.reset_index(drop=True)
        x = []
        y = []
        z = []        
        for j in range(0, len(z_raw)):
            z.append([z_raw[j], z_raw[j]])
            y.append([y_raw[j], y_raw[j]])
            x.append([count, count+1])            
        hovertext = list() #modify text box on hover
        for yi, yy in y:
            hovertext.append(list())
            hovertext[-1].append('{}h00<br /> {:.3f} kVA'.format(yy, z[yi][0]))
            
        traces.append(dict(
            z=z,
            x=x,
            y=y,
            name=m,
            showscale=False,
            type='surface',                               
            text = hovertext,
            hoverinfo="name+text"
        ))
        count += 1
        
    #set layout    
    layout = go.Layout(
            title= daytype + ' hourly load profile for "' + customer_class + '" customers ' + str(years_electrified) +' years after electrification',
            margin=go.Margin(t=50,r=50,b=50,l=50, pad=10),
            height= 700,
            scene=dict(
                xaxis=dict(
                        title = 'month',
                        type = 'category',
                        ticktext = months,
                        tickvals = np.arange(0.5, 12.5, 1),
                        tickwidth = 1.5
                        ),
                yaxis=dict(
                        title = 'time of day',
                        tickvals = np.arange(0, 24, 2)),
                zaxis=dict(
                        title = 'demand (kVA)',
                        tickvals = np.arange(0, ceil(maxdemand*10)/10, 0.1),
                        rangemode = "tozero")
                )
    )
                
    fig = { 'data':traces, 'layout':layout }
    
    return offline.iplot(fig, filename='hourly-class-profile')