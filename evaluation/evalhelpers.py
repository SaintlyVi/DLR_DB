#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 09:49:43 2017

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
import features.ts as ts
import features.socios as socios
import expertmod.excore as expert
    
def inferredClasses(year, experiment_dir):
    """
    This function gets the inferred class for each AnswerID from 'DLR_DB/classmod/out/experiment_dir'.
    """
    dirpath = os.path.join(classout_dir, experiment_dir)
    filename = 'classes_' + str(year) + '.csv'
    classes = pd.read_csv(os.path.join(dirpath, filename), header=None, names=['AnswerID','class'])
    
    return classes

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

def observedDemandSummary(year, experiment_dir):
    
    data = ts.avgMonthlyDemand(year)
    classes = inferredClasses(year, experiment_dir)
    yearselect = yearsElectrified(year)
    
    richprofiles = data.merge(classes.merge(yearselect, on='AnswerID'), on='AnswerID')
    richprofiles.columns = ['RecorderID', 'AnswerID', 'mean_monthly_kvah', 'class', 'YearsElectrified', 'Valid']  
    
    profiles = richprofiles.groupby(['class','YearsElectrified']).mean().drop(columns=['AnswerID'], axis=1)
    
    return profiles.reset_index()

def observedHourlyProfiles(year, experiment_dir):
    
    data = ts.avgDaytypeDemand(year)
    classes = inferredClasses(year, experiment_dir)
    yearselect = yearsElectrified(year)
    
    richprofiles = data.merge(classes.merge(yearselect, on='AnswerID'), on='AnswerID')
    
    profiles = richprofiles.groupby(['class','YearsElectrified','month','daytype','hour']).mean().drop(columns=['AnswerID'], axis=1)
    
    return profiles.reset_index()

def plotDemandSummary(customer_class):
    """
    This function plots the average monthly energy consumption for a specified customer class from 
    1 to 15 years since electrification. Data is based on the DPET model.
    """
    
    summary = expert.dpetDemandSummary()
    df = summary.loc[summary['class'] == customer_class, ['Year','Energy [kWh]']] 
    data = [go.Bar(
                x=df['Year'],
                y=df['Energy [kWh]'],
                name=customer_class
            )]
     
    layout = go.Layout(
                title='Annualised Monthly Energy Consumption for "' + customer_class + '" Customer Class',
                xaxis=dict(
                    title='years since electrification',
                    tickfont=dict(
                    size=14,
                    color='rgb(107, 107, 107)'
                    )
                ),
                yaxis=dict(
                    title='average annual kWh/month',
                    titlefont=dict(
                        size=16,
                        color='rgb(107, 107, 107)'
                    )
                )
    )
    
    return offline.iplot({"data":data, "layout":layout})

def plot15YearDemandSummary():
    """
    This function plots the average monthly energy consumption for a specified customer class from 
    1 to 15 years since electrification. Data is based on the DPET model.
    """
    
    summary = expert.dpetDemandSummary()
    df = summary.loc[:, ['class','Year','Energy [kWh]']].sort_values(by='Energy [kWh]')
    data = []
    
    for c in df['class'].unique():

        trace = go.Scatter(
                x=df.loc[df['class'] == c, 'Year'],
                y=df.loc[df['class'] == c, 'Energy [kWh]'],
                name=c,
                fill='tonexty',
                mode='lines'
        )
        data.append(trace)
     
    layout = go.Layout(
                title='Annualised Monthly Energy Consumption for Domestic Energy Consumers',
                xaxis=dict(
                    title='years since electrification',
                    tickfont=dict(
                    size=14,
                    color='rgb(107, 107, 107)'
                    )
                ),
                yaxis=dict(
                    title='average annual kWh/month',
                    titlefont=dict(
                        size=16,
                        color='rgb(107, 107, 107)'
                    )
                ),
    )
    
    return offline.iplot({"data":data, "layout":layout})

def plotHourlyProfiles(customer_class, daytype='Weekday', years_electrified=7, **kwargs):
    """
    This function plots the hourly load profile for a subset of a customer class for a specified year since electrification. The function requires either a data model or an expert model as input.
    
    **kwargs must specify:
        model = 'data' OR model = 'expert'
    if model = data, then **kwargs must also contain year(int) of observations and experiment_dir(str)
    """
    
    try:
        model = kwargs.get('model')
    except:
        return print('You must specify model = "expert" or model = "data" within your kwargs')
    
    if model == 'expert':        
        df = expert.dpetHourlyProfiles()
        df.columns = ['years_electrified','mean_monthly_kvah','month','daytype','hour','mean_kva','std_kva','class']
    elif model == 'data':
        try:
            year = kwargs.get('year')
            experiment_dir = kwargs.get('experiment_dir')
        except:
            return print('You must specify year and experiment_dir within your kwargs')
        df = observedHourlyProfiles(year, experiment_dir)
        df.columns = ['class', 'years_electrified', 'month', 'daytype', 'hour', 'mean_kva', 'std_kva']
        
    df = df[(df['daytype']==daytype) & (df['years_electrified']==years_electrified) & (df['class']==customer_class)]
    maxdemand = df['mean_kva'].max()
    
    #generate plot data
    traces = []
    y_raw = df.loc[:, 'hour']
    y_raw = y_raw.reset_index(drop=True)
    months = np.flipud(df['month'].unique())
    count = 0
    for m in months:
        z_raw = df.loc[df['month'] == m, 'mean_kva']
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

def plot15YearHourlyProfiles(customer_class, daytype='Weekday'):
    
    df = expert.dpetHourlyProfiles()
    df = df[(df['Day Type']==daytype) & (df['class']==customer_class)]
    
    ncol = 3
    nrow = 5
   
    fig = py.tools.make_subplots(rows=nrow, cols=ncol, subplot_titles=['Year ' + str(x) for x in range(1, 16)], horizontal_spacing = 0.1, print_grid=False)
    
    r = 1
    c = 1
    maxdemand = df['Mean [kVA]'].max()
        
    for y in range(1, 16):
        count = 1
        if c == ncol + 1: 
            c = 1
        ro = int(ceil(r/ncol))
        
        try:
            data = df[df['Year']==y]
            z = data['Mean [kVA]'].reset_index(drop=True)
            x = data['Time of day [hour]']
            y = data.Month
            hovertext = list()
            for yi, yy in enumerate(y.unique()):
                hovertext.append(list())
                for xi, xx in enumerate(x.unique()):
                    hovertext[-1].append('hour: {}<br />month: {}<br />{:.3f} kVA'.format(xx, yy, z[24 * yi + xi]))
            trace = go.Heatmap(z = z, 
                               x = x, 
                               y = y,
                               zmin = 0,
                               zmax = maxdemand,
                               text = hovertext,
                               hoverinfo="text")
            fig.append_trace(trace, ro, c)
            
        except:
            pass
        
        c += 1
        r += 1
        count += 1

    fig['layout'].update(showlegend=False, 
                           title='Mean estimated energy demand for "' + customer_class + '" customers from 1 to 15 years after electrification', 
                           height=1500)
    for k in range(1,16):
          fig['layout'].update({'yaxis{}'.format(k): go.YAxis(type = 'category',
                                                              ticktext = data.Month.unique(),
                                                              tickvals = np.arange(0.5, 12.5, 1),
                                                              tickangle = -20,
                                                              tickwidth = 1.5
                                                             ),
                                
                                'xaxis{}'.format(k): go.XAxis(title = 'Time of day (hours)', 
                                                              tickvals = np.arange(0, 24, 2))
                                })    
                                
    return offline.iplot(fig)
