#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 12:17:34 2017

@author: saintlyvi
"""
import pandas as pd    
import numpy as np
from math import ceil, floor

import plotly.offline as offline
import plotly.graph_objs as go
import plotly as py
offline.init_notebook_mode(connected=True) #set for plotly offline plotting

import expertmod.excore as expert

def plotAnswerIDCount(submodel):

    data = []
    yrs = list(range(1,16))
    
    #Get mean AnswerID count for number of years electrified
    for c in submodel['class'].unique():
        selectdata = submodel[submodel['class']==c][['YearsElectrified',
                         'AnswerID_count']].groupby('YearsElectrified').mean().applymap(
                         lambda x: ceil(x))
        t = selectdata.reindex(yrs, fill_value=0).reset_index()
        
        trace = go.Bar(
                x=yrs,
                y=t['AnswerID_count'],
                name=c
                )
        data.append(trace)
    
    layout = go.Layout(
                barmode='stack',
                title = 'Number of AnswerIDs inferred for each customer class for 1 - 15+ years after electrification',
                xaxis = dict(title='Years Electrified',
                                tickvals = yrs),
                yaxis = dict(title='AnswerID count'),
                margin = dict(t=100,r=150,b=50,l=150))
    
    fig = go.Figure(data=data, layout=layout)

    return offline.iplot(fig, filename='answer_id_count')

def plotValidObsRatio(ohp, daytype):
        
    lenx = 15 * 12 * 24 # = years * months * hours
    
    d = ohp.loc[ohp['daytype']==daytype][['class', 'YearsElectrified', 'month', 'hour', 'valid_obs_ratio']]
    d['tix'] = 12*24*(d.YearsElectrified-1) + 24*(d.month-1) + d.hour
    z = d['valid_obs_ratio']*100
    x = d['tix']
    y = d['class']
    hovertext = list() #modify text box on hover
    for row in d.iterrows():
        hovertext.append(list())
        hovertext[-1].append(
                'Year: {}<br />month: {}<br />time: {}h00<br />valid: {}%'.format(
                        row[1]['YearsElectrified'], row[1]['month'], 
                        row[1]['hour'], row[1]['valid_obs_ratio']*100))
    data = [go.Heatmap(z = z, 
                       x = x, 
                       y = y,
                       name = daytype,
                       zmin = 0,
                       zmax = 100,
                       text = hovertext,
                       hoverinfo ="text" , 
                       colorscale=[[0.0, 'rgb(165,0,38)'], 
                                   [0.1111111111111111,'rgb(215,48,39)'],
                                   [0.2222222222222222,'rgb(244,109,67)'],
                                   [0.3333333333333333, 'rgb(253,174,97)'],
                                   [0.4444444444444444, 'rgb(254,224,144)'],
                                   [0.5555555555555556, 'rgb(224,243,248)'],
                                   [0.6666666666666666, 'rgb(171,217,233)'],
                                   [0.7777777777777778, 'rgb(116,173,209)'],
                                   [0.8888888888888888, 'rgb(69,117,180)'],
                                   [1.0, 'rgb(49,54,149)']]
                       )]

    layout = go.Layout(showlegend=False, 
                       title='Percentage valid ' + daytype + ' observations for all inferred classes in data model',
                       margin = dict(t=150,r=150,b=50,l=150),
                       height = 400,
                       yaxis = dict(
                               type = 'category',
                               ticktext = d['class'],
                               tickwidth = 1.5),
                       xaxis = dict(                        
                               title = 'Years Electrified',
                               ticktext = list(range(1, 16)),
                               tickvals = np.arange(12*24/2, lenx+1, 12*24),
                               ),
                               )
                       
    fig = go.Figure(data=data, layout=layout)
                                
    return offline.iplot(fig, filename='valid_obs_ratio_'+daytype)

def plotHourlyProfiles(customer_class, model_cat, daytype='Weekday', years_electrified=7, 
                             model_dir=None, data=None):
    """
    This function plots the hourly load profile for a subset of a customer class for a specified year since electrification. The function requires either a data model or an expert model as input.

    """
    
    if model_cat == 'expert':
        if model_dir is None:        
            df = expert.expertHourlyProfiles()
        else:
            df = expert.expertHourlyProfiles(model_dir)
        df.columns = ['YearsElectrified', 'mean_monthly_kw', 'month', 'daytype', 'hour', 
                      'kva_mean', 'kva_std', 'class']
    elif model_cat == 'data':
        if data is None:
            return(print('Specify the observed hourly load profile dataframe to be used for this graphic.'))
        else:
            df = data[['class', 'YearsElectrified', 'month', 'daytype', 
                         'hour', 'kva_mean', 'kva_std']]
  
    df = df[(df['daytype']==daytype) & (df['YearsElectrified']==years_electrified) & (df['class']==customer_class)]
    if df.empty:
        return(print('Cannot retrieve data for the given submodel parameters. Please specify a different submodel.'))
    else:
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
            hovertext[-1].append('{}h00<br />{:.3f} kVA'.format(yy, z[yi][0]))
            
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
            title= daytype + ' hourly load profile for "' + customer_class + '" customers ' +
                    str(years_electrified) +' years after electrification',
            margin=go.Margin(t=50,r=50,b=50,l=50, pad=10),
            height= 700,
            scene=dict(
                xaxis=dict(
                        title = 'month',
                        type = 'category',
                        ticktext = months,
                        tickvals = np.arange(0.5, 12.5, 1),
                        tickwidth = 1.5,
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
   
    return offline.iplot(fig, filename='annual_hourly_profiles')

def plotProfileSimilarity(merged_hp, customer_class, daytype):
    """
    daytype = one of [Weekday, Saturday, Sunday]
    """
    
    d = merged_hp.loc[(merged_hp['daytype']==daytype) & (merged_hp['class']==customer_class)][['YearsElectrified', 'month', 'hour', 'Mean [kVA]', 'kva_mean', 'kva_std']]
    d['tix'] = 12*24*(d.YearsElectrified-1) + 24*(d.month-1) + d.hour
    d['tixnames'] = d.apply(lambda xd: 'Year '+str(int(xd.YearsElectrified))+
        '<br />Month '+str(int(xd.month))+'<br />'+str(int(xd.hour))+'h00', axis=1)

    trace0 = go.Scatter(
        showlegend=False,
        opacity=0,
        x=d['tix'],
        y=list([0]*len(d)),
        mode='lines',
        name='new-model',
        line=dict(
            color='black',
            width=0.5),
        text=d['tixnames'],
        hoverinfo = 'text',
        hoverlabel = dict(
                bgcolor='white')
        )
    trace1 = go.Scatter(
            x=d['tix'],
            y=d['Mean [kVA]'],
            fill= None,
            mode='lines',
            name='ex model mean',
            line=dict(
                    color='green'),
            hoverinfo='y'
            )
    trace2 = go.Scatter(
        x=d['tix'],
        y=d['kva_mean'],
        fill='tonexty',
        fillcolor='rgb(255, 204, 255)',
        mode='lines',
        name='new model mean',
        line=dict(
            color='purple'),
        hoverinfo='y'
            )
    trace3 = go.Scatter(
        x=d['tix'],
        y=d['kva_std'] + d['kva_mean'],
        mode='lines',
        name='new model std dev',
        line=dict(
            color='purple',
            dash = 'dot'),
        hoverinfo='none'
            )
    
    y4 = [(y>0)*y for y in d['kva_mean'] - d['kva_std']]
    trace4 = go.Scatter(
        x=d['tix'],
        y=y4,
        mode='lines',
        showlegend=False,
        line=dict(
            color='purple',
            dash = 'dot'),
        hoverinfo='none'
            )
    
    data = [trace0, trace1, trace2, trace3, trace4]
    
    layout = go.Layout(showlegend=True, 
                title=daytype + ' load profile model similarity for ' + customer_class + ' customers ',
                margin = dict(t=150,r=150,b=50,l=150),
                height = 400,
                yaxis = dict(
                        title = 'mean hourly demand (kVA)',
                        ticksuffix=' kVA'),
                xaxis = dict(                        
                        title = 'time electrified (years)',
                        ticktext = list(range(0, 16)),
                        tickvals = np.arange(0, (15*12*24)+1, 12*24),
                        rangeslider=dict(),)
                        )
    fig = go.Figure(data=data, layout=layout)
    
    return offline.iplot(fig, filename='profile-similarity')

def plotDemandSimilarity(merged_ds):
    """
    daytype = one of [Weekday, Saturday, Sunday]
    """
    data = []
    lay = []

    #generate existing and new model traces for each customer subclass
    count=1
    for c in merged_ds['class'].unique():
        d = merged_ds.loc[(merged_ds['class']==c)][['YearsElectrified','Energy [kWh]','M_kw_mean','M_kw_std']]

        trace0 = go.Bar(
                x=d['YearsElectrified'],
                y=d['Energy [kWh]'],
                xaxis='x'+str(count),
                yaxis='y'+str(count),
                marker=dict(
                        color='green'),
                name='ex-model'
                )
                
        trace1 = go.Bar(
            x=d['YearsElectrified'],
            y=d['M_kw_mean'],            
            name='new-model',
            opacity=0.5
            )
        
        lay.append({'yaxis{}'.format(count): go.YAxis(type = 'linear',
                            title='annual mean monthly<br /> consumption (kWh)'),
                    'xaxis{}'.format(count): go.XAxis(title = 'time electrified (years)',
                            ticktext = list(range(0, d.YearsElectrified.max()+1)), 
                            tickvals = np.arange(0, d.YearsElectrified.max()+1, 1))
                     })
 
        data.append(trace0)
        data.append(trace1)
        count+=1

    #create subplot graph objects
    rows = int(len(data)/2)
    fig = py.tools.make_subplots(rows=rows, cols=1, subplot_titles=list(merged_ds['class'].unique()), horizontal_spacing = 0.1, print_grid=False)    

    for i in list(range(0,len(data))):
        r = floor(i/2)+1
        fig.append_trace(data[i],r,1)

    fig['layout'].update(showlegend=False, 
                title='15 year annualised monthly demand model similarity',
                barmode='stack')
    
    #update layout for all subplots
    for k in range(0,rows):
        fig['layout'].update(lay[k])
                                
    return offline.iplot(fig, filename='demand-similarity')

def plotMaxDemandSpread(md):

    data = []
    
    for c in md['class'].unique():
        d = md[md['class']==c]
        trace = dict(
            type = 'scatter',
            x=d['month'],
            y=d['hour'],
            mode = 'markers', 
            name = c,
            marker = dict(size = d['Unitsread_kva']*3)
            )
    
        data.append(trace)
    
    return offline.iplot({'data': data}, filename='max-demand-spread')

def plotMonthlyMaxDemand(md):

    data = []
    
    for c in md['class'].unique():
        d = md[md['class']==c]
        trace = dict(
            type = 'scatter',
            x=d['month'],
            y=d['Unitsread_kva'],
            mode = 'markers', 
            name = c,
            marker = dict(size = d['Unitsread_kva']*3
            ))
    
        data.append(trace)
    
    return offline.iplot({'data': data}, filename='monthly-max-demand')

def plotHourlyMaxDemand(md):

    data = []
    
    for c in md['class'].unique():
        d = md[md['class']==c]
        trace = dict(
            type = 'scatter',
            x=d['hour'],
            y=d['Unitsread_kva'],
            mode = 'markers', 
            name = c,
            marker = dict(size = d['Unitsread_kva']*3
            ))
    
        data.append(trace)
    
    return offline.iplot({'data': data}, filename='hourly-max-demand')