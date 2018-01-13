#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 10:01:03 2017

@author: SaintlyVi

This module creates a graphical visualisation of a Bayesian Network. It requires the daft package for graph construction. Install the most recent version with 'pip install daft'.

More info on daft is available here: http://daft-pgm.org
"""

from math import ceil
import numpy as np

#initiate offline plotting for plotly
import colorlover as cl
import plotly.offline as offline
import plotly.graph_objs as go
from plotly import tools
offline.init_notebook_mode(connected=True)

from experiment.algorithms.bn import inferCustomerClasses
    
def plotClassDist(model, evidence_dir, year):
    """
    This function plots the probability distribution over all the inferred classes for all the AnswerIDs 
    in a given year.
    """
    colors = cl.flipper()['div']['5']['RdGy']
    scl = [[0,colors[2]],[0.25,colors[3]],[0.5,colors[4]],[0.75,colors[1]],[1,colors[0]]]
 
    df, bn = inferCustomerClasses(model, evidence_dir, year)
    melt = df.reset_index().melt(id_vars='AnswerID')
    melt['tixnames'] = melt.apply(lambda x: 'AnswerID: '+ x['AnswerID']+'<br />class: '+ x['variable']+'<br />likelihood: '+"{0:.3f}".format(x['value']), axis=1)
    trace = go.Heatmap(z=melt.value,
                       x=melt.AnswerID,
                       y=melt.variable,
                       colorscale = scl,
                       colorbar=dict(
                                   title='likelihood'),
                       text=melt['tixnames'],
                       hoverinfo='text'
                       )
     
    data=[trace]
    layout = go.Layout(title='Probability Distribution of Customer Classes for ' + str(year),
                    xaxis=dict(
                        title='household IDs',
                        type = 'category',
                        showticklabels=False,
                        ticks = '',
                        showline=True,
                        ),
                    yaxis=dict(
                        type = 'category',
                        showline=True,),
                    margin=go.Margin(
                                    l=175,
                                    r=75,
                                    b=50,
                                    t=100
                                )
                    )
    
    return offline.iplot({"data":data, "layout":layout})

def plotClassYearRange(yearstart, yearend, model, evidence_dir):
    """
    This function creates subplots of the probability distribution over all the inferred classes 
    for a range of years.
    """
    colors = cl.flipper()['div']['5']['RdGy']
    scl = [[0,colors[2]],[0.25,colors[3]],[0.5,colors[4]],[0.75,colors[1]],[1,colors[0]]]

    ncol = 3
    nplots = yearend - yearstart + 1
    nrow = int(ceil(nplots/ncol))
   
    fig = tools.make_subplots(rows=nrow, cols=int(ncol), subplot_titles=list(range(yearstart, yearend + 1)), print_grid=False)
    
    r = 1
    c = 1
        
    for y in range(yearstart, yearend + 1):
        if c == ncol + 1: 
            c = 1
        ro = int(ceil(r/ncol))
        
        if r == 1: #toggle colorscale
            scl_switch=True
        else:
            scl_switch=False
        
        try:
            df, bn = inferCustomerClasses(model, evidence_dir, y)
            melt = df.reset_index().melt(id_vars='AnswerID')
            melt['tixnames'] = melt.apply(lambda x: 'AnswerID: '+ x['AnswerID']+'<br />class: '+ x['variable']+'<br />likelihood: '+"{0:.3f}".format(x['value']), axis=1)
            trace = go.Heatmap(z=melt.value,
                               x=melt.AnswerID,
                               y=melt.variable,
                               text=melt['tixnames'],
                               hoverinfo='text',
                               colorscale = scl,                               
                               showscale=scl_switch,
                               colorbar=dict(
                                           title='likelihood',
                                           len=0.5,
                                           yanchor='bottom'))
            fig.append_trace(trace, ro, c)
            
        except:
            pass
        
        c += 1
        r += 1
    
    fig['layout'].update(showlegend=False, title='Probability Distribution of Customer Classes from' + str(yearstart)+'-'+str(yearend),
      height=350+300*(nrow-1))

    for k in np.arange(1, yearend+1, 3):
          fig['layout'].update({'yaxis{}'.format(k): go.YAxis(type = 'category',
                                                              showline=True),
                                'xaxis{}'.format(k): go.XAxis(#title = 'household IDs', 
                                                              type = 'category',
                                                              showticklabels=False,
                                                              ticks = '',
                                                              showline=True)
                                                                })  
    
    for k in np.setdiff1d(np.arange(1, 8),np.arange(1, 8, 3)):
          fig['layout'].update({'yaxis{}'.format(k): go.YAxis(showticklabels=False,
                                                              ticks = '',
                                                              showline=True),
                                'xaxis{}'.format(k): go.XAxis(#title = 'household IDs', 
                                                              type = 'category',
                                                              showticklabels=False,
                                                              ticks = '',
                                                              showline=True)
                                                                })        
      
    
    return offline.iplot(fig)