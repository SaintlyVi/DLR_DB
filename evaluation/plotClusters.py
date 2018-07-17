#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 09:31:48 2018

@author: saintlyvi
"""

import pandas as pd
import numpy as np

import plotly.plotly as py
import plotly.offline as po
import plotly.graph_objs as go
import plotly.tools as tools
import colorlover as cl
#from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
cf.go_offline()

import matplotlib. pyplot as plt
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap

import evaluation.evalClusters as ec

def plotClusterIndex(index, title, ylog=False):
    cluster_results = ec.readResults()
    experiments = cluster_results.experiment_name.unique()
    colours = dict(zip(experiments, cl.scales[str(len(experiments))]['qual']['Paired']))
    df = cluster_results.set_index(['experiment_name','series'])[[index,'clusters']]
    data = pd.pivot_table(df[[index,'clusters']], index='clusters', columns=df.index, values=index)

    #generate plot data
    traces = []
    for c in data.columns:
        x = data.index
        y = data[c]
        n = '_'.join([c[0].split('_',1)[1],str(c[1])])
        hovertext = list()
        for i in x:
            hovertext.append('{}<br />{}: {:.3f}<br />{} clusters<br />'.format(n, index, y[i], i))

        traces.append(dict(
            x=x,
            y=y,
            name=n,
            legendgroup=c[0].split('_',1)[1],
            mode='lines+markers',
            marker=dict(size=3),
            line=dict(color=colours[c[0]]),
            text = hovertext,
            hoverinfo='text',
            connectgaps=True
        ))

    #set layout
    if ylog == True:
        yax = dict(title = index+' (log scale)' , type='log')
    else:
        yax = dict(title = index)
    layout = go.Layout(
            title= title,
            margin=go.Margin(t=50,r=50,b=50,l=50, pad=10),
            height= 700,
            xaxis=dict(title = 'clusters (log scale)', type='log'),
            yaxis=yax,
            hovermode = "closest"
            )

    fig = {'data':traces, 'layout':layout }
    return po.iplot(fig)

def plotClusterCentroids(centroids, cluster_size, meta, n_best=1):
    
    traces = centroids.T
    n_clust = traces.columns[-1]+1
    traces.columns = ['cluster ' + str(k+1) for k in traces.columns.values]   
    largest = 'cluster '+str(cluster_size.idxmax()+1)
    
    n_best = meta['n_best']
    experiment_name = meta['experiment_name']
    
    colours = cl.interp(cl.scales['12']['qual']['Paired'], 100)[:n_clust]
    
    fig = tools.make_subplots(rows=3, cols=1, shared_xaxes=False, specs=[[{'rowspan': 2}],[None],[{}]],
                              subplot_titles=['cluster profiles '+experiment_name+' (n='+str(n_clust)+
                                              ') TOP '+str(n_best),'cluster sizes'], print_grid=False)  
    i = 0
    for col in traces.columns:
        if col == largest:
            width = 3
        else:
            width = 1
        fig.append_trace({'x': traces.index, 'y': traces[col], 'line':{'color':colours[i],'width':width}, 'type': 'scatter', 'name': col}, 1, 1)
        i+=1
    fig.append_trace({'x': traces.columns, 'y': cluster_size, 'type': 'bar', 'name': str(n_clust)+' clusters'} , 3, 1)
    
    fig['layout']['xaxis1'].update(title='time of day')
    fig['layout']['xaxis2'].update(tickangle=270)
    fig['layout']['yaxis1'].update(title='normalised load profile')
    fig['layout']['yaxis2'].update(title='profile count')
    fig['layout']['margin'].update(t=50,r=80,b=100,l=90,pad=10),
    fig['layout'].update(height=700, hovermode = "closest")
    
    po.iplot(fig)
    
def plotClusterLabels(label_data, year, n_clust=None, som_dim=0):
    
    if n_clust is None:
        c = label_data.columns[0]
    else:
        c = str(som_dim)+'_'+str(n_clust)
    df = label_data.loc[pd.IndexSlice[:,str(year)],[c]].reset_index()
    df.date = df.date.dt.date
    
    fig = df.iplot(kind='heatmap', title='Daily cluster labels for profiles in '+str(year)+
                   ' (cluster ='+c+')', x='date', y='ProfileID', z=c, colorscale='spectral', asFigure=True)

    fig['layout']['yaxis'].update(dict(type='category',title='ProfileID'))
    fig['layout']['xaxis'].update(dict(title='Date'))
    for i, trace in enumerate(fig['data']):
        hovertext = list()
        for j in range(len(trace['x'])):
            hovertext.append('date: {}<br />cluster label: {}<br />ProfileID: {}<br />'.format(trace['x'][j], trace['z'][j]+1, trace['y'][j]))
        trace['text'] = hovertext
        trace['hoverinfo']='text'
    
    return po.iplot(fig)

#---------------------------
# Prepping the colorbar
#---------------------------

def display_cmap(cmap): #Display  a colormap cmap
    plt.imshow(np.linspace(0, 100, 256)[None, :],  aspect=25, interpolation='nearest', cmap=cmap) 
    plt.axis('off')
    
def colormap_to_colorscale(cmap):
    #function that transforms a matplotlib colormap to a Plotly colorscale
    return [ [k*0.1, colors.rgb2hex(cmap(k*0.1))] for k in range(11)]

def colorscale_from_list(alist, name): 
    # Defines a colormap, and the corresponding Plotly colorscale from the list alist
    # alist=the list of basic colors
    # name is the name of the corresponding matplotlib colormap
    
    cmap = LinearSegmentedColormap.from_list(name, alist)
#    display_cmap(cmap)
    colorscale=colormap_to_colorscale(cmap)
    return cmap, colorscale

def normalize(x,a,b): #maps  the interval [a,b]  to [0,1]
    if a>=b:
        raise ValueError('(a,b) is not an interval')
    return float(x-a)/(b-a)

def asymmetric_colorscale(data,  div_cmap, ref_point=0.0, step=0.05):
    #data: data can be a DataFrame, list of equal length lists, np.array, np.ma.array
    #div_cmap is the symmetric diverging matplotlib or custom colormap
    #ref_point:  reference point
    #step:  is step size for t in [0,1] to evaluate the colormap at t
   
    if isinstance(data, pd.DataFrame):
        D = data.values
    elif isinstance(data, np.ma.core.MaskedArray):
        D=np.ma.copy(data)
    else:    
        D=np.asarray(data, dtype=np.float) 
    
    dmin=np.nanmin(D)
    dmax=np.nanmax(D)
    if not (dmin < ref_point < dmax):
        raise ValueError('data are not appropriate for a diverging colormap')
        
    if dmax+dmin > 2.0*ref_point:
        left=2*ref_point-dmax
        right=dmax
        
        s=normalize(dmin, left,right)
        refp_norm=normalize(ref_point, left, right)# normalize reference point
        
        T=np.arange(refp_norm, s, -step).tolist()+[s]
        T=T[::-1]+np.arange(refp_norm+step, 1, step).tolist()
        
        
    else: 
        left=dmin
        right=2*ref_point-dmin
        
        s=normalize(dmax, left,right) 
        refp_norm=normalize(ref_point, left, right)
        
        T=np.arange(refp_norm, 0, -step).tolist()+[0]
        T=T[::-1]+np.arange(refp_norm+step, s, step).tolist()+[s]
        
    L=len(T)
    T_norm=[normalize(T[k],T[0],T[-1]) for k in range(L)] #normalize T values  
    return [[T_norm[k], colors.rgb2hex(div_cmap(T[k]))] for k in range(L)]

#---------------------------
# end colorbar functions
#---------------------------


def plotClusterSpecificity(data, corr_list, n_clust=None):
    
    print('n_clust options: \n', data.columns.values)
    
    if n_clust is None:
        n_clust = data.columns[0]

    n_corr = len(corr_list)    
    
    #Create dataframes for plot
#    df = data.reset_index()
    
    subplt_titls = ()
    titles = []
    for corr in corr_list:
        title = '"Greater than random" probability of '+corr+' assigned to cluster'
        titles.append((title, None))    
    for t in titles:
        subplt_titls += t
    
    #Initialise plot
    fig = tools.make_subplots(rows=n_corr, cols=2, shared_xaxes=False, print_grid=False, 
                              subplot_titles=subplt_titls)
    #Create colour scale
    smarties = cl.scales['5']['div']['Spectral']
    slatered=['#232c2e', '#ffffff','#c34513']
    label_cmap, label_cs = colorscale_from_list(slatered, 'label_cmap') 
    
    i = 1
    for corr in corr_list:
        function = 'ec.'+corr+'Corr(data, n_clust)'
        rndm_lklhd, lbls2 = eval(function)   

        #Create colorscales
        colorscl= asymmetric_colorscale(lbls2, label_cmap, ref_point=1.0)
#        colorscl=[[0.0, 'rgb(112,138,144)'],[white, 'rgb(255,255,255)'],[1.0, 'rgb(239,138,98)']]

        #Create traces
        heatmap = go.Heatmap(z = lbls2.T.values, x = lbls2.index, y = lbls2.columns, name = corr, 
                          colorscale=colorscl, colorbar=dict(title='likelihood',len=0.9/n_corr, y= 1-i/n_corr+0.05/i, yanchor='bottom'))
        bargraph = lbls2.iplot(kind='bar', colors=smarties, showlegend=False, asFigure=True)

        fig.append_trace(heatmap, i, 1)
        for b in bargraph['data']:
            fig.append_trace(b, i, 2)
        random_likelihood=dict(type='scatter', x=[lbls2.index[0], lbls2.index[-1]], y=[1, 1], 
                                       mode='lines', line=dict(color='black',dash='dash'))
        fig.append_trace(random_likelihood, i, 2)
        
        fig['layout']['yaxis'+str(i*2)].update(title='greater than random Passignment')
        fig['layout']['annotations'].extend([dict(x = lbls2.index[int(len(lbls2.index)*0.5)], y = 1, showarrow=True, yshift=5,
                                              text="random assignment",ax=10, ay=-70, xref='x'+str(i*2), yref='y'+str(i*2))])
        
        i += 1

    #Update layout
    fig['layout'].update(title='Temporal specificity of k clusters', height=n_corr*400, hovermode = "closest", showlegend=False) 

    po.iplot(fig)