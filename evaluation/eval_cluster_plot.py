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

import evaluation.eval_clusters as ec

def plotPrettyColours(data, grouping):
   
    if grouping == 'experiments':
        colour_seq = ['Reds','Oranges','YlOrBr','YlGn','Greens','BuGn','Blues',
                      'PuBu','Purples','PuRd']
        df = pd.DataFrame(data.experiment_name.unique(), columns=['name'])
        df['root'] = df.applymap(lambda x: '_'.join(x.split('_',2)[0:2]))
        
    elif grouping == 'elec_bin':
        colour_seq = ['YlGn','PuRd','Blues','YlOrBr','Greens','Oranges',
                      'Purples','PuBu','BuGn','Reds']
        df = data['elec_bin'].reset_index().rename({'elec_bin':'root', 'k':'name'}, axis=1)

    df['root'] = df.root.astype('category')                
    df.root.cat.rename_categories(colour_seq[:len(df.root.cat.categories)], inplace=True)
    col_temp = df.groupby('root').apply(lambda x: len(x))
    
    my_cols = list()
    for c, v in col_temp.items():
        try:
            i = 0
            gcol=list()
            while i < v:
                gcol.append(cl.scales['9']['seq'][c][2+i])
                i+=1
        except:
            i = 0
            gcol=list()
            jump = int(80/v)
            while i < v:
                gcol.append(cl.to_rgb(cl.interp(cl.scales['9']['seq'][c], 100))[-1-jump*i])
                i+=1
        my_cols+=gcol
    
    colours = dict(zip(df.name, my_cols))
    
    return colours

def plotClusterIndex(index, title, experiments, groupby='algorithm', ylog=False):
    cluster_results = ec.readResults()
    cluster_results = cluster_results.groupby(['experiment_name','som_dim','n_clust']).mean().reset_index() 
    cluster_results = cluster_results[cluster_results.experiment_name.isin(experiments)]
    cluster_results['series'] = cluster_results['som_dim'].where(
         (cluster_results['n_clust'] != 0) & 
         (cluster_results['som_dim'] != 0), '')
    
    colours = plotPrettyColours(cluster_results,'experiments')
    df = cluster_results.set_index(['experiment_name','series'])[[index,'clusters']]
    data = pd.pivot_table(df[[index,'clusters']], index='clusters', columns=df.index, values=index)

    #generate plot data
    groupdict = dict(zip(['experiment','algorithm'],[0,1]))
    traces = []
    for c in data.columns:
        t = c[0].split('_',1)+[str(c[1])]
        x = data.index
        y = data[c]
        n = ' '.join(t)
        hovertext = list()
        for i in x:
            hovertext.append('{}<br />{}: {:.3f}<br />{} clusters<br />'.format(n, index, y[i], i))

        traces.append(dict(
            x=x,
            y=y,
            name=n,
            legendgroup=t[groupdict[groupby]],
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

def plotClusterCentroids(centroids, n_best=1):

    traces = centroids.iloc[:, 0:24].T
    n_clust = traces.columns[-1]
    traces.columns = ['cluster ' + str(k) for k in traces.columns.values]   
    largest = 'cluster '+str(centroids.cluster_size.idxmax())
    
    n_best = centroids['n_best'].unique()[0]
    experiment_name = centroids['experiment'].unique()[0]
    
    colours =  plotPrettyColours(centroids, 'elec_bin')    
    fig = tools.make_subplots(rows=3, cols=1, shared_xaxes=False, specs=[[{'rowspan': 2}],[None],[{}]],
                              subplot_titles=['cluster profiles '+experiment_name+' (n='+str(n_clust)+
                                              ') TOP '+str(n_best),'cluster sizes'], print_grid=False)  
    i = 0
    for col in traces.columns:
        if col == largest:
            width = 3
        else:
            width = 1
        fig.append_trace({'x': traces.index, 'y': traces[col], 'line':{'color':colours[i+1],'width':width}, 
                          'type': 'scatter', 'legendgroup':centroids['elec_bin'][i+1], 'name': col}, 1, 1)
#        fig.append_trace({'x': col, 'y': cluster_size[i+1], 'type': 'bar', 
#                          'legendgroup':centroids['elec_bin'][i+1], 'name': col} , 3, 1)
        i+=1
    for b in centroids['elec_bin'].unique():
        t = centroids.cluster_size[centroids.index[centroids.elec_bin==b]]
        fig.append_trace({'x': t.index.values, 'y': t.values, 'type': 'bar', 'legendgroup':b, 'name': b, 
                          'marker': dict(color=[colours[k] for k in t.index.values])} , 3, 1)
#        fig.append_trace({'x': traces.columns, 'y': cluster_size, 'type': 'bar', 'legendgroup':centroids['elec_bin'][i+1], 'name': str(n_clust)+' clusters'} , 3, 1)
    
    fig['layout']['xaxis1'].update(title='time of day')
    fig['layout']['xaxis2'].update(tickangle=270)
    fig['layout']['yaxis1'].update(title='load profile')
    fig['layout']['yaxis2'].update(title='profile count')
    fig['layout']['margin'].update(t=50,r=80,b=100,l=90,pad=10),
    fig['layout'].update(height=700, hovermode = "closest")
    
    po.iplot(fig)
    
def plotClusterLabels(label_data, year, n_clust=None, som_dim=0):
    
#    if n_clust is None:
#        c = label_data.columns[0]
#    else:
#        c = str(som_dim)+'_'+str(n_clust)
    df = label_data.loc[pd.IndexSlice[:,str(year)],'k'].reset_index()
    df.date = df.date.dt.date
    
    fig = df.iplot(kind='heatmap', title='Daily cluster labels for profiles in '+str(year), x='date', y='ProfileID', z='k', colorscale='spectral', asFigure=True)

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
        function = 'ec.'+corr+'Corr(data)'
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
    
def plotClusterMetrics(metrics_dict, title, metric=None, make_area_plot=False, ylog=False):

    #format plot attributes
    if make_area_plot == True:
        fillme = 'tozeroy'
    else:
        fillme = None
    
    colours = ['Red','Green','Orange','Blue','Purple','Brown','Black','Yellow','PuBu','BuGn',]
#    colours = cl.scales[str(len(metrics_dict.keys()))]['div']['Spectral']

    #generate plot data
    traces = []
    s = 0
    for k,v in metrics_dict.items():
        for i,j in v.items():
            if metric is None:
                grouped=i
                pass
            elif i in metric:
                grouped=None
                pass
            else:
                continue
            x = j.index
            y = j
            traces.append(dict(
                x=x,
                y=y,
                name=k+' | '+i,
                legendgroup=grouped,
                mode='lines',
                marker=dict(size=3),
                line=dict(color=colours[s]),
                fill=fillme,
                connectgaps=True
            ))
        s += 1

    #set layout
    if ylog == True:
        yax = dict(title = 'metric (log scale)' , type='log')
    else:
        yax = dict(title = 'metric')
    layout = go.Layout(
            title= 'Comparison of '+title+' for different model runs',
            margin=go.Margin(t=50,r=50,b=50,l=50, pad=10),
            height= 300+len(traces)*15,
            xaxis=dict(title = 'clusters'),
            yaxis=yax,
            hovermode = "closest"
            )

    fig = {'data':traces, 'layout':layout }
    return po.iplot(fig)