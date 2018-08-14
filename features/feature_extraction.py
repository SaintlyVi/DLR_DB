#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 13:59:37 2017

@author: Wiebke Toussaint
"""

import pandas as pd
import os
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

from support import writeLog, image_dir, data_dir, experiment_dir
from features.feature_socios import genS
from evaluation.eval_clusters import getLabels, getExpDetails

def plotF(F, columns, save_name=None):
    """
    Plots column category counts in F. Columns must be a list.
    """
    if save_name is None:
        dir_path = os.path.join(image_dir, 'experiment')
    else:
        dir_path = os.path.join(image_dir, 'experiment', save_name)
    os.makedirs(dir_path, exist_ok=True)
    
    data = F[['ProfileID']+columns]

    if len(columns) == 1:
        fig, axes = plt.subplots(nrows=1, ncols=1)
        data.groupby(columns).ProfileID.count().plot('bar', 
                          title='Count of observations in ' + columns[0] + ' bins', figsize=(10, 6))
        plt.tight_layout()
        fig.savefig(os.path.join(dir_path, columns[0]+'.png'))
        plt.close()
    
    elif len(columns) > 1:
        fig, axes = plt.subplots(nrows=len(data.columns)-1, ncols=1)
        r = 0
        for c in data.columns.drop('ProfileID'):
            data.groupby(c).ProfileID.count().plot('bar', 
                        title='Count of observations in ' + c + ' bins', 
                        ax=axes[r], figsize=(10-len(data.columns)/4, len(data.columns)*4))
            r += 1
            plt.tight_layout()            

        fig.savefig(os.path.join(dir_path, save_name+'.png'))
        plt.close()
    
    return fig

def genFProfiles(experiment, socios, n_best=1, savefig=False):
    """
    generates a socio-demographic feature set
    """
    
    year_start, year_end, drop_0, prepro, exp_root = getExpDetails(experiment)
    
    kf_dir = os.path.join(data_dir, 'cluster_evaluation','k_features', experiment+'_'+socios+'BEST'+
                          str(n_best))
    kf_path = kf_dir+'.csv'
    if os.path.exists(kf_path) is True:
        F = pd.read_csv(kf_path)

    else:
        os.makedirs(kf_dir, exist_ok=True)
        print('Extracting and creating feature data...')
        # Get cluster labels
        XL = getLabels(experiment, n_best)
#        XL['DD'] = XL.iloc[:,list(range(0,24))].sum(axis=1)
        XL = XL.drop(columns=[str(i) for i in range(0,24)], axis=0).reset_index()
    
        # Add temporal features
        XL['year']=XL.date.dt.year
        XL['month']=XL.date.dt.month_name()
        XL['weekday']=XL.date.dt.weekday_name
        
        winter = ['May','June','July','August']
        work_week = ['Monday','Tuesday','Wednesday','Thursday']
        
        XL['season'] = XL.month.where(XL.month.isin(winter), 'summer')
        XL['season'] = XL.season.where(XL.season=='summer', 'winter')
        XL['daytype'] = XL.weekday.where(~XL.weekday.isin(work_week), 'weekday')
#        XL.drop(columns=['date','month','weekday','elec_bin'], inplace=True)

        kXL = XL.groupby(['ProfileID','year','season','daytype'])['k'].value_counts().reset_index(name='k_count')
        kXL = kXL[kXL.k_count>1] #keep rows with two or more occurences of k
        
        S = genS(socios, year_start, year_end, 'feather').reset_index()  
        
        #merge socio-demographic, geographic and temporal features
        F = pd.merge(kXL, S, how='inner',on='ProfileID')
        del XL, S
        
#        monthly_consumption_bins = [5, 50, 150, 400, 600, 1200, 2500, 4000]
#        daily_demand_bins = [x /30*1000/230 for x in monthly_consumption_bins]
#        bin_labels = ['{0:.0f}-{1:.0f}'.format(x,y) for x, y in zip(daily_demand_bins[:-1], daily_demand_bins[1:])]
#        F['DD'] = pd.cut(F.DD, daily_demand_bins, labels=bin_labels, right=False)
#        F['DD'] = F.DD.where(~F.DD.isna(), 0)
#        F['DD'] = F.DD.astype('category')
#        F.DD.cat.reorder_categories([0]+bin_labels, ordered=True,inplace=True)
        
#        F.drop('ProfileID', axis=1, inplace=True)
        columns = F.columns.tolist()
        columns.remove('k')
        F = F[columns + ['k']]
        F['k'] = F['k'].astype('category')
        
        F.to_csv(kf_path, index=False)
        
        for y in range(year_start, year_end+1):
            for c in F.columns:
                F[c] = F[c].astype('category')
            Y = F[F.year==y]
            Y.drop(columns=['ProfileID','year'], inplace=True)
            Y.to_csv(os.path.join(kf_dir, str(y)+'.csv'), index=False)
    
    if savefig is True:
        for c in F.columns:#.drop(['ProfileID']):
            plotF(F, [c], socios)
    
    return F

def genArffFile(experiment, socios, skip_cat=None, weighted=True, n_best=1):
    
    kf_name = experiment+'_'+socios+'BEST'+ str(n_best)
    kf_dir = os.path.join(data_dir, 'cluster_evaluation','k_features', kf_name)
    if weighted == True:
        kf_path = kf_dir+'.arff'
    elif weighted == False:
        kf_path = kf_dir+'noW.arff'
    
    F = genFProfiles(experiment, socios, n_best)
    F.drop('ProfileID', axis=1, inplace=True)
    attributes = [] 
    for c in F.columns:
        if c == 'k_count':
            pass
        if c in skip_cat:
            att = '@attribute ' + c + ' numeric'
            attributes.append(att)
        else:
            att = '@attribute ' + c
            cats = F[c].astype('category')
            att += ' {'+",".join(map(str, cats.cat.categories))+'}'
            attributes.append(att)
    
    F.fillna('?', inplace=True)
            
    with open(kf_path, 'a+') as myfile:
        myfile.write('@relation ' + kf_name + '\n\n')
        for a in attributes:  
            myfile.write(a+'\n')
        myfile.write('\n@data\n')        
        for r in F.iterrows(): 
            if weighted == True:
                weight = r[1]['k_count']
            elif weighted == False:
                weight = ''
            vals = r[1].drop('k_count')
            myfile.write(','.join(map(str,vals)) + ',{'+str(weight)+'}\n')

    return print('Successfully created',experiment, socios, 'arff file.')

def genFHouseholds(experiment, socios, n_best=1):
    
    F = genFProfiles(experiment, socios, n_best, savefig=False)  
    Fhh = F.iloc[F.groupby(['ProfileID','season','daytype'])['k_count'].idxmax()]     
    
#    Fsub.to_csv(kf_path, index=False)
    
    return Fhh

def features2dict(data):      
    """            
    This function converts a dataframe into a dict formatted for use as evidence in libpgm BN inference.
    
    Unsure if this function has much use going forward ...19 July 2018
    
    if drop_0 is False:
        f_path = os.path.join(data_dir, 'cluster_evaluation', 'k_features', 
                                  experiment+'_'+socios+str(year_start)+str(year_end)+'.feather')
    elif drop_0 is True:
        f_path = os.path.join(data_dir, 'cluster_evaluation', 'k_features', 
                                  experiment+'_'+socios+'drop0'+str(year_start)+str(year_end)+'.feather')  

    if os.path.exists(f_path) is True:
        F = feather.read_dataframe(f_path)
    """
    for c in data.columns:
        data[c].replace(np.nan, '', regex=True, inplace=True) #remove nan as BN inference cannot deal 
    featuredict = data.to_dict('index') 
    e = []
    for f in featuredict.values(): 
        d = dict()
        for k, v in f.items():
            if v is not str(''):
                d[k] = v
        e.append(d)  
    evidence = dict(zip(featuredict.keys(), e))
    
    return evidence

def checkFeatures(data, appliances):
    """
    This function error checks appliance features for records that indicate appliance usage but no ownership.
    """
    
    err = pd.DataFrame()
    for a in appliances:
        try:
            e = data.loc[(data[a]==0)&(data[a+'_use']>0), [a,a+'_use',a+'_broken']]
            print(e)
            err = err.append(e)
        except:
            pass
        
    return err