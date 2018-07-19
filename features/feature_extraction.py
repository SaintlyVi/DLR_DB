# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 13:59:37 2017

@author: Wiebke Toussaint
"""

import pandas as pd
import os
import numpy as np

from support import fdata_dir, writeLog
import features.feature_socios as socios

def features2dict(data):      
    """            
    This function converts a dataframe into a dict formatted for use as evidence in libpgm BN inference.
    
    Unsure if this function has much use going forward ...19 July 2018
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