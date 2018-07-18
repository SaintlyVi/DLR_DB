# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 13:59:37 2017

@author: Wiebke Toussaint
"""

import pandas as pd
import json
import os
from glob import glob
import numpy as np

from support import feature_dir, fdata_dir, InputError, writeLog, validYears
import features.feature_socios as socios

def generateFeatureSetSingle(year, spec_file, set_id='ProfileID'):
    """
    This function generates a json formatted evidence text file compatible with the syntax for providing evidence to the python library libpgm for the specified year. The function requires a json formatted text file with feature specifications as input.
    
    """
    #Get feature specficiations
    files = glob(os.path.join(feature_dir, 'specification', spec_file + '*' + '.txt'))

    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                featurespec = json.load(f)        
            year_range = featurespec['year_range']
        except:
            raise InputError(year, 'Problem reading the spec file.')
            
        if year >= int(year_range[0]) and year <= int(year_range[1]):            
            validYears(year) #check if year input is valid
            break
        else:
            continue
            
    searchlist = featurespec['searchlist']
    features = featurespec['features']
    transform = featurespec['transform']
    bins = featurespec['bins']
    labels = featurespec['labels']
    cut = featurespec['cut']
    if len(featurespec['geo'])==0:
        geo = None
    else:
        geo = featurespec['geo']
    
    #Get data and questions from socio-demographic survey responses
    data = socios.extractFeatures(searchlist, year, col_names=searchlist, geo=geo)
    missing_cols = list(set(searchlist) - set(data.columns))
    data = data.append(pd.DataFrame(columns=missing_cols)) #add columns dropped during feature extraction
    data.fillna(0, inplace=True) #fill na with 0 to allow for further processing
    data['AnswerID'] = data.AnswerID.astype(int)
    data['ProfileID'] = data.ProfileID.astype(int)

    if len(data) is 0:
        raise InputError(year, 'No survey data collected for this year')
    
    else:
        #Transform and select BN nodes from dataframe 
        for k, v in transform.items():
            data[k] = data.apply(lambda x: eval(v), axis=1)
        try:
            data = data[[set_id, geo] + features]
        except:
            data = data[[set_id] + features]
            
    #adjust monthly income for inflation. 
    #Important that this happens here, after columns have been renamed and before income data is binned
    if 'monthly_income' in features:
        cpi_percentage=(0.265,0.288,0.309,0.336,0.359,0.377,0.398,0.42,0.459,0.485,0.492,0.509,
                    0.532,0.57,0.636,0.678,0.707,0.784,0.829,0.88,0.92,0.979,1.03)
        cpi = dict(zip(list(range(1994,2015)),cpi_percentage))
        data['monthly_income'] = data['monthly_income']/cpi[year]
    
        #Cut columns into datatypes that match factors of BN node variables    
        for k, v in bins.items():
            bin_vals = [int(b) for b in v]
            try:
                data[k] = pd.cut(data[k], bins = bin_vals, labels = labels[k],
                    right=eval(cut[k]['right']), 
                               include_lowest=eval(cut[k]['include_lowest']))
            except KeyError:
                data[k] = pd.cut(data[k], bins = bin_vals, labels = labels[k])                                  
        
        data.set_index(set_id, inplace=True) #set AnswerID column as index

    return data
                 
def generateFeatureSetMulti(spec_files, year_start=1994, year_end=2014):

    if isinstance(spec_files, list):
        pass
    else:
        spec_files = [spec_files]
    
    ff = pd.DataFrame()    
    for spec in spec_files:
        gg = pd.DataFrame()
        for year in range(year_start, year_end+1):
            try:
                gg = gg.append(generateFeatureSetSingle(year, spec))
            except Exception:
                ## TODO this should be logged
                print('Could not extract features for ' + str(year) + ' with spec ' + spec)
            pass
        ff = ff.merge(gg, left_index=True, right_index=True, how='outer')

    ff = ff[~ff.index.duplicated(keep='first')] #problem with profile_id 8396, answer id 2000458
    
    return ff

def features2dict(data):      
    """            
    This function converts a dataframe into a dict formatted for use as evidence in libpgm BN inference.
    """
    for c in data.columns:
            data[c] = data[c].astype(int)
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

def saveFeatures(spec_files, year_start, year_end):
    """
    This function saves an evidence dataset with observations in the data directory.
    
    """
    loglines = []

    if isinstance(spec_files, list):
        pass
    else:
        spec_files = [spec_files]
        
    #Save data to disk
    root_name = '_'.join(spec_files)
    file_name =  root_name + '_' + str(year_start) + '+'+ str(year_end-year_start) + '.txt'
    dir_path = os.path.join(fdata_dir, root_name)
    os.makedirs(dir_path , exist_ok=True)
    file_path = os.path.join(dir_path, file_name)

    #Generate evidence data
    evidence = generateFeatureSetMulti(spec_files, year_start, year_end)
    status = 1      
    message = 'Success!'
    evidence.to_json(file_path)
    print('Success! Saved to data/feature_data/' + root_name + '/' + file_name) ## TODO this message must move
        
    l = ['featureExtraction', year_start, year_end, status, message, spec_files, file_name]
    loglines.append(l)
        
    logs = pd.DataFrame(loglines, columns = ['process','from year','to year','status','message',
                                             'features', 'output_file'])
    writeLog(logs,'log_generateData')
            
    return
