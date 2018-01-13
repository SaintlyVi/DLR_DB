# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 13:59:37 2017

@author: Wiebke Toussaint
"""

import pandas as pd
import json
import os
import numpy as np

from support import feature_dir, fdata_dir, InputError, writeLog
import features.feature_socios as socios

def generateData(year, spec_file):
    """
    This function generates a json formatted evidence text file compatible with the syntax for providing evidence the python library libpgm for the specified year. The function requires a json formatted text file with feature specifications as input.
    
    """
    #Get feature specficiations
    file_path = os.path.join(feature_dir, 'specification', spec_file + '.txt')
    with open(file_path, 'r') as f:
        featurespec = json.load(f)
    
    year_range = featurespec['year_range']
    
    if year >= int(year_range[0]) and year <= int(year_range[1]):
        
        searchlist = featurespec['searchlist']
        features = featurespec['features']
        transform = featurespec['transform']
        bins = featurespec['bins']
        labels = featurespec['labels']
        cut = featurespec['cut']
        
        #Get data and questions from socio-demographic survey responses
        data, questions = socios.buildFeatureFrame(searchlist, year)
        
        try:
            #Transform and select BN nodes from dataframe 
            for k, v in transform.items():
                data[k] = data.apply(lambda x: eval(v), axis=1)
            data = data[['AnswerID'] + features]
            
            #Cut columns into datatypes that match factors of BN node variables    
            for k, v in bins.items():
                bin_vals = [int(b) for b in v]
                try:
                    data[k] = pd.cut(data[k], bins = bin_vals, labels = labels[k],
                        right=eval(cut[k]['right']), 
                                   include_lowest=eval(cut[k]['include_lowest']))
                except KeyError:
                    data[k] = pd.cut(data[k], bins = bin_vals, labels = labels[k])                                  
                
            for c in data.columns:
                if c not in bins.keys():
                    data[c] = data[c].map("{:.0f}".format, na_action='ignore')
            
            data.set_index('AnswerID', inplace=True) #set AnswerID column as index
            
            #Convert dataframe into a dict formatted for use as evidence in libpgm BN inference
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
        
        except ValueError:
            pass
        
    else:
        raise InputError(year, 'The input year is out of range of the specification.') 
        
def saveData(yearstart, yearend, spec_file, output_name='evidence'):
    """
    This function saves an evidence dataset with observations in the data directory.
    
    """
    for year in range(yearstart, yearend + 1):
        
        #Save data to disk
        root_name = spec_file.split('_')[0]
        file_name =  root_name + '_' + output_name + '_' + str(year) + '.txt'
        dir_path = os.path.join(fdata_dir, root_name)
        os.makedirs(dir_path , exist_ok=True)
        file_path = os.path.join(dir_path, file_name)
        
        try:
            #Generate evidence data
            evidence = generateData(year, spec_file)
            print('Saving data to feature_data/' + root_name + '/' + file_name)
        except InputError as e:
            print(e)
            print('Saving empty file')
            evidence = None
        
        with open(file_path, 'w') as f:
            json.dump(evidence, f)
        
    logline = [yearstart, yearend, spec_file]
    log_lines = pd.DataFrame([logline], columns = ['from_year','to_year', 'specification_name'])
    writeLog(log_lines,'log_feature_ext')
            
    return
