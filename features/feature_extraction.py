# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 13:59:37 2017

@author: Wiebke Toussaint
"""

import pandas as pd
import json
import os

from support import feature_dir, fdata_dir
import features.feature_socios as socios

def generateData(year, spec_file):
    """
    This function generates a json formatted evidence text file compatible with the syntax for providing evidence the python library libpgm for the specified year. The function requires a json formatted text file with feature specifications as input.
    
    """
    #Get feature specficiations
    file_path = os.path.join(feature_dir, 'specification', spec_file + '.txt')
    with open(file_path, 'r') as f:
        featurespec = json.load(f)

    searchlist = featurespec['searchlist']
    features = featurespec['features']
    transform = featurespec['transform']
    bins = featurespec['bins']
    labels = featurespec['labels']
    cut = featurespec['cut']
    
    #Get data and questions from socio-demographic survey responses
    data, questions = socios.buildFeatureFrame(searchlist, year)
    
    #Transform and select BN nodes from dataframe 
    for k, v in transform.items():
        data[k] = data.apply(lambda x: eval(v), axis=1)
    data = data[['AnswerID'] + features]
    
    #Cut columns into datatypes that match factors of BN node variables    
    for k, v in bins.items():
        bin_vals = [int(b) for b in v]
        try:
            cut_right = eval(cut[k]['right'])
            cut_include_lowest = eval(cut[k]['include_lowest'])
        except KeyError:
            pass
        data[k] = pd.cut(data[k], bins = bin_vals, labels = labels[k], right=cut_right, include_lowest=cut_include_lowest)
        
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
        
def saveData(yearstart, yearend, spec_file, output_name='evidence'):
    """
    This function saves an evidence dataset with observations in the data directory.
    
    """
    for year in range(yearstart, yearend + 1):
        
        #Generate evidence data
        evidence = generateData(year, spec_file)
        root_name = spec_file.split('_')[0]
        
        #Save data to disk
        file_name =  root_name + '_' + output_name + '_' + str(year) + '.txt'
        dir_path = os.path.join(fdata_dir, root_name)
        os.makedirs(dir_path , exist_ok=True)
        file_path = os.path.join(dir_path, file_name)
        
        with open(file_path, 'w') as f:
            json.dump(evidence, f)
        print('Successfully saved to ' + file_path)
    
    return