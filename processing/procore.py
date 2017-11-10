#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 17:55:45 2017

@author: saintlyvi
"""

import numpy as np
import pandas as pd
import feather
from glob import glob
import os
from pathlib import Path

from support import rawprofiles_dir, hourlyprofiles_dir, table_dir

def reduceRawProfiles(year):
    """
    This function uses a rolling window to reduce all raw load profiles to hourly mean values. Monthly load profiles are then concatenated into annual profiles and returned as a dictionary object.
    The data is structured as follows:
        dict[unit:{year:[list_of_profile_ts]}]
    
    """
    p = Path(os.path.join(rawprofiles_dir, str(year)))
    
    for unit in ['A', 'V', 'kVA', 'Hz', 'kW']:
        #create empty directory to save files   
        dir_path = os.path.join(hourlyprofiles_dir, unit)
        os.makedirs(dir_path, exist_ok=True)
        #initialise empty dataframe to concatenate annual timeseries
        ts = pd.DataFrame()
        #iterate through all data files to combine 5min monthly into hourly reduced annual timeseries
        for child in p.iterdir():  
            try:
                childpath = glob(os.path.join(child, '*_' + unit + '.feather'))[0]
                data = feather.read_dataframe(childpath)
                data.Datefield = np.round(data.Datefield.astype(np.int64), -9).astype('datetime64[ns]')
                data['Valid'] = data['Valid'].map(lambda x: x.strip()).map({'Y':True, 'N':False})
                if unit in ['A','V','Hz','kVA','kW']:
                    hourlydata = data.groupby(['RecorderID', 'ProfileID']).resample('H', on='Datefield').mean()
                else:
                    print("Unit must be one of 'A', 'V', 'kVA', 'Hz', 'kW'")
                hourlydata.reset_index(inplace=True)
                hourlydata = hourlydata.loc[:, hourlydata.columns != 'Active']
                ts = ts.append(hourlydata)
                print(child, unit)
            except:
                print('Could not add data for ' + str(child) + ' ' + unit) #skip if feather file does not exist 
    #write to reduced data to file
        if ts.empty:
            pass
        else:
            wpath = os.path.join(dir_path, str(year) + '_' + unit + '.feather')
            feather.write_dataframe(ts, wpath)
            print('Write success')
    return

def loadProfiles(year, unit):
    """
    This function loads a year's hourly unit profiles into a dataframe and returns it together with the year and unit concerned.
    
    """
    data = feather.read_dataframe(os.path.join(hourlyprofiles_dir, unit, str(year) + '_' + unit + '.feather')) #load data
    return data, year, unit

def loadTables(filepath = table_dir):
    """
    This function loads all feather tables in filepath into workspace.
    
    """
    try:
        files = glob(os.path.join(filepath, '*.feather'))
        names = [f.rpartition('.')[0] for f in os.listdir(filepath)]
        tables = {}
        for n, f in zip(names, files):
            tables[n] = feather.read_dataframe(f)
            
    except:
## TODO: get data from energydata.uct.ac.za
        pass
    
    return tables

def csvTables(savepath):
    """
    This function fetches tables saved as feather objects and saves them as csv files.
    """
    
    tables = loadTables() #get data
    filenames = [os.path.join(savepath, x + '.csv') for x in list(tables.keys())] #generate list of filenames
    
    for name, path in zip(list(tables.keys()), filenames):
        df = tables[name]
        df.to_csv(path, index=False)
        print('Successfully saved to ' + path)        
    return

def data2ckan(csvpath):
    """
    This function uploads csv tables to the energydata.uct.ac.za data portal for online access. 
    """
    return
        