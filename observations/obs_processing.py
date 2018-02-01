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

import ckanapi

import plotly as py
from plotly.offline import offline
import plotly.graph_objs as go

from support import rawprofiles_dir, profiles_dir, table_dir, writeLog, validYears

def reduceRawProfiles(year, unit, interval):
    """
    This function uses a rolling window to reduce all raw load profiles to hourly mean values. Monthly load profiles are then concatenated into annual profiles and returned as a dictionary object.
    The data is structured as dict[unit:{year:[list_of_profile_ts]}]
    
    """
    validYears(year) #check if year input is valid
    p = Path(os.path.join(rawprofiles_dir, str(year)))
    
    #initialise empty dataframe to concatenate annual timeseries
    ts = pd.DataFrame()

    #iterate through all data files to combine 5min monthly into hourly reduced annual timeseries
    for child in p.iterdir():  
        try:
            childpath = glob(os.path.join(child, '*_' + unit + '.feather'))[0]
            data = feather.read_dataframe(childpath)
            data.Datefield = np.round(data.Datefield.astype(np.int64), -9).astype('datetime64[ns]')
            data['Valid'] = data['Valid'].map(lambda x: x.strip()).map({'Y':1, 'N':0})
            data['Valid'].fillna(0, inplace=True)
            if unit in ['A','V','Hz','kVA','kW']:
                hourlydata = data.groupby(['RecorderID', 'ProfileID']).resample(interval, on='Datefield').mean()
            else:
                print("Unit must be one of 'A', 'V', 'kVA', 'Hz', 'kW'")
            hourlydata.reset_index(inplace=True)
            hourlydata = hourlydata.loc[:, hourlydata.columns != 'Active']
            ts = ts.append(hourlydata)
            print(child, unit)
        except:
            print('Could not add data for ' + str(child) + ' ' + unit) #skip if feather file does not exist 
        
    ts.drop_duplicates(inplace=True)
        
    return ts

def saveReducedProfiles(yearstart, yearend, interval):
    """
    This function iterates through profile units, reduces all profiles with reduceRawProfiles() and saves the result as a feather object in a directory tree.
    
    """ 
    for year in range(yearstart, yearend + 1):   
        for unit in ['A', 'V', 'kVA', 'Hz', 'kW']:
            
            #create empty directory to save files   
            dir_path = os.path.join(profiles_dir, interval, unit)
            os.makedirs(dir_path, exist_ok=True)
        
            ts = reduceRawProfiles(year, unit, interval)
            
            #write to reduced data to file
            if ts.empty:
                pass
            else:
                wpath = os.path.join(dir_path, str(year) + '_' + unit + '.feather')
                feather.write_dataframe(ts, wpath)
                print('Write success')
    
    logline = [yearstart, yearend, interval]
    log_lines = pd.DataFrame([logline], columns = ['from_year','to_year', 'resample_interval'])
    writeLog(log_lines,'log_reduce_profiles')

    return

def loadProfiles(year, unit, dir_name):
    """
    This function loads a year's unit profiles from the dir_name in profiles directory into a dataframe and returns it together with the year and unit concerned.
    
    """
    validYears(year) #check if year input is valid
    
    #load data
    data = feather.read_dataframe(os.path.join(profiles_dir, dir_name, unit,
                                               str(year)+'_'+unit+'.feather'))
    data.drop_duplicates(inplace=True)
    
    return data, year, unit

def loadTable(name, query=None, columns=None):
    """
    This function loads all feather tables in filepath into workspace.
    
    """
    dir_path = os.path.join(table_dir, 'feather')
    
    try:
        file = os.path.join(dir_path, name +'.feather')
        d = feather.read_dataframe(file)
        if columns is None:
            table = d
        else:
            table = d[columns]
            
    except:
        #fetch tables from energydata.uct.ac.za
        ckan = ckanapi.RemoteCKAN('http://energydata.uct.ac.za/', get_only=True)
        resources = ckan.action.package_show(id='dlr-database-tables-94-14')        
        for i in range(0, len(resources['resources'])):
            if resources['resources'][i]['name'] == name:
                print('... fetching table from energydata.uct.ac.za')
                r_id = resources['resources'][i]['id']
                d = ckan.action.datastore_search(resource_id=r_id, q=query, fields=columns, limit=1000000)['records']
                table = pd.DataFrame(d)
            else:
                pass

    try: 
        return table

    except UnboundLocalError:
        return('Could not find table with name '+name)    

def csvTables():
    """
    This function fetches tables saved as feather objects and saves them as csv files.
    """

    os.makedirs(os.path.join(table_dir, 'csv') , exist_ok=True)
    
    #get data
    feather_path = os.path.join(table_dir, 'feather')
    names = [f.rpartition('.')[0] for f in os.listdir(feather_path)]

    for n in names:    
        table = loadTable(n)
        path = os.path.join(table_dir, 'csv', n + '.csv')
        table.to_csv(path, index=False)
        print('Successfully saved to ' + path)
        
        #generate list of filenames
#        filenames = [os.path.join(dir_path, 'csv', x + '.csv') for x in list(tables.keys())]
#        
#        for name, path in zip(list(tables.keys()), filenames):
#            df = tables[name]
#            df.to_csv(path, index=False)
#            print('Successfully saved to ' + path)        
    return
        
def shapeProfiles(year, unit):
    """
    This function reshapes a year's unit profiles into a dataframe indexed by date, with profile IDs as columns and units read as values.
    annualunitprofile variable should be a pandas data frame constructed with the loadProfiles() function.
    Rows with Valid=0 are removed.
    
    The function returns [shaped_profile_df, year, unit]; a tuple containing the shaped dataframe indexed by hour with aggregated unit values for all profiles, the year and unit concerned.
    
    """
    data, year, unit = loadProfiles(year, unit)
    
    data.loc[(data.Unitsread.notnull())&(data.Valid != 1), 'Unitsread'] = np.nan
    data.ProfileID = data.ProfileID.astype(str)
    data.set_index(['Datefield','ProfileID'], inplace=True, drop=True)
    data = data[~data.index.duplicated(keep='first')]
    
    profile_matrix = data.unstack()['Unitsread'] #reshape dataframe
    valid_matrix = data.unstack()['Valid']
    
    return profile_matrix, year, unit, valid_matrix

def nanAnalysis(year, unit, threshold = 0.95):
    """
    This function displays information about the missing values for all customers in a load profile unit year.
    threshold - float between 0 and 1: user defined value that specifies the percentage of observed hours that must be valid for the profile to be considered useable.
    
    The function returns:
        * two plots with summary statistics of all profiles
        * the percentage of profiles and measurement days with full observational data above the threshold value.
    """
    
    data, year, unit, valid_matrix = shapeProfiles(year, unit)

    #prep data
    fullrows = data.count(axis=1)/data.shape[1]
    fullcols = data.count(axis=0)/data.shape[0]
    
    trace1 = go.Scatter(name='% valid profiles',
                        x=fullrows.index, 
                        y=fullrows.values)
    trace2 = go.Bar(name='% valid hours',
                    x=fullcols.index, 
                    y=fullcols.values)
#    thresh = go.Scatter(x=fullrows.index, y=threshold, mode = 'lines', name = 'threshold', line = dict(color = 'red'))
    
    fig = py.tools.make_subplots(rows=2, cols=1, subplot_titles=['Percentage of ProfileIDs with Valid Observations for each Hour','Percentage of Valid Observational Hours for each ProfileID'], print_grid=False)
    
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 2, 1)
#    fig.append_trace(thresh, 2, 1)
    fig['layout']['xaxis2'].update(title='ProfileIDs', type='category', exponentformat="none")
    fig['layout']['yaxis'].update(domain=[0.55,1])
    fig['layout']['yaxis2'].update(domain=[0, 0.375])
    fig['layout'].update(title = "Visual analysis of valid DLR load profile data for " + str(year) + " readings (units: " + unit + ")", height=850)
      
    goodhours = len(fullcols[fullcols > threshold]) / len(fullcols) * 100
    goodprofiles = len(fullrows[fullrows > threshold]) / len(fullrows) * 100
    
    print('{:.2f}% of hours have over {:.0f}% fully observed profiles.'.format(goodhours, threshold * 100))
    print('{:.2f}% of profiles have been observed over {:.0f}% of time.'.format(goodprofiles, threshold * 100))
    
    offline.iplot(fig)
    
    return 