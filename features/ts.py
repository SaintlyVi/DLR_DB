# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 08:31:00 2017

@author: Wiebke Toussaint

Load Subclass Model
    based on data from DTPET up to 2009 and the GLF 
- load shape over time
- trajectory of load growth over time
- distinct name of customers whome the load subclass represents

model attributes
ATTRIBUTES
* power (kW)
* power factor
* load factor
TYPE
* hour of the day
* day type: weekday / weekend
* season: high / low

process
* exclude public holidays
* normalise all profiles in subclass by annual energy
* average annual curves to arrive at a subclass load shape
* aggregate

"""

import pandas as pd
import numpy as np
import feather
import os

import plotly as py
from plotly.offline import offline
import plotly.graph_objs as go

import features.socios as socios
from features.support import hourlyprofiles_dir, classout_dir   

def loadProfiles(year, unit):
    """
    This function loads a year's hourly unit profiles into a dataframe and returns it together with the year and unit concerned.
    
    """
    data = feather.read_dataframe(os.path.join(hourlyprofiles_dir, unit, str(year) + '_' + unit + '.feather')) #load data
    return data, year, unit

def shapeProfiles(annualunitprofile):
    """
    This function reshapes a year's unit profiles into a dataframe indexed by date, with profile IDs as columns and units read as values.
    annualunitprofile variable should be a pandas data frame constructed with the loadProfiles() function.
    Rows with Valid=0 are removed.
    
    The function returns [shaped_profile_df, year, unit]; a tuple containing the shaped dataframe indexed by hour with aggregated unit values for all profiles, the year and unit concerned.
    
    """
    data = annualunitprofile[0]
    year = annualunitprofile[1]
    unit = annualunitprofile[2]
    
    valid_data = data[data.Valid > 0] #remove invalid data - valid for 10min readings = 6, valid for 5min readings = 12
    sorted_data = valid_data.sort_values(by='Datefield') #sort by date
    sorted_data.ProfileID = sorted_data.ProfileID.apply(lambda x: str(x))
    pretty_data = sorted_data.set_index(['Datefield','ProfileID']).unstack()['Unitsread'] #reshape dataframe
    return pretty_data, year, unit

def nanAnalysis(year, unit, threshold = 0.95):
    """
    This function displays information about the missing values for all customers in a load profile unit year.
    shapedprofile is a dataframe that has been created with shapeProfiles.
    threshold 
    
    The function returns:
        * two plots with summary statistics of all profiles
        * the percentage of profiles and measurement days with full observational data above the threshold value.
    """
    
    data = shapeProfiles(loadProfiles(year, unit))[0]
    year = shapeProfiles(loadProfiles(year, unit))[1]
    unit = shapeProfiles(loadProfiles(year, unit))[2]

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
    
#investigating one location
def tsAgg(year, unit, interval, locstring=None):
    """
    This function returns the aggregated mean or total load profile for all ProfileIDs for a year in a given location.
    Use socios.recorderLocations() to get locstrings for locations of interest. 
    Interval should be 'D' for calendar day frequency, 'M' for month end frequency or 'A' for annual frequency. Other interval options are described here: http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    
    The aggregate function for kW and kVA is sum().
    The aggregate function for A, V and Hz is mean().
    """
    #load data
    try:
        data = loadProfiles(year, unit)[0]
        data['ProfileID'] = data['ProfileID'].astype('category')
    except:
        return print("Invalid unit")
    
    #subset dataframe by location
    if locstring is None:
        loc = data
    else:
        loc = data[data.RecorderID.str.contains(locstring.upper())]
        
    #specify aggregation function for different units    
    if unit in ['kW','kVA']:
        aggregated = loc.groupby(['RecorderID','ProfileID']).resample(interval, on='Datefield').sum()
    elif unit in ['A', 'V', 'Hz']:
        aggregated = loc.groupby(['RecorderID','ProfileID']).resample(interval, on='Datefield').mean()
    
    return aggregated

def getProfilePower(year):
    #get list of AnswerIDs in variable year
    a_id = socios.loadID(year, id_name = 'AnswerID')
    
    #get dataframe of linkages between AnswerIDs and ProfileIDs
    links = socios.loadTables().get('links')
    year_links = links[links.AnswerID.isin(a_id)]
    year_links = year_links.loc[year_links.ProfileID != 0, ['AnswerID','ProfileID']]    
    
    #get profile metadata (recorder ID, recording channel, recorder type, units of measurement)
    profiles = socios.loadTables().get('profiles')
    #add AnswerID information to profiles metadata
    profile_meta = year_links.merge(profiles, left_on='ProfileID', right_on='ProfileId').drop('ProfileId', axis=1)        
    VI_profile_meta = profile_meta.loc[(profile_meta['Unit of measurement'] == 2), :] #select current profiles only
        
    #get profile data for year
    iprofile = loadProfiles(year, 'A')[0]    
    vprofile = loadProfiles(year, 'V')[0]
    
    if year < 2009: #pre-2009 recorder type is set up so that up to 12 current profiles share one voltage profile
        #get list of ProfileIDs in variable year
        p_id = socios.loadID(year, id_name = 'ProfileID')
        year_profiles = profiles[profiles.ProfileId.isin(p_id)]        
        vchan = year_profiles.loc[year_profiles['Unit of measurement']==1, ['ProfileId','RecorderID']] #get metadata for voltage profiles

        iprofile = iprofile.merge(vchan, on='RecorderID', suffixes=('_i','_v'))
        iprofile.rename(columns={"ProfileId": "matchcol"}, inplace=True)        
        power = iprofile.merge(vprofile, left_on=['matchcol', 'Datefield'], right_on=['ProfileID','Datefield'], suffixes=['_i', '_v'])

    elif 2009 <= year <= 2014: #recorder type is set up so that each current profile has its own voltage profile
        vprofile['matchcol'] = vprofile['ProfileID'] + 1
        power_temp = vprofile.merge(iprofile, left_on=['matchcol', 'Datefield'], right_on=['ProfileID','Datefield'], suffixes=['_v', '_i'])
        power_temp.drop(['RecorderID_v','RecorderID_i', 'matchcol', 'Valid_i', 'Valid_v'], axis=1, inplace=True)
        
        pprofile = loadProfiles(year, 'kW')[0] #get kW readings
        pprofile['matchcol'] = pprofile['ProfileID'] - 3 #UoM = 5, ChannelNo = 5, 9 or 13
        power = power_temp.merge(pprofile, right_on=['matchcol', 'Datefield'], left_on=['ProfileID_v','Datefield'])
                
    else:
        return print('Year is out of range. Please select a year between 1994 and 2014')
    
    power = power.loc[:,['RecorderID_v', 'ProfileID_v', 'Datefield', 'Unitsread_v', 'ProfileID_i', 'Unitsread_i']]
    power.columns = ['RecorderID','V_ProfileID','Datefield','V_Unitsread','I_ProfileID','I_Unitsread']
    power['kWh_calculated'] = power.V_Unitsread*power.I_Unitsread*0.001
    output = power.merge(VI_profile_meta.loc[:,['AnswerID','ProfileID']], left_on='I_ProfileID', right_on='ProfileID').drop('ProfileID', axis=1)
    
    return output

def classProfilePower(year, experiment_dir = 'exp'):
    """
    This function gets the inferred class for each AnswerID from 'DLR_DB/classmod/out/experiment_dir' and aggregates the profiles by month, day type and hour of the day.
    """
    
    dirpath = os.path.join(classout_dir, experiment_dir)
    filename = 'classes_' + str(year) + '.csv'
    
    #get data
    classes = pd.read_csv(os.path.join(dirpath, filename), header=None, names=['AnswerID','class'])
    profiles = getProfilePower(year)
    
    #add class label to profile IDs
    df = classes.merge(profiles, on='AnswerID')
    #select subset of columns that is of interest
    classpower = df.loc[:, ['AnswerID','class','Datefield','kWh_calculated']]
    
    return classpower

def annualPower(year, experiment_dir = 'exp'):
    
    df = classProfilePower(year, experiment_dir)
    mean_hourly_id = df.groupby('AnswerID')['kWh_calculated'].mean().reset_index() #get mean annual hourly kWh value for each AnswerID
    
    #daily power profiles
    sum_daily_id = df.groupby(['class','AnswerID',df['Datefield'].dt.year,df['Datefield'].dt.month,df['Datefield'].dt.day]).sum()
    sum_daily_id.index.names = ['class','AnswerID','Year','Month','Day']
    sum_daily_id.reset_index(inplace=True)
    mean_daily_id = sum_daily_id.groupby(['class','AnswerID'])['kWh_calculated'].mean().reset_index()
    
    #monthly power profiles
    sum_monthly_id = df.groupby(['class','AnswerID',df['Datefield'].dt.year,df['Datefield'].dt.month]).sum()
    sum_monthly_id.index.names = ['class','AnswerID','Year','Month']
    sum_monthly_id.reset_index(inplace=True)
    mean_monthly_id = sum_monthly_id.groupby(['class','AnswerID'])['kWh_calculated'].mean().reset_index()
    
    mean_id0 = mean_daily_id.merge(mean_hourly_id, on='AnswerID', suffixes=['_d','_h'])
    mean_id = mean_id0.merge(mean_monthly_id, on=['AnswerID','class'])
    mean_id.columns = ['class','AnswerID','kWh_daily_mean','kWh_hourly_mean','kWh_monthly_mean']
    
    #manipulate dataframe to match DPET hourly summary output
    dfnorm = df.merge(mean_hourly_id, how='outer', on='AnswerID', suffixes=['','_mean'])
    dfnorm['kWh_norm'] = dfnorm.kWh_calculated / dfnorm.kWh_calculated_mean
    dfnorm = dfnorm.loc[:, ['AnswerID','class','Datefield','kWh_norm']]
    
    dfnorm['Month'] = dfnorm['Datefield'].dt.month
    daytypebins = [0, 5, 6, 7]
    daytypelabels = ['Weekday', 'Saturday', 'Sunday']
    dfnorm['DayType'] = pd.cut(dfnorm.Datefield.dt.weekday, bins = daytypebins, labels = daytypelabels, right=False, include_lowest=True)
    dfnorm['Hour'] = dfnorm['Datefield'].dt.hour
    
    #group and normalise dataframe to get mean and std of power profiles by customer class, 
    grouped = dfnorm.groupby(['class','Month','DayType','Hour'])
    class_norm = grouped['kWh_norm'].agg([np.mean, np.std]).rename(columns={'mean': 'mean_kWh_norm','std': 'std_kWh_norm'}).reset_index()
    
    grouped_id = dfnorm.groupby(['class','AnswerID','Month','DayType','Hour'])
    id_norm = grouped_id['kWh_norm'].agg([np.mean, np.std]).rename(columns={'mean': 'mean_kWh_norm','std': 'std_kWh_norm'}).reset_index()
    #load factor
    
    return sum_daily_id, sum_monthly_id, mean_id, id_norm, class_norm

#plotting
#ap = annualPower(2012, class_dir = 'exp1')
#idprofile = ap[3]
#data = idprofile.loc[(idprofile['AnswerID'] == 1004031) & (idprofile['DayType'] == 'Weekday'), :]
#plotdata = data.pivot(columns='Month', index='Hour', values='mean_kWh_norm')
#plotdata.plot()

links = socios.loadTables().get('links')
p = socios.loadTables().get('profiles')
activeprofiles = p[(p.ProfileId.isin(links.ProfileID[(links.ProfileID.isin(p.ProfileId[(p['Unit of measurement'] == 2)])) & (links.GroupID == 1000105)])) & (p.Active == True)]
len(links[(links.GroupID == 1000105) & (links.AnswerID != 0)])
len(links[(links.GroupID == 1000105) & (links.ProfileID != 0)])