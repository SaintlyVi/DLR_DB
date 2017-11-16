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
from pandas.tseries.frequencies import to_offset

import features.socios as socios
from processing.procore import loadProfiles, loadTables

tables = loadTables()

#investigating one location
def aggTs(year, unit, interval, locstring=None):
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
        data.set_index('Datefield', inplace=True)
    except:
        return print("Invalid unit")
    
    #subset dataframe by location & remove invalid readings
    if locstring is None:
        loc = data[(data.Valid==1)]
    else:
        loc = data[(data.RecorderID.str.contains(locstring.upper())) & (data.Valid==1)]
        
    #specify aggregation function for different units    
    if unit in ['kW','kVA']:
        aggregated = loc.groupby(['RecorderID','ProfileID']).resample(interval, on='Datefield').sum()
    elif unit in ['A', 'V', 'Hz']:
        aggregated = loc.groupby(['RecorderID','ProfileID']).resample(interval).agg({'Unitsread':'mean','Valid':'sum'})
        
    validhours = pd.to_timedelta(to_offset(interval)) / np.timedelta64(1, 'h')
    aggregated['Valid'] = aggregated['Valid']/validhours
    
    return aggregated

def getProfilePower(year):
    #get list of AnswerIDs in variable year
    a_id = socios.loadID(year, id_name = 'AnswerID')
    
    #get dataframe of linkages between AnswerIDs and ProfileIDs
    links = tables.get('links')
    year_links = links[links.AnswerID.isin(a_id)]
    year_links = year_links.loc[year_links.ProfileID != 0, ['AnswerID','ProfileID']]    
    
    #get profile metadata (recorder ID, recording channel, recorder type, units of measurement)
    profiles = tables.get('profiles')
    #add AnswerID information to profiles metadata
    profile_meta = year_links.merge(profiles, left_on='ProfileID', right_on='ProfileId').drop('ProfileId', axis=1)        
    VI_profile_meta = profile_meta.loc[(profile_meta['Unit of measurement'] == 2), :] #select current profiles only
        
    #get profile data for year
    iprofile = loadProfiles(year, 'A')[0]    
    vprofile = loadProfiles(year, 'V')[0]
    
    if year <= 2009: #pre-2009 recorder type is set up so that up to 12 current profiles share one voltage profile
        #get list of ProfileIDs in variable year
        p_id = socios.loadID(year, id_name = 'ProfileID')
        year_profiles = profiles[profiles.ProfileId.isin(p_id)]        
        vchan = year_profiles.loc[year_profiles['Unit of measurement']==1, ['ProfileId','RecorderID']] #get metadata for voltage profiles

        iprofile = iprofile.merge(vchan, on='RecorderID', suffixes=('_i','_v'))
        iprofile.rename(columns={"ProfileId": "matchcol"}, inplace=True)        
        power = iprofile.merge(vprofile, left_on=['matchcol', 'Datefield'], right_on=['ProfileID','Datefield'], suffixes=['_i', '_v'])
        power.drop(['RecorderID_i', 'matchcol'], axis=1, inplace=True)
        power.rename(columns={'RecorderID_v':'RecorderID'}, inplace=True)

    elif 2009 < year <= 2014: #recorder type is set up so that each current profile has its own voltage profile
        vprofile['matchcol'] = vprofile['ProfileID'] + 1
        power_temp = vprofile.merge(iprofile, left_on=['matchcol', 'Datefield'], right_on=['ProfileID','Datefield'], suffixes=['_v', '_i'])
        power_temp.drop(['RecorderID_v','RecorderID_i', 'matchcol'], axis=1, inplace=True)
        
        pprofile = loadProfiles(year, 'kVA')[0] #get kW readings
        pprofile['matchcol'] = pprofile['ProfileID'] - 2 #UoM = 4, ChannelNo = 4, 8 or 12
        power = power_temp.merge(pprofile, right_on=['matchcol', 'Datefield'], left_on=['ProfileID_v','Datefield'])
        power.rename(columns={'ProfileID':'ProfileID_kva', 'Unitsread':'Unitsread_kva'}, inplace=True)
        power.drop(['matchcol'], axis=1, inplace=True)
        
    else:
        return print('Year is out of range. Please select a year between 1994 and 2014')
    
    power['kVAh_calculated'] = power.Unitsread_v*power.Unitsread_i*0.001
    power['valid_calculated'] = power.Valid_i * power.Valid_v
    output = power.merge(VI_profile_meta.loc[:,['AnswerID','ProfileID']], left_on='ProfileID_i', right_on='ProfileID').drop(['ProfileID','Valid_i','Valid_v'], axis=1)
    output = output[output.columns.sort_values()]
    
    return output

def aggProfilePower(year, interval):
    """
    This function returns the aggregated mean or total load profile for all AnswerIDs for a year.
    Interval should be 'D' for calendar day frequency, 'M' for month end frequency or 'A' for annual frequency. Other interval options are described here: http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    
    The aggregate function for kVA and kVA_calculated is sum().
    The aggregate function for A, V is mean().
    """

    data = getProfilePower(year)
    data.set_index('Datefield', inplace=True)
    data.dropna(subset=['Valid','valid_calculated'], inplace=True)
    
    try:
        aggregated = data.groupby(['RecorderID','AnswerID']).resample(interval).agg({'Unitsread_i': np.mean, 'Unitsread_v': np.mean, 'Unitsread_kva': np.sum, 'kVAh_calculated': np.sum})
        aggregated = aggregated[['kVAh_calculated', 'Unitsread_kva', 'Unitsread_i', 'Unitsread_v']]
    
    except:
        aggregated = data.groupby(['RecorderID','AnswerID']).resample(interval).agg({'Unitsread_i': np.mean, 'Unitsread_v': np.mean, 'kVAh_calculated': np.sum})
        aggregated = aggregated[['kVAh_calculated', 'Unitsread_i', 'Unitsread_v']]
    
    return aggregated

## TODO
def maxDemand(year):
    
    data = getProfilePower(year)
    maxdemand = data.iloc[data.reset_index().groupby(['AnswerID'])['Unitsread_i'].idxmax()].reset_index(drop=True)
    
    maxdemand['month'] = maxdemand['Datefield'].dt.month
    maxdemand['daytype'] = maxdemand['Datefield'].dt.dayofweek
    maxdemand['hour'] = maxdemand['Datefield'].dt.hour
                             
    return maxdemand[['AnswerID','RecorderID','Unitsread_i','month','daytype','hour']]

def avgDailyDemand(year):
    
    data = getProfilePower(year)
    data.set_index('Datefield', inplace=True)
    
    return

def avgMonthlyDemand(year):
    
    data = aggProfilePower(year, 'M')
    try:
        avgmonthlydemand = data.groupby(['RecorderID','AnswerID'])['Unitsread_kva'].mean()
    except:
        avgmonthlydemand = data.groupby(['RecorderID','AnswerID'])['kVAh_calculated'].mean()
    
    return avgmonthlydemand.reset_index()

def avgDaytypeDemand(year):
    
    data = getProfilePower(year)
    data['month'] = data['Datefield'].dt.month
    data['dayix'] = data['Datefield'].dt.dayofweek
    data['hour'] = data['Datefield'].dt.hour
    cats = pd.cut(data.dayix, bins = [0, 5, 6, 7], right=False, labels= ['Weekday','Saturday','Sunday'], include_lowest=True)
    data['daytype'] = cats
    
    try:
        daytypedemand = data.groupby(['AnswerID', 'month', 'daytype', 'hour'])['Unitsread_kva'].agg(['mean', 'std'])
    except:
        daytypedemand = data.groupby(['AnswerID', 'month', 'daytype', 'hour'])['kVAh_calculated'].agg(['mean', 'std'])

    return daytypedemand.reset_index()