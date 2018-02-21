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

import features.feature_socios as socios
from observations.obs_processing import loadProfiles, loadTable
from support import InputError

#investigating one location
def aggTs(year, unit, interval, dir_name='H', locstring=None):
    """
    This function returns the aggregated mean or total load profile for all ProfileIDs for a year in a given location.
    Use socios.recorderLocations() to get locstrings for locations of interest. 
    Interval should be 'D' for calendar day frequency, 'M' for month end frequency or 'A' for annual frequency. Other interval options are described here: http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    
    The aggregate function for kW and kVA is sum().
    The aggregate function for A, V and Hz is mean().
    """
    #load data
    try:
        data = loadProfiles(year, unit, dir_name)[0]
        data['ProfileID'] = data['ProfileID'].astype('category')
        data.set_index('Datefield', inplace=True)
        
    except:
        raise InputError(unit, "Invalid unit")
    
    #subset dataframe by location & remove invalid readings
    if locstring is None:
        loc = data
    else:
        loc = data[data.RecorderID.str.contains(locstring.upper())]
        
    #specify aggregation function for different units    
    if unit in ['kW','kVA']:
        aggregated = loc.groupby(['RecorderID','ProfileID']).resample(interval).sum()
    elif unit in ['A', 'V', 'Hz']:
        aggregated = loc.groupby(['RecorderID','ProfileID']).resample(interval).agg({
                'Unitsread':'mean',
                'Valid':'sum'})
    
    aggregated.reset_index(inplace=True)    
    
    validhours = aggregated['Datefield'].apply(lambda x: (x - pd.date_range(end=x, periods=2, freq = interval)[0]) / np.timedelta64(1, 'h'))
    aggregated['Valid'] = aggregated['Valid']/validhours
    
    return aggregated

def getProfilePower(year, dir_name='H'):
    """
    This function retrieves and computes kW and kVA readings for all profiles in a year.
    """
    #get list of AnswerIDs in variable year
    a_id = socios.loadID(year, id_name = 'AnswerID')['id']
    
    #get dataframe of linkages between AnswerIDs and ProfileIDs
    links = loadTable('links')
    year_links = links[links.AnswerID.isin(a_id)]
    year_links = year_links.loc[year_links.ProfileID != 0, ['AnswerID','ProfileID']]    
    
    #get profile metadata (recorder ID, recording channel, recorder type, units of measurement)
    profiles = loadTable('profiles')
    #add AnswerID information to profiles metadata
    profile_meta = year_links.merge(profiles, left_on='ProfileID', right_on='ProfileId').drop('ProfileId', axis=1)        
    VI_profile_meta = profile_meta.loc[(profile_meta['Unit of measurement'] == 2), :] #select current profiles only
        
    #get profile data for year
    iprofile = loadProfiles(year, 'A', dir_name)[0]    
    vprofile = loadProfiles(year, 'V', dir_name)[0]
    
    if year <= 2009: #pre-2009 recorder type is set up so that up to 12 current profiles share one voltage profile
        #get list of ProfileIDs in variable year
        p_id = socios.loadID(year, id_name = 'ProfileID')
        year_profiles = profiles[profiles.ProfileId.isin(p_id.id)]        
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

        kwprofile = loadProfiles(year, 'kW', dir_name)[0] #get kW readings
        kwprofile['matchcol'] = kwprofile['ProfileID'] - 3 #UoM = 5, ChannelNo = 5, 9, 13

        kvaprofile = loadProfiles(year, 'kVA', dir_name)[0] #get kVA readings
        kvaprofile['matchcol'] = kvaprofile['ProfileID'] - 2 #UoM = 4, ChannelNo = 4, 8 or 12        
        kvaprofile.drop(columns='RecorderID',inplace=True)
        
        power_temp2 = power_temp.merge(kwprofile, right_on=['matchcol', 'Datefield'], left_on=['ProfileID_v','Datefield'])
        power = power_temp2.merge(kvaprofile, right_on=['matchcol', 'Datefield'], left_on=['matchcol','Datefield'], suffixes=['_kw','_kva'])
        
        power.drop(['matchcol'], axis=1, inplace=True)
        
    else:
        return print('Year is out of range. Please select a year between 1994 and 2014')
    
    power['kw_calculated'] = power.Unitsread_v*power.Unitsread_i*0.001
    power['valid_calculated'] = power.Valid_i * power.Valid_v

    output = power.merge(VI_profile_meta.loc[:,['AnswerID','ProfileID']], left_on='ProfileID_i', right_on='ProfileID').drop(['ProfileID','Valid_i','Valid_v'], axis=1)
    output = output[output.columns.sort_values()]
    output.fillna({'valid_calculated':0}, inplace=True)
    
    return output

def aggProfilePower(profilepowerdata, interval):
    """
    This function returns the aggregated mean or total load profile for all AnswerIDs for a year.
    Interval should be 'D' for calendar day frequency, 'M' for month end frequency or 'A' for annual frequency. Other interval options are described here: http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    
    The aggregate function for kW and kW_calculated is sum().
    The aggregate function for A, V is mean().
    """

    data = profilepowerdata.set_index('Datefield')
    
    try:
        aggprofile = data.groupby(['RecorderID','AnswerID']).resample(interval).agg({
                'Unitsread_i': np.mean, 
                'Unitsread_v': np.mean, 
                'Unitsread_kw': np.sum,
                'Unitsread_kva': np.mean,
                'kw_calculated': np.sum, 
                'valid_calculated': np.sum})
    
    except:
        aggprofile = data.groupby(['RecorderID','AnswerID']).resample(interval).agg({
                'Unitsread_i': np.mean, 
                'Unitsread_v': np.mean, 
                'kw_calculated': np.sum,  
                'valid_calculated': np.sum})
        
    aggprofile.reset_index(inplace=True)    
    
    aggprofile['interval_hours'] = aggprofile['Datefield'].apply(lambda x: (x - pd.date_range(end=x, periods=2, freq = interval)[0]) / np.timedelta64(1, 'h'))
    aggprofile['valid_obs_ratio'] = aggprofile['valid_calculated']/aggprofile['interval_hours']
    
    aggprofile['interval'] = interval

    return aggprofile

def annualIntervalDemand(aggprofilepowerdata):
    """
    This function computes the mean annual power consumption for the interval aggregated in aggprofilepowerdata.
    """
    
    interval = aggprofilepowerdata.interval[0]
    
    try:
        aggdemand = aggprofilepowerdata.groupby(['RecorderID','AnswerID']).agg({
                'Unitsread_kw': ['mean', 'std'], 
                'Unitsread_kva': ['mean', 'std'],
                'valid_calculated':'sum',
                'interval_hours':'sum'})
        aggdemand.columns = ['_'.join(col).strip() for col in aggdemand.columns.values]
        aggdemand.rename(columns={
                'Unitsread_kw_mean':interval+'_kw_mean', 
                'Unitsread_kw_std':interval+'_kw_std', 
                'Unitsread_kva_mean':interval+'_kva_mean', 
                'Unitsread_kva_std':interval+'_kva_std', 
                'valid_calculated_sum':'valid_hours'}, inplace=True)

    except:
        aggdemand = aggprofilepowerdata.groupby(['RecorderID','AnswerID']).agg({
                'kw_calculated': ['mean', 'std'], 
                'valid_calculated':'sum',
                'interval_hours':'sum'})        
        aggdemand.columns = ['_'.join(col).strip() for col in aggdemand.columns.values]
        aggdemand.rename(columns={
                'kw_calculated_mean':interval+'_kw_mean', 
                'kw_calculated_std':interval+'_kw_std', 
                'valid_calculated_sum':'valid_hours'}, inplace=True) 
        
    aggdemand['valid_obs_ratio'] = aggdemand['valid_hours']/aggdemand['interval_hours_sum']    
    aggdemand['interval'] = interval
    
    return aggdemand.reset_index()

def aggDaytypeDemand(profilepowerdata):   
    """
    This function generates an hourly load profile for each AnswerID.
    The model contains aggregate hourly kW readings for the parameters:
        Month
        Daytype [Weekday, Sunday, Monday]
        Hour
    """
    
    data = profilepowerdata
    data['month'] = data['Datefield'].dt.month
    data['dayix'] = data['Datefield'].dt.dayofweek
    data['hour'] = data['Datefield'].dt.hour
    cats = pd.cut(data.dayix, bins = [0, 5, 6, 7], right=False, labels= ['Weekday','Saturday','Sunday'], include_lowest=True)
    data['daytype'] = cats
    data['total_hours'] = 1    
    
    try:
        daytypedemand = data.groupby(['AnswerID', 'month', 'daytype', 'hour']).agg({
                'Unitsread_kw': ['mean', 'std'], 
                'Unitsread_kva': ['mean', 'std'],
                'valid_calculated':'sum', 
                'total_hours':'sum'})
        daytypedemand.columns = ['_'.join(col).strip() for col in daytypedemand.columns.values]
        daytypedemand.rename(columns={
                'Unitsread_kw_mean':'kw_mean', 
                'Unitsread_kw_std':'kw_std', 
                'Unitsread_kva_mean':'kva_mean', 
                'Unitsread_kva_std':'kva_std', 
                'valid_calculated_sum':'valid_hours'}, inplace=True)

    except: #for years < 2009 where only V and I were observed
        daytypedemand = data.groupby(['AnswerID', 'month', 'daytype', 'hour']).agg({
                'kw_calculated': ['mean', 'std'],
                'valid_calculated':'sum', 
                'total_hours':'sum'})
        daytypedemand.columns = ['_'.join(col).strip() for col in daytypedemand.columns.values]
        daytypedemand.rename(columns={
                'kw_calculated_mean':'kw_mean', 
                'kw_calculated_std':'kw_std', 
                'valid_calculated_sum':'valid_hours'}, inplace=True)

    daytypedemand['valid_obs_ratio'] = daytypedemand['valid_hours'] / daytypedemand['total_hours_sum']
        
    return daytypedemand.reset_index()
