#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 09:49:43 2017

@author: SaintlyVi
"""

    
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

links = loadTables().get('links')
p = loadTables().get('profiles')
activeprofiles = p[(p.ProfileId.isin(links.ProfileID[(links.ProfileID.isin(p.ProfileId[(p['Unit of measurement'] == 2)])) & (links.GroupID == 1000105)])) & (p.Active == True)]
len(links[(links.GroupID == 1000105) & (links.AnswerID != 0)])
len(links[(links.GroupID == 1000105) & (links.ProfileID != 0)])