#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Wiebke Toussaint

This file contains functions to fetch data from the Domestic Load Research SQL Server database. It must be run from a server with a DLR database installation.

The following functions are defined:
    getObs 
    getProfileID
    getMetaProfiles
    profileFetchEst
    getProfiles
    getSampleProfiles
    profilePeriod
    getGroups
    getLocation
    saveTables
    saveAllProfiles
    anonAns
    
SOME EXAMPLES 

# Using getObs with SQL queries:

query = 'SELECT * FROM [General_LR4].[dbo].[linktable] WHERE ProfileID = 12005320'
df = getObs(querystring = query)
    
"""

import pandas as pd
import numpy as np
import pyodbc 
import feather
import os

import shapefile as shp
from shapely.geometry import Point
from shapely.geometry import shape

from support import rawprofiles_dir, table_dir, obs_dir, writeLog, validYears, data_dir

def getObs(db_cnx, tablename = None, querystring = 'SELECT * FROM tablename', chunksize = 10000):
    """
    Fetches a specified table from the DLR database and returns it as a pandas dataframe.

    """
    #connection object:
    try:
        with open(os.path.join(obs_dir, db_cnx), 'r') as f: 
            cnxnstr = f.read().replace('\n', '')
            
    except FileNotFoundError as err:
        print("File not found error: {0}".format(err))
        raise
    else:
        try:
            cnxn = pyodbc.connect(cnxnstr)  
            #specify and execute query(ies):
            if querystring == "SELECT * FROM tablename":
                if tablename is None:
                    return print('Specify a valid table from the DLR database')
                elif tablename == 'Profiletable':
                    return print('The profiles table is too large to read into python in one go. Use the getProfiles() function.') 
                else:
                    query = "SELECT * FROM [General_LR4].[dbo].%s" % (tablename)
            else:
                query = querystring
                
            df = pd.read_sql(query, cnxn)   #read to dataframe   
            return df
        except Exception:
            raise
            
def geoMeta():
    """
    This function generates geographic metadata for groups by combining GroupID Lat, Long co-ords with municipal boundaries dataset
    """
    # Download the SHP, DBF and SHX files from http://energydata.uct.ac.za/dataset/2016-municipal-boundaries-south-africa
    munic2016 = os.path.join(data_dir, 'obs_datasets', 'geo_meta', '2016-Boundaries-Local','Local_Municipalities_2016') 
    site_ref = pd.read_csv(os.path.join(data_dir, 'obs_datasets', 'geo_meta', 'DLR Site coordinates.csv'))
    #response = requests.get(ckanurl)
    
    sf = shp.Reader(munic2016)
    all_shapes = sf.shapes() # get all the polygons
    all_records = sf.records()
    
    g = list()
    
    for i in range(0, len(site_ref)):
        for j in range(0, len(all_shapes)):
            boundary = all_shapes[j]
            if Point(tuple([site_ref.loc[i,'Long'],site_ref.loc[i,'Lat']])).within(shape(boundary)):
                g.append([all_records[j][k] for k in (1, 5, 9)])
                
    geo_meta = pd.DataFrame(g, columns = ['Province','Municipality','District'])
    geo_meta.loc[geo_meta.Province == 'GT', 'Province'] = 'GP'
    
    site_geo = pd.concat([site_ref, geo_meta], axis = 1)
    site_geo = site_geo[['GPSName','Lat','Long','Province','Municipality','District']].drop_duplicates()
    site_geo.to_csv(os.path.join(data_dir, 'obs_datasets', 'geo_meta', 'site_geo.csv'), index=False)

def getGroups(db_cnx, year = None):
    """
    This function performs some massive Groups wrangling
    
    """
    groups = getObs(db_cnx, 'Groups')
    groups['ParentID'].fillna(0, inplace=True)
    groups['ParentID'] = groups['ParentID'].astype('int64').astype('category')
    groups['GroupName'] = groups['GroupName'].map(lambda x: x.strip())
    #TRY THIS groups['GroupName'] = groups['GroupName'].str.strip()
    
    #Deconstruct groups table apart into levels
    #LEVEL 1 GROUPS: domestic/non-domestic
    groups_level_1 = groups[groups['ParentID']==0] 
    #LEVEL 2 GROUPS: Eskom LR, NRS LR, Namibia, Clinics, Shops, Schools
    groups_level_2 = groups[groups['ParentID'].isin(groups_level_1['GroupID'])]
    #LEVLE 3 GROUPS: Years
    groups_level_3 = groups[groups['ParentID'].isin(groups_level_2['GroupID'])]
    #LEVLE 4 GROUPS: Locations
    groups_level_4 = groups[groups['ParentID'].isin(groups_level_3['GroupID'])]
    
    #Slim down the group levels to only include columns requried for merging
    g1 = groups.loc[groups['ParentID']==0,['GroupID','ParentID','GroupName']].reset_index(drop=True)
    g2 = groups.loc[groups['ParentID'].isin(groups_level_1['GroupID']), ['GroupID','ParentID','GroupName']].reset_index(drop=True)
    g3 = groups.loc[groups['ParentID'].isin(groups_level_2['GroupID']), ['GroupID','ParentID','GroupName']].reset_index(drop=True)
    
    #reconstruct group levels as one pretty, multi-index table
    recon3 = pd.merge(groups_level_4, g3, left_on ='ParentID', right_on = 'GroupID' , how='left', suffixes = ['_4','_3'])
    recon2 = pd.merge(recon3, g2, left_on ='ParentID_3', right_on = 'GroupID' , how='left', suffixes = ['_3','_2'])
    recon1 = pd.merge(recon2, g1, left_on ='ParentID', right_on = 'GroupID' , how='left', suffixes = ['_2','_1'])
    prettyg = recon1[['ContextID','GroupID_1','GroupID_2','GroupID_3','GroupID_4','GroupName_1','GroupName_2','GroupName_3','GroupName_4']]
    prettynames = ['ContextID', 'GroupID_1','GroupID_2','GroupID_3','GroupID','Dom_NonDom','Survey','Year','Location']
    prettyg.columns = prettynames
    
    #create multi-index dataframe
    allgroups = prettyg.set_index(['GroupID_1','GroupID_2','GroupID_3']).sort_index()
    allgroups['LocName'] = allgroups['Location'].apply(lambda x:x.partition(' ')[2])
    
    
    #add geographic metadata: lat, long municipality, district, province
    try:
        geo_meta = pd.read_csv(os.path.join(data_dir, 'obs_datasets', 'geo_meta', 'site_geo.csv'))
    except:
        geoMeta()
        geo_meta = pd.read_csv(os.path.join(data_dir, 'obs_datasets', 'geo_meta', 'site_geo.csv'))
        
    output = allgroups.merge(geo_meta[['GPSName','Lat','Long','Province','Municipality','District']], left_on='LocName', right_on='GPSName', how='left')
    output.drop(labels='GPSName', axis=1, inplace=True)
    
    if year is None:
        return output

    #filter dataframe on year
    else:
        validYears(year) #check if year is valid
        return output[output.Year.astype(int) == year] 

def getProfileID(db_cnx, year = None):
    """
    Fetches all profile IDs for a given year. None returns all profile IDs.
    
    """
    links = getObs(db_cnx, 'LinkTable')
    allprofiles = links[(links.GroupID != 0) & (links.ProfileID != 0)]
    if year is None:
        return allprofiles
    #match GroupIDs to getGroups to get the profile years:
    else:
        profileid = pd.Series(allprofiles.loc[allprofiles.GroupID.isin(getGroups(db_cnx, year).GroupID), 'ProfileID'].unique())
    return profileid

def getMetaProfiles(db_cnx, year, units = None):
    """
    Fetches profile meta data. Units must be one of  V or A. From 2009 onwards kVA, Hz and kW have also been measured.
    
    """
    #list of profiles for the year:
    pids = pd.Series(map(str, getProfileID(db_cnx, year))) 
    #get observation metadata from the profiles table:
    metaprofiles = getObs(db_cnx, 'profiles')[['Active','ProfileId','RecorderID','Unit of measurement']]
    metaprofiles = metaprofiles[metaprofiles.ProfileId.isin(pids)] #select subset of metaprofiles corresponding to query
    metaprofiles.rename(columns={'Unit of measurement':'UoM'}, inplace=True)
    metaprofiles.loc[:,['UoM', 'RecorderID']] = metaprofiles.loc[:,['UoM', 'RecorderID',]].apply(pd.Categorical)
    puom = getObs(db_cnx, 'ProfileUnitsOfMeasure').sort_values(by=['UnitsID'])
    cats = list(puom.loc[puom.UnitsID.isin(metaprofiles['UoM'].cat.categories), 'Description'])
    metaprofiles['UoM'].cat.categories = cats

    if units is None:
        plist = metaprofiles['ProfileId']
    elif units in ['V','A','kVA','kW']:
        uom = units.strip() + ' avg'
        plist = metaprofiles[metaprofiles.UoM == uom]['ProfileId']
    elif units=='Hz':
        uom = 'Hz'
        plist = metaprofiles[metaprofiles.UoM == uom]['ProfileId']
    else:
        return print('Check spelling and choose V, A, kVA, Hz or kW as units, or leave blank to get profiles of all.')
    return metaprofiles, plist

def getProfiles(db_cnx, group_year, month, units):
    """
    This function fetches load profiles for one calendar year. 
    It takes the year as number and units as string:
        [A, V] for 1994 - 2008 
        [A, V, kVA, Hz, kW] for 2009 - 2014
    
    """
    ## Get metadata
    mp, plist = getMetaProfiles(db_cnx, group_year, units)
    
    ## Get profiles from server
    subquery = ', '.join(str(x) for x in plist)
    query = "SELECT pt.ProfileID \
     ,pt.Datefield \
     ,pt.Unitsread \
     ,pt.Valid \
    FROM [General_LR4].[dbo].[Profiletable] pt \
    WHERE pt.ProfileID IN (" + subquery + ") AND MONTH(Datefield) =" + str(month) + " \
    ORDER BY pt.Datefield, pt.ProfileID"
    profiles = getObs(db_cnx, querystring = query)
    
    #data output:    
    df = pd.merge(profiles, mp, left_on='ProfileID', right_on='ProfileId')
    df.drop('ProfileId', axis=1, inplace=True)
    #convert strings to category data type to reduce memory usage
    df.loc[:,['ProfileID','Valid']] = df.loc[:,['ProfileID','Valid']].apply(pd.Categorical)
    
    head_year = df.head(1).Datefield.dt.year[0]
    tail_year = df.tail(1).Datefield.dt.year[len(df)-1]
    
    return df, head_year, tail_year
    
def writeProfiles(db_cnx, group_year, month, units):
    """
    Creates folder structure and saves profiles as feather file.
    """
    df, head_year, tail_year = getProfiles(db_cnx, group_year, month, units)
    
    dir_path = os.path.join(rawprofiles_dir, str(group_year), str(head_year) + '-' + str(month))
    os.makedirs(dir_path , exist_ok=True)
    path = os.path.join(dir_path, str(head_year) + '-' + str(month) + '_' + str(units) + '.feather')
    
    if head_year == tail_year: #check if dataframe contains profiles for two years
        print(path)
        feather.write_dataframe(df, path)
        print('Write success')
        
    else:
        #split dataframe into two years and save separately
        head_df = df[df.Datefield.dt.year == head_year].reset_index(drop=True)
        print(path)
        feather.write_dataframe(head_df, path) 
        print('Write success')
        
        #create directory for second year
        dir_path = os.path.join(rawprofiles_dir, str(group_year), str(tail_year) + '-' + str(month))
        os.makedirs(dir_path , exist_ok=True)
        path = os.path.join(dir_path, str(tail_year)+'-'+str(month)+'_'+str(units)+'.feather')
        tail_df = df[df.Datefield.dt.year == tail_year].reset_index(drop=True)
        print(path)
        feather.write_dataframe(tail_df, path)
        print('Write success')
    
    logline = [group_year, month, units]
    log_lines = pd.DataFrame([logline], columns = ['group_year','observation_month', 'unit'])
    writeLog(log_lines,'log_retrieve_observations')

def writeTables(names, dataframes): 
    """
    This function saves a list of names with an associated list of dataframes as feather files.
    The getObs() and getGroups() functions can be used to construct the dataframes.
    
    """
    #create data directories
    os.makedirs(os.path.join(table_dir, 'feather') , exist_ok=True)
    os.makedirs(os.path.join(table_dir, 'csv') , exist_ok=True)

    #get data        
    datadict = dict(zip(names, dataframes))
    for k in datadict.keys():
        if datadict[k].size == datadict[k].count().sum():
            data = datadict[k]
        else:  
            data = datadict[k].fillna(np.nan) #feather doesn't write None type
        
        feather_path = os.path.join(table_dir, 'feather', k + '.feather')
        feather.write_dataframe(data, feather_path)
        
        csv_path = os.path.join(table_dir, 'csv', k + '.csv')
        data.to_csv(csv_path, index=False)
        print('successfully saved table '  + k)
        
    return

def saveTables(db_cnx):
    """
    This function fetches tables from the SQL database and saves them as a feather object. 
    """
    #get and save important tables
    groups = getGroups(db_cnx) 
    questions = getObs(db_cnx, 'Questions')
    questionaires = getObs(db_cnx, 'Questionaires')
    qdtype = getObs(db_cnx, 'QDataType')
    qredundancy = getObs(db_cnx, 'QRedundancy')
    qconstraints = getObs(db_cnx, 'QConstraints')
    answers = getObs(db_cnx, 'Answers')
    links = getObs(db_cnx, 'LinkTable')
    profiles = getObs(db_cnx, 'Profiles')
    profilesummary = getObs(db_cnx, 'ProfileSummaryTable')
    recorderinstall = getObs(db_cnx, 'RECORDER_INSTALL_TABLE')
    
    tablenames = ['groups', 'questions', 'questionaires', 'qdtype', 'qredundancy', 'qconstraints', 'answers', 'links', 'profiles' ,'profilesummary','recorderinstall']
    tabledata = [groups, questions, questionaires, qdtype, qredundancy, qconstraints, answers, links, profiles, profilesummary, recorderinstall]
    
    writeTables(tablenames, tabledata)
 
def saveAnswers(db_cnx):
    """
    This function fetches survey responses and anonymises them to remove all discriminating personal information of respondents. The anonymised dataset is returned and saved as a feather object.
    Details for questions to anonymise are contained in two csv files, anonymise/blobQs.csv and anonymise/charQs.csv.
    
    """
    anstables = {'Answers_blob':'blobQs.csv', 'Answers_char':'charQs.csv', 'Answers_Number':None}    
    for k,v in anstables.items():
        a = getObs(db_cnx, k) #get all answers
        if v is None:
            pass
        else:
            qs = pd.read_csv(os.path.join(obs_dir, 'anonymise', v))
            qs = qs.loc[lambda qs: qs.anonymise == 1, :]
            qanon = pd.merge(getObs(db_cnx, 'Answers'), qs, left_on='QuestionaireID', right_on='QuestionaireID')[['AnswerID','ColumnNo','anonymise']]
            for i, rows in qanon.iterrows():
                a.set_value(a[a.AnswerID == rows.AnswerID].index[0], str(rows.ColumnNo),'a')
        
        writeTables([k.lower() + '_anonymised'],[a]) #saves answers as feather object
    return
    
def saveRawProfiles(yearstart, yearend, db_cnx):
    """
    This function iterates through all profiles and saves them in a ordered directory structure by year and unit.
    """
    
    if yearstart < 2009:
        for year in range(yearstart, yearend + 1):
            for unit in ['A','V']:
                for month in range(1, 13):
                    writeProfiles(db_cnx, year, month, unit)
    elif yearstart >= 2009 and yearend <= 2014:       
        for year in range(yearstart, yearend + 1):
            for unit in ['A', 'V', 'kVA', 'Hz', 'kW']:
                for month in range(1, 13):
                    writeProfiles(db_cnx, year, month, unit)