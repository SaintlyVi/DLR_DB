#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 12:48:14 2018

@author: saintlyvi
"""

import ckanapi
import pandas as pd
import os
import feather

from support import aggprofiles_dir, table_dir
from features_ts import readAggProfiles

def appData():
    
    data = pd.DataFrame()
    
    try:
        for y in range(1994, 2015):
            d = readAggProfiles(y, 'adtd')
            data = data.append(d)

    except FileNotFoundError:
            
        feather_path = os.path.join(aggprofiles_dir, 'adtd', 'feather')
        csv_path = os.path.join(aggprofiles_dir, 'adtd', 'csv')
        os.makedirs(feather_path, exist_ok=True)
        os.makedirs(csv_path, exist_ok=True)    
        
        #fetch tables from energydata.uct.ac.za
        apikey = input('Enter your APIKEY from http://energydata.uct.ac.za/user/YOUR_USERNAME: ')
        ckan = ckanapi.RemoteCKAN('http://energydata.uct.ac.za/', apikey=apikey, get_only=True)
        tables = ckan.action.package_show(id='dlr-database-tables-94-14')        
        for i in range(0, len(tables['resources'])):
    #        if tables['resources'][i]['name'] == name:
                print('... fetching ' + tables['resources'][i]['name'] + ' from energydata.uct.ac.za')
                r_id = tables['resources'][i]['id']
                d = ckan.action.datastore_search(resource_id=r_id, limit=1000000)['records']
                table = pd.DataFrame(d)
                table = table.iloc[:,:-1]
                
                feather.write_dataframe(table, table_dir)
                table.to_csv(table_dir, index=False)
        
        profiles = ckan.action.package_show(id='dlr-average-day-type-demand-profiles')        
        for i in range(0, len(profiles['resources'])):
    #        if tables['resources'][i]['name'] == name:
                print('... fetching ' + profiles['resources'][i]['name'] + ' from energydata.uct.ac.za')
                r_id = profiles['resources'][i]['id']
                d = ckan.action.datastore_search(resource_id=r_id, limit=1000000)['records']
                adtd = pd.DataFrame(d)
                #write profiles to disk                
                feather.write_dataframe(adtd, feather_path)
                adtd.to_csv(csv_path, index=False)
            
        try:
            for y in range(1994, 2015):
                d = readAggProfiles(y, 'adtd')
                data = data.append(d)
        except FileNotFoundError:
            print('Check your internet connection and rerun the app.')
    
    return data

