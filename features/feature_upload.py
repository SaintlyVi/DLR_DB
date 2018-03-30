#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 15:00:01 2018

@author: saintlyvi
"""

import ckanapi
from pathlib import Path
import os

from support import aggprofiles_dir

def profiles2ckan(year_start, year_end, directory, package_name):
    
    apikey = input('Enter your APIKEY from http://energydata.uct.ac.za/user/YOUR_USERNAME: ')
    ckan = ckanapi.RemoteCKAN('http://energydata.uct.ac.za/', apikey=apikey)
     
    path = Path(os.path.join(aggprofiles_dir, directory, 'csv'))
    for child in path.iterdir():
        n = child.name
        filename = n.split('.')[0]
        nu = filename.split('_')[-1]
        for y in range(year_start, year_end + 1):  
            if int(nu)==y:
                file_upload = ckan.call_action('resource_create',
                                 {'package_id': package_name, 'name':filename},
                                 files={'upload': open(str(child), 'rb')})
                view = ckan.call_action('resource_view_create',
    {'resource_id': file_upload['id'], 'title':'Data', 'view_type':'recline_view'})
                print(file_upload['name'] + ' uploaded and ' + view['title'] + ' view created.')        
        
        
def bulkDelete(package_name):
    
    apikey = input('Enter your APIKEY from http://energydata.uct.ac.za/user/YOUR_USERNAME: ')
    ckan = ckanapi.RemoteCKAN('http://energydata.uct.ac.za/', apikey=apikey)
         
    p = ckan.call_action('package_show',{'id':package_name})
    for i in range(0, len(p['resources'])): 
        ckan.call_action('resource_delete', {'id': p['resources'][i]['id']})