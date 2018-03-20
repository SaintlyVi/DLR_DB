#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 10:04:28 2018

@author: saintlyvi
"""

import shapefile as shp
from shapely.geometry import Point
from shapely.geometry import shape
import pandas as pd
import ckanapi

#ckan = ckanapi.RemoteCKAN('http://energydata.uct.ac.za/', get_only=True)
#ckanurl = ckan.action.resource_show(id="c92cea6d-9c2b-46e2-af49-a71b2b0b9270")['url']

#munic2011 = "/home/saintlyvi/Downloads/meso_2010_base_dd"

# Download the SHP, DBF and SHX files from http://energydata.uct.ac.za/dataset/2016-municipal-boundaries-south-africa
munic2016 = "/home/saintlyvi/Downloads/2016-Boundaries-Local/Local_Municipalities_2016"
filename = 'data/site_reference.csv'   
site_ref = pd.read_csv('data/site_reference.csv')
#response = requests.get(ckanurl)

sf = shp.Reader(munic2016)
sfRec = sf.records()

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
site_geo.to_csv('data/site_geo.csv')

