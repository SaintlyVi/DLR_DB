#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 12:17:16 2018

This is a shell script for retrieving and saving observations from the DLR MSSQL database.

@author: SaintlyVi
"""

import optparse

from observations.obs_processing import saveReducedProfiles, csvTables

parser = optparse.OptionParser()

parser.add_option('-i', '--intervalresample', dest='interval', default='H', type=str, help='Reduce load profiles to interval')
parser.add_option('-s', '--startyear', dest='startyear', type=int, help='Start year for profile data retrieval')
parser.add_option('-e', '--endyear', dest='endyear', type=int, help='End year for profile data retrieval')
parser.add_option('-c', '--csvdir', action='store_true', dest='csv', help='Format and save tables as csv files')

parser.set_defaults(csv=False)

(options, args) = parser.parse_args()
    
if options.startyear is None:
    options.startyear = int(input('Enter observation start year: '))
if options.endyear is None:
    options.endyear = int(input('Enter observation end year: '))
saveReducedProfiles(options.startyear, options.endyear, options.interval)

if options.csv == True:
    csvTables()
    
print('>>>obsProcess end<<<')