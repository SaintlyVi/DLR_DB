#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 12:17:16 2018

This is a shell script for retrieving and saving observations from the DLR MSSQL database.

@author: SaintlyVi
"""

import optparse

from features.feature_extraction import saveData

parser = optparse.OptionParser()

parser.add_option('-s', '--startyear', dest='startyear', type=int, help='Start year for profile data retrieval')
parser.add_option('-e', '--endyear', dest='endyear', type=int, help='End year for profile data retrieval')
parser.add_option('-f', '--specfile', dest='specfile', type=str, help='Feature specification file name')
parser.add_option('-n', '--name', dest='name', type=str,  default='evidence', help='Output file naming convention')

(options, args) = parser.parse_args()
    
if options.startyear is None:
    options.startyear = int(input('Enter observation start year: '))
if options.endyear is None:
    options.endyear = int(input('Enter observation end year: '))
saveData(options.startyear, options.endyear, options.specfile, options.name)

print('>>>featureExtraction end<<<')