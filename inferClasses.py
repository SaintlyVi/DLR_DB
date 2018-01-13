#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 12:17:16 2018

This is a shell script for retrieving and saving observations from the DLR MSSQL database.

@author: SaintlyVi
"""

import optparse

parser = optparse.OptionParser()

parser.add_option('-p', '--parameters', dest='param', help='Parameters of implementation')
parser.add_option('-d', '--data_dir', dest='data', help='Name of directory with evidence / data')
parser.add_option('-s', '--startyear', dest='startyear', type=int, help='Start year class inference')
parser.add_option('-e', '--endyear', dest='endyear', type=int, help='End year class inference')

(options, args) = parser.parse_args()

experiment, algorithm, run = options.param.split('_')
exec('from experiment.algorithms.'+algorithm+' import saveClasses')
    
if options.startyear is None:
    options.startyear = int(input('Enter observation start year: '))
if options.endyear is None:
    options.endyear = int(input('Enter observation end year: '))
    
saveClasses(options.startyear, options.endyear, options.param, options.data)

print('>>>inferClasses end<<<')