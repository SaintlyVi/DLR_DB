#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 12:17:16 2018

This is a shell script for retrieving and saving observations from the DLR MSSQL database.

@author: SaintlyVi
"""

import optparse

from observations.obs_retrieval import saveTables, saveAnswers, saveRawProfiles

parser = optparse.OptionParser()

parser.add_option('-c', '--cnxn', dest='cnxn', default='cnxnstr.txt', help='Database connection filename')
parser.add_option('-t', '--tables', action='store_true', dest='tables', help='Save tables to disk')
parser.add_option('-a', '--answers', action='store_true', dest='answers', help='Save answers to disk')
parser.add_option('-p', '--profiles', action='store_true', dest='profiles', help='Save profiles to disk')
parser.add_option('-s', '--startyear', dest='startyear', type=int, help='Start year for profile data retrieval')
parser.add_option('-e', '--endyear', dest='endyear', type=int, help='End year for profile data retrieval')

parser.set_defaults(tables=False, answers=False, profiles=False)

(options, args) = parser.parse_args()
    
if options.tables == True:
    saveTables(options.cnxn)
    
if options.answers == True:
    saveAnswers(options.cnxn)

if options.profiles == True:
    if options.startyear is None:
        options.startyear = int(input('Enter observation start year: '))
    if options.endyear is None:
        options.endyear = int(input('Enter observation end year: '))
    saveRawProfiles(options.startyear, options.endyear, options.cnxn)
    
print('>>>script end<<<')