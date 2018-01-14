#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 12:17:16 2018

This is a shell script for retrieving and saving observations from the DLR MSSQL database.

@author: SaintlyVi
"""

import optparse

from experiment.experimental_model import saveExpModel

parser = optparse.OptionParser()

parser.add_option('-y', '--year', dest='year', type=int, help='Database connection filename')
parser.add_option('-x', '--experiment', dest='experiment', help='Save tables to disk')
parser.add_option('-a', '--algorithm', dest='algorithm', help='Save answers to disk')
parser.add_option('-r', '--run', dest='run', help='Save profiles to disk')

(options, args) = parser.parse_args()
    
saveExpModel(options.year, options.experiment, options.algorithm, options.run)

print('>>>genExpModel end<<<')