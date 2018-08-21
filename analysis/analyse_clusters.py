#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 18:04:22 2018

@author: saintlyvi
"""

import pandas as pd
import numpy as np
import datetime as dt
from math import ceil, log
import feather
import os
from glob import glob
import peakutils

import evaluation.eval_clusters as ec
import features.feature_socios as soc


experiment = 'exp7_kmeans_unit_norm'
xlabel = ec.getLabels(experiment)
hr = ec.householdReliability(xlabel)
S = soc.genS('features3', 1994, 2014, 'feather')
longterm = S.merge(hr, on = 'ProfileID')
L = longterm.groupby(['k','Year']).agg({'monthly_income':'mean','adults':'mean','unemployed':'mean','entropy':'mean','k_count':['sum','count']}).reset_index()


def plotMonthlyIncome():
    
    'RdBu'
    
    return