#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 12:17:16 2018

This is a shell script for running timeseries clustering arguments from terminal.

@author: SaintlyVi
"""

import argparse
import os
import pandas as pd
import time

from support import experiment_dir, writeLog, log_dir
from experiment.algorithms.clusters import som, kmeans, saveLabels
from features.feature_ts import genX

# Set up argument parser to run from terminal
parser = argparse.ArgumentParser(description='Cluster DLR timeseries data.')
parser.add_argument('params', type=str, help='Parameter file with clustering specifications')
parser.add_argument('-top', type=int, help='Save labels for top n results')
parser.add_argument('-skip', type=int, help='Skip runs from top in parameter file')
args = parser.parse_args()

param_dir = os.path.join(experiment_dir, 'parameters')
param_path = os.path.join(param_dir, args.params + '.txt')
header = open(param_path,'r')
param = list()
for line in header:
    if line.strip() != '':                # ignore blank lines
        param.append(eval(line))

for i in range(args.skip+1, len(param)): #skip first line with header info
    # Extract all parameter values
    algorithm = param[i][0]
    start = param[i][1]
    end = param[i][2]
    preprocessing = param[i][3]
    range_n_dim = param[i][4]
    transform = param[i][5]
    range_n_clusters = param[i][6]
    
    print(param[i])
    
    tic = time.time()
    
    X = genX([start, end])

    if algorithm == 'som':
        stats, centroids, cluster_lbls = som(X, range_n_dim, args.top, preprocessing, transform, args.params,
                                             n_clusters=range_n_clusters)  
    if algorithm == 'kmeans':
        stats, centroids, cluster_lbls = kmeans(X, range_n_clusters, args.top, preprocessing, args.params)
#    if args.top:
#        saveLabels(X, cluster_lbls, stats, args.top)
        
    toc = time.time()
    
    log_line = param[i]
    logs = pd.DataFrame([[args.params, (toc-tic)/60] + list(log_line)], columns = ['experiment','runtime','algorithm', 
                         'start', 'end', 'preprocessing', 'range_n_dim', 'transform', 'range_n_clusters'])
    writeLog(logs, os.path.join(log_dir,'log_runClusters'))

print('\n>>>genClusters end<<<')