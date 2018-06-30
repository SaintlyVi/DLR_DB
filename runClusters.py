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

from support import experiment_dir, writeLog, log_dir
from experiment.algoritms.clusters import som, kmeans, saveResults, saveLabels
from features.feature_ts import genX

# Set up argument parser to run from terminal
parser = argparse.ArgumentParser(description='Cluster DLR timeseries data.')
parser.add_argument('p', dest='param_file', type=str, help='Parameter file with clustering specifications')
parser.add_argument('-l', dest='save_labels', type=int, help='Save cluster labels of top (int) results')
args = parser.parse_args()

param_dir = os.path.join(experiment_dir, 'parameters')
param_path = os.path.join(param_dir, args.param_file + '.txt')
header = open(param_path,'r')
param = list()
for line in header:
    if line.strip() != '':                # ignore blank lines
        param.append(eval(line))

for i in range(1, len(param)): #skip first line with header info
    # Extract all parameter values
    algorithm = param[i][0]
    start = param[i][1]
    end = param[i][2]
    preprocessing = param[i][3]
    range_n_dim = param[i][4]
    transform = param[i][5]
    range_n_clusters = param[i][6]

    X = genX([start, end])

    if algorithm == 'som':
        cluster_stats, cluster_centroids, cluster_lbls = som(X, range_n_dim, preprocessing, 
                                                             transform, range_n_clusters)    
    if args.algorithm == 'kmeans':
        cluster_stats, cluster_centroids, cluster_lbls = kmeans(X, range_n_clusters, preprocessing)

    eval_results = saveResults(args.param_file, cluster_stats, cluster_centroids)
    
    if type(args.save_labels) is int:
        saveLabels(X, cluster_lbls, args.save_labels, eval_results, args.param_file)
    
    log_line = param[i]
    logs = pd.DataFrame(log_line, columns = ['algorithm', 'start', 'end', 'preprocessing', 
                                             'range_n_dim', 'transform', 'range_n_clusters'])
    writeLog(logs, os.path.join(log_dir,'log_runClusters.csv'))

print(eval_results)
print('\n>>>genClusters end<<<')