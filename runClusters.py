#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 12:17:16 2018

This is a shell script for retrieving and saving observations from the DLR MSSQL database.

@author: SaintlyVi
"""

import optparse

from experiment.algoritms.clusters import som, kmeans, saveResults
from features.feature_ts import genX

parser = optparse.OptionParser()

parser.add_option('-s', '--start', dest='start', type=int, help='Year start')
parser.add_option('-e', '--end', dest='end', type=int, help='Year end')
parser.add_option('-a', '--algorithm', dest='algorithm', help='som or kmeans')
parser.add_option('-d', '--range_n_dim', dest='range_n_dim', help='range_n_dim - som only')
parser.add_option('-n', '--range_n_clusters', dest='range_n_clusters', help='range_n_clusters')
parser.add_option('-p', '--preprocessing', dest='preprocessing', help='preprocessing')

(options, args) = parser.parse_args()

if options.start is None:
    options.start = int(input('Enter observation start year: '))
if options.end is None:
    options.end = int(input('Enter observation end year: '))
    
X = genX([options.start, options.end])

if options.algorithm == 'som':
    cluster_stats, cluster_centroids, cluster_lbls = som(X, options.range_n_dim, options.preprocessing, 
                                                         transform=False, options.range_n_clusters)

if options.algorithm == 'kmeans':
    cluster_stats, cluster_centroids, cluster_lbls = kmeans(X, options.range_n_clusters, options.preprocessing)

print('>>>genClusters end<<<')