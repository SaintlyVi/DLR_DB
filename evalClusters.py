#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 17:26:20 2018

@author: saintlyvi
"""

import argparse
import os
import pandas as pd
import time

from evaluation.eval_clusters import getLabels, realCentroids, consumptionError, peakCoincidence
from features.feature_extraction import genF

# Set up argument parser to run from terminal
parser = argparse.ArgumentParser(description='Evaluate DLR timeseries clusters.')
parser.add_argument('experiment', type=str, help='Experiment_algorithm_preprocessing')
parser.add_argument('socios', type=str, help='Specification of socio_demographic features')
args = parser.parse_args()


xl = getLabels(args.experiment)
centroids = realCentroids(xl, args.experiment)
consumptionError(xl, centroids, compare='total')
consumptionError(xl, centroids, compare='peak')
peakCoincidence(xl, centroids)
F = genF(args.experiment, args.socios, savefig=False)