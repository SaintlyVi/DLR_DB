#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 15:21:07 2018

@author: saintlyvi
"""

from experiment.prediction import features2array, learnBN
from features.feature_extraction import genF

F = genF('exp5_kmeans_unit_norm','features1')
fdata = features2array(F)
bn = learnBN(fdata, 'predict1_bn')