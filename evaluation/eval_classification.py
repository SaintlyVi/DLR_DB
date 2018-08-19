#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 09:52:39 2018

@author: saintlyvi
"""

import pandas as pd
import os
from glob import glob

from support import data_dir, results_dir, experiment_dir

mod = pd.DataFrame()
p = os.path.join(results_dir,'classification_results')
for file in os.listdir(p): 
    if 'output_mod3' in file: 
        data = pd.read_csv(os.path.join(p, file))
#        print(mod.columns.difference(data.columns))
        mod = pd.concat([mod, data], axis=0, sort=True)
#        print(file, len(mod.columns))
mod.sort_values(by=['Key_Dataset','Key_Run','Key_Scheme_options'], inplace=True)
mod.to_csv(os.path.join(p, 'classification_output3.csv'), index=False)