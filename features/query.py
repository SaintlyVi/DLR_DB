# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 16:35:39 2017

@author: CKAN
"""

import pandas as pd
from socios import *

wall_material_dist = answerSearch('wall', 3)[0].groupby(answerSearch('wall', 3)[0].iloc[:,-1]).size()

roof_material_dist = answerSearch('roof', 3)[0].groupby(answerSearch('roof', 3)[0].iloc[:,-1]).size()

    
    files = glob(os.path.join(data_dir, '*.feather'))
    names = [f.rpartition('.')[0] for f in os.listdir(data_dir)]
    tables = {}
    for n, f in zip(names, files):
        try:
            tables[n] = feather.read_dataframe(f)
        except:
            pass
    return tables