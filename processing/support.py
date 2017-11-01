# -*- coding: utf-8 -*-
"""
@author: Wiebke Toussaint

Support functions for the src module
"""

import os
from pathlib import Path

# root dir
dlrdb_dir = str(Path(__file__).parents[1])

# level 1
src_dir = str(Path(__file__).parents[0])
classmodel_dir = os.path.join(dlrdb_dir, 'class_model')
obs_dir = os.path.join(dlrdb_dir, 'observations')
data_dir = os.path.join(dlrdb_dir, 'data')

# level 2 & 3 DATA
rawprofiles_dir = os.path.join(data_dir, 'profiles', 'raw')
hourlyprofiles_dir = os.path.join(data_dir, 'profiles', 'hourly')
table_dir = os.path.join(data_dir, 'tables')

# level 2 & 3 INFERENCE
evidence_dir = os.path.join(classmodel_dir, 'evidence')
classout_dir = os.path.join(classmodel_dir, 'out')