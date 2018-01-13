# -*- coding: utf-8 -*-
"""
@author: Wiebke Toussaint

Support functions for the src module
"""

import os
from pathlib import Path

# root dir
dlrdb_dir = str(Path(__file__).parents[0])

# level 1
experimental_model_dir = os.path.join(dlrdb_dir, 'experiment')
obs_dir = os.path.join(dlrdb_dir, 'observations')
feature_dir = os.path.join(dlrdb_dir, 'features')
data_dir = os.path.join(dlrdb_dir, 'data')
eval_dir = os.path.join(dlrdb_dir, 'evaluation')
image_dir = os.path.join(dlrdb_dir, 'images')

# level 2 & 3 DATA
dpet_dir = os.path.join(data_dir, 'benchmark_model', 'dpet')
table_dir = os.path.join(data_dir, 'obs_datasets', 'tables')
profiles_dir = os.path.join(data_dir, 'obs_datasets', 'profiles')
rawprofiles_dir = os.path.join(profiles_dir, 'raw')
fdata_dir = os.path.join(data_dir, 'feature_data')
cdata_dir = os.path.join(dlrdb_dir, 'class_data')

class InputError(ValueError):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message