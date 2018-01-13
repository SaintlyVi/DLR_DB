# -*- coding: utf-8 -*-
"""
@author: Wiebke Toussaint

Support functions for the src module
"""

import os
from pathlib import Path
import datetime as dt

# root dir
dlrdb_dir = str(Path(__file__).parents[0])

# level 1
experimental_model_dir = os.path.join(dlrdb_dir, 'experiment')
obs_dir = os.path.join(dlrdb_dir, 'observations')
feature_dir = os.path.join(dlrdb_dir, 'features')
data_dir = os.path.join(dlrdb_dir, 'data')
eval_dir = os.path.join(dlrdb_dir, 'evaluation')
image_dir = os.path.join(dlrdb_dir, 'images')
log_dir = os.path.join(dlrdb_dir, 'log')

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
        
def writeLog(log_line, file_name):    
    """Adds timestamp column to dataframe, then writes dataframe to csv log file. 
    """
    #Create log_dir and file to log path
    os.makedirs(log_dir , exist_ok=True)
    log_path = os.path.join(log_dir, file_name+'.csv')
    
    #Add timestamp
    log_line.insert(0, 'timestamp', dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    #Write log data to file
    if os.path.isfile(log_path):
        log_line.to_csv(log_path, mode='a', header=False, columns = log_line.columns, index=False)
        print('\nSuccess! Log entry added to log/' + file_name + '.csv\n')
    else:
        log_line.to_csv(log_path, mode='w', columns = log_line.columns, index=False)
        print('\nSuccess! Log file created and log entry added to log/' + file_name + '.csv\n')    
    return log_line