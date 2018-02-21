"""
Support file for generating helper functions for Bayesian networks, inference and subsequent data visualisation.

NOTE: Apply the following fix before using libpgm with python 3:
    
    Execute `2to3 -w /home/user/anaconda3/lib/python3.5/site-packages/utils/bntextutils.py` from the command prompt.

@author: Wiebke Toussaint
"""
import pandas as pd
import json
import os
import numpy as np
import daft

from libpgm.graphskeleton import GraphSkeleton
from libpgm.nodedata import NodeData
from libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
from libpgm.tablecpdfactorization import TableCPDFactorization

from support import fdata_dir, experiment_dir, cdata_dir, InputError, writeLog, validYears

def loadbn(param_file):
    """
    This function loads the bn model into the workspace from its associated .txt file.
    """
    file_path = os.path.join(experiment_dir, 'parameters', param_file + '.txt')
    
    nd = NodeData()
    skel = GraphSkeleton()
    nd.load(file_path)
    skel.load(file_path)
    skel.toporder()
    bn = DiscreteBayesianNetwork(skel, nd)
    return bn

def readEvidence(year, dir_name):
    """
    This function reads the data from an evidence file saved in the evidence directory. The naming convention for the file is `bn_evidence_(year).txt`. The function returns a list of evidence dicts and a list of corresponding AnswerIDs.
    """
    
    validYears(year) #check if year input is valid
    
    dir_path = os.path.join(fdata_dir, dir_name)
    file_name = [s for s in os.listdir(dir_path) if str(year) in s][0]
    file_path = os.path.join(dir_path, file_name)
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    if data is None:
        raise InputError(year, 'No evidence data available for this year')
    else:
        answerid = list(data.keys())
        evidence = list(data.values())
        
        return evidence, answerid

def inferCustomerClasses(param_file, evidence_dir, year):
    """
    This function uses the variable elimination algorithm from libpgm to infer the customer class of each AnswerID, given the evidence presented in the socio-demographic survey responses. 
    
    It returns a tuple of the dataframe with the probability distribution over all classes for each AnswerID and the BN object.
    """   
    bn = loadbn(param_file)
    evidence, a_id = readEvidence(year, evidence_dir)
    query = {"customer_class":''}
    
    cols = bn.Vdata.get('customer_class')['vals']
    result = pd.DataFrame(columns=cols) #create empty dataframe in which to store inferred probabilities
    
    count = 0 #set counter
    for e in evidence:
        bn = loadbn(param_file)
        fn = TableCPDFactorization(bn)
        try:
            inf = fn.condprobve(query, e)
            classprobs = list(inf.vals)
            result.loc[count] = classprobs
            count += 1
        except:
            result.loc[count] = [None] * len(cols)
            count += 1
    
    result['AnswerID'] = a_id
    result.set_index(keys='AnswerID',inplace=True)
    
    return result

def saveClasses(yearstart, yearend, param_file, evidence_dir):
    """
    This function saves the inferred probability distribution over each class for each AnswerID in a csv file. 
    """
    
    loglines = []
    
    for year in range(yearstart, yearend + 1):
        
        try:
            dir_path = os.path.join(cdata_dir, param_file)
            os.makedirs(dir_path , exist_ok=True)
            file_path = os.path.join(dir_path, 'classes_'+str(year)+'.csv')  
            classes = inferCustomerClasses(param_file, evidence_dir, year)
            
            status = 1      
            message = 'Success!'
            print('Success! Saving to data/class_data/' + param_file + '/classes_'+str(year)+'.csv' )            

        except InputError as e:
            pass
            classes = pd.DataFrame()
            status = 0 
            message = e.message
            print(e)
            print('Saving empty file') 

        classes.to_csv(file_path)
                       
        l = ['classInference', year, status, message, param_file, evidence_dir+'_spec']
        loglines.append(l)
        
    logs = pd.DataFrame(loglines, columns = ['process', 'year','status','message', 
                                             'parameter_settings', 'feature_specification'])
    writeLog(logs,'log_inferClasses')
            
def graphViz(param_file):
    """
    This function generates a png image of the Bayesian Network graph. The model input parameter must be a string that specifies the name of a libpgm compatible, json formatted text file. 
    """
    
    bn = loadbn(param_file) 
    bn.toporder()
    parents = []
    roots = []
    for v in bn.V:
        vp = bn.getparents(v)
        if len(vp) ==0:
            roots+=[v]
        parents+= vp
    parents = set(parents)

    child_hierarchy = []
    for p in parents:
        child_hierarchy+=[bn.Vdata.get(p)['children']]

    node_unit = 1.25 #set node height
    aspect = 3.0 
    node_width=node_unit*aspect #set node width
    width = max([len(i) for i in child_hierarchy])*node_width*1.2 #set image width to 1.2 x node_width x max_nodes_row
    height=(2*len(parents)+3)*node_unit #set image height

    # Instantiate the PGM.
    pgm = daft.PGM([width, height], origin=[0, 0], grid_unit=1, aspect=aspect, node_unit=node_unit)

    # Root nodes
    root_y = height - node_unit*1.5
    root_x = width/(len(roots)+1)
    i=1
    for r in roots:
        pgm.add_node(daft.Node(r, r, x=root_x*i, y=root_y))
        i+=1

    # Child nodes
    i=1
    for children in child_hierarchy:
        child_y = root_y - (1+i)*node_unit
        j=1
        for c in children:
            child_x = node_width*0.6 + node_width*1.2*(j-1)
            pgm.add_node(daft.Node(c, c, x=child_x, y=child_y))
            j+=1
        i+=1

    # Add in the edges
    for e in bn.E:
        pgm.add_edge(e[0], e[1])

    # Render and save
    dir_path = os.path.join(experiment_dir, 'images')
    os.makedirs(dir_path , exist_ok=True)
    file_path = os.path.join(dir_path, param_file + ".png")    
    
    pgm.render()
    pgm.figure.savefig(file_path, dpi=150)
