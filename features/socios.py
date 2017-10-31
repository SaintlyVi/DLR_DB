#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 09:34:08 2017

@author: Wiebke Toussaint

Answer query script: This script contains functions to query and manipulate DLR survey answer sets. It references datasets that must be stored in a /data/tables subdirectory in the parent directory.

"""

import numpy as np
import pandas as pd
import feather
from glob import glob
import os

from features.support import table_dir

def loadTables(filepath = table_dir):
    """
    This function loads all feather tables in filepath into workspace.
    
    """
    files = glob(os.path.join(filepath, '*.feather'))
    names = [f.rpartition('.')[0] for f in os.listdir(filepath)]
    tables = {}
    for n, f in zip(names, files):
        try:
            tables[n] = feather.read_dataframe(f)
        except:
            pass
    return tables
    
def csvTables(savepath):
    """
    This function fetches tables saved as feather objects and saves them as csv files.
    """
    
    tables = loadTables() #get data
    filenames = [os.path.join(savepath, x + '.csv') for x in list(tables.keys())] #generate list of filenames
    
    for name, path in zip(list(tables.keys()), filenames):
        df = tables[name]
        df.to_csv(path, index=False)
        print('Successfully saved to ' + path)        
    return        

def loadID(year = None, id_name = 'AnswerID'):
    """
    This function subsets Answer or Profile IDs by year. Year input can be number or string. id_name is AnswerID or ProfileID.
    """
    groups = loadTables().get('groups')
    links = loadTables().get('links')
    all_ids = links[(links.GroupID != 0) & (links[id_name] != 0)]
    if year is None:
        ids = pd.Series(all_ids.loc[:, id_name].unique())
    else:      
        stryear = str(year)
        id_select = groups[groups.Year==stryear]['GroupID']
        ids = pd.Series(all_ids.loc[all_ids.GroupID.isin(id_select), id_name].unique())
    return ids

def loadQuestions(dtype = None):
    """
    This function gets all questions.
    
    """
    qu = loadTables().get('questions').drop(labels='lock', axis=1)
    qu.Datatype = qu.Datatype.astype('category')
    qu.Datatype.cat.categories = ['blob','char','num']
    if dtype is None:
        pass
    else: 
        qu = qu[qu.Datatype == dtype]
    return qu

def loadAnswers(dtype = None):
    """
    This function returns all answer IDs and their question responses for a selected data type. If dtype is None, answer IDs and their corresponding questionaire IDs are returned instead.
    
    """
    if dtype is None:
        ans = loadTables().get('answers').drop(labels='lock', axis=1)
    elif dtype == 'blob':
        ans = loadTables().get('answers_blob_anon')
        ans.fillna(np.nan, inplace = True)
    elif dtype == 'char':
        ans = loadTables().get('answers_char_anon').drop(labels='lock', axis=1)
    elif dtype == 'num':
        ans = loadTables().get('answers_number_anon').drop(labels='lock', axis=1)
    return ans

def searchQuestions(searchterm = '', qnairid = None, dtype = None):
    """
    Searches questions for a search term, taking questionaire ID and question data type (num, blob, char) as input. 
    A single search term can be specified as a string, or a list of search terms as list.
    
    """
    if isinstance(searchterm, list):
        pass
    else:
        searchterm = [searchterm]
    searchterm = [s.lower() for s in searchterm]
    qcons = loadTables().get('qconstraints').drop(labels='lock', axis=1)
    qu = loadQuestions(dtype)
    qdf = qu.join(qcons, 'QuestionID', rsuffix='_c') #join question constraints to questions table
    qnairids = list(loadTables().get('questionaires')['QuestionaireID']) #get list of valid questionaire IDs
    if qnairid is None: #gets all relevant queries
        pass
    elif qnairid in qnairids: #check that ID is valid if provided
        qdf = qdf[qdf.QuestionaireID == qnairid] #subset dataframe to relevant ID
    else:
        return print('Please select a valid QuestionaireID', qnairids)
    result = qdf.loc[qdf.Question.str.lower().str.contains('|'.join(searchterm)), ['Question', 'Datatype','QuestionaireID', 'ColumnNo', 'Lower', 'Upper']]
    return result

def typeSplitQuestions(qnid = 3):
    """
    Creates a dict with items containing questions by type (num, blob, char).
    
    """
    d = {i : searchQuestions(qnairid = qnid, dtype=i) for i in ['num','blob','char']}
    return d

def searchAnswers(searchterm = '', qnairid = 3, dtype = 'num'):
    """
    This function returns the answers IDs and responses for a list of search terms
    
    """
    allans = loadAnswers() #get answer IDs for questionaire IDs
    ans = loadAnswers(dtype) #retrieve all responses for data type
    questions = searchQuestions(searchterm, qnairid, dtype) #get column numbers for query
    result = ans[ans.AnswerID.isin(allans[allans.QuestionaireID == qnairid]['AnswerID'])] #subset responses by answer IDs
    result = result.iloc[:, [0] +  list(questions['ColumnNo'])]
    return [result, questions[['ColumnNo','Question']]]

def buildFeatureFrame(searchlist, year):
    """
    This function creates a dataframe containing the data for a set of selected features for a given year.
    
    """
    data = pd.DataFrame(data = loadID(year), columns=['AnswerID']) #get AnswerIDs for year
    questions = pd.DataFrame() #construct dataframe with feature questions
    
    for s in searchlist:
        if year <= 1999:
            ans = searchAnswers(s, qnairid = 6, dtype = 'num')
        else:
            ans = searchAnswers(s, qnairid = 3, dtype = 'num')
        d = ans[0]
        q = ans[1]
        q['searchterm'] = s
        newdata = d[d.AnswerID.isin(data.AnswerID)]
        data = pd.merge(data, newdata, how='outer', on = 'AnswerID')
        questions = pd.concat([questions, q])
    questions.reset_index(drop=True, inplace=True)
        
    return [data, questions]

def checkAnswer(answerid, features):
    """
    This function returns the survey responses for an individuals answer ID and list of search terms.
    
    """
    links = loadTables().get('links')
    groupid = links.loc[links['AnswerID']==answerid].reset_index(drop=True).get_value(0, 'GroupID')
    groups = loadTables().get('groups')
    year = int(groups.loc[groups.GroupID == groupid, 'Year'].reset_index(drop=True)[0])
    
    ans = buildFeatureFrame(features, year)[0].loc[buildFeatureFrame(features, year)[0]['AnswerID']==answerid]
    return ans

def recorderLocations(year = 2014):
    """
    This function returns all survey locations and recorder abbreviations for a given year. Only valid from 2009 onwards.
    
    """
    stryear = str(year)
    groups = loadTables().get('groups')
    groups['loc'] = groups['Location'].apply(lambda x:x.partition(' ')[2])
    recorderids = loadTables().get('recorderinstall')
    
    reclocs = groups.merge(recorderids, left_on='GroupID', right_on='GROUP_ID')
    reclocs['recorder_abrv'] = reclocs['RECORDER_ID'].apply(lambda x:x[:3])
    yearlocs = reclocs.loc[reclocs['Year']== stryear,['GroupID','loc','recorder_abrv']].drop_duplicates()
    
    locations = yearlocs.sort_values('loc')
    return locations 

def lang(code = None):
    """
    This function returns the language categories.
    
    """
    language = dict(zip(searchAnswers(qnairid=5)[0].iloc[:,1], searchAnswers(qnairid=5,dtype='char')[0].iloc[:,1]))
    if code is None:
        pass
    else:
        language = language[code]
    return language

def altE(code = None):
    """
    This function returns the alternative fuel categories.
    
    """
    altenergy = dict(zip(searchAnswers(qnairid=8)[0].iloc[:,1], searchAnswers(qnairid=8,dtype='char')[0].iloc[:,1]))
    if code is None:
        pass
    else:
        altenergy = altenergy[code]
    return altenergy


#QnID = ans[ans['AnswerID'] == 34]['QuestionaireID']