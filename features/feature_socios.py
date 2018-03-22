#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 09:34:08 2017

@author: Wiebke Toussaint

Answer query script: This script contains functions to query and manipulate DLR survey answer sets. It references datasets that must be stored in a /data/tables subdirectory in the parent directory.

"""

import numpy as np
import pandas as pd

from observations.obs_processing import loadTable
from support import validYears

def loadID(year = None, id_name = 'AnswerID'):
    """
    This function subsets Answer or Profile IDs by year. Tables variable can be constructred with loadTables() function. Year input can be number or string. id_name is AnswerID or ProfileID. 
    """
    groups = loadTable('groups')
    links = loadTable('links')
    links = links[(links.GroupID != 0) & (links[id_name] != 0)]
    join = links.merge(groups, on='GroupID')    
    all_ids = join[join['Survey'] != 'Namibia'] # take Namibia out
    clean_ids = all_ids.drop_duplicates(id_name)[['Year', 'LocName', 'GroupID', id_name]]
    clean_ids.columns = ['Year', 'LocName', 'GroupID', 'id']
    clean_ids.Year = clean_ids.Year.astype(int)
    if year is None:
        return clean_ids
    else:   
        validYears(year) #check if year input is valid
        ids = clean_ids[clean_ids.Year == int(year)]
        return ids

def loadQuestions(dtype = None):
    """
    This function gets all questions.
    
    """
    qu = loadTable('questions').drop(labels='lock', axis=1)
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
        ans = loadTable('answers', columns=['AnswerID', 'QuestionaireID'])
    elif dtype == 'blob':
        ans = loadTable('answers_blob_anonymised')
        ans.fillna(np.nan, inplace = True)
    elif dtype == 'char':
        ans = loadTable('answers_char_anonymised').drop(labels='lock', axis=1)
    elif dtype == 'num':
        ans = loadTable('answers_number_anonymised').drop(labels='lock', axis=1)
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
    qcons = loadTable('qconstraints').drop(labels='lock', axis=1)
    qu = loadQuestions(dtype)
    qdf = qu.join(qcons, 'QuestionID', rsuffix='_c') #join question constraints to questions table
    qnairids = list(loadTable('questionaires')['QuestionaireID']) #get list of valid questionaire IDs
    if qnairid is None: #gets all relevant queries
        pass
    elif qnairid in qnairids: #check that ID is valid if provided
        qdf = qdf[qdf.QuestionaireID == qnairid] #subset dataframe to relevant ID
    else:
        return print('Please select a valid QuestionaireID', qnairids)
    
    result = qdf.loc[qdf.Question.str.lower().str.contains('|'.join(searchterm)), ['Question', 'Datatype','QuestionaireID', 'ColumnNo', 'Lower', 'Upper']]
    return result

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
    year = int(year)
    data = pd.DataFrame(loadID(year, 'AnswerID')['id']) #get AnswerIDs for year
    data.columns = ['AnswerID']
    questions = pd.DataFrame() #construct dataframe with feature questions

    if isinstance(searchlist, list):
        pass
    else:
        searchlist = [searchlist]
        
    for s in searchlist:
        try:
            if year <= 1999:
                d, q = searchAnswers(s, qnairid = 6, dtype = 'num')
            else:
                d, q = searchAnswers(s, qnairid = 3, dtype = 'num')
                d.columns = ['AnswerID', s]
            q['searchterm'] = s
            newdata = d[d.AnswerID.isin(data.AnswerID)]
            data = pd.merge(data, newdata, on = 'AnswerID')
            questions = pd.concat([questions, q])
        except:
            pass
    questions.reset_index(drop=True, inplace=True)
        
    return data, questions

def checkAnswer(answerid, features):
    """
    This function returns the survey responses for an individuals answer ID and list of search terms.
    
    """
    links = loadTable('links')
    groupid = links.loc[links['AnswerID']==answerid].reset_index(drop=True).get_value(0, 'GroupID')
    groups = loadTable('groups')
    year = int(groups.loc[groups.GroupID == groupid, 'Year'].reset_index(drop=True)[0])
    
    ans = buildFeatureFrame(features, year)[0].loc[buildFeatureFrame(features, year)[0]['AnswerID']==answerid]
    return ans

def recorderLocations(year = 2014):
    """
    This function returns all survey locations and recorder abbreviations for a given year. Only valid from 2009 onwards.
    
    """
    if year > 2009:
        stryear = str(year)
        groups = loadTable('groups')
        recorderids = loadTable('recorderinstall')
        
        reclocs = groups.merge(recorderids, left_on='GroupID', right_on='GROUP_ID')
        reclocs['recorder_abrv'] = reclocs['RECORDER_ID'].apply(lambda x:x[:3])
        yearlocs = reclocs.loc[reclocs['Year']== stryear,['GroupID','LocName','recorder_abrv']].drop_duplicates()
        
        locations = yearlocs.sort_values('LocName')
        return locations 
    
    else:
        print('Recorder locations can only be returned for years after 2009.')

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
