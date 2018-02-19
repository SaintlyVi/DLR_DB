#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:09:33 2017

@author: saintlyvi
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
from dash.dependencies import Input, Output#, State

import pandas as pd
import os
import base64
from collections import OrderedDict

import features.feature_socios as socios 
from support import dlrdb_dir

image_dir = os.path.join(dlrdb_dir,'img')
erc_logo = os.path.join(image_dir, 'erc_logo.jpg')
erc_encoded = base64.b64encode(open(erc_logo, 'rb').read())
sanedi_logo = os.path.join(image_dir, 'sanedi_logo.jpg')
sanedi_encoded = base64.b64encode(open(sanedi_logo, 'rb').read())

app = dash.Dash()

external_css = ["https://fonts.googleapis.com/css?family=Overpass:300,300i",
                "https://cdn.rawgit.com/plotly/dash-app-stylesheets/dab6f937fd5548cebf4c6dc7e93a10ac438f5efb/dash-technical-charting.css"]

for css in external_css:
    app.css.append_css({"external_url": css})

app.layout = html.Div([
        html.Div([
            html.Div([
                html.Img(src='data:image/png;base64,{}'.format(erc_encoded.decode()), 
                         style={'width': '100%', 'paddingLeft':'5%', 'marginTop':'20%' })    
            ],
                className='three columns',
                style={'margin_top':'20'}
            ),
            html.Div([
                html.H2('South African Domestic Load Research',
                        style={'textAlign': 'center'}
                ),
                html.H1('Data eXplorer',
                        style={'textAlign':'center'}
                )                    
            ],
                className='six columns'
            ),        
            html.Div([
                 html.Img(src='data:image/png;base64,{}'.format(sanedi_encoded.decode()), 
                          style={'width':'100%','margin-left':'-5%','marginTop':'10%' })                       
            ],
                className='three columns'
            ),              
        ],
            className='row',
            style={'background':'white',
                   'margin-bottom':'40'}
        ), 
        html.Div([
            html.Div([
                html.H3('Survey Locations'
                ),
                html.Div([
                    dcc.RangeSlider(
                        id = 'input-years-explore',
                        marks={i: i for i in range(1994, 2015, 2)},
                        min=1994,
                        max=2014,
                        step=1,
                        value=[2011, 2011],
                        dots = True
                    )                        
                ],
                    className='ten columns'
                ),
            ],
                className='columns',
                style={'margin-bottom':'10',
                       'width':'55%',
                       'float':'left'}
            ),
            html.Div([
                dt.DataTable(
                    id='output-location-list',
                    rows=[{}], # initialise the rows
                    row_selectable=True,
                    filterable=True,
                    sortable=True,
                    column_widths=100,
                    resizable=True,
                    selected_row_indices=[],)     
            ],
                className='columns',
                style={'margin-bottom':'10',
                       'margin-top':'50',
                       'width':'40%',
                       'float':'right'}
            ),
        ],
            className='row',
        ),
        html.Hr(),
        html.Div([ 
            html.Div([
                html.H3('Survey Questions'
                ),
                html.P('The DLR socio-demographic survey was updated in 2000. Select the surveys that you want to search.'),
                html.Div([
                    dcc.Checklist(
                        id = 'input-survey',
                        options=[
                                {'label': '1994 - 1999', 'value': 6},
                                {'label': '2000 - 2014', 'value': 3}
                                ],
                        values=[3]
                    )
                ],
                    className='container',
                    style={'margin': '10'}
                ),
                html.Div([
                    dcc.Input(
                        id='input-search-word',
                        placeholder='search term',
                        type='text',
                        value=''
                    )
                ],
                    className='container',
                    style={'margin': '10'}
                ),
                dt.DataTable(
                    id='output-search-word-questions',
                    rows=[{}], # initialise the rows
                    row_selectable=True,
                    filterable=False,
                    sortable=True,
                    selected_row_indices=[],)
            ],
                className='columns',
                style={'margin':10,
                       'width':'40%',
                       'float':'right'}
            ),
            html.Div([
                html.H3('Question Response Summary'
                ),
                html.P('Select a question and set of locations to see the distribution of responses.'),
                html.Div([
                    dcc.RadioItems(
                        id = 'input-summarise',
                        options=[
                                {'label': 'stats', 'value': 'stats'},
                                {'label': 'count', 'value': 'count'}
                                ],
                        value='count',
                        labelStyle={'display': 'inline-block'}
                    )
                ],
                    className='container',
                    style={'margin': '10'}
                ),
                dt.DataTable(
                    id='output-locqu-summary',
                    rows=[{}], # initialise the rows
                    row_selectable=True,
                    filterable=True,
                    sortable=True,
                    column_widths=40,
                    selected_row_indices=[],)
            ],
                className='columns',
                style={'margin':10,
                       'width':'55%',
                       'float':'left'}
            )
        ],
            className='row'
        ),
        html.Hr(),        
        html.Div([
            html.H3('Download Data'
            ),
            html.Div([
                html.Label('Select year range'
                ),
                dcc.RangeSlider(
                    id = 'input-years-download',
                    marks={i: i for i in range(1994, 2015, 2)},
                    min=1994,
                    max=2014,
                    step=1,
                    value=[2011, 2011],
                    dots = True
                )
            ],
                className='seven columns',
                style={'margin-bottom': '50'}
            ),
            html.P(),
            html.Div([
                html.Label('Specify comma-separated list of search terms to select question responses'
                ),
                dcc.Input(
                    id='search-list',
                    placeholder='search term',
                    type='text',
                    value=''
                )
            ],
                className='seven columns',
                style={'margin-bottom': '10'}
            )
        ],
            className='container',
            style={'margin': 10,
                   'padding': 0}
        ),
    ],
    #Set the style for the overall dashboard
    style={
        'width': '100%',
        'max-width': '1200',
        'margin-left': 'auto',
        'margin-right': 'auto',
        'font-family': 'overpass',
        'background-color': '#F3F3F3',
        'padding': '40',
        'padding-top': '20',
        'padding-bottom': '20',
    },
)

#Define outputs
@app.callback(
        Output('output-location-list','rows'),
        [Input('input-years-explore','value')]
        )
def update_locations(years_explore):
    ids = socios.loadID()
    dff = pd.DataFrame()
    for u in range(years_explore[0], years_explore[1]+1):
        df = ids[ids.Year.astype(int) == u].groupby(['Year','LocName'])['id'].count().reset_index()
        dff = dff.append(df)
    dff.reset_index(inplace=True, drop=True)
    dff.rename(columns={'LocName':'Location', 'id':'# households'}, inplace=True)
    return dff.to_dict('records', OrderedDict)
            
@app.callback(
        Output('output-search-word-questions','rows'),
        [Input('input-search-word','value'),
         Input('input-survey','values')]
        )
def update_questions(search_word, surveys):
    if isinstance(surveys, list):
        pass
    else:
        surveys = [surveys]
    df = socios.searchQuestions(search_word)[['Question','QuestionaireID']]
    dff = df.loc[df['QuestionaireID'].isin(surveys)]
    questions = pd.DataFrame(dff['Question'])
    return questions.to_dict('records')

@app.callback(
        Output('output-locqu-summary','rows'),
        [Input('output-location-list','selected_row_indices'),
         Input('output-location-list','rows'),
         Input('output-search-word-questions','selected_row_indices'),
         Input('output-search-word-questions','rows'),
         Input('input-summarise', 'value')]
        )

def update_locqu_summary(loc_select, loc_rows, qu_select, qu_rows, summarise):

    locations = pd.DataFrame(loc_rows).iloc[loc_select]
    years = locations.Year.unique()
    ids = socios.loadID()
    idselect = ids.loc[(ids.Year.isin(years)) & (ids.LocName.isin(locations.Location.unique())), 'id']
    
    searchterms = list(pd.DataFrame(qu_rows).loc[qu_select, 'Question'])
    
    d = pd.DataFrame()
    for y in years.astype(int):
        df = socios.buildFeatureFrame(searchterms, y)[0]
        d = d.append(df)

    locqu = d[d.AnswerID.isin(idselect)]
    
    if summarise == 'count':
        locqu_summary = pd.DataFrame()
        for c in locqu.columns[1:]:
            l = locqu[c].value_counts()
            locqu_summary = locqu_summary.append(l)
            
#        locqu_summary.columns = pd.MultiIndex.from_product([['Values'], locqu_summary.columns])
        locqu_sum_cols = ['count ' + str(i) for i in locqu_summary.columns]
        locqu_summary = locqu_summary.reset_index()
        locqu_summary.columns = ['Question'] + locqu_sum_cols
#        locqu_summary.columns = locqu_summary.columns.set_levels(['Values',              'Questions'],level=0)
            
    elif summarise == 'stats':
        locqu_summary = locqu.iloc[:,1:].describe().T
    
    return locqu_summary.to_dict('records', OrderedDict)

# Run app from script. Go to 127.0.0.1:8050 to view
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)