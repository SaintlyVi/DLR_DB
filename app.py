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
from collections import OrderedDict

import features.feature_socios as socios 

app = dash.Dash()

external_css = ["https://fonts.googleapis.com/css?family=Overpass:300,300i",
                "https://cdn.rawgit.com/plotly/dash-app-stylesheets/dab6f937fd5548cebf4c6dc7e93a10ac438f5efb/dash-technical-charting.css"]

for css in external_css:
    app.css.append_css({"external_url": css})

app.layout = html.Div(
        [
        html.Div([
            html.H2('South African Domestic Load Research',
                    style={'textAlign': 'center'}
            ),
            html.H1('Data eXplorer',
                    style={'textAlign': 'center'}
            )              
        ],
            className='twelve columns',
            style={'background': 'white',
                   'margin-bottom': '40'}
        ), 
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
                ),
            ],
                className='seven columns',
                style={'margin-bottom': '10'}
            ),
            html.Div([
                html.Div([
                        html.P(),
                        dt.DataTable(
                            id='output-location-list',
                            rows=[{}], # initialise the rows
                            row_selectable=True,
                            filterable=True,
                            sortable=True,
                            min_width=400,
                            selected_row_indices=[],)
                    ],
                        className='four columns'
                    ),        
            ],
                className='four columns',
                style={'margin-bottom': '10',
                       'padding-left': '40'}
            ),
        ],
            className='container',
            style={'margin': 10}
        ),
        html.Hr(),
        html.Div([
            html.H3('Survey Questions'
            ), 
            html.Div([
                html.P('The DLR socio-demographic survey was updated in 2000. Select the surveys that you want to search.'),
                html.Div([
                    dcc.Checklist(
                        id = 'input-survey',
                        options=[
                                {'label': '1994 - 1999', 'value': 6},
                                {'label': '2000 - 2014', 'value': 3}
                                ],
                        values=[3]
                    ),              
                    html.P(),   
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
                    ),              
                    html.P(),
                ],
                    className='four colmns'
                ),
                dt.DataTable(
                    id='output-search-word-questions',
                    rows=[{}], # initialise the rows
                    row_selectable=True,
                    filterable=False,
                    sortable=True,
                    selected_row_indices=[],)
            ],
                className='six columns'
            )
        ],
            className='container',
            style={'margin': 10,
                   'padding': 0}
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
def update_locations(user_selection):
    ids = socios.loadID()
    dff = pd.DataFrame()
    for u in range(user_selection[0], user_selection[1]+1):
        df = ids[ids.Year == str(u)].groupby(['Year','LocName'])['id'].count().reset_index()
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

# Run app from script. Go to 127.0.0.1:8050 to view
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)