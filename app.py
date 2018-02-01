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
            className='row' 
        ), 
        html.Div([
            html.H3('Survey Locations'
            ), 
            html.Div([
                dcc.RangeSlider(
                    id = 'year-range',
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
                html.H3('Table'
                ),
                html.H3('Table'
                ),        
            ],
                className='four columns',
                style={'margin-bottom': '10'}
            ),
        ],
            className='container',
            style={'margin': 10,
                   'padding': 0}
        ),
        html.Hr(),
        html.Div([
            html.H3('Survey Questions'
            ), 
            html.Div([
                dcc.Input(
                    id='search-word',
                    placeholder='search term',
                    type='text',
                    value=''
                ),
                html.P(),
                dt.DataTable(
                    id='search-word-questions',
                    rows=[{}], # initialise the rows
                    row_selectable=True,
                    filterable=False,
                    sortable=True,
                    selected_row_indices=[],)
            ],
                className='six columns')
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
                    id = 'year-range',
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
        Output('search-word-questions','rows'),
        [Input('search-word','value')]
        )
def update_questions(user_selection):
    df = socios.searchQuestions(user_selection, dtype='num')[['Question','QuestionaireID']]
    dff = df.loc[~df['QuestionaireID'].isin([1, 2, 4])]
    dff.columns=['Question','Questionaire']
    return dff.to_dict('records')

# Run app from script. Go to 127.0.0.1:8050 to view
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)