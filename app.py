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
import plotly.graph_objs as go
import pandas as pd
import os
import base64

import features.feature_socios as socios 
from support import dlrdb_dir

image_dir = os.path.join(dlrdb_dir,'img')
erc_logo = os.path.join(image_dir, 'erc_logo.jpg')
erc_encoded = base64.b64encode(open(erc_logo, 'rb').read())
sanedi_logo = os.path.join(image_dir, 'sanedi_logo.jpg')
sanedi_encoded = base64.b64encode(open(sanedi_logo, 'rb').read())

# Load datasets
site_geo = pd.read_csv('data/site_geo.csv')
site_geo.rename(columns={'GPSName':'Location'}, inplace=True)
ids = socios.loadID()
ids_summary = ids.groupby(['Year','LocName','GroupID'])['id'].count().reset_index()
ids_summary.rename(columns={'GroupID':'GroupId', 'id':'# households'}, inplace=True)
loc_summary = site_geo.merge(ids_summary[['GroupId','# households']], on='GroupId')

mapbox_access_token = 'pk.eyJ1Ijoic2FpbnRseXZpIiwiYSI6ImNqZHZpNXkzcjFwejkyeHBkNnp3NTkzYnQifQ.Rj_C-fOaZXZTVhTlliofMA'

app = dash.Dash()
app.config['suppress_callback_exceptions']=True

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
                html.H1('Data Explorer',
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
            html.H3('Survey Locations'
            ),
            html.Div([
                html.Div([
                    dcc.Graph(
                        animate=True,
                        style={'height': 450},
                        id='map'
                    ),
                    dcc.RangeSlider(
                        id = 'input-years',
                        marks={i: i for i in range(1994, 2015, 2)},
                        min=1994,
                        max=2014,
                        step=1,
                        included=True,
                        value= [1994, 2014],
                        updatemode='drag',
                        dots = True
                    )       
                ],
                    className='Eleven columns'
                ),
            ],
                className='columns',
                style={'margin-bottom':'10',
                       'margin-left':'0',
                       'width':'50%',
                       'float':'left'}
            ),
            html.Div([
                dt.DataTable(
                    id='output-location-list',
                    rows=[{}], # initialise the rows
                    row_selectable=False,
                    columns = ['Year','Province','Municipality','Location','# households'],
                    filterable=True,
                    sortable=True,
                    column_widths=100,
                    min_height = 450,
                    resizable=True,
                    selected_row_indices=[],),
                html.P('"# households" is the number of households surveyed at that location',
                       style={'font-style': 'italic'}
                       )     
            ],
                className='columns',
                style={'margin-bottom':'10',
                       'margin-top':'30',
                       'width':'45%',
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
                       'float':'left'}
            ),
            html.Div([
                html.H3('Survey Responses Overview'
                ),
                html.P('Select a question and set of locations to see the distribution of responses.'),
                html.Div([
                    dcc.RadioItems(
                        id = 'input-summarise',
                        options=[
                                {'label': 'mean', 'value': 'mean'},
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
                       'width':'50%',
                       'float':'right'}
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
#                    dots = True,
#                    included=True
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
        [Input('input-years','value')]
        )
def update_locations(input_years):
    dff = pd.DataFrame()
    for y in range(input_years[0], input_years[1]+1):
        df = loc_summary.loc[loc_summary.Year.astype(int) == y, ['Year','Province','Municipality','Location', '# households']]
        dff = dff.append(df)
    dff.reset_index(inplace=True, drop=True)
    return dff.to_dict('records')
            
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
        Output('map','figure'),        
        [Input('output-location-list','rows')]
        )

def update_map(input_locations):

    loc_list = pd.DataFrame(input_locations)
    keys=['Year','Location']
    i_loc = loc_list.set_index(keys).index
    i_site = loc_summary.set_index(keys).index
    
    georef = loc_summary[i_site.isin(i_loc)]
        
    traces = []
    for y in range(georef.Year.min(), georef.Year.max()+1):
        lat = georef.loc[(georef.Year==y), 'Lat']
        lon = georef.loc[(georef.Year==y), 'Long']
        text = georef.loc[(georef.Year==y), 'Location'] + ', ' + georef.loc[(georef.Year==y), 'Municipality']
#        marker_size = site_geo.loc[site_geo.Year==y,'# households']
        trace=go.Scattermapbox(
                name=y,
                lat=lat,
                lon=lon,
                mode='markers',
                marker=go.Marker(
                    size=12
                ),
                text=text,
            )
        traces.append(trace)
    figure=go.Figure(
        data=go.Data(traces),
        layout = go.Layout(
                autosize=True,
                hovermode='closest',
                mapbox=dict(
                    accesstoken=mapbox_access_token,
                    bearing=0,
                    center=dict(
                        lat=site_geo[site_geo.Location=='Ikgomotseng'] ['Lat'].unique()[0],
                        lon=site_geo[site_geo.Location=='Ikgomotseng']['Long'].unique()[0]
                    ),
                    pitch=0,
                    zoom=4.2,
                    style='light'
                ),
                margin = go.Margin(
                        l = 10,
                        r = 10,
                        t = 20,
                        b = 30
                ),
                showlegend=False
            )
    )
    return figure
   
@app.callback(
        Output('output-locqu-summary','rows'),
        [#Input('output-location-list','selected_row_indices'),
         Input('output-location-list','rows'),
         Input('output-search-word-questions','selected_row_indices'),
         Input('output-search-word-questions','rows'),
         Input('input-summarise', 'value')
         ]
        )
def update_locqu_summary(loc_rows, qu_selected_ix, qu_rows, summarise):

    locations = pd.DataFrame(loc_rows)
    years = locations.Year.unique()
    idselect = ids.loc[(ids.Year.isin(years)) & (ids.LocName.isin(locations.Location.unique())), ['id','Year','LocName']]
    idselect.reset_index(inplace=True, drop=True)
    
    searchterms = list(pd.DataFrame(qu_rows).loc[qu_selected_ix, 'Question'])
    
    d = pd.DataFrame()
    for y in years.astype(int):
        df = socios.buildFeatureFrame(searchterms, y)[0]
        d = d.append(df)

    locqu = idselect.merge(d, how='left',left_on='id',right_on='AnswerID')
    locqu.drop('id', axis=1, inplace=True)
    aggcols = locqu.columns[3::]
    group_locqu = locqu.groupby(by=['Year','LocName'])[aggcols].describe()
 
    
    if summarise == 'mean':
        locqu_summary = group_locqu.iloc[:, (group_locqu.columns.get_level_values(1)=='mean')|(group_locqu.columns.get_level_values(1)=='50%')|(group_locqu.columns.get_level_values(1)=='std')]
        locqu_summary.reset_index(inplace=True)
            
    elif summarise == 'count':
        locqu_summary = group_locqu.iloc[:, (group_locqu.columns.get_level_values(1)=='count')]
        locqu_summary.reset_index(inplace=True)
        
    return locqu_summary.to_dict('records')

# Run app from script. Go to 127.0.0.1:8050 to view
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)