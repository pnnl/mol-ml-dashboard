import json
import pandas as pd
from dash import Dash, dcc, html, Output, Input, State, no_update, ctx, dash_table, ALL
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem import Draw

import plotly.express as px
import base64
import dash_bio
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import pairwise
from sklearn.preprocessing import StandardScaler
import glob
from data_transformers import *
from utils import *

#load data from files
all_data, property_list = load_data()

#generate molecular structure images
generate_images(all_data)

#calculate descriptors and UMAP
all_data, mol_descriptors, umap_model = process_data(all_data)


color_palletes = {"Plotly3":px.colors.sequential.Plotly3,"Viridis":px.colors.sequential.Viridis,"Cividis":px.colors.sequential.Cividis,
                  "Plasma":px.colors.sequential.Plasma,"Turbo":px.colors.sequential.Turbo,"Blackbody":px.colors.sequential.Blackbody,"Bluered":px.colors.sequential.Bluered,
                  "Jet":px.colors.sequential.Jet,"Rainbow":px.colors.sequential.Rainbow,"RdBu":px.colors.sequential.RdBu,"RdPu":px.colors.sequential.RdPu,
                  "YlGnBu":px.colors.sequential.YlGnBu,"YlOrRd":px.colors.sequential.YlOrRd,"haline":px.colors.sequential.haline,"ice":px.colors.sequential.ice,
                  "deep":px.colors.sequential.deep,"dense":px.colors.sequential.dense,"Pinkyl":px.colors.sequential.Pinkyl,"Darkmint":px.colors.sequential.Darkmint,
                  "Bluyl":px.colors.sequential.Bluyl,"Teal":px.colors.sequential.Teal,"Tealgrn":px.colors.sequential.Tealgrn,"Purp":px.colors.sequential.Purp,
                  "Purpor":px.colors.sequential.Purpor,"Sunset":px.colors.sequential.Sunset,"Sunsetdark":px.colors.sequential.Sunsetdark,"Agsunset":px.colors.sequential.Agsunset,
                  "BrBG":px.colors.diverging.BrBG,"PRGn":px.colors.diverging.PRGn,"PiYG":px.colors.diverging.PiYG,"PuOr":px.colors.diverging.PuOr,
                  "RdBu":px.colors.diverging.RdBu,"RdYlBu":px.colors.diverging.RdYlBu,"RdYlGn":px.colors.diverging.RdYlGn,"Spectral":px.colors.diverging.Spectral,
                  "delta":px.colors.diverging.delta,"curl":px.colors.diverging.curl,"Temps":px.colors.diverging.Temps,"Picnic":px.colors.diverging.Picnic}

app = Dash(__name__)

all_data['select'] = ''
all_data['filter'] = np.nan
all_data['selected_error'] = 0
all_data['random_error'] = 0

fig_umap = px.scatter(all_data,x='dim1',y='dim2')
fig_close_up = px.scatter(None)
blank_fig = px.scatter(None)
blank_fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
blank_bar = px.bar(None)
blank_bar.update_layout(margin=dict(l=20, r=20, t=20, b=20))

fig_umap.update_traces(hoverinfo="none", hovertemplate=None)

initial_table_data = all_data.set_index('SMILES')[mol_descriptors].transpose().reset_index().iloc[2:,:]

initial_property_predictions = pd.DataFrame({'Molecule':['']}.update({prop:[''] for prop in property_list}),index=['Property Predictions'])
initial_molecule_table_data = pd.DataFrame(data={'formula':property_list,str(all_data.loc[1,'formula']):[np.nan]*len(property_list)})
initial_store_data = all_data.head()

error_tolerance = 0.0000000001
init_sample_size = len(all_data)

#load saved ML models
all_data, saved_models = load_models(all_data)

model_cols = [c for c in all_data.columns if '_pred' in c or '_truth' in c or '_error' in c]

initial_similarity_table_data = pd.DataFrame(columns=['id','SMILES','similarity'] + list(saved_models.keys()),index=range(20))


#Dashboard layout
app.layout = html.Div([
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Molecule Editor', children=[
            html.Div([
                html.Br(),
                html.Label('max sample size'),
                dcc.Slider(id='me-sample-size-umap', min=0, max=len(all_data), step=1,
                            marks={str(size): str(size) for size in range(0,len(all_data),1000)},
                            value=init_sample_size),
                            
                html.Label('Color pallete:  ',style={'display':'inline-block','width':'20vh','vertical-align':'middle','text-align':'right'}),
                dcc.Dropdown(list(color_palletes.keys()),'Plasma', id='me-dropdown-pallete',style={'display':'inline-block','width':'20vh','vertical-align':'middle'}),
                html.Label('Toggle hover data:   ',style={'display':'inline-block','width':'15vh','vertical-align':'middle','text-align':'right'}),
                dcc.RadioItems(['On','Off'],'On',id='me-toggle-hover',style={'display':'inline-block','width':'15vh','vertical-align':'middle'}),

                html.Label('Filter molecules:  ',style={'display':'inline-block','width':'20vh','vertical-align':'middle','text-align':'right'}),
                dcc.Dropdown([''] + mol_descriptors + model_cols,id='me-dropdown-property',placeholder='property',style={'display':'inline-block','width':'20vh','vertical-align':'middle'}),
                dcc.Dropdown(['','<','>','=','!='],'',id='me-dropdown-operator',placeholder='operator',style={'display':'inline-block','width':'20vh','vertical-align':'middle'}),
                dcc.Input(placeholder='value or item',id='me-input-value'),

                html.Label('Data:   ',style={'display':'inline-block','width':'10vh','vertical-align':'middle','text-align':'right'}),
                dcc.Dropdown(['all'] + property_list,'all',id='me-dropdown-data',placeholder='operator',style={'display':'inline-block','width':'20vh','vertical-align':'middle'}),
                html.Button('Filter', id='me-filter-button',style={'display':'inline-block','vertical-align':'middle'}),

                html.Br(),
                html.Br(),

            ],style={'width': '100%'}),
            
            html.Div([
                html.Div([
                    html.Br(),
                    html.Br(),
                    html.Label('Toggle NaN values:',style={'display':'inline-block','width':'15vh','vertical-align':'middle','text-align':'right'}),
                    dcc.RadioItems(['Show','Hide'],'Show',id='me-nan-scatter-umap',style={'display':'inline-block','width':'15vh','vertical-align':'middle'}),
                    dcc.Dropdown(['select'] + mol_descriptors + model_cols,'select',id='me-dropdown-color',style={'display':'inline-block','width':'60vh','vertical-align':'middle'}),
                    dcc.Graph(figure=fig_umap, id='me-scatter-umap', clear_on_unhover=True, style={'width': '90vh', 'height': '75vh'}),
                ], style={'float': 'left','display':'inline-block','height':'50vh'}),

                html.Div([
                    dash_bio.Jsme(style={'margin-top': '60px'},width = '40vh',height='40vh',id='me-jsme'),
                    dash_table.DataTable(data=initial_table_data.iloc[:,[0,2]].to_dict('records'), page_size=6, id='me-descriptor-table')
                ], style={'float':'left','display':'inline-block','height':'75vh'}),

                html.Div([
                    html.Br(),
                    dcc.Dropdown(['dim1','dim2'] + [f'{model_id}_pred' for model_id in saved_models.keys()],
                                 'dim1',id='me-close-up-x-axis',style={'display':'inline-block','width':'25vh'}),
                    dcc.Dropdown(['dim1','dim2'] + [f'{model_id}_pred' for model_id in saved_models.keys()],
                                 'dim2',id='me-close-up-y-axis',style={'display':'inline-block','width':'25vh'}),
                    dcc.Graph(figure=fig_close_up, id='me-scatter-close-up', clear_on_unhover=True, 
                                style={'width': '50vh', 'height': '55vh'}),
                    html.Div([
                        dash_table.DataTable(data=initial_property_predictions.to_dict('records'), page_size=10, id='me-property-predictions')
                    ],style={'width':'40vh','padding-right':'5vh','padding-left':'5vh'})
                ], style={'float': 'left', 'display': 'inline-block','height':'75vh'}),
                
                html.Div([
                    dcc.Dropdown([model_id for model_id in saved_models.keys()],
                                        list(saved_models.keys())[0],id='me-pred-prop-sel',style={'display':'inline-block','width':'25vh'}), 
                    dcc.Graph(figure=blank_fig, id=f'me-scatter', clear_on_unhover=True, 
                                style={'width': '50vh', 'height': '40vh'})
                ],
                     style={'width':'60vh','padding-right':'5vh','padding-left':'5vh'}),
            ],
                     style={'width': '100%','height':'75vh'}),
            
            

            dcc.Store(id='me-filtered-data',data=initial_store_data.to_json(orient="split")),
            dcc.Store(id='me-selected-data',data=initial_store_data.to_json(orient="split")),
            dcc.Store(id='me-shared-data-close-up'),
        ]),

        dcc.Tab(label='Data Exploration', children=[
            html.Div([
                html.Div([
                    html.Br(),
                    html.Label('Sample size'),
                    dcc.Slider(
                    id='de-sample-size-umap',
                    min=0, max=len(all_data), step=1,
                    marks={str(size): str(size) for size in range(0,len(all_data),1000)},
                    value=init_sample_size),

                    html.Label('Color pallete:  ',style={'display':'inline-block','width':'20vh','vertical-align':'middle','text-align':'right'}),
                    dcc.Dropdown(list(color_palletes.keys()),'Plasma', id='de-dropdown-pallete',style={'display':'inline-block','width':'20vh','vertical-align':'middle'}),
                    html.Label('Toggle hover data:   ',style={'display':'inline-block','width':'15vh','vertical-align':'middle','text-align':'right'}),
                    dcc.RadioItems(['On','Off'],'On',id='de-toggle-hover',style={'display':'inline-block','width':'15vh','vertical-align':'middle'}),

                    html.Label('Filter molecules:  ',style={'display':'inline-block','width':'20vh','vertical-align':'middle','text-align':'right'}),
                    dcc.Dropdown([''] + mol_descriptors + model_cols,id='de-dropdown-property',placeholder='property',style={'display':'inline-block','width':'20vh','vertical-align':'middle'}),
                    dcc.Dropdown(['','<','>','=','!='],'',id='de-dropdown-operator',placeholder='operator',style={'display':'inline-block','width':'20vh','vertical-align':'middle'}),
                    dcc.Input(placeholder='value or item',id='de-input-value'),

                    html.Label('Data:   ',style={'display':'inline-block','width':'10vh','vertical-align':'middle','text-align':'right'}),
                    dcc.Dropdown(['all'] + property_list + model_cols,'all',id='de-dropdown-data',placeholder='operator',style={'display':'inline-block','width':'20vh','vertical-align':'middle'}),
                    html.Button('Filter', id='de-filter-button',style={'display':'inline-block','vertical-align':'middle'}),

                    dcc.Store(id='de-filtered-data',data=all_data.copy().to_json(orient="split")),

                    html.Br(),
                    html.Br(),
                ]),
                html.Div([
                    html.Label('Toggle NaN values',style={'display':'inline-block','width':'15vh','vertical-align':'middle','text-align':'right'}),
                    dcc.RadioItems(['Show','Hide'],'Show',id='de-nan-scatter-umap',style={'display':'inline-block','width':'15vh','vertical-align':'middle'}),
                    html.Label('Color by property',style={'display':'inline-block','width':'15vh','vertical-align':'middle','text-align':'right'}),
                    dcc.Dropdown(['filter','select'] + mol_descriptors + model_cols,'filter',id="de-color-scatter-umap",style={'display':'inline-block','width':'50vh','vertical-align':'middle'}),
                    html.Label('x-axis variable',style={'display':'inline-block','width':'15vh','vertical-align':'middle','text-align':'right'}),
                    dcc.Dropdown(['dim1','dim2'] + mol_descriptors + model_cols,'dim1',id="de-x-axis-scatter-close-up",style={'display':'inline-block','width':'35vh','vertical-align':'middle'}),
                    html.Label('y-axis variable',style={'display':'inline-block','width':'15vh','vertical-align':'middle','text-align':'right'}),
                    dcc.Dropdown(['dim1','dim2'] + mol_descriptors + model_cols,'dim2',id="de-y-axis-scatter-close-up",style={'display':'inline-block','width':'35vh','vertical-align':'middle'}),
                ]),
                html.Div([
                    dcc.Graph(figure=fig_umap, id='de-scatter-umap', clear_on_unhover=True, 
                              style={'width': '100vh', 'height': '75vh'}),

                    dcc.Store(id="de-shared-data-umap",data=all_data.copy().to_json(orient="split"))
                ], style={'float': 'left', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(figure=fig_close_up, id='de-scatter-close-up', clear_on_unhover=True, 
                              style={'width': '100vh', 'height': '60vh'}),

                    html.Div([
                        html.Label('Statistics'),
                    ], id='de-close-up-statistics'),

                    dcc.Store(id="de-shared-data-close-up",data=all_data.copy().to_json(orient="split")),
                    dcc.Store(id="de-real-shared-data-close-up",data=all_data.copy().to_json(orient="split")),
                ], style={'width': '32%','float': 'left', 'display': 'inline-block'})
            ])
        ]),
        
        dcc.Tab(label='Molecule Recommendations', children=[
            html.Div([
                html.Br(),
                html.Label('max sample size'),
                dcc.Slider(id='re-sample-size-umap', min=0, max=len(all_data), step=1,
                            marks={str(size): str(size) for size in range(0,len(all_data),1000)},
                            value=init_sample_size),
                            
                html.Label('Color pallete:  ',style={'display':'inline-block','width':'20vh','vertical-align':'middle','text-align':'right'}),
                dcc.Dropdown(list(color_palletes.keys()),'Plasma', id='re-dropdown-pallete',style={'display':'inline-block','width':'20vh','vertical-align':'middle'}),
                html.Label('Toggle hover data:   ',style={'display':'inline-block','width':'15vh','vertical-align':'middle','text-align':'right'}),
                dcc.RadioItems(['On','Off'],'On',id='re-toggle-hover',style={'display':'inline-block','width':'15vh','vertical-align':'middle'}),

                html.Label('Filter molecules:  ',style={'display':'inline-block','width':'20vh','vertical-align':'middle','text-align':'right'}),
                dcc.Dropdown([''] + mol_descriptors + model_cols,id='re-dropdown-property',placeholder='property',style={'display':'inline-block','width':'20vh','vertical-align':'middle'}),
                dcc.Dropdown(['','<','>','=','!='],'',id='re-dropdown-operator',placeholder='operator',style={'display':'inline-block','width':'20vh','vertical-align':'middle'}),
                dcc.Input(placeholder='value or item',id='re-input-value'),

                html.Label('Data:   ',style={'display':'inline-block','width':'10vh','vertical-align':'middle','text-align':'right'}),
                dcc.Dropdown(['all'] + property_list,'all',id='re-dropdown-data',placeholder='operator',style={'display':'inline-block','width':'20vh','vertical-align':'middle'}),
                html.Button('Filter', id='re-filter-button',style={'display':'inline-block','vertical-align':'middle'}),

                html.Br(),
                html.Br(),

            ],style={'width': '100%'}),

            html.Div([
                html.Div([
                    html.Label('Toggle NaN values:',style={'display':'inline-block','width':'15vh','vertical-align':'middle','text-align':'right'}),
                    dcc.RadioItems(['Show','Hide'],'Show',id='re-nan-scatter-umap',style={'display':'inline-block','width':'15vh','vertical-align':'middle'}),
                    dcc.Dropdown(['select'] + mol_descriptors + model_cols,'select',id='re-dropdown-color',style={'display':'inline-block','width':'60vh','vertical-align':'middle'}),
                    dcc.Graph(figure=fig_umap, id='re-scatter-umap', clear_on_unhover=True, style={'width': '90vh', 'height': '75vh'}),
                ], style={'float': 'left','display':'inline-block','height':'75vh'}),

                html.Div([
                    html.Div([
                        html.Br(),
                        html.Br(),
                        html.Br(),
                        html.Br(),
                        html.Br(),
                        html.Label('Find similar molecules to: ' + str(all_data.loc[0,'SMILES']),style={'vertical-align':'middle'},id='re-find-similar-molecules-label'),
                        html.Br(),
                        html.Img(src=None,id='re-image-init-molecule',style={'float':'left','display':'inline-block','width':'20vh','height':'20vh'}),
                        html.Div([
                            dash_table.DataTable(data=initial_molecule_table_data.to_dict('records'), page_size=4, id='re-init-descriptor-table'),
                        ],style={'float':'left','display':'inline-block','width':'25vh'}),
                    ],style={'float':'left','display':'inline-block','width':'50vh'}),
                        
                    html.Div([
                        html.Label('Selected:',style={'vertical-align':'middle'},id='re-selected-molecule-label'),
                        html.Br(),
                        html.Img(src=None,id='re-image-similar-molecule',style={'float':'left','display':'inline-block','width':'20vh','height':'20vh'}),
                        html.Div([
                            dash_table.DataTable(data=initial_molecule_table_data.to_dict('records'), page_size=4, id='re-similar-descriptor-table'),
                        ],style={'float':'left','display':'inline-block','width':'25vh'})
                    ],style={'float':'left','display':'inline-block','width':'50vh'}),

                    html.Br(),
                    html.Br(),
                    html.Label("Note: all values are machine learning predictions, not ground truth"),
                ], style={'float':'left','display':'inline-block','width':'50vh'}),
                
                html.Div(get_div1(list(saved_models.keys()), initial_similarity_table_data), 
                         style={'float':'left','display':'inline-block','width':'60vh'})

            ],style={'width': '100%','height':'75vh'}),
            dcc.Store(id='re-filtered-data',data=initial_store_data.to_json(orient="split")),
            dcc.Store(id='re-selected-data',data=initial_store_data.to_json(orient="split")),
            dcc.Store(id='re-descriptor-table-data',data=initial_store_data.to_json(orient="split")),
        ]),

        dcc.Tab(label='Prediction Comparison', children=[
            html.Div([
                html.Div([
                    html.Br(),

                    html.Label('Color pallete:  ',style={'display':'inline-block','width':'20vh','vertical-align':'middle','text-align':'right'}),
                    dcc.Dropdown(list(color_palletes.keys()),'Plasma', id='pe-dropdown-pallete',style={'display':'inline-block','width':'20vh','vertical-align':'middle'}),
                    html.Label('Toggle hover data:   ',style={'display':'inline-block','width':'15vh','vertical-align':'middle','text-align':'right'}),
                    dcc.RadioItems(['On','Off'],'On',id='pe-toggle-hover',style={'display':'inline-block','width':'10vh','vertical-align':'middle'}),

                    html.Label('Property:   ',style={'display':'inline-block','width':'10vh','vertical-align':'middle','text-align':'right'}),
                    dcc.Dropdown(property_list,property_list[0],id='pe-dropdown-property',placeholder='operator',style={'display':'inline-block','width':'20vh','vertical-align':'middle'}),

                    html.Label('Color by property',style={'display':'inline-block','width':'15vh','vertical-align':'middle','text-align':'right'}),
                    dcc.Dropdown(['selected_error'] + mol_descriptors + model_cols,'selected_error',id="pe-color-scatter-umap",style={'display':'inline-block','width':'20vh','vertical-align':'middle'}),

                    html.Label('# of Random Samples',style={'display':'inline-block','width':'20vh','vertical-align':'middle','text-align':'right'}),
                    dcc.Input(placeholder='Number of Random Samples',id="pe-number-random-samples",style={'display':'inline-block','width':'10vh','vertical-align':'middle'}),

                    html.Label('Random sample',style={'display':'inline-block','width':'15vh','vertical-align':'middle','text-align':'right'}),
                    dcc.Dropdown([1,'Full Model'],1,id="pe-dropdown-scatter-random-sample",style={'display':'inline-block','width':'20vh','vertical-align':'middle'}),

                    html.Br(),
                    html.Br(),
                ]),
                html.Div([
                    dcc.Graph(figure=blank_fig, id='pe-scatter-umap', clear_on_unhover=True, 
                              style={'width': '80vh', 'height': '80vh'}),

                    dcc.Store(id="pe-shared-data-umap",data=all_data.copy().to_json(orient="split"))
                ], style={'float': 'left', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(figure=blank_fig, id='pe-scatter-close-up', clear_on_unhover=True, 
                              style={'width': '60vh', 'height': '45vh','display':'inline-block'}),
                    dcc.Graph(figure=blank_fig, id='pe-scatter-close-up-2', clear_on_unhover=True, 
                              style={'width': '60vh', 'height': '45vh','display':'inline-block'}),

                    html.Br(),

                    dcc.Graph(figure=blank_bar, id='pe-bar-mae', clear_on_unhover=True, 
                              style={'width': '60vh', 'height': '35vh','display':'inline-block'}),
                    dcc.Graph(figure=blank_bar, id='pe-bar-r-squared', clear_on_unhover=True, 
                              style={'width': '60vh', 'height': '35vh','display':'inline-block'}),

                    dcc.Store(id="pe-real-shared-data-close-up",data=pd.DataFrame().to_json(orient="split")),
                    
                ], style={'float': 'left', 'display': 'inline-block'})
            ])
        ]),
    ]),
    dcc.Tooltip(id="tooltip"),
])
    

@app.callback(
    Output("me-filtered-data", "data"),
    Output("me-input-value","value"),
    Input("me-filter-button","n_clicks"),
    Input('me-sample-size-umap','value'),
    State("me-dropdown-property","value"),
    State("me-dropdown-operator","value"),
    State("me-input-value","value"),
    State("me-dropdown-data","value"),
    prevent_initial_callback = True
)
def filter_data(button,sample_size,property,operator,value,data):
    new_data = all_data.copy()

    input_id = ctx.triggered_id
    
    if (input_id == 'me-filter-button'):

        if (property != '') and (operator != '') and (value != ''):
            try:
                if operator == '<':
                    new_data = new_data.loc[new_data[property] < float(value)]
                elif operator == '>':
                    new_data = new_data.loc[new_data[property] > float(value)]
                elif operator == '=':
                    new_data = new_data.loc[new_data[property] == float(value)]
                elif operator == '!=':
                    new_data = new_data.loc[new_data[property] != float(value)]
            except:
                value = 'invalid'
                new_data = all_data.copy().sample(sample_size)
        if len(new_data) > sample_size:
            new_data = new_data.sample(sample_size)
            
    else:
        new_data = all_data.copy().sample(sample_size)
    
    return new_data['id'].to_json(orient='split'), value
    

@app.callback(
    Output("me-scatter-umap", "figure"),
    Output("me-jsme","smiles"),
    Output("me-selected-data","data"),
    Input("me-scatter-umap","clickData"),
    Input("me-filtered-data","data"),
    Input("me-dropdown-color","value"),
    Input("me-dropdown-pallete","value"),
    Input("me-nan-scatter-umap","value"),
    prevent_initial_callback = True
)
def update_scatter_umap(clickData,filtered_data,color,pallete,nan):
    new_data_ids = pd.read_json(filtered_data)
    new_data_ids.index = new_data_ids['data']
    new_data = all_data[all_data.id.map(lambda id: id in new_data_ids['data'])]
    if nan == "Hide":
        new_data = new_data[new_data[color]==new_data[color]]

    if clickData is None:
        return px.scatter(new_data,x='dim1',y='dim2',color=color,color_continuous_scale=color_palletes[pallete]), None, None

    new_data['select'] = ''
    pt = clickData["points"][0]
    x = pt['x']
    new_data.loc[abs(new_data['dim1'] - x) < error_tolerance,'select'] = 'selected'
    selected_data = new_data[abs(new_data['dim1'] - x) < error_tolerance]
    data_row = selected_data.squeeze()
    smiles = data_row['SMILES']

    fig_umap = px.scatter(new_data,x='dim1',y='dim2',color=color,color_continuous_scale=color_palletes[pallete])
    
    return fig_umap, smiles, selected_data.to_json(orient='split')


@app.callback(
        Output("me-descriptor-table","data"),
        Output("me-shared-data-close-up","data"),
        Output("me-scatter-close-up","figure"),
        Output("me-property-predictions","data"),
        Output("me-scatter","figure"),
        Input("me-jsme","eventSmiles"),
        State("me-shared-data-close-up","data"),
        State("me-selected-data","data"),
        Input("me-close-up-x-axis","value"),
        Input("me-close-up-y-axis","value"),
        Input("me-dropdown-color","value"),
        Input("me-dropdown-pallete","value"),
        Input("me-pred-prop-sel","value"),
        prevent_initial_callback = True
)
def update_descriptor_table(smiles,close_up_data,selected_data,x_axis,y_axis,color,pallete,pred_prop):
    input_id = ctx.triggered_id
    if (input_id != 'me-jsme') and (close_up_data is not None):
        new_data = pd.read_json(close_up_data, orient='split')
        fig_close_up = px.scatter(new_data,x=x_axis,y=y_axis, 
                    color=new_data.index,hover_data=['SMILES','formula'],color_continuous_scale=color_palletes[pallete])

        fig = px.scatter(new_data,x='formula',y=f'{pred_prop}_pred',hover_data=['SMILES','formula'],color_continuous_scale=color_palletes[pallete],
                           color=color if len(new_data[color].dropna())==len(new_data) else None)
       
        return no_update, close_up_data, fig_close_up, no_update, fig
    elif ((smiles == '') or (smiles is None)):
        return no_update, close_up_data, no_update, no_update, no_update
    new_data = pd.read_json(selected_data, orient='split')

    try:
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        
        if (close_up_data is None) or (smiles==new_data.loc[new_data.index[0],'SMILES']):
            descriptor_table_data = new_data.loc[new_data.index[0],
                                                 'MaxAbsEStateIndex':'fr_urea'].transpose().reset_index().rename(columns={new_data.index[0]:new_data.loc[new_data.index[0],'formula'],
                                                                                                                          'index':'Descriptor'}).to_dict('records')
        else:
            new_data = pd.read_json(close_up_data, orient='split')
            molecule = Chem.MolFromSmiles(smiles)
            data = pd.DataFrame.from_dict(getMolDescriptors(molecule),orient='index')

            formula = str(CalcMolFormula(molecule))

            embedding = umap_model.transform(data.transpose().values)
            
            new_data = new_data.merge(data.transpose(),how='outer')
            index = len(new_data)-1
            new_data.loc[index,'SMILES'] = smiles
            new_data.loc[index,'formula'] = formula
            new_data.loc[index,'dim1'] = embedding[:,0]
            new_data.loc[index,'dim2'] = embedding[:,1]

            for model_id, model in saved_models.items():
                preds = model.predict(new_data.SMILES)
                
                new_data.loc[index,f'{model_id}_pred'] = round(preds[-1]*1000000)/1000000
           
            descriptor_table_data = data.reset_index().rename(columns={0:formula,'index':'Descriptor'}).to_dict('records')

        fig_close_up = px.scatter(new_data,x=x_axis,y=y_axis, 
                    color=new_data.index,hover_data=['SMILES','formula'],color_continuous_scale=color_palletes[pallete])
        
        output_df = {'Molecule':new_data['formula'].to_list()}
        for model_id, model in saved_models.items():
            if model_id == pred_prop:
                fig = px.scatter(new_data,x='formula',y=f'{model_id}_pred',hover_data=['SMILES','formula'],color_continuous_scale=color_palletes[pallete],
                           color=color if len(new_data[color].dropna())==len(new_data) else None)
            property = model_id.split('_')[0]
            output_df[property] = new_data[f'{model_id}_pred'].to_list()
      
        property_predictions = pd.DataFrame(output_df).to_dict('records')

        return descriptor_table_data, new_data.to_json(orient='split'), fig_close_up, property_predictions, fig
    except:
        import traceback
        traceback.print_exc()
        descriptor_table_data = initial_table_data.iloc[:,[0,2]].rename(columns={'index':'invalid smiles!','BrBr':'try again!'}).to_dict('records')
        print("invalid smiles!")
        return descriptor_table_data, close_up_data, no_update, no_update, no_update


@app.callback(
    Output("de-filtered-data", "data"),
    Output("de-input-value","value"),
    Input("de-filter-button","n_clicks"),
    Input('de-sample-size-umap','value'),
    State("de-dropdown-property","value"),
    State("de-dropdown-operator","value"),
    State("de-input-value","value"),
    State("de-dropdown-data","value"),
    prevent_initial_callback = True
)
def filter_data(button,sample_size,property,operator,value,data):
    new_data = all_data.copy()

    input_id = ctx.triggered_id
    
    if (input_id == 'de-filter-button'):

        if (property != '') and (operator != '') and (value != ''):
            try:
                if operator == '<':
                    new_data = new_data.loc[new_data[property] < float(value)]
                elif operator == '>':
                    new_data = new_data.loc[new_data[property] > float(value)]
                elif operator == '=':
                    new_data = new_data.loc[new_data[property] == float(value)]
                elif operator == '!=':
                    new_data = new_data.loc[new_data[property] != float(value)]
            except:
                value = 'invalid'
                new_data = all_data.copy().sample(sample_size)
        if len(new_data) > sample_size:
            new_data = new_data.sample(sample_size)
            
    else:
        new_data = all_data.copy().sample(sample_size)
    
    return new_data['id'].to_json(orient='split'), value


@app.callback(
    Output("de-scatter-umap", "figure"),
    Output("de-shared-data-umap",'data'),
    Input("de-filtered-data","data"),
    Input("de-dropdown-pallete","value"),
    Input('de-sample-size-umap','value'),
    Input("de-color-scatter-umap", 'value'),
    Input('de-nan-scatter-umap','value'),
)
def update_scatter_umap(filtered_data,pallete,sample_size,color_umap,nan):
    new_data_ids = pd.read_json(filtered_data).copy() 
    new_data_ids.index = new_data_ids['data']
    new_data = all_data.copy()
    new_data.loc[all_data.id.map(lambda id: id in new_data_ids['data']),'filter'] = 1
    new_data.index = range(len(new_data))
    
    if nan == 'Hide':
        no_nan_data = new_data[new_data[color_umap]==new_data[color_umap]]
        no_nan_data = no_nan_data[no_nan_data['filter']==1]
        fig_umap = px.scatter(no_nan_data,x='dim1',y='dim2',
            color='filter' if color_umap == 'filter' else color_umap,color_continuous_scale=color_palletes[pallete])
        return fig_umap, no_nan_data['id'].to_json(orient='split')
    else:
        if (color_umap == 'filter') or (len(new_data[new_data['filter']==1])==len(new_data)):
            fig_umap = px.scatter(new_data,x='dim1',y='dim2',
                color='filter' if color_umap == 'filter' else color_umap,color_continuous_scale=color_palletes[pallete])
        else:
            new_data = new_data[new_data['filter']==1]
            fig_umap = px.scatter(new_data,x='dim1',y='dim2',
                color='filter' if color_umap == 'filter' else color_umap,color_continuous_scale=color_palletes[pallete])
        return fig_umap, new_data['id'].to_json(orient='split')


@app.callback(
        Output("de-shared-data-close-up","data"),
        Input("de-scatter-umap","selectedData"),
        Input("de-shared-data-umap","data"),
)
def generate_scatter_close_up(selectedData,shared_data):
        
    new_data_ids = pd.read_json(shared_data)
    new_data_ids.index = new_data_ids['data']
    new_data = all_data[all_data.id.map(lambda id: id in new_data_ids['data'])]
    new_data.index = range(len(new_data))
    

    if selectedData is not None:
        pts = pd.DataFrame(selectedData["points"])
        
        if len(pts) != 0:
            new_data = new_data.loc[list(pts['pointIndex'].values)]
            

    return new_data['id'].to_json(orient='split')


@app.callback(
        Output("de-scatter-close-up","figure"),
        Output("de-close-up-statistics","children"),
        Output("de-real-shared-data-close-up","data"),
        Input("de-shared-data-close-up",'data'),
        Input("de-dropdown-pallete","value"),
        Input("de-x-axis-scatter-close-up",'value'),
        Input("de-y-axis-scatter-close-up",'value'),
        Input("de-color-scatter-umap",'value'),
        Input("de-filtered-data","data")
)
def update_scatter_close_up(selected_data,pallete,x_axis,y_axis,color_umap,filtered_data):
    if (x_axis is None) or (y_axis is None):
        return no_update, no_update, selected_data


    if color_umap == 'select':
        filtered_data_ids = pd.read_json(filtered_data)
        filtered_data_ids.index = filtered_data_ids['data']
        new_data = all_data[all_data.id.map(lambda id: id in filtered_data_ids['data'])]    
        new_data['select'] = np.nan

        #get rid of nan
        stat_data = new_data[new_data[x_axis]==new_data[x_axis]].head(5)
        stat_data = stat_data[stat_data[y_axis]==stat_data[y_axis]]

        new_data_ids = pd.read_json(selected_data).copy() 
        new_data_ids.index = new_data_ids['data']
        stat_data.loc[all_data.id.map(lambda id: id in new_data_ids['data']),'select'] = 1
        stat_data.index = range(len(stat_data))
        fig_close_up = px.scatter(stat_data,x=x_axis,y=y_axis, 
                                    color='select',hover_data=['SMILES','formula'],color_continuous_scale=color_palletes[pallete])
    else:
        new_data_ids = pd.read_json(selected_data).copy() 
        new_data_ids.index = new_data_ids['data']
        new_data = all_data[all_data.id.map(lambda id: id in new_data_ids['data'])]

        #get rid of nan
        stat_data = new_data[new_data[x_axis]==new_data[x_axis]]
        stat_data = stat_data[stat_data[y_axis]==stat_data[y_axis]]
        if color_umap == 'filter':
            filtered_data_ids = pd.read_json(filtered_data)
            filtered_data_ids.index = filtered_data_ids['data']
            stat_data.loc[all_data.id.map(lambda id: id in filtered_data_ids['data']),'filter'] = 1
            
            stat_data.index = range(len(stat_data))
            fig_close_up = px.scatter(stat_data,x=x_axis,y=y_axis, 
                                    color='filter',hover_data=['SMILES','formula'],color_continuous_scale=color_palletes[pallete])
        else:
            stat_data.index = range(len(stat_data))
            fig_close_up = px.scatter(stat_data,x=x_axis,y=y_axis, 
                                    color=color_umap,hover_data=['SMILES','formula'],color_continuous_scale=color_palletes[pallete])
        
    if (len(stat_data) != 0) and is_numeric_dtype(stat_data[x_axis]) and is_numeric_dtype(stat_data[y_axis]):
        children = [
            html.Label('*NaN values in the x and y axis are not graphed'),
            html.Br(),
            html.Label('*When coloring by discrete values, hover data may be inaccurate'),
            html.Br(),
            html.Br(),
            html.Label('Statistics'),
            html.Br(),
            html.Label('Mean Squared Error: ' + str(mean_squared_error(stat_data[x_axis],stat_data[y_axis]))),
            html.Br(),
            html.Label('R-Squared: ' + str(r2_score(stat_data[x_axis],stat_data[y_axis]))),
        ]
    else:
        children = [
            html.Label('*NaN values in the x and y axis are not graphed'),
            html.Br(),
            html.Br(),
            html.Label('Statistics'),
            html.Br(),
            html.Label('Mean Squared Error: N/A'),
            html.Br(),
            html.Label('R-Squared: N/A'),
        ]
        
    return fig_close_up, children, stat_data['id'].to_json(orient='split')

@app.callback(
    Output("re-filtered-data", "data"),
    Output("re-input-value","value"),
    Input("re-filter-button","n_clicks"),
    Input('re-sample-size-umap','value'),
    State("re-dropdown-property","value"),
    State("re-dropdown-operator","value"),
    State("re-input-value","value"),
    State("re-dropdown-data","value"),
    prevent_initial_callback = True
)
def filter_data(button,sample_size,property,operator,value,data):
    new_data = all_data.copy()

    input_id = ctx.triggered_id
    
    if (input_id == 're-filter-button'):
        
        if (property != '') and (operator != '') and (value != ''):
            try:
                if operator == '<':
                    new_data = new_data.loc[new_data[property] < float(value)]
                elif operator == '>':
                    new_data = new_data.loc[new_data[property] > float(value)]
                elif operator == '=':
                    new_data = new_data.loc[new_data[property] == float(value)]
                elif operator == '!=':
                    new_data = new_data.loc[new_data[property] != float(value)]
            except:
                value = 'invalid'
                new_data = all_data.copy().sample(sample_size)
        if len(new_data) > sample_size:
            new_data = new_data.sample(sample_size)
            
    else:
        new_data = all_data.copy().sample(sample_size)
    
    return new_data['id'].to_json(orient='split'), value
    


@app.callback(
    Output("re-scatter-umap", "figure"),
    Output("re-selected-data","data"),
    Output("re-image-init-molecule","src"),
    Output("re-init-descriptor-table","data"),
    Output("re-similarity-descriptor-table","data"),
    Output("re-find-similar-molecules-label","children"),
    Output("re-descriptor-table-data",'data'),
    Output("re-similarity-descriptor-table","tooltip_data"),
    Input("re-scatter-umap","clickData"),
    Input("re-filtered-data","data"),
    Input("re-dropdown-color","value"),
    Input("re-dropdown-pallete","value"),
    Input("re-nan-scatter-umap","value"),
    Input({"type": "re-dropdown", "index": ALL}, "value"),
    Input({"type": "re-input", "index": ALL}, "value"),
    prevent_initial_callback = True
)
def update_scatter_umap(clickData,filtered_data,color,pallete,nan,operators,inputs):

    new_data_ids = pd.read_json(filtered_data)

    new_data_ids.index = new_data_ids['data']
    new_data = all_data[all_data.id.map(lambda id: id in new_data_ids['data'])]
    
    
    if nan == "Hide":
        new_data = new_data[new_data[color]==new_data[color]]

    if (clickData is None):
        return px.scatter(new_data,x='dim1',y='dim2',color=color,color_continuous_scale=color_palletes[pallete]), None, None, None, None, 'Find similar molecules to: ', None, None

    new_data['select'] = 0
    pt = clickData["points"][0]['pointIndex']
    
    new_data.loc[pt,'select'] = 1
    selected_data = new_data.loc[pt]
    length = len(selected_data)
    data_row = selected_data.squeeze()
    fig_umap = px.scatter(new_data,x='dim1',y='dim2',color=color,color_continuous_scale=color_palletes[pallete])


    if length == 0:
        return fig_umap, selected_data.to_json(orient='split'), None, None, None, 'Find similar molecules to: ', None, None

    src = 'data:image/png;base64,{}'.format(base64.b64encode(open(r'image_assets/molecule_structures//' + str(data_row['id']) + '-' + str(data_row['formula']) + '.png','rb').read()).decode())
    
    init_descriptor_table_data = pd.DataFrame(data={'formula': list(saved_models.keys()),
                                                    str(data_row['formula']):[str(data_row[f'{model_id}_pred']) for model_id in saved_models.keys()]}).to_dict('records')

    molecule = str(data_row['SMILES'])

    X = all_data.fillna(0).set_index('SMILES').loc[molecule,'MaxAbsEStateIndex':'fr_urea']
    Y = all_data.fillna(0).loc[:,'MaxAbsEStateIndex':'fr_urea']

    similarities = pd.DataFrame(pairwise.cosine_similarity(X.values.reshape(1, -1),Y))

    molecules = all_data.SMILES.to_list()
    similarities = similarities.rename(columns={i:molecules[i] for i in range(len(similarities.columns))})
    similarities = similarities.rename(index={0:'similarity'}).transpose()
    similarities = similarities.reset_index().rename(columns={'index':'SMILES'})
    
    new_data = all_data.loc[:,['SMILES','id'] + [f'{model_id}_pred' for model_id in saved_models.keys()]].merge(similarities,on='SMILES',how='outer')


    for m, model_id in enumerate(saved_models.keys()):
        if operators[m] == '<':
            new_data = new_data.loc[new_data[f'{model_id}_pred'] < float(inputs[m])]
        elif operators[m] == '>':
            new_data = new_data.loc[new_data[f'{model_id}_pred'] > float(inputs[m])]

    new_data = new_data.sort_values(by='similarity',ascending=False)[:256]

    similarity_descriptor_table_data = pd.DataFrame(data={'id':new_data.id.to_list(),
                                                          'SMILES':new_data.SMILES.to_list(),
                                                          'similarity':new_data.similarity.to_list()})
    
    for model_id in list(saved_models.keys()):
        similarity_descriptor_table_data[model_id] = new_data[f'{model_id}_pred'].astype(float).round(decimals=3).to_list()
    
    
    tooltip_data=[{column: {'value': str(value), 'type': 'markdown'}
                            for column, value in row.items()
                        } for _, row in similarity_descriptor_table_data.iterrows()]
    

    return fig_umap, selected_data.to_json(orient='split'), src, init_descriptor_table_data, similarity_descriptor_table_data.to_dict('records'), 'Find similar molecules to: ' + str(data_row['SMILES']), new_data['id'].to_json(orient='split'), tooltip_data


@app.callback(
    Output("re-image-similar-molecule","src"),
    Output("re-similar-descriptor-table","data"),
    Output("re-selected-molecule-label","children"),
    Input("re-similarity-descriptor-table","active_cell"),
    Input("re-descriptor-table-data","data"),
    prevent_initial_callback = True
)
def select_similar_molecule(active_cell,table_data):
    if not active_cell:
        return None, no_update, 'Selected: '
    try:
        active_row_id = active_cell['row_id']

        new_data_ids = pd.read_json(table_data).copy() 
        new_data_ids.index = new_data_ids['data']
        new_data = all_data[all_data.id.map(lambda id: id in new_data_ids['data'])]


        new_data = new_data.set_index('id')

        data_row = pd.DataFrame(new_data.loc[active_row_id,:]).transpose().reset_index().rename(columns={'index':'id'}).squeeze()


        src = 'data:image/png;base64,{}'.format(base64.b64encode(open(r'image_assets/molecule_structures//' + str(data_row['id']) + '-' + str(data_row['formula']) + '.png','rb').read()).decode())


        similar_descriptor_table_data = pd.DataFrame(data={'formula': list(saved_models.keys()),
                                                    str(data_row['formula']):[str(data_row[f'{model_id}_pred']) for model_id in saved_models.keys()]}).to_dict('records')


        return src, similar_descriptor_table_data, 'Selected: ' + str(data_row['SMILES'])
    except Exception as e: 
        return None, no_update, 'Selected: '
    
@app.callback(
    Output("pe-scatter-umap", "figure"),
    Input("pe-dropdown-property","value"),
    Input("pe-dropdown-pallete","value"),
    Input("pe-color-scatter-umap", 'value'),
    Input("pe-real-shared-data-close-up","data"),
)
def update_scatter_umap(property,pallete,color_umap,error_data):
    property_truth = str(property + '_truth')

    new_data = all_data[all_data[property_truth]==all_data[property_truth]]

    if (color_umap == 'selected_error') or (color_umap == 'random_error') and (error_data is not None):
        
        error_data = pd.read_json(error_data,orient='split')
        if not error_data.empty:
            
            if color_umap in new_data.columns:
                new_data = new_data.drop(columns=[color_umap])
                
            new_data = new_data.merge(error_data[['SMILES',color_umap]],on='SMILES',how='left')
                        

    fig_umap = px.scatter(new_data,x='dim1',y='dim2',color=color_umap,color_continuous_scale=color_palletes[pallete])
    fig_umap.update_layout(margin=dict(l=20, r=20, t=20, b=20))

    return fig_umap

@app.callback(
        Output("pe-real-shared-data-close-up","data"),
        Output("pe-dropdown-scatter-random-sample","options"),
        Input("pe-scatter-umap","selectedData"),
        State("pe-dropdown-property","value"),
        State("pe-real-shared-data-close-up","data"),
        State("pe-number-random-samples","value"),
        State("pe-dropdown-scatter-random-sample","options"),
        prevent_initial_callback = True
)
def detect_selected_data(selectedData,property,output_sample_data,n_samples,current_options):


    if selectedData is not None:
        pts = selectedData["points"][:]

        x = [round(point['x']*1000000000000)/1000000000000 for point in pts]
        y = [round(point['y']*1000000000000)/1000000000000 for point in pts]
        if len(x) != 0:
            property_truth = str(property + '_truth')
            new_data = all_data[all_data[property_truth]==all_data[property_truth]].copy()
            new_data = new_data.fillna(0)

            train_data = new_data[new_data.dim1.map(lambda dim1: round(dim1*1000000000000)/1000000000000 in x)]
            train_data = new_data[new_data.dim2.map(lambda dim2: round(dim2*1000000000000)/1000000000000 in y)]
            test_data = train_data.merge(new_data,how='outer',indicator=True)
            test_data = test_data.loc[test_data['_merge']=='right_only']
            
            RFR_selected_sample = RFR_to_dataframe(new_data,train_data,test_data,property_truth,'selected_RFR_' + property + '_pred','selected_error')


            if (n_samples is not None) and (n_samples != '') and (int(n_samples) > 1):
                RFR_combined = RFR_selected_sample.drop(property_truth,axis=1)
                for i in range(1,int(n_samples)+1):
                    train_data = new_data.copy().sample(len(train_data))
                    test_data = train_data.merge(new_data,how='outer',indicator=True)
                    test_data = test_data.loc[test_data['_merge']=='right_only']
                    RFR_random_sample = RFR_to_dataframe(new_data,train_data,test_data,property_truth,'random_RFR_' + property + '_pred_' + str(i),'random_error_' + str(i))

                    RFR_combined = RFR_combined.merge(RFR_random_sample.drop(property_truth,axis=1),how='outer',on='SMILES')
                    
                scatter_random_sample_options = list(range(1,int(n_samples) + 1)) + ['Full Model']
            else:
                train_data = new_data.copy().sample(len(train_data))
                test_data = train_data.merge(new_data,how='outer',indicator=True)
                test_data = test_data.loc[test_data['_merge']=='right_only']
                RFR_random_sample = RFR_to_dataframe(new_data,train_data,test_data,property_truth,'random_RFR_' + property + '_pred_1','random_error_1')

                RFR_combined = RFR_selected_sample.drop(property_truth,axis=1).merge(RFR_random_sample.drop(property_truth,axis=1),how='outer',on='SMILES')
                
           
                scatter_random_sample_options = [1,'Full Model']

       
            if 'selected_error' in new_data.iloc[:,:-2].columns:
                new_data = new_data.iloc[:,:-2].drop(columns=['selected_error']).merge(RFR_combined,how='inner',on='SMILES')
            else:
                new_data = new_data.iloc[:,:-2].merge(RFR_combined,how='inner',on='SMILES')
            new_data.index = new_data.id - 1


            return new_data.to_json(orient='split'),scatter_random_sample_options

    return output_sample_data,current_options

@app.callback(
        Output("pe-scatter-close-up-2","figure"),
        Input("pe-real-shared-data-close-up","data"),
        Input("pe-dropdown-pallete","value"),
        State("pe-dropdown-property","value"),
        Input("pe-dropdown-scatter-random-sample","value"),
        prevent_initial_callback = True
)
def update_scatter_selected_and_bars(sample_data,pallete,property,scatter_random_sample_number):
    if (sample_data is None):
        return no_update

    sample_data = pd.read_json(sample_data,orient='split').copy()

    if (sample_data.empty):
        return no_update
    
    if scatter_random_sample_number == 'Full Model':

        for k in saved_models.keys():
            if property in k:
                x_axis = str(f'{k}_pred')
                color = str(f'{k}_error')
        y_axis = str(property+'_truth')

        fig_random_sample = px.scatter(sample_data,x=str(x_axis),y=y_axis,color=color,color_continuous_scale=color_palletes[pallete])
        maximum = min([max(sample_data[x_axis]),max(sample_data[y_axis])])
        minimum = max([min(sample_data[x_axis]),min(sample_data[y_axis])])
        fig_random_sample.add_shape(type="line",x0=minimum,y0=minimum,x1=maximum,y1=maximum)
        fig_random_sample.update_layout(margin=dict(l=20, r=20, t=20, b=20))

        return fig_random_sample
    
    random_x_axis = str('random_RFR_'+property+'_pred')
    random_y_axis = str(property+'_truth')
    random_color = str('random_error')

    random_data = sample_data[sample_data[str(random_x_axis + '_' + str(scatter_random_sample_number))]==sample_data[str(random_x_axis + '_' + str(scatter_random_sample_number))]]

    fig_random_sample = px.scatter(random_data,x=str(random_x_axis + '_' + str(scatter_random_sample_number)),y=random_y_axis,color=str(random_color + '_' + str(scatter_random_sample_number)),color_continuous_scale=color_palletes[pallete])
    maximum = min([max(random_data[str(random_x_axis + '_' + str(scatter_random_sample_number))]),max(random_data[random_y_axis])])
    minimum = max([min(random_data[str(random_x_axis + '_' + str(scatter_random_sample_number))]),min(random_data[random_y_axis])])
    fig_random_sample.add_shape(type="line",x0=minimum,y0=minimum,x1=maximum,y1=maximum)
    fig_random_sample.update_layout(margin=dict(l=20, r=20, t=20, b=20))

    return fig_random_sample

@app.callback(
        Output("pe-scatter-close-up","figure"),
        Output("pe-bar-mae","figure"),
        Output("pe-bar-r-squared","figure"),
        Input("pe-real-shared-data-close-up","data"),
        Input("pe-dropdown-pallete","value"),
        State("pe-dropdown-property","value"),
        State("pe-number-random-samples","value"),
        prevent_initial_callback = True
)
def update_scatter_selected_and_bars(sample_data,pallete,property,n_samples):
    if (sample_data is None):
        return no_update, no_update, no_update

    sample_data = pd.read_json(sample_data,orient='split').copy()

    if (sample_data.empty):
        return no_update, no_update, no_update
    
    models = saved_models.keys()
    for model in models:
        if property in model:
            pred_col = f'{model}_pred'

    selected_x_axis = str('selected_RFR_'+property+'_pred')
    selected_y_axis = str(property+'_truth')
    selected_color = str('selected_error')

    selected_data = sample_data[sample_data[selected_x_axis]==sample_data[selected_x_axis]]

    fig_selected_sample = px.scatter(selected_data,x=selected_x_axis,y=selected_y_axis,color=selected_color,color_continuous_scale=color_palletes[pallete])
    maximum = min([max(selected_data[selected_x_axis]),max(selected_data[selected_y_axis])])
    minimum = max([min(selected_data[selected_x_axis]),min(selected_data[selected_y_axis])])
    fig_selected_sample.add_shape(type="line",x0=minimum,y0=minimum,x1=maximum,y1=maximum)
    fig_selected_sample.update_layout(margin=dict(l=20, r=20, t=20, b=20))

    random_x_axis = str('random_RFR_'+property+'_pred')
    random_y_axis = str(property+'_truth')
    random_color = str('random_error')

    if (n_samples is not None) and (n_samples != '') and (int(n_samples) > 1):
        mae_list = [mean_squared_error(selected_data[selected_x_axis],selected_data[selected_y_axis])] + [mean_squared_error(sample_data.loc[sample_data[str(random_x_axis + '_' + str(i))]==sample_data[str(random_x_axis + '_' + str(i))],str(random_x_axis + '_' + str(i))],
                                                                                                                             sample_data.loc[sample_data[str(random_x_axis + '_' + str(i))]==sample_data[str(random_x_axis + '_' + str(i))],random_y_axis]) for i in range(1,int(n_samples) + 1)] + [mean_squared_error(sample_data['gnn_solubility_pred' if property == 'solubility' else str('RFR_'+property+'_pred')],sample_data[str(property+'_truth')])]
        r2_list = [r2_score(selected_data[selected_x_axis],selected_data[selected_y_axis])] + [r2_score(sample_data.loc[sample_data[str(random_x_axis + '_' + str(i))]==sample_data[str(random_x_axis + '_' + str(i))],str(random_x_axis + '_' + str(i))],
                                                                                                        sample_data.loc[sample_data[str(random_x_axis + '_' + str(i))]==sample_data[str(random_x_axis + '_' + str(i))],random_y_axis]) for i in range(1,int(n_samples) + 1)] + [r2_score(sample_data['gnn_solubility_pred' if property == 'solubility' else str('RFR_'+property+'_pred')],sample_data[str(property+'_truth')])]
        idx_list = ['Selected Sample'] + ['Random Sample ' + str(i) for i in range(1,int(n_samples) + 1)] + ['Full Model']

        stats = pd.DataFrame({'Mean Squared Error':mae_list,
                            'R-Squared':r2_list},
                            index=idx_list)
    else:
        random_data = sample_data[sample_data[str(random_x_axis + '_' + str(1))]==sample_data[str(random_x_axis + '_' + str(1))]]
        
        stats = pd.DataFrame({'Mean Squared Error':[mean_squared_error(selected_data[selected_x_axis],selected_data[selected_y_axis]),
                                                    mean_squared_error(random_data[str(random_x_axis + '_1')],random_data[random_y_axis]),
                                                    mean_squared_error(sample_data[pred_col],sample_data[str(property+'_truth')])],
                        'R-Squared':[r2_score(selected_data[selected_x_axis],selected_data[selected_y_axis]),
                                     r2_score(random_data[str(random_x_axis + '_1')],random_data[random_y_axis]),
                                     r2_score(sample_data[pred_col],sample_data[str(property+'_truth')])]},
                        index=['Selected Sample','Random Sample','Full Model'])

    fig_bar_mae = px.bar(stats,x=stats.index,y='Mean Squared Error')
    fig_bar_mae.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    fig_bar_r_squared = px.bar(stats,x=stats.index,y='R-Squared')
    fig_bar_r_squared.update_layout(margin=dict(l=20, r=20, t=20, b=20))

    return fig_selected_sample, fig_bar_mae, fig_bar_r_squared

@app.callback(
    Output("tooltip","show"),
    Output("tooltip","bbox"),
    Output("tooltip","children"),
    Input("me-scatter-umap","hoverData"),
    Input("de-scatter-umap","hoverData"),
    Input("de-scatter-close-up","hoverData"),
    Input("re-scatter-umap","hoverData"),
    State("de-real-shared-data-close-up","data"),
    State("me-toggle-hover","value"),
    State("de-toggle-hover","value"),
    State("re-toggle-hover","value"),
    Input("pe-scatter-umap","hoverData"),
    Input("pe-scatter-close-up","hoverData"),
    Input("pe-scatter-close-up-2","hoverData"),
    State("pe-real-shared-data-close-up","data"),
    State("pe-toggle-hover","value"),
    State("pe-dropdown-property","value"),
    State("pe-dropdown-scatter-random-sample","value"),
    prevent_initial_callback = True
)
def display_hover_data(hover_me_umap, hover_de_umap, hover_de_close_up, hover_re_umap, shared_data_close_up, me_hover, de_hover, re_hover,hover_pe_umap, hover_pe_close_up, hover_pe_close_up_2, sample_data, pe_hover,property,scatter_random_sample_number):
    if (hover_me_umap is None) and (hover_de_close_up is None) and (hover_de_umap is None) and (hover_re_umap is None) and (hover_pe_close_up is None) and (hover_pe_umap is None) and (hover_pe_close_up_2 is None):
        return False, None, None
    
    input_id = ctx.triggered_id

    if (input_id == 'de-scatter-close-up') and (de_hover == 'On'):
        if (de_hover == 'On'):
            pt = hover_de_close_up["points"][0]
            bbox = pt["bbox"]
            num = pt['pointNumber']

            new_data_ids = pd.read_json(shared_data_close_up)
            new_data_ids.index = new_data_ids['data']
            new_data = all_data[all_data.id.map(lambda id: id in new_data_ids['data'])]
            
            data_row = new_data.iloc[num]
    elif (input_id == 'pe-scatter-close-up') and (pe_hover == 'On') and (sample_data is not None):
        pt = hover_pe_close_up["points"][0]

        sample_data = pd.read_json(sample_data,orient='split')
        pred_name = str('selected_RFR_'+property+'_pred')
        truth_name = str(property+'_truth')
        sample_data = sample_data[sample_data[pred_name]==sample_data[pred_name]]

        bbox = pt["bbox"]
        x = pt['x']
        y = pt['y']
        data_row = sample_data[abs(sample_data.loc[:,pred_name] - x) < error_tolerance]
        data_row = data_row[abs(sample_data.loc[:,truth_name] - y) < error_tolerance].squeeze()
    elif (input_id == 'pe-scatter-close-up-2') and (pe_hover == 'On') and (sample_data is not None):
        pt = hover_pe_close_up_2["points"][0]

        sample_data = pd.read_json(sample_data,orient='split')
        
        if scatter_random_sample_number == 'Full Model':
            
            for k in saved_models.keys():
                if property in k:
                    pred_name = str(f'{k}_pred')
            
        else:
            pred_name = str('random_RFR_'+property+'_pred_' + str(scatter_random_sample_number))
        
        truth_name = str(property+'_truth')
        sample_data = sample_data[sample_data[pred_name]==sample_data[pred_name]]

        bbox = pt["bbox"]
        x = pt['x']
        y = pt['y']
        data_row = sample_data[abs(sample_data.loc[:,pred_name] - x) < error_tolerance]
        data_row = data_row[abs(sample_data.loc[:,truth_name] - y) < error_tolerance].squeeze()
    else:
        if (input_id == 'me-scatter-umap') and (me_hover == 'On'):
            pt = hover_me_umap["points"][0]
        elif (input_id == 'de-scatter-umap') and (de_hover == 'On'):
            pt = hover_de_umap["points"][0]
        elif (input_id == 're-scatter-umap') and (re_hover == 'On'):
            pt = hover_re_umap["points"][0]
        elif (input_id == 'pe-scatter-umap') and (pe_hover == 'On'):
            pt = hover_pe_umap["points"][0]
        else:
            return False, None, None

        new_data = all_data.copy()


        bbox = pt["bbox"]
        x = pt['x']
        y = pt['y']

        data_row = new_data[abs(new_data['dim1'] - x) < error_tolerance]
        data_row = data_row[abs(new_data['dim2'] - y) < error_tolerance].squeeze()
            
    
    name = data_row['formula']
    molecule_id = data_row['id']
    desc = data_row['SMILES']

    if molecule_id == molecule_id:
        img_src = r'image_assets/molecule_structures//' + str(data_row['id']) + '-' + data_row['formula'] + '.png'
    else:
        img_src = None

    encoded_image = base64.b64encode(open(img_src,'rb').read())

    children = [
        html.Div([
            html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode())),
            html.H2(f"{name}",style={"color":"darkblue","overflow-wrap":"break-word"}),
            html.P(f"{molecule_id}",style={"overflow-wrap":"break-word"}),
            html.P(f"{desc}",style={"overflow-wrap":"break-word"}),
        ], style={'width':'300px','white-space':'normal'})
    ]

    return True, bbox, children

print('Running server...')
app.run_server(debug=True,use_reloader=True)  # Turn off reloader if inside Jupyter
