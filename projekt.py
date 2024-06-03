# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 14:01:23 2024

@author: marek & radek
"""
import pandas as pd
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import linregress

#link do danych
#https://www.kaggle.com/datasets/ahmettyilmazz/fuel-consumption
df = pd.read_csv('Fuel_Consumption_2000-2022.csv')
app = Dash(__name__)


#Zamiana wielkości liter
df['MAKE'] = df['MAKE'].str.upper()

#etykietowanie: klas auta, typów paliwa, modeli aut
df['VEHICLE CLASS LABEL'], _ = pd.factorize(df['VEHICLE CLASS'])
df['FUEL LABEL'], _ = pd.factorize(df['FUEL'])
df_make = df['MAKE'].sort_values()

df_make = df_make.drop_duplicates()
df_make = df_make.reset_index(drop=True)
df_make = pd.DataFrame(df_make, columns=['MAKE'])
df_make['MAKE LABEL'] = df_make.index
df_make = df_make[['MAKE LABEL', 'MAKE']]


df_unique_class = df[['VEHICLE CLASS LABEL', 'VEHICLE CLASS']].drop_duplicates()
df_unique_fuel = df[['FUEL LABEL', 'FUEL']].drop_duplicates()
df_car_model = df[['MAKE', 'MODEL']].drop_duplicates()
df_car_model['LABEL'] = df_car_model.index
df_car_model = df_car_model[['LABEL', 'MAKE', 'MODEL']]
df_make = df_make.drop_duplicates()

# Zastosowanie nowej kolejności kolumn
new_order = ['YEAR',
             'MAKE',
             'MODEL',
             'VEHICLE CLASS',
             'ENGINE SIZE',
             'CYLINDERS',
             'TRANSMISSION',
             'FUEL',
             'FUEL CONSUMPTION',
             'HWY (L/100 km)',
             'COMB (L/100 km)',
             'COMB (mpg)',
             'EMISSIONS']


df = df[new_order]


#przygotowane na podstawie: https://dash.plotly.com/tutorial
# App layout
app.layout = html.Div([
    html.Div(children='FUEL CONSUMPTION ANALAYSIS', style={'textAlign': 'center', 'color': 'blue', 'fontSize': '26px', 'fontWeight': 'bold', 'margin': '20px 0'}),
    #histogram
    html.Div(children='Bar plot', style={'textAlign': 'center', 'fontSize': '24px', 'fontWeight': 'bold', 'margin': '20px 0'}),
    html.Div([
        html.Div([
            html.Label('X:'),
            dcc.Dropdown(
                options=[col for col in df.columns],
                value='MAKE',
                id='group-by-dropdown'
            ),
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),

        html.Div([
            html.Label('Y'),
            dcc.Dropdown(
                options=[col for col in df.columns],
                value='FUEL CONSUMPTION',
                id='controls-and-dropdown-item'
            ),
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),

        html.Div([
            html.Label('Aggregation method'),
            dcc.Dropdown(
                options=[{'label': 'Average', 'value': 'avg'}, 
                         {'label': 'Sum', 'value': 'sum'}, 
                         {'label': 'Count', 'value': 'count'}],
                value='avg',
                id='controls-and-histfunc'
            ),
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
    ], style={'display': 'flex', 'justifyContent': 'space-between'}),

    dcc.Graph(figure={}, id='controls-and-graph'),
    #wykres regresji
    html.Hr(),
    html.Div(children='Regression Analysis', style={'textAlign': 'center', 'fontSize': '24px', 'fontWeight': 'bold', 'margin': '20px 0'}),
    html.Div([
        html.Div([
            html.Label('X:'),
            dcc.Dropdown(
                options=[{'label': col, 'value': col} for col in df.columns if df[col].dtype != 'object'],
                value='ENGINE SIZE',
                id='regression-x-dropdown'
            ),
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),

        html.Div([
            html.Label('Y'),
            dcc.Dropdown(
                options=[{'label': col, 'value': col} for col in df.columns if df[col].dtype != 'object'],
        value='EMISSIONS',
        id='regression-y-dropdown'
            ),
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),

    ], style={'display': 'flex', 'justifyContent': 'space-between'}),

    dcc.Graph(figure={}, id='regression-graph'),
    html.Div(id='regression-info', style={'padding': '20px', 'fontSize': '16px'}),
    html.Hr(),
    #Tabele
    html.Div(children='Data set in tables', style={'textAlign': 'center', 'fontSize': '20px', 'volor': 'blue', 'fontWeight': 'bold', 'margin': '20px 0'}),
    html.Hr(),
    html.Div(children='All car makes in data set', style={'textAlign': 'left', 'fontSize': '16px', 'fontWeight': 'bold', 'margin': '20px 0'}), 
    dash_table.DataTable(data=df_make.to_dict('records'), page_size=50, style_data={'whiteSpace': 'normal', 'height': 'auto'}, fill_width=False),
    html.Div(children='All car models in the dataset', style={'textAlign': 'left', 'fontSize': '16px', 'fontWeight': 'bold', 'margin': '20px 0'}),
    dash_table.DataTable(data=df_car_model.to_dict('records'), page_size=50, style_data={'whiteSpace': 'normal', 'height': 'auto'}, fill_width=False),
    html.Div(children='All fuel type in data set', style={'textAlign': 'left', 'fontSize': '16px', 'fontWeight': 'bold', 'margin': '20px 0'}),
    dash_table.DataTable(data=df_unique_fuel.to_dict('records'), page_size=50, style_data={'whiteSpace': 'normal', 'height': 'auto'}, fill_width=False),
    html.Div(children='All vehicle class in data set', style={'textAlign': 'left', 'fontSize': '16px', 'fontWeight': 'bold', 'margin': '20px 0'}),
    dash_table.DataTable(data=df_unique_class.to_dict('records'), page_size=50, style_data={'whiteSpace': 'normal', 'height': 'auto'}, fill_width=False),
    html.Div(children='Data set', style={'textAlign': 'left', 'fontSize': '16px', 'fontWeight': 'bold', 'margin': '20px 0'}),
    dash_table.DataTable(data=df.to_dict('records'), page_size=50),
], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif'})

#funkcja do robienia box plot
@callback(
    Output(component_id='controls-and-graph', component_property='figure'),
    [Input(component_id='controls-and-dropdown-item', component_property='value'),
     Input(component_id='controls-and-histfunc', component_property='value'),
     Input(component_id='group-by-dropdown', component_property='value')]
)
def update_graph(y, histfunc, group_by):
    fig = px.histogram(df, x=group_by, y=y, histfunc=histfunc)
    
    fig.update_layout(
        title=f"GROUP BY {group_by.upper()}",
        xaxis_title=group_by.upper(),
        yaxis_title=y
    )
    
    fig.update_xaxes(categoryorder="total descending")
    return fig

#funkcja do robienia wykresów i analizy regresji
@callback(
    [Output(component_id='regression-graph', component_property='figure'),
     Output(component_id='regression-info', component_property='children')],
    [Input(component_id='regression-x-dropdown', component_property='value'),
     Input(component_id='regression-y-dropdown', component_property='value')]
)
def update_regression_graph(x_col, y_col):
    # Linia regresji
    slope, intercept, r_value, p_value, std_err = linregress(df[x_col], df[y_col])
    
    regression_line = slope * df[x_col] + intercept
    
    # Wykres punktowy - scatter plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode='markers', name='Data Points'))
    fig.add_trace(go.Scatter(x=df[x_col], y=regression_line, mode='lines', name='Regression Line'))
    
    fig.update_layout(
        title=f"Regression Analysis between {x_col} and {y_col}",
        xaxis_title=x_col,
        yaxis_title=y_col
    )
    
    # Info o regresji
    regression_equation = f"Equation: y = {slope:.2f}x + {intercept:.2f}"
    r_value_text = f"r = {r_value:.2f}"
    r_squared_text = f"R² = {r_value**2:.2f}"
    p_value_text = f"p = {p_value:.2f}"
    regression_info = f"{regression_equation} | {r_value_text} | {r_squared_text} | {p_value_text}"
    
    return fig, regression_info


if __name__ == '__main__':
    app.run(debug=True)
    
    
