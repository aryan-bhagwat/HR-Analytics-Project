import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import pickle
import os

# Load data and model
data_path = "c:\\Aryan\\HR Analytics Project\\data\\hr_data.csv"
model_path = "c:\\Aryan\\HR Analytics Project\\models\\best_attrition_model.pkl"
df = pd.read_csv(data_path)

# Create a Dash application
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("HR Analytics Dashboard - Employee Attrition"),
    
    html.Div([
        html.H2("Attrition Overview"),
        dcc.Graph(
            id='attrition-overview',
            figure=px.pie(df, names='Attrition', title='Employee Attrition Distribution')
        )
    ]),
    
    html.Div([
        html.H2("Attrition by Department"),
        dcc.Graph(
            id='attrition-by-department',
            figure=px.histogram(df, x='Department', color='Attrition', 
                               barmode='group', title='Attrition by Department')
        )
    ]),
    
    html.Div([
        html.H2("Attrition by Age Group"),
        dcc.Graph(
            id='attrition-by-age',
            figure=px.histogram(df, x='Age', color='Attrition', 
                               nbins=10, title='Attrition by Age Group')
        )
    ]),
    
    html.Div([
        html.H2("Job Satisfaction vs Attrition"),
        dcc.Graph(
            id='satisfaction-vs-attrition',
            figure=px.box(df, x='Attrition', y='JobSatisfaction', 
                         title='Job Satisfaction vs Attrition')
        )
    ]),
    
    html.Div([
        html.H2("Monthly Income vs Attrition"),
        dcc.Graph(
            id='income-vs-attrition',
            figure=px.box(df, x='Attrition', y='MonthlyIncome', 
                         title='Monthly Income vs Attrition')
        )
    ])
])

if __name__ == '__main__':
    app.run(debug=True)  # Changed from run_server() to run()