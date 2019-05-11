#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:55:44 2019

@author: Home
"""

import pandas as pd
import time
import os


from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import DataTable, TableColumn, Paragraph, RadioGroup, Tabs, Panel
from bokeh.layouts import row, column

from tornado import gen
from threading import Thread
from functools import partial

from joblib import dump, load

main_path = os.getcwd()
main_path = main_path+'/Data/model_data/'

df = pd.read_csv(main_path+'model_performance.csv')



# this must only be modified from a Bokeh session callback
source = ColumnDataSource(data=df.to_dict(orient='list'))

# This is important! Save curdoc() to make sure all threads
# see the same document.
doc = curdoc()

@gen.coroutine
def update(df):
    source.stream(df.to_dict(orient='list'))

def blocking_task():
    while True:
        # do some blocking computation
        time.sleep(30)
        
        df = pd.read_csv(main_path+'model_performance.csv')

        # but update the document from callback
        doc.add_next_tick_callback(partial(update, df))
        
        
df.Score[0].split(',')[1][:10].strip()
        
def accuracy(source):
    df = ColumnDataSource.to_df(source)
    
    df = df[['Model #', 'Score']]
            
    df['Training Score'] = df.Score.apply(lambda x: float(str(x).split(',')[0][1:]))
    
    df['Training Score'] = round(df['Training Score'],3)
    
    df['Test Score'] = df.Score.apply(lambda x: float(str(x).split(',')[1][:10].strip()))
    
    df['Test Score'] = round(df['Test Score'],3)
    
    df = df.sort_values('Training Score',ascending=True).reset_index(drop=True)
    df = df.reset_index().rename(str, columns=({'index':'Count'}))
    
    line_graph = figure(title="Model Generalization Power", width = 900, height= 275, y_range=(0, 1))
        
    line_graph.line(x= df['Count'], y=df['Training Score'], color='Purple')
    line_graph.line(x= df['Count'], y=df['Test Score'], color='Blue')
    
    #TODO COLOR LEGEND
    #TODO HOVER TOOL FOR MODEL NUMBER
    #TODO TITLE Y AXIS
    #TODO REMOVE NUMBERS FROM X AXIS
    return line_graph


def model_table(source):
    df = ColumnDataSource.to_df(source)  
            
    df['Training Score'] = df.Score.apply(lambda x: float(str(x).split(',')[0][1:]))
    
    df['Training Score'] = round(df['Training Score'],3)
    
    df['Test Score'] = df.Score.apply(lambda x: float(str(x).split(',')[1][:10].strip()))
    
    df['Test Score'] = round(df['Test Score'],3)
    
    df = df.sort_values('Test Score',ascending=False).reset_index(drop=True)

    df = df.drop(['Score'],axis=1)
        
    col_list = [x for x in df.columns]
    
    column_titles = list()
    
    for i in range(0, len(col_list)):
        column_titles.append(TableColumn(field=col_list[i], title=col_list[i]))

    source = ColumnDataSource(data=df.to_dict(orient='list'))
    
    datatable = DataTable(source = source, columns = column_titles, width = 650)
    
    #TODO ORDER COLUMNS
    
    return datatable



def model_select(source):
    df = ColumnDataSource.to_df(source)
             
    df['Training Score'] = df.Score.apply(lambda x: float(str(x).split(',')[0][1:]))
    
    df['Training Score'] = round(df['Training Score'],3)
    
    df['Test Score'] = df.Score.apply(lambda x: float(str(x).split(',')[1][:10].strip()))
    
    df['Test Score'] = round(df['Test Score'],3)
    
    df = df.sort_values('Test Score',ascending=False).reset_index(drop=True)
     
    df = df[['Model', 'Model #']]
             
    df['Combo'] = df.apply(lambda x: x['Model'] +'_'+ str(x['Model #']),1) 
      
    labels = [x for x in df.Combo]

    model_selection = RadioGroup( labels=labels, active=0)

    return model_selection


def update_simulation_model(attr, old, new):
    selected = model_selection.labels[model_selection.active]
        
    model = load(main_path+selected+'.joblib') 
    dump(model, main_path+'simulation_model'+'.joblib') 
    
    #shap.image_url(url=['/Data/model_data/'+selected+'_shap.png'],x=x_range[0], y=y_range[1],w=x_range[1]-x_range[0],h=y_range[1]-y_range[0])


#df = ColumnDataSource.to_df(source)
             
     
#df = df[['Model', 'Model #']]
             
#df['Combo'] = df.apply(lambda x: x['Model'] +'_'+ str(x['Model #']),1) 
      
#labels = [x for x in df.Combo]

#x_range = (0,1) # could be anything - e.g.(0,1)
#y_range = (0,1)
#shap = figure(x_range=x_range, y_range=y_range)
#shap.image_url(url=['/Data/model_data/'+labels[0]+'_shap.png'],x=x_range[0], y=y_range[1],w=x_range[1]-x_range[0],h=y_range[1]-y_range[0])



table_title = Paragraph(text="All Models")

model_select_title = Paragraph(text="Select Model for Simulator")

model_selection = model_select(source)
#must be after model_selection
model_selection.on_change('active', update_simulation_model)


line_graph = accuracy(source)
datatable = model_table(source)



top_row = row(line_graph)


selection_col = column(model_select_title, model_selection)

#tab1 = Panel(child = shap ,title = 'Shap Values')
tab2 = Panel(child = datatable, title = 'Scatter Plot')


bottom_tabs = Tabs(tabs=[ tab2], width=700, height=450)  


bottom_row = row(bottom_tabs, selection_col)

dash = column(top_row, bottom_row)


doc.add_root(dash)
doc.title = 'Model Evaluation'

thread = Thread(target=blocking_task)
thread.start()

    