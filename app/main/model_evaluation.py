#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:55:44 2019

@author: Home
"""

import pandas as pd
import time
import os
from ast import literal_eval
import numpy as np

from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import DataTable, TableColumn, Paragraph, RadioGroup, Tabs, Panel, Select, PreText
from bokeh.layouts import row, column, gridplot
from sklearn.model_selection import train_test_split
from tornado import gen
from threading import Thread
from functools import partial
import shap
from joblib import dump, load
from sklearn.preprocessing import RobustScaler


main_path = os.getcwd()
main_path = main_path+'/Data/'

df = pd.read_csv(main_path+'model_data/model_performance.csv')



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
        
        df = pd.read_csv(main_path+'model_data/model_performance.csv')

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
    
    datatable = DataTable(source = source, columns = column_titles, width = 900)
    
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

    model_selection = Select(title = 'Select Top Model', value=labels[0], options = labels, width=200)

    return model_selection


def update_simulation_model(attr, old, new):
    df = ColumnDataSource.to_df(source)
    selected = model_selection.value
    
    split = selected.split('_')
    
    table = df[df.Model == split[0]]
    table = table[table['Model #']== int(split[1])]
    
    mini_table = pd.DataFrame(literal_eval(table['Parameters'][0]),index= [0])                       
                        
    printed.text = str(mini_table.transpose())

    scenario_1 = pd.read_csv(main_path+'scenario_1.csv')    
        
    model = load(main_path+selected+'.joblib') 

   
    prediction_1 = pd.Series(model.predict(scenario_1))
    prediction_1.to_csv(main_path+'prediction_1.csv', index=False)
    
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

printed = PreText(text="")

table_title = Paragraph(text="All Models")

model_select_title = Paragraph(text="Select Model for Simulator")

model_selection = model_select(source)
#must be after model_selection
model_selection.on_change('value', update_simulation_model)


line_graph = accuracy(source)
datatable = model_table(source)



top_row = row(line_graph)

mid_row = row(model_selection)




tab2 = Panel(child = printed ,title = 'Specific Model')
tab1 = Panel(child = datatable, title = 'All View')


bottom_tabs = Tabs(tabs=[ tab1, tab2], width=900, height=450)  


bottom_row = column(mid_row, bottom_tabs)

dash = column(top_row, bottom_row)


doc.add_root(dash)
doc.title = 'Model Evaluation'

thread = Thread(target=blocking_task)
thread.start()


"""

   
    empty_df =pd.DataFrame()
    predicts = []
    for i in range(0, len(scenario_1)):
        empty_df = scenario_1[i:i+1]
        predicts.append(model.predict(empty_df))
        
        
    scenario_1['Predicted'] = scenario_1.apply(lambda x: model.predict(x))
"""
    