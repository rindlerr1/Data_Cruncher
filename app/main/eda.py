#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 09:40:44 2019

@author: Home
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import DataTable, TableColumn, Paragraph, Select, PreText, Panel, Tabs
from bokeh.layouts import row, column


categorical_vars = []
numeric_vars = []
PATH = '/users/home/desktop/data_cruncher/app/Data/app_data.csv'

df = pd.read_csv(PATH)




for i in range(0, len(df.columns)):
    if (type(df[df.columns[i]][0]) == np.int64 or type(df[df.columns[i]][0]) == np.float64):
        numeric_vars.append(df.columns[i])
    elif type(df[df.columns[i]][0]) == str:
        categorical_vars.append(df.columns[i])

def datetime(x):
    return np.array(x, dtype=np.datetime64)

def record_line(df):
    table = df.groupby(['Date'])['Date'].count().rename('Record Counts').reset_index()
    sd = np.std(table['Record Counts'])
    mean = np.mean(table['Record Counts'])
    
    y_max = max(table['Record Counts']) + sd*2
    y_min = min(table['Record Counts']) - sd*2
    
    table['Upper Bound'] = table['Record Counts'].apply(lambda x : mean+(sd*2))
    table['Mid-Upper Bound'] = table['Record Counts'].apply(lambda x : mean+(sd))
    
    
    table['Lower Bound'] = table['Record Counts'].apply(lambda x : mean-(sd*2))
    table['Mid-Lower Bound'] = table['Record Counts'].apply(lambda x : mean-(sd))
    
    records = figure(x_axis_type="datetime", title="File Record Counts", width = 850, height= 300, y_range=(y_min, y_max))

    records.xaxis.axis_label = 'Date'
    records.yaxis.axis_label = 'Record Counts'
    
    records.circle(datetime(table['Date']), table['Record Counts'], color='#6D358A', fill_alpha=0.8, size=10)
    
    records.line(datetime(table['Date']), table['Upper Bound'], color='#34ADFF')
    #records.line(datetime(table['Date']), table['Mid-Upper Bound'], color='red')
    records.line(datetime(table['Date']), table['Lower Bound'], color='#34ADFF')
    #records.line(datetime(table['Date']), table['Mid-Lower Bound'], color='red')       
            
    
    #potential broken days
    p1 = Paragraph(text="Dates With High/Low Record Counts", width=200)
    table['Outliers'] = table.apply(lambda x: 1 if (x['Record Counts'] > x['Upper Bound'] or x['Record Counts'] < x['Lower Bound']) else 0, 1)

    outliers = table[table['Outliers']== 1].reset_index(drop=True)
    outliers = outliers[['Date']]
    
    data_table = PreText(text='', width=175)
    data_table.text = str(outliers)

    return records, p1, data_table






def describe_data(df, numeric_vars):
    table = df[numeric_vars]
    table = table.describe().reset_index()
    data_dict= {}
    for i in range(0, len(table.columns)):
        data_dict[table.columns[i]] = table[table.columns[i]]

    data_source = ColumnDataSource(data_dict)

    return data_source





def create_catdata(df, categorical_vars):
    cats = []
    for i in range(0, len(categorical_vars)):
        cats.append(list())
    for i in range(0, len(cats)):
        for q in range(0, len(cats)):
            col1 = df[categorical_vars[i]].rename('Cat1')
            col2 = df[categorical_vars[q]].rename('Cat2')
            df_chi = pd.concat([col1, col2], axis=1)
            chi_tab = pd.crosstab(df_chi.Cat1, df_chi.Cat2)
            cats[i].append(round(chi2_contingency(observed= chi_tab)[1],4))
    
    data_dict = {}
    for i in range(0, len(cats)):
        data_dict[categorical_vars[i]] = cats[i]
    table = pd.DataFrame.from_dict(data_dict)
    table = table[categorical_vars]
    table.index = categorical_vars
    table = table.reset_index()
    
    
    data_dict= {}
    for i in range(0, len(table.columns)):
        data_dict[table.columns[i]] = table[table.columns[i]]

    data_source = ColumnDataSource(data_dict)

    return data_source



def create_table(datasource, size):   
    column_list = list(datasource.column_names)
    column_titles = list()
    
    for i in range(0, len(column_list)):
        column_titles.append(TableColumn(field=column_list[i], title=column_list[i]))
    
    datatable_ = DataTable(source = datasource, columns = column_titles, width = size, height= 300)
    return datatable_

def create_numerictable(df, numeric_vars):
    corr_coeffs = np.corrcoef(df[numeric_vars], rowvar=0)
    corr_coeffs = pd.DataFrame(corr_coeffs)
    corr_coeffs.columns = numeric_vars
    corr_coeffs.index= np.array(numeric_vars)
    corr_coeffs = corr_coeffs.reset_index()
    data_dict= {}
    for i in range(0, len(corr_coeffs.columns)):
        data_dict[corr_coeffs.columns[i]] = corr_coeffs[corr_coeffs.columns[i]]

    data_source = ColumnDataSource(data_dict)
    return data_source


def scatter_data(df, x_widget, y_widget):
    table = df

    data_dict = {
            'X_Axis': table[x_widget],
            'Y_Axis': table[y_widget]}
    
    scatter_datasource = ColumnDataSource(data_dict)
    return scatter_datasource

def scatter_plot(scatter_data):
    s_plot = figure(title="Numeric Variable Relationships", width = 350, height= 350)
    s_plot.circle(x='X_Axis',y='Y_Axis', color= '#302782', source=scatter_data)
    return s_plot
    


records, p1, data_table = record_line(df)

scatter_datasource = scatter_data(df, numeric_vars[0], numeric_vars[1] )  
scatter = scatter_plot(scatter_datasource)


p2 = Paragraph(text="P-Values Categorical Var Relationships", width=250)
p3 = Paragraph(text="Descriptive Statistics", width=250)


describe_datasource = describe_data(df, numeric_vars)
describe_datatable = create_table(describe_datasource,900)

describe_catdata = create_catdata(df, categorical_vars)
cat_datatable = create_table(describe_catdata, 425)

numeric_tabledatasource = create_numerictable(df, numeric_vars)
numeric_datatable = create_table(numeric_tabledatasource, 450 )




def update_scatter(attr, old, new):
    new_scatter = scatter_data(df, x_widget = x_select.value, y_widget = y_select.value)
    scatter_datasource.data.update(new_scatter.data)
         
x_select = Select(title="X_Axis", value=numeric_vars[0], options= numeric_vars)
x_select.on_change('value', update_scatter)

y_select = Select(title="Y_Axis", value=numeric_vars[1], options= numeric_vars)
y_select.on_change('value', update_scatter)








scatter_layout = row(scatter, column(x_select, y_select))

tab1 = Panel(child = numeric_datatable ,title = 'Corr Coeffs')
tab2 = Panel(child = scatter_layout, title = 'Scatter Plot')
tabs = Tabs(tabs=[ tab2, tab1], width=450, height=300)    
 
cat_layout = column(p2,cat_datatable,width=450)
broken_col = column(p1, data_table, width = 350)
descrip_layout = column(p3, describe_datatable, width=905)

top_row = row(records,broken_col, width = 900, height= 325)
middle_row = row(cat_layout, tabs)
bottom_row = row(descrip_layout, width = 825)

basic_layout = column(top_row,middle_row, bottom_row)
    





curdoc().add_root(basic_layout)
curdoc().title = "Store Sales"        



"""

import mysql.connector
#connect to db and pull in dataset
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="Izzie722", database='icom_demo')


cursor = mydb.cursor()
query = ("SELECT * FROM file_metadata")
cursor.execute(query)
db_data=cursor.fetchall()

PATH = db_data[len(db_data)-1][1]



def corr_data(df, cat1_widget, cat2_widget):
    table = df[[cat1_widget, cat2_widget]]
    table = table.groupby([cat1_widget, cat2_widget])[cat1_widget].count().rename('Count').reset_index()
    table['Count'] = table['Count'].apply(lambda x: x/table['Count'].sum()  )
    table = pd.pivot_table(data=table, values='Count', index=cat1_widget, columns=cat2_widget)

    return table
    
    
def update_corr(attr, old, new):
    new_corr = corr_data(df, cat1_widget.value, cat2_widget.value)
    table_correlation.text = str(new_corr)



cat1_widget = Select(title="X_Axis", value=categorical_vars[0], options= categorical_vars)
cat1_widget.on_change('value', update_corr)

cat2_widget = Select(title="Y_Axis", value=categorical_vars[1], options= categorical_vars)
cat2_widget.on_change('value', update_corr)

corr_table = create_catdata(df, categorical_vars)
table_correlation = PreText(text='', width=500)
table_correlation.text = str(corr_table)
"""