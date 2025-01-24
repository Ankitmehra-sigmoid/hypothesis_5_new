#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


st.set_page_config(layout="wide")

def add_weekend_data(df, year_col='Lst.datum', week_col='week_of_year', totpal_col='TOTPAL'):
    """
    Adds weekend dates (Saturday and Sunday) to a dataset for each unique combination 
    of grouping columns and sets 'TOTPAL' values to 0 for these dates.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing date, week number, and grouping columns.
    year_col (str): Column name for the date column. Defaults to 'Lst.datum'.
    week_col (str): Column name for the week number column. Defaults to 'week_of_year'.
    totpal_col (str): Column name for the TOTPAL values. Defaults to 'TOTPAL'.

    Returns:
    pd.DataFrame: Updated DataFrame with weekend data added.
    """

    # Ensure the date column is in datetime format
    df[year_col] = pd.to_datetime(df[year_col])

    # Get unique combinations of the grouping columns
    grouping_columns = ['Customer Clients data', 'Postal Code clients data', 'Street', 'DC', week_col]
    unique_combinations = df[grouping_columns].drop_duplicates()

    # Function to calculate weekend dates for a given week number
    def get_weekend_dates(year, week_number):
        # Get the Monday of the given week
        monday = datetime.strptime(f"{year}-W{week_number:02d}-1", "%Y-W%U-%w")
        saturday = monday + timedelta(days=5)
        sunday = monday + timedelta(days=6)
        return saturday, sunday

    # Create a new DataFrame to store the results
    result_df = pd.DataFrame()

    # Iterate through each unique combination
    for _, group in unique_combinations.iterrows():
        # Filter data for this combination
        group_data = df[
            (df['Customer Clients data'] == group['Customer Clients data']) &
            (df['Postal Code clients data'] == group['Postal Code clients data']) &
            (df['Street'] == group['Street']) &
            (df['DC'] == group['DC']) &
            (df[week_col] == group[week_col])
        ]

        # Get the weekend dates for the week
        year = group_data[year_col].dt.year.iloc[0]
        week_number = group[week_col]
        saturday, sunday = get_weekend_dates(year, week_number)

        # Create rows for weekend dates
        weekend_data = pd.DataFrame({
            'Customer Clients data': [group['Customer Clients data']] * 2,
            'Postal Code clients data': [group['Postal Code clients data']] * 2,
            'Street': [group['Street']] * 2,
            'DC': [group['DC']] * 2,
            year_col: [saturday, sunday],
            week_col: [group[week_col]] * 2,
            'Month_orig': [saturday.month, sunday.month],
            totpal_col: ['Weekend', '']
        })

        # Append original group data and weekend data to the result
        result_df = pd.concat([result_df, group_data, weekend_data], ignore_index=True)

    # Sort the result DataFrame by combination and date
    result_df.sort_values(by=['Customer Clients data', 'Postal Code clients data', 'Street', 'DC', year_col], inplace=True)

    # Reset index for the result DataFrame
    result_df.reset_index(drop=True, inplace=True)

    return result_df

###########################################

# File: streamlit_app.py

data=pd.read_csv('df_original_scenario.csv',parse_dates=['Lst.datum'])
data2=pd.read_csv('df_updated_scenario_2.csv',parse_dates=['updated_delivery_date'])

data_orig_nitish=pd.read_csv('df_original_scenario_nitish.csv')
data_up_nitish=pd.read_csv('df_updated_scenario_nitish.csv',parse_dates=['updated_delivery_date'])

df_best_scenario=pd.read_csv('df_best_scenario_info_approach_2.csv')


data2['week_of_year'] = data2['updated_delivery_date'].apply(lambda x: x.isocalendar().week)
data_up_nitish['week_of_year'] = data_up_nitish['updated_delivery_date'].apply(lambda x: x.isocalendar().week)

# Filters
# st.sidebar.header("Filters")
# customer_cluster = st.sidebar.selectbox("Customer Cluster", options=data['customer cluster'].unique(), index=0)
# filtered_data = data[data['customer cluster'] == customer_cluster]
# filtered_data2 = data2[data2['customer cluster'] == customer_cluster]



customers = st.sidebar.selectbox("Customer Clients data", options=data['Customer Clients data'].unique(), index=0)
filtered_data = data[data['Customer Clients data'] == customers]
filtered_data2=data2[data2['Customer Clients data']==customers]

filtered_data_n = data_orig_nitish[data_orig_nitish['Customer Clients data'] == customers]
filtered_data_n2=data_up_nitish[data_up_nitish['Customer Clients data']==customers]


postal_code = st.sidebar.selectbox("Postal Code", options=filtered_data['Postal Code clients data'].unique())

filtered_data = filtered_data[filtered_data['Postal Code clients data'] == postal_code]
filtered_data2= filtered_data2[filtered_data2['Postal Code clients data'] == postal_code]


filtered_data_n = filtered_data_n[filtered_data_n['Postal Code clients data'] == postal_code]
filtered_data_n2=filtered_data_n2[filtered_data_n2['Postal Code clients data']==postal_code]

City = st.sidebar.selectbox("City", options=filtered_data['city'].unique())

                    
street = st.sidebar.selectbox("Street", options=filtered_data['Street'].unique())
filtered_data = filtered_data[filtered_data['Street'] == street]
filtered_data2= filtered_data2[filtered_data2['Street'] == street]

filtered_data_n = filtered_data_n[filtered_data_n['Street'] == street]
filtered_data_n2=filtered_data_n2[filtered_data_n2['Street']==street]


                     
dc = st.sidebar.selectbox("DC", options=filtered_data['DC'].unique())
filtered_data = filtered_data[filtered_data['DC'] == dc]
filtered_data2= filtered_data2[filtered_data2['DC'] == dc]

filtered_data_n = filtered_data_n[filtered_data_n['DC'] == dc]
filtered_data_n2=filtered_data_n2[filtered_data_n2['DC']==dc]

original_cost=round(filtered_data['Total cost orig'].sum(),0)
updated_cost=round(filtered_data2['Total cost updated'].sum())
savings=filtered_data['Total cost orig'].sum()-filtered_data2['Total cost updated'].sum()

original_cost_nitish=original_cost
updated_cost_nitish=round(filtered_data_n2['Total cost updated'].sum(),0)

savings_nitish=original_cost_nitish-updated_cost_nitish
if savings_nitish<5:
    original_cost_nitish=original_cost
    updated_cost_nitish=original_cost
    savings_nitish=0
    
                     
months = st.sidebar.multiselect(
    "Month_orig",
    options=filtered_data['Month_orig'].unique(),
    default=filtered_data['Month_orig'].unique()
)
# filtered_data = filtered_data[filtered_data['Month_orig'].isin(months)]

# filtered_data = filtered_data[filtered_data['Month_orig'] == month]

weeks = filtered_data[filtered_data['Month_orig'].isin(months)]['week_of_year'].unique().tolist() 

# weeks=filtered_data['week_of_year'].unique().tolist()
filtered_data = filtered_data[filtered_data['week_of_year'].isin(weeks)]

start_date=filtered_data['Lst.datum'].min()
last_date=filtered_data['Lst.datum'].max()

filtered_data2= filtered_data2[filtered_data2['updated_delivery_date']>=start_date]
filtered_data2= filtered_data2[filtered_data2['updated_delivery_date']<=last_date]

filtered_data_n2=filtered_data_n2[filtered_data_n2['updated_delivery_date']>=start_date]
filtered_data_n2= filtered_data_n2[filtered_data_n2['updated_delivery_date']<=last_date]

##save_data
# filtered_data.to_csv('filtered_data_before.csv',index=False)
# filtered_data2.to_csv('filtered_data_after.csv',index=False)
filtered_data_n2.to_csv('filtered_data_after_nitish.csv',index=False)

# Display tiles
# st.title("Cost Dashboard")

filtered_data=add_weekend_data(filtered_data,year_col='Lst.datum', week_col='week_of_year', totpal_col='TOTPAL')
filtered_data2=add_weekend_data(filtered_data2, year_col='updated_delivery_date', week_col='week_of_year', totpal_col='TOTPAL')
filtered_data_n2=add_weekend_data(filtered_data_n2,year_col='updated_delivery_date',week_col='week_of_year', totpal_col='TOTPAL')

# Bar plot function
def create_bar_plot(df, date_col, qty_col, title, width=900, height=350):
    df[date_col] = df[date_col].astype(str)
    fig = px.bar(
        df, 
        x=date_col, 
        y=qty_col, 
        labels={date_col: 'Delivery Date', qty_col: 'Total Pallets'}, 
        title=title,
        text=qty_col,  # Add text labels on the bars
        width=width,    # Set the width of the graph
        height=height   # Set the height of the graph
    )
    
    # Customize text position for better visibility
    fig.update_traces(textposition='outside')  # Places text above the bars
    
    return fig


# Graphs

graph1 = create_bar_plot(filtered_data, 'Lst.datum', 'TOTPAL',title='Shipment Profile Without Consolidation')
graph2 = create_bar_plot(filtered_data2, 'updated_delivery_date', 'TOTPAL',title='Shipment Profile After Consolidation', width=900, height=350)
graph3 = create_bar_plot(filtered_data_n2, 'updated_delivery_date', 'TOTPAL',title='Shipment Profile After Consolidation Approach 2', width=900, height=350)

# Display the graphs


# st.header("Before Consolidation")
# st.plotly_chart(graph1)
st.header("Consolidation Approach-1")
col1, col2,col3 = st.columns(3)

col1.metric("Total cost original (2023)", f"${original_cost:,.2f}")
col2.metric("Total cost updated (2023)", f"${updated_cost:,.2f}")
col3.metric("Total Savings (2023)", f"${savings:,.2f}")
# st.plotly_chart(graph2)

st.header("Consolidation Approach-2")
col4, col5,col6 = st.columns(3)

col4.metric("Total cost original (2023)", f"${original_cost_nitish:,.2f}")
col5.metric("Total cost updated (2023)", f"${updated_cost_nitish:,.2f}")
col6.metric("Total Savings (2023)", f"${savings_nitish:,.2f}")
# st.plotly_chart(graph3)
                        
                        
                        
                        
input_date_col = 'Lst.datum'
input_y_col = 'TOTPAL' # total_cost_cost_sheet_as_is
output_date_col = 'updated_delivery_date'
output_y_col = 'TOTPAL' # total_cost_cost_sheet_after
y_title = 'Total Pallets' # Must be 'Shipment Cost' or 'Pallets'
# Hover items
input_shipment_cost_col = 'total_cost_cost_sheet_as_is'
output_shipment_cost_col = 'total_cost_cost_sheet_after'
input_pallets_col = 'TOTPAL'
output_pallets_col = 'TOTPAL'
best_scenario=df_best_scenario[(df_best_scenario["Customer Clients data"]==customers) & (df_best_scenario['Postal Code clients data']==postal_code) & (df_best_scenario['Street']==street) & (df_best_scenario['DC']==dc)]['Best Scenario'].iloc[0]

                        
date_format = "%d-%b"
# Date format
# filtered_data["date_format"] = pd.to_datetime(filtered_data[input_date_col]).dt.strftime(date_format)
# filtered_data2["date_format"] = pd.to_datetime(filtered_data2[output_date_col]).dt.strftime(date_format)
# filtered_data_n2["date_format"] = pd.to_datetime(filtered_data_n2[output_date_col]).dt.strftime(date_format)
filtered_data["date_format"] = pd.to_datetime(filtered_data[input_date_col]).dt.strftime(f"{date_format} (%A)")
filtered_data2["date_format"] = pd.to_datetime(filtered_data2[output_date_col]).dt.strftime(f"{date_format} (%A)")
filtered_data_n2["date_format"] = pd.to_datetime(filtered_data_n2[output_date_col]).dt.strftime(f"{date_format} (%A)")

                        
# Plotly figure
fig = make_subplots(
    rows=3,
    cols=1,
    subplot_titles=("Before Consolidation", "Consolidation Approach-1", "Consolidation Approach-2: "+best_scenario),
    shared_xaxes="all",
    shared_yaxes="all",
    y_title=y_title,
    vertical_spacing=0.1,
)
if len(filtered_data.index) > 0:
    fig.add_trace(
        go.Bar(
            x=filtered_data["date_format"],
            y=filtered_data[input_y_col],
            hovertemplate="<b>Date: </b>%{customdata[1]}<br><b>Pallets: </b>%{customdata[0]}",
            customdata=filtered_data[[input_pallets_col,'date_format']],
            xaxis="x1",
            marker={"color": "#3366CC"},
            text=filtered_data[input_y_col],  # Add data labels
            texttemplate="%{text}",         # Format the data labels
            textposition="outside",         # Position the labels outside the bar
            showlegend=False,
        ),
        row=1,
        col=1,
    )
if len(filtered_data2.index) > 0:
    fig.add_trace(
        go.Bar(
            x=filtered_data2["date_format"],
            y=filtered_data2[output_y_col],
            hovertemplate="<b>Date: </b>%{customdata[1]}<br><b>Pallets: </b>%{customdata[0]}",
            customdata=filtered_data2[[input_pallets_col,'date_format']],
            xaxis="x2",
            marker={"color": "#109618"},
            text=filtered_data2[output_y_col],  # Add data labels
            texttemplate="%{text}",         # Format the data labels
            textposition="outside",         # Position the labels outside the bar
            showlegend=False,
        ),
        row=2,
        col=1,
    )
if len(filtered_data_n2.index) > 0:
    fig.add_trace(
        go.Bar(
            x=filtered_data_n2["date_format"],
            y=filtered_data_n2[output_y_col],
            hovertemplate="<b>Date: </b>%{customdata[1]}<br><b>Pallets: </b>%{customdata[0]}",
            customdata=filtered_data_n2[[input_pallets_col,'date_format']],
            xaxis="x2",
            marker={"color": "#109618"},
            text=filtered_data_n2[output_y_col],  # Add data labels
            texttemplate="%{text}",         # Format the data labels
            textposition="outside",         # Position the labels outside the bar
            showlegend=False,
        ),
        row=3,
        col=1,
    )
xaxis_array = []
if (len(filtered_data.index) > 0) or (len(filtered_data2.index) > 0):
    xaxis_array = sorted(
        list(
            set(
                pd.to_datetime(filtered_data[input_date_col]).to_list() + pd.to_datetime(filtered_data2[output_date_col]).to_list() +pd.to_datetime(filtered_data_n2[output_date_col]).to_list()
            )
        )
    )
    xaxis_array = [v.strftime(date_format) for v in xaxis_array]

fig.update_layout(
    {
        "title": y_title,
        "xaxis3": {
            "title": "Date",
            "tickangle": -45,
            "categoryorder": "array",
            "categoryarray": xaxis_array,
        },
        "template": "plotly_white",
        "width": 1500,  # Set the desired width of the graph
        "height": 700,  # Set the desired height of the graph
    }
)
st.plotly_chart(fig, use_container_width=True)
                        
