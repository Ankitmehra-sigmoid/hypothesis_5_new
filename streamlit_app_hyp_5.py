#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import plotly.express as px

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
# In[ ]:


# File: streamlit_app.py

data=pd.read_csv('df_original_scenario.csv')
data2=pd.read_csv('df_updated_scenario_2.csv')

data_orig_nitish=pd.read_csv('df_original_scenario_nitish.csv')
data_up_nitish=pd.read_csv('df_updated_scenario_nitish.csv')


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
# filtered_data_n2.to_csv('

# Display tiles
# st.title("Cost Dashboard")



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


st.header("Before Consolidation")
st.plotly_chart(graph1)
st.header("Consolidation Approach-1")
col1, col2,col3 = st.columns(3)

col1.metric("Total cost original (2023)", f"${original_cost:,.2f}")
col2.metric("Total cost updated (2023)", f"${updated_cost:,.2f}")
col3.metric("Total Savings (2023)", f"${savings:,.2f}")
st.plotly_chart(graph2)

st.header("Consolidation Approach-2")
col4, col5,col6 = st.columns(3)

col4.metric("Total cost original (2023)", f"${original_cost_nitish:,.2f}")
col5.metric("Total cost updated (2023)", f"${updated_cost_nitish:,.2f}")
col6.metric("Total Savings (2023)", f"${savings_nitish:,.2f}")
st.plotly_chart(graph3)
                        
                        
                        
                        
# input_date_col = 'Lst.datum'
# input_y_col = 'TOTPAL' # total_cost_cost_sheet_as_is
# output_date_col = 'updated_delivery_date'
# output_y_col = 'TOTPAL' # total_cost_cost_sheet_after
# y_title = 'Shipment Cost' # Must be 'Shipment Cost' or 'Pallets'
# # Hover items
# input_shipment_cost_col = 'total_cost_cost_sheet_as_is'
# output_shipment_cost_col = 'total_cost_cost_sheet_after'
# input_pallets_col = 'TOTPAL'
# output_pallets_col = 'TOTPAL'
                        
# date_format = "%d-%b"
# # Date format
# filtered_data["date_format"] = pd.to_datetime(filtered_data[input_date_col]).dt.strftime(date_format)
# filtered_data2["date_format"] = pd.to_datetime(filtered_data2[output_date_col]).dt.strftime(date_format)
# filtered_data_n2["date_format"] = pd.to_datetime(filtered_data_n2[output_date_col]).dt.strftime(date_format)

                        
# # Plotly figure
# fig = make_subplots(
#     rows=3,
#     cols=1,
#     subplot_titles=("Before Consolidation", "After Consolidation", "After Consolidation 2"),
#     shared_xaxes="all",
#     shared_yaxes="all",
#     y_title=y_title,
#     vertical_spacing=0.1,
# )
# if len(filtered_data.index) > 0:
#     fig.add_trace(
#         go.Bar(
#             x=filtered_data["date_format"],
#             y=filtered_data[input_y_col],
#             hovertemplate="<b>Pallets: </b>%{customdata[0]}<br><b>Cost: </b>€%{customdata[1]:,.1f}",
#             customdata=filtered_data[[input_pallets_col]],
#             xaxis="x1",
#             marker={"color": "#3366CC"},
#             showlegend=False,
#         ),
#         row=1,
#         col=1,
#     )
# if len(filtered_data2.index) > 0:
#     fig.add_trace(
#         go.Bar(
#             x=filtered_data2["date_format"],
#             y=filtered_data2[output_y_col],
#             hovertemplate="<b>Pallets: </b>%{customdata[0]}<br><b>Cost: </b>€%{customdata[1]:,.1f}",
#             customdata=filtered_data2[[input_pallets_col]],
#             xaxis="x2",
#             marker={"color": "#109618"},
#             showlegend=False,
#         ),
#         row=2,
#         col=1,
#     )
# if len(filtered_data_n2.index) > 0:
#     fig.add_trace(
#         go.Bar(
#             x=filtered_data_n2["date_format"],
#             y=filtered_data_n2[output_y_col],
#             hovertemplate="<b>Pallets: </b>%{customdata[0]}<br><b>Cost: </b>€%{customdata[1]:,.1f}",
#             customdata=filtered_data_n2[[input_pallets_col]],
#             xaxis="x2",
#             marker={"color": "#109618"},
#             showlegend=False,
#         ),
#         row=3,
#         col=1,
#     )
# xaxis_array = []
# if (len(filtered_data.index) > 0) or (len(filtered_data2.index) > 0):
#     xaxis_array = sorted(
#         list(
#             set(
#                 pd.to_datetime(filtered_data[input_date_col]).to_list() + pd.to_datetime(filtered_data2[output_date_col]).to_list()
#             )
#         )
#     )
#     xaxis_array = [v.strftime(date_format) for v in xaxis_array]

# fig.update_layout(
#     {
#         "title": y_title,
#         "xaxis3": {
#             "title": "Date",
#             "tickangle": -45,
#             "categoryorder": "array",
#             "categoryarray": xaxis_array,
#         },
#         "template": "plotly_white",
#     }
# )
# # fig.show()
                        
# # fig_bytes = io.BytesIO()
# # go.Figure(fig).write_image(fig_bytes, format='png')
# # fig_bytes.seek(0)
                        
# st.plotly_graph(fig, use_column_width=True)
