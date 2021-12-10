#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from collections import Counter
import math
import webbrowser
import os


# In[2]:


import dash
from dash import dcc, html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots

print('Execution Started.....')

# In[3]:


df = pd.read_csv('CAERS_ASCII_2004_2017Q2.csv')
df.head()


# In[4]:


temp_df = df.copy()


# # Q1

# In[5]:


temp_df['AEC_Event Start Date'] = pd.to_datetime(temp_df['AEC_Event Start Date'])


# In[6]:


temp_df['CI_Gender'].unique()


# In[7]:


for name in temp_df['CI_Gender'].unique():
    if(name == 'Female'):
        temp_df['CI_Gender'] = temp_df['CI_Gender'].replace(name,0)
    elif(name == 'Male'):
        temp_df['CI_Gender'] = temp_df['CI_Gender'].replace(name,1)
    else:
        temp_df['CI_Gender'] = temp_df['CI_Gender'].replace(name,2)
temp_df = temp_df.replace(' ',np.nan)


# In[8]:


temp_df['CI_Gender'].unique()


# In[9]:


# df['CI_Gender'] = df['CI_Gender'].replace('Female',0)
# df['CI_Gender'] = df['CI_Gender'].replace('Male',1)


# In[10]:


temp_df['AEC_Event Start Date'].replace('', np.nan, inplace=True)
temp_df.dropna(subset=['AEC_Event Start Date'], inplace=True)
temp_df.shape


# In[11]:


data = temp_df[['AEC_Event Start Date','CI_Gender']]


# In[12]:


data.to_csv('q1.csv',index=False)


# In[13]:


dt = pd.read_csv('q1.csv')


# In[ ]:





# In[14]:


years = []
for i in range(0,dt['AEC_Event Start Date'].shape[0]):
    date = pd.to_datetime(dt['AEC_Event Start Date'][i])
    date = str(date).split()[0]
    year = pd.to_numeric(date.split('-')[0])
    years.append(year)


# In[15]:


dt['years'] = years


# In[16]:


dt.head(2)


# In[17]:


data_count_by_year = dt.groupby('years')['CI_Gender'].count()


# In[18]:


data_count_by_year = data_count_by_year.to_dict()


# In[19]:


# dt = dt[(dt['CI_Gender'].str.contains('0', na=False)) | (dt['CI_Gender'].str.contains('1', na=False))]
# dt.info()


# In[20]:


dt['CI_Gender'] = pd.to_numeric(dt['CI_Gender'])
# dt.info()


# In[21]:


dt.index = dt['years']


# In[22]:


dt_zeros = dt.where(dt['CI_Gender'] == 0)
dt_ones = dt.where(dt['CI_Gender'] == 1)
dt_unknown = dt.where(dt['CI_Gender'] == 2)


# In[23]:


dt_zeros = dt_zeros.dropna()
dt_ones = dt_ones.dropna()
dt_unknown = dt_unknown.dropna()


# In[24]:


dt_zeros = dt_zeros.groupby(dt_zeros.index)['CI_Gender'].count()
dt_ones = dt_ones.groupby(dt_ones.index)['CI_Gender'].count()
dt_unknown = dt_unknown.groupby(dt_unknown.index)['CI_Gender'].count()


# In[25]:


dt_zeros_female = dt_zeros.to_dict()
dt_ones_male = dt_ones.to_dict()
dt_twos_unknown = dt_unknown.to_dict()


# In[26]:


years = []
total_entries = []
male = []
female = []
unknown = []

for key,value in data_count_by_year.items():
    #Female
    if key in dt_zeros_female:
        female.append(dt_zeros_female[key])
    else:
        female.append(0)

    #Male
    if key in dt_ones_male:
        male.append(dt_ones_male[key])
    else:
        male.append(0)

    #Unknown
    if key in dt_twos_unknown:
        unknown.append(dt_twos_unknown[key])
    else:
        unknown.append(0)
    years.append(key)
    total_entries.append(value)


# In[27]:


d = {'Years' : years,
    'Male' : male,
    'Female' : female,
    'Unknown' : unknown,
    'Total_entries' : total_entries}
q1_final_df = pd.DataFrame(d)


# In[28]:


def percentage(part, whole):
    percentage = 100 * float(part)/float(whole)
    return round(percentage,2)

# print(percentage(3, 5))


# In[29]:


male_percent_list = []
female_percent_list = []
unknown_percent_list = []
for index, row in q1_final_df.iterrows():
    male_percent = percentage(row['Male'], row['Total_entries'])
    female_percent = percentage(row['Female'], row['Total_entries'])
    unknown_percent = percentage(row['Unknown'], row['Total_entries'])
    
    male_percent_list.append(male_percent)
    female_percent_list.append(female_percent)
    unknown_percent_list.append(unknown_percent)


# In[30]:


q1_final_df['Male_percentage'] = male_percent_list
q1_final_df['Female_percentage'] = female_percent_list
q1_final_df['Unknown_percentage'] = unknown_percent_list


# # Q2

# In[31]:


dt_symptoms = df['SYM_One Row Coded Symptoms']
dt_symptoms


# In[32]:


symptoms_list  = []
for i in range(0,len(dt_symptoms)):
    temp = str(dt_symptoms[i])
    temp = temp.split(',')
    
    for j in range(0,len(temp)):
        txt = temp[j].strip()
        temp[j] = txt
    symptoms_list.extend(temp)


# In[33]:


symptoms_count = dict(Counter(symptoms_list))
# print(symptoms_count)


# In[34]:


q2_final_df = pd.DataFrame([symptoms_count])


# In[35]:


q2_final_df = q2_final_df.transpose()


# In[36]:


q2_final_df.rename(columns={0: "values"},inplace=True)


# In[37]:


q2_final_df.sort_values('values',ascending=False,inplace=True)


# # Q3

# In[38]:


third_df = df.copy()


# In[39]:


third_df.columns


# In[40]:


q3_final_df = pd.DataFrame()


# In[ ]:





# In[41]:


for industry in third_df['PRI_FDA Industry Name'].unique():
    temp_looped_df = pd.DataFrame()
    temp_df = third_df[third_df['PRI_FDA Industry Name'] == industry]

    events_splitted_lst = []
    for colval in temp_df['AEC_One Row Outcomes']:
        temp = colval.split(',')
        
        for j in range(0,len(temp)):
            txt = temp[j].strip()
            temp[j] = txt 
        events_splitted_lst.extend(temp)

    Events_count = dict(Counter(events_splitted_lst))

    Industry_name = []
    event = []
    counts = []
    for key,value in Events_count.items():
        Industry_name.append(industry)
        event.append(key)
        counts.append(value)

    dic = {
    'Industry Name' : Industry_name,
    'Events' : event,
    'Count' : counts
    }

    temp_looped_df = pd.DataFrame(dic)
    q3_final_df = q3_final_df.append(temp_looped_df)


# In[42]:


q3_final_df


# In[43]:


filter_list = ['Not Available', 'Bakery Prod/Dough/Mix/Icing']
q3_final_df[q3_final_df['Industry Name'].isin(filter_list)]


# In[44]:


sum_df = pd.DataFrame(columns = ['Industries','Total no of Events'])
for indus in q3_final_df['Industry Name'].unique(): 
    temp = q3_final_df[q3_final_df['Industry Name'] == indus]
    sum_df = sum_df.append({'Industries' : indus,
                  'Total no of Events' : temp.Count.sum()},
                  ignore_index=True)


# In[45]:


sum_df.sort_values('Total no of Events',ascending=False,inplace=True)


# In[46]:


sum_df.head()


# In[ ]:


print('Wait for the browser to load....')


# # Plotly

# In[47]:


BS = "https://cdn.jsdelivr.net/npm/bootstrap@5.1.0/dist/css/bootstrap.min.css"
app = dash.Dash(__name__,external_stylesheets=[BS,dbc.themes.BOOTSTRAP])
app.layout = html.Div([
    ###########Q1 Header###########
    html.H2(
        className='q1Header',
        children=''),
    ###########Q1 Filter Table###########
    html.Div([
        html.Table([
            html.Tr([
                html.Td(html.P(className='From', children='From : ')),
                html.Td(
                    dcc.Dropdown(className='from_year_dropdown',
                                 id='from-year-dropdown',
                                 options=[{
                                     'label': year,
                                     'value': year
                                 } for year in q1_final_df.Years],
                                 value=q1_final_df.Years.max() - 10,
                                 clearable=False)),
                html.Td(html.P(className='To', children='To : ')),
                html.Td(
                    dcc.Dropdown(className='to_year_dropdown',
                                 id='to-year-dropdown',
                                 options=[{
                                     'label': year,
                                     'value': year
                                 } for year in q1_final_df.Years],
                                 value=q1_final_df.Years.max(),
                                 clearable=False))
            ])
        ])
    ]),
    ###########Q1 Charts###########
    html.Div([
        html.Div(className="q1_bar_chart",
                 children=[dcc.Graph(id='q1_bar_graph')]),
        html.Div(className="q1_pie_chart",
                 children=[dcc.Graph(id='q1_pie_graph')])
    ]),
    html.Div([html.Hr(className='Q_hr')]),
    html.Div([
        ###########Q2 Header############
        html.H2(
            className='q2Header',
            children=''),
        ############Q2 Graph############
        html.Div(className='q2_graph',
                 children=[dcc.Graph(id='q2_bar_chart')]),
        ###########Q2 graph Filters###########
        html.Div(className='q2_graph_filters',
                 children=[
                     html.Div([html.P(className='filter',
                                      children='Range')]),
                     html.Div([
                     html.Div(className='range',
                              children=[
                                  dcc.Input(id='range',
                                            type='text',
                                            placeholder='Eg.1-10',
                                            value='1-10')
                              ]),
                     html.Div(className='end',
                              children=[
                                  html.Div([
                                      html.Ul(className = 'q2_graph_text',
                                              children = [
                                                  html.Li(['Mention the range to be displayed in graph.']),
                                                  html.Li(['Symptoms have been arranged in descending order.']),
                                                  html.Li(['For better readability use the range of difference is less than or equal to 10.'])
                                      ])
                                  ])
                              ])
                     ])
                 ])
    ]),
    html.Div([html.Hr(className='Q_hr')]),
    html.Div([
        ###########Q3 Header###########
        html.H2(
            className='q3Header',
            children= ''
        ),
        ###########Q3 Top 10 Graph###########
        html.Div([
            dcc.Graph(
                figure = px.bar(sum_df[0:10],
                                y = 'Total no of Events',
                                x = 'Industries',
                               title='Top 10 Industries with Most Adverse Events')
            )
        ]),
        html.Div([html.Br()]),
        html.Div([html.Br()]),
        html.Div(
            className = 'q3_subplot_main_div',
            children=[
                html.Div(
                    className = 'industries_drop_down',
                    children = [
                    dbc.DropdownMenu(
                        label = 'Select the Industries',
                        children=[
                            dcc.Checklist(
                                id = 'industries_drop_down',
                                options=[{
                                    'label': industry, 
                                    'value': industry
                                } for industry in q3_final_df['Industry Name'].unique()
                                ],
                                value=['Bakery Prod/Dough/Mix/Icing','Ice Cream Prod'],
                                labelStyle={'display': 'block'}
                            )
                        ]
                    )
                ]),
                html.Div(
                    className = 'events_div',
                    id = 'event_drop_down'
                )
            ]),
        html.Div([
            html.Div([
                dcc.Graph(id = 'compare_subplots')
            ])
        ])
    ])
])


@app.callback(
    [Output('q1_bar_graph', 'figure'),
     Output('q1_pie_graph', 'figure')],
    [Input('from-year-dropdown', 'value'),
     Input('to-year-dropdown', 'value')])
def update_q1_bar_pie_chart(from_value, to_value):
    
    if from_value > to_value:
        raise PreventUpdate
    
    colors = ['rgb(239, 85, 59)', 'rgb(99, 110, 250)', 'rgb(0, 204, 150)']

    temp_df = q1_final_df[(q1_final_df.Years >= int(from_value))
                       & (q1_final_df.Years <= int(to_value))]

    #Bar Chart
    fig_bar = go.Figure(data=[
        go.Bar(name='Male', x=temp_df.Years, y=temp_df.Male_percentage),
        go.Bar(name='Female', x=temp_df.Years, y=temp_df.Female_percentage),
        go.Bar(name='Unknown', x=temp_df.Years, y=temp_df.Unknown_percentage)
    ],
                       layout = {
                           'title' : 'Percentage of adverse events reported each year by gender'
                       })
    fig_bar.update_xaxes(title_text = 'Years')
    fig_bar.update_yaxes(title_text = 'Percentage')
    fig_bar.update_layout(barmode='group')
    fig_bar.update_layout(showlegend=False)
    

    #Pie Chart
    Count_list = [
        temp_df.Male.sum(),
        temp_df.Female.sum(),
        temp_df.Unknown.sum()
    ]
    Name = ['Male', 'Female', 'Unknown']
    fig_pie = px.pie(q1_final_df,
                     names=Name,
                     values=Count_list,
                     color_discrete_sequence=colors,
                     title = 'All-over percentage of the adverse events by<br>gender over selected time period')

    fig_pie.update_layout(legend_title_text='Gender')
    fig_pie.update_traces(hole=.4, hoverinfo="label+percent+name")

    return fig_bar, fig_pie


@app.callback(
    Output('q2_bar_chart', 'figure'),
    Input('range', 'value'))
#      Input('end', 'value')])
def Update_q2_bar_chart(range):

    if range is None:
        raise PreventUpdate
        
    range = range.split('-')
    
    if len(range) != 2:
        raise PreventUpdate
        
    start = int(range[0])
    end = int(range[1])
    
    if start > end:
        raise PreventUpdate
    
    slice_df = q2_final_df[start-1:end + 1]
    slice_df.sort_values('values', ascending=True, inplace=True)


    #Q2 Bar chart
    fig_bar = px.bar(slice_df,
                     x='values',
                     y=slice_df.index,
                     orientation='h',
                     labels = dict(index = 'Symptoms', 
                                  values = 'No of observations recorded'),
                    title = 'Most frequently Reported Symptoms over entire span')
    return fig_bar


#################          Q3           ############################
@app.callback(
    Output('event_drop_down', 'children'), 
    Input('industries_drop_down', 'value'))
def update_events(industries):
    if industries is None:
        raise PreventUpdate
        
    events = q3_final_df[q3_final_df['Industry Name'].isin(industries)]
    unique_events = events.Events.unique()
    
    return dbc.DropdownMenu(
                        label = 'Select the Events',
                        className = 'events_dropdown',
                        children=[
                            dcc.Checklist(
                                id = 'events_checklist',
                                options=[{
                                    'label': events, 
                                    'value': events
                                } for events in unique_events
                                ],
                                value=['VISITED AN ER',
                                       'VISITED A HEALTH CARE PROVIDER',
                                       'REQ. INTERVENTION TO PRVNT PERM. IMPRMNT.', 'HOSPITALIZATION',
                                       'NON-SERIOUS INJURIES/ ILLNESS', 'LIFE THREATENING', 'DEATH',
                                       'SERIOUS INJURIES/ ILLNESS', 'NONE',
                                       'OTHER SERIOUS (IMPORTANT MEDICAL EVENTS)', 'DISABILITY',
                                       'CONGENITAL ANOMALY'],
                                labelStyle={'display': 'block'}
                            )]
                        )

@app.callback(Output('compare_subplots', 'figure'),
              [Input('industries_drop_down', 'value'),
             Input('events_checklist','value')])
def update_compare_charts(indus_drop,events_drop):
        
    fig = go.Figure()
    for indus in indus_drop:
        compare_df = q3_final_df[q3_final_df['Industry Name'] == indus]
        compare_df = compare_df[compare_df['Events'].isin(events_drop)]
        
        hover_text = []
        bubble_size = []

        for index, row in compare_df.iterrows():
            hover_text.append(('Industry Name: {Industry_Name}<br>'+
                              'Adverse Events: {Events}<br>'+
                              'No of Records: {Count}<br>').format(Industry_Name=row['Industry Name'],
                                                Events=row['Events'],
                                                Count=row['Count']))

            bubble_size.append(math.sqrt(row['Count']))

        compare_df['text'] = hover_text
        compare_df['size'] = bubble_size
        
        
        fig.add_trace(go.Scatter(x=compare_df['Events'],
                                 y=compare_df['Count'],
                                 mode='markers',
                                 text=compare_df['text'],
                                 marker_size=compare_df['size'],
                                 name = indus
                                ))
        
    fig.update_layout(height=600, title_text="Compare Industries the industries with their respective events")
    fig.update_xaxes(title = 'Adverse Events')
    fig.update_yaxes(title = 'No Of Records')
    
    return fig

    
if __name__ == '__main__':
    # def open_browser():
    #     webbrowser.open_new('http://127.0.0.1:8080/')
    # open_browser()
    # app.run_server(debug=False, use_reloader=False,port=os.environ.get('PORT', '8080'))
    app.run_server(debug=False)


# In[ ]:





# In[ ]:




