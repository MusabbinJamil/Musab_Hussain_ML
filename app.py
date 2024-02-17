import dash
from dash import dcc, html
from dash import dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import json

# Load your machine learning data into a pandas DataFrame
df1 = pd.read_csv('Data/data.csv', delimiter=';')  
df2 = pd.read_csv('Data/NPHA-doctor-visits.csv', delimiter=',')

with open('Data/reviews.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

df3 = pd.json_normalize(data, record_path=['paper', 'review'], 
                       meta=[['paper', 'id'], ['paper', 'preliminary_decision']],
                       meta_prefix='paper_')
df3.head()
df3.columns

# Create a simple figure
fig = px.line(df1, x='Marital status', y='Target', title='Marital Status vs Target')

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[
    {
        'href': 'https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/cyborg/bootstrap.min.css',
        'rel': 'stylesheet',
    },
])

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        html.H1("Welcome to My Machine Learning Data App!"),
        html.H2("Where data is the new oil, and we're drilling!"),
    ], style={'textAlign': 'center', 'margin': '50px', 'color': 'white'}),
    html.Div(id='page-content')
], style={'backgroundColor': '#060606'})

navbar = dbc.Navbar(
        dbc.Row(
            dbc.Col(
                dbc.NavbarSimple(
                    children=[
                        dbc.NavItem(dbc.NavLink("Home", href="/")),
                        dbc.NavItem(dbc.NavLink("Page 1", href="/page-1")),
                        dbc.NavItem(dbc.NavLink("Page 2", href="/page-2")),
                        dbc.NavItem(dbc.NavLink("Page 3", href="/page-3")),
                    ],
                    color="dark",
                    dark=True
                ),
                width={"size": 16},
            ),
        ),
)

index_page = html.Div([
    navbar,
    html.Div([
        html.P("Dive into the Student Dropout dataset to uncover patterns and predictors of academic discontinuation. Explore the National Hospital Poll dataset to understand patient satisfaction and healthcare quality across the nation. Delve into the Research Report Acceptance and Denials dataset to discern the factors influencing the acceptance or rejection of research papers. Harness the power of Machine Learning to unearth hidden patterns, generate predictions and drive informed decision-making."),
        html.P("The dataset available at the UCI Machine Learning Repo  sitory under the title 'Predict Students Dropout And Academic Success' is a rich source of information for understanding student performance and predicting academic outcomes. This dataset contains a variety of features related to students' academic and personal life, such as their grades, attendance, socio-economic background, and more. It provides a comprehensive view of the factors that can influence a student's academic success or lead to dropout. By applying machine learning techniques to this dataset, we can identify patterns and correlations that can help educational institutions implement strategies to improve student retention and academic performance."),
        html.A("Access the 'Predict students dropout and academic success' dataset here.", href="https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success", target="_blank"),
        html.P("The National Poll on Healthy Aging (NPHA) dataset, available at the UCI Machine Learning Repository, is a valuable resource for studying the health and well-being of older adults. The dataset includes responses from a diverse group of adults aged 50 to 80, covering a wide range of topics such as physical health, mental health, access to healthcare, and lifestyle factors. The data collected provides a comprehensive snapshot of the health and lifestyle of the aging population in the United States. By applying machine learning techniques to this dataset, we can gain insights into the factors that contribute to healthy aging and identify areas where healthcare services for older adults can be improved."),
        html.A("Access the 'National Poll on Healthy Aging (NPHA)' dataset here.", href="https://archive.ics.uci.edu/dataset/936/national+poll+on+healthy+aging+(npha)", target="_blank"),
        html.P("The Paper Reviews dataset, available at the UCI Machine Learning Repository, offers a deep dive into the world of academic paper reviews. This dataset includes detailed information about the review process of papers submitted to a scientific conference. It contains data about the papers themselves, the reviewers' feedback, and the final decision made by the review committee. By applying machine learning techniques to this dataset, we can uncover patterns and trends that influence the acceptance or rejection of a paper. This can provide valuable insights to authors aiming to increase their chances of having their work accepted in future submissions."),
        html.A("Access the 'Paper Reviews' dataset here.", href="https://archive.ics.uci.edu/dataset/410/paper+reviews", target="_blank")
    ], style={'textAlign': 'center', 'margin': '50px', 'color': 'white'}),
    html.Div(id='page-content')
])

# Page 1
fig1 = px.bar(df1['Target'])
fig1.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='white'
)
fig2 = px.histogram(df1[['Previous qualification', 'Mother\'s qualification', 'Father\'s qualification']])
fig2.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='white'
)
page_1_layout = html.Div([
    html.H3('Predicting students dropout based on background', style={'textAlign': 'center', 'color': 'white'}),
    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df1.columns],
        data=df1.to_dict('records'),
        page_size=10,
        style_cell={'backgroundColor': '#1e2130', 'color': 'white'},
    ),
    dcc.Graph(
        id='bar1',
        figure=fig1
    ),
    dcc.Graph(
        id='bar2',
        figure=fig2
    ),
    html.H4('Model Used: Logistic Regression', style={'color': 'white'}),
    html.H4('Accuracy: 47.89%', style={'color': 'white'}),
    dcc.Link('Go back to home', href='/', style={'color': 'white'}),
], style={'backgroundColor': '#060606'})

# Page 2
fig1 = px.box(df2['Phyiscal Health'])
fig1.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='white'
)
fig2 = px.line(df2['Number of Doctors Visited'])
fig2.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='white'
)
fig2.update_xaxes(range=[0, 100])

page_2_layout = html.Div([
    html.H3('Predicting Physical Health after Doctor visits', style={'textAlign': 'center', 'color': 'white'}),
    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df2.columns],
        data=df2.to_dict('records'),
        page_size=10,
        style_cell={'backgroundColor': '#1e2130', 'color': 'white'},
    ),
    dcc.Graph(
        id='box1',
        figure=fig1,
        style={'margin': '50px'}
    ),
    dcc.Graph(
        id='box2',
        figure=fig2,
        style={'margin': '50px'}
    ),
    html.H4('Model Used: Linear Regression', style={'color': 'white'}),
    html.H4('Accuracy: 76.26%', style={'color': 'white'}),
    dcc.Link('Go back to home', href='/', style={'color': 'white'}),
], style={'backgroundColor': '#060606'})

# Page 3
fig1 = px.bar(df3['confidence'])
fig1.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='white'
)
fig2 = px.bar(df3['paper_paper.preliminary_decision'])
fig2.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='white'
)

page_3_layout = html.Div([
    html.H3('Predicting Paper Acceptence', style={'textAlign': 'center', 'color': 'white'}),
    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df3.columns],
        data=df3.to_dict('records'),
        page_size=10,
        style_cell={'backgroundColor': '#1e2130', 'color': 'white'},
    ),
    dcc.Graph(
        id='bar1',
        figure=fig1
    ),
    dcc.Graph(
        id='bar2',
        figure=fig2
    ),
    html.H4('Model Used: Random Forest Classifier', style={'color': 'white'}),
    html.H4('Accuracy: 67.47%', style={'color': 'white'}),
    
    dcc.Link('Go back to home', href='/', style={'color': 'white'}),
], style={'backgroundColor': '#060606'})

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return page_1_layout
    elif pathname == '/page-2':
        return page_2_layout
    elif pathname == '/page-3':
        return page_3_layout
    else:
        return index_page

if __name__ == '__main__':
    app.run_server(debug=True)