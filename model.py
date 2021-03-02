import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import scipy.stats as stats
import random
import pickle
import math
import warnings
warnings.filterwarnings('ignore')

pd.options.display.float_format = '{:.2f}'.format


@st.cache(allow_output_mutation=True)
def load_pickle():
    return pickle.load(open("EDA.pickle", 'rb'))

@st.cache(allow_output_mutation=True)
def load_data():
    return pd.read_csv("adult.csv",na_values='?')

@st.cache(allow_output_mutation=True)
def plot_prob_cutof(cutof_df,data_type,metric):
    algo_list = cutof_df.Algorithm.unique().tolist()
    fig = go.Figure()
    color = ['#2E4053','#EDBB99','#7DCEA0','#F1948A','#5DADE2','#D35400','#A93226','#229954','#626567'
                 ,'#1A5276','#145A32','#0B5345','#7D6608','#4D5656','#AF601A','#626567'
                 ,'#512E5F','#641E16','#17202A']
    random_color = random.sample(color,len(algo_list))
    
    for i,alg in enumerate(algo_list):
        df = cutof_df[(cutof_df.Data==data_type) & (cutof_df.Algorithm==alg) ]
        fig.add_trace(go.Scatter(x=df.prob
                                    , y=df[metric]
                             , name=alg,
                         line=dict(color=random_color[i], width=2)))
    fig.update_layout(width=800,height=600)
    # fig.show()
    return(fig)


EDA_Pickle = pickle.load(open("EDA.pickle", 'rb'))

# Data
# df = pd.read_csv("adult.csv",na_values='?')
df = pd.read_csv("Update_adult.csv",na_values='?')
cat_cols = df.select_dtypes(exclude=np.number).columns
num_cols = df.select_dtypes(include=np.number).columns

# Sidebar
st.sidebar.header("**Walkthrough**")
# data = st.sidebar.button("Data")
# eda = st.sidebar.button("EDA")
# viz = st.sidebar.button("Visualization")

flow=st.sidebar.selectbox("",['Overview',"Data","EDA","Statistical Tests","Model Building","Metrics"])

# Main Area

if flow=='Overview':
    st.header("Overview")
    st.image(
        "main.gif")

if flow=='Data':
    st.header("Dataset")
    st.write("""
    **Rows:** """ + str(df.shape[0])+""" **Attributes:** """+str(df.shape[1]))
    st.dataframe(df)
    # st.dataframe(df.style.set_caption('Top 10 Fields of Research by Aggregated Funding Amount'))
    # st.dataframe(df.style.highlight_max(axis=0))

if flow=='EDA':
    st.header("EDA and Visualization")
    eda=st.beta_expander("EDA")
    c1,c2=eda.beta_columns((1,1))
    c1.header("Info")
    c1.dataframe(EDA_Pickle['df_head'],height=225)
    c2.header("Describe")
    c2.dataframe(EDA_Pickle['df_summary'],height=225)
    vizs=st.beta_expander("Visualization")

    c1,c2,c3,c4,c5,c6,c7,c8=vizs.beta_columns((8))
    b1=c1.button("Frequency by Income");b2=c2.button("Gender vs Income Sankey");b3=c3.button("Workclass vs Income");b4=c4.button("Gen vs Workc vs Income")
    b5=c5.button("Education vs Income");b6=c6.button("Marital Status vs Income");b7=c7.button("Occ vs Rel vs Race vs Income");b8=c8.button("Box Plots for Numerical")
    # for viz in EDA_Pickle.keys():
    #     vizs.header("")
    #     vizs.write(EDA_Pickle[viz])

    if b1:
        vizs.header("Frequency by Income")
        vizs.plotly_chart(EDA_Pickle['target_count_fig'])
    if b2:
        vizs.header("Gender vs Income")
        vizs.plotly_chart(EDA_Pickle['income_vs_gender_fig'])
    if b3:
        vizs.header("Workclass vs Income")
        vizs.plotly_chart(EDA_Pickle['income_vs_workclass_fig'])
    if b4:
        vizs.header("Gender vs Workclass vs Income")
        vizs.plotly_chart(EDA_Pickle['Workclass_vs_Gender_vs_Income_group'])
    if b5:
        vizs.header("Education vs Income")
        vizs.plotly_chart(EDA_Pickle['education_vs_Income'])
    if b6:
        vizs.header("Marital Status vs Income")
        vizs.plotly_chart(EDA_Pickle['marital_status_income'])
    if b7:
        vizs.header("Occupation vs Relationship vs Race vs Income")
        vizs.plotly_chart(EDA_Pickle['Occ_vs_Relationship_vs_Race_vs_Income_Group'])
    if b8:
        vizs.header("Box Plots")
        vizs.plotly_chart(EDA_Pickle['num_var_boxplot'])

if flow=='Statistical Tests':
    st.header("Statistical Tests")
    chi_square_cols=df[cat_cols]
    anova_cols=df[num_cols]
    chi_square=st.beta_expander("Chi-Square Test")
    ## ChiSquare Indep
    chi_square.markdown("""**Null hypothesis** : There is no significant difference between the observed and expected frequencies from the expected distribution.  
    **Alternative hypothesis**: There is a significant difference between the observed and expected frequencies from the expected distribution.""")
    select_chi_square=chi_square.selectbox("ChiSquare Columns",cat_cols)
    values_data=df[select_chi_square].value_counts().sort_values(ascending=True)
    plotly_data=[
        go.Bar(
            y=values_data.keys(),
            x=values_data,
            orientation='h',
            text=values_data,
            textposition='auto'
            )
        ]
    layout = go.Layout(
    height=500,
    # width=1000,
    title=select_chi_square+" vs Counts",
    hovermode='closest',
    xaxis=dict(title='Count', ticklen=5, zeroline=False, gridwidth=2, domain=[0.1, 1]),
    yaxis=dict(title=select_chi_square, ticklen=5, gridwidth=2),
    showlegend=False
    )
    fig = go.Figure(data=plotly_data, layout=layout)
    chi_square.plotly_chart(fig,use_container_width=True)

    chi_square.dataframe(values_data)
    test_statistic_indep,p_val_indep=stats.chisquare(values_data)
    chi_square.write("chi-squared test statistic is **"+str(round(test_statistic_indep,2))+"** and p-values is **"+str(round(p_val_indep,2))+"**")

    contingency=st.beta_expander("Chi-Square Contingent Test")
    ## ChiSquare Contingency
    c1,c2=contingency.beta_columns((1,1))
    chi_square_var1=c1.selectbox("Chisquare First Variable",cat_cols)
    chi_square_var2=c2.selectbox("Chisquare Second Variable",cat_cols[~(cat_cols==chi_square_var1)])

    contingency.markdown("""**Null hypothesis** : Assumes that there is no association between the two variables.  
    **Alternative hypothesis**: There is a significant difference between the observed and expected frequencies from the expected distribution""")
    values_cont=pd.crosstab(df[chi_square_var1],df[chi_square_var2])
    contingency.dataframe(values_cont)
    # contingency.write(df[chi_square_var1].unique())
    pickle.dump(values_cont,open("values count",'wb'))
    plotly_data_cont=[
        go.Heatmap(
            z=values_cont,
            y=values_cont.index,
            x=values_cont.columns,
            hoverongaps = False,
            text=values_cont
            )
        ]
    layout = go.Layout(
        height=500,
        title=chi_square_var1+" vs "+chi_square_var2,
        hovermode='closest',
        xaxis=dict(title=chi_square_var1,zeroline=False),
        yaxis=dict(title=chi_square_var2,gridwidth=2),
        showlegend=False
        )
    fig=go.Figure(data=plotly_data_cont,layout=layout)
    # fig = ff.create_annotated_heatmap(z=np.array(values_cont),
    #         y=values_cont.index.tolist(),
    #         x=values_cont.columns.tolist(),colorscale='Viridis')
    contingency.plotly_chart(fig,use_container_width=True)
    test_statistic_cont,p_val_cont,dof_cont,expected_cont=stats.chi2_contingency(values_cont.values)
    contingency.write("chi-squared test statistic is **"+str(round(test_statistic_cont,2))+"** , p-values is **"+str(round(p_val_cont,2))+"** and dof are **"+str(round(dof_cont,2))+"**")


    # f_oneway=st.beta_expander("Oneway Anova")
    # two_oneway=st.beta_expander("Twoway Anova")
    
if flow=="Metrics":

    df_cutoff=pd.read_csv('cutoffdata.csv')
    cutoff_expander=st.beta_expander("Cutoff Chart")
    algs=df_cutoff['Algorithm'].unique().tolist()
    # st.write(algs)
    c1,c2,c3=cutoff_expander.beta_columns(3)
    alg=c1.multiselect("Algorithm",algs)
    dt=c2.selectbox("Data Type",["Train","Test"])
    met=c3.selectbox("Metric",df_cutoff.columns.tolist()[2:-2])
    formatted_data=df_cutoff[(df_cutoff['Algorithm'].isin(alg))&(df_cutoff['Data']==dt)]
    cutoff_expander.plotly_chart(plot_prob_cutof(formatted_data,dt,met),use_container_width=True)
    
    

