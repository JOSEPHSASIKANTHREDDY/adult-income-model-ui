import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import  pickle
import math
import warnings
warnings.filterwarnings('ignore')

EDA_Pickle=pickle.load(open("EDA.pickle",'rb'))

# Data
df = pd.read_csv("adult.csv")
cat_cols=df.select_dtypes(exclude=np.number).columns
num_cols=df.select_dtypes(include=np.number).columns

# Sidebar
st.sidebar.header("**Walkthrough**")
data=st.sidebar.button("Data")
eda=st.sidebar.button("EDA")
# st.sidebar.button("Visualization")

# Main Area
## DataTab
# data_expander=st.beta_expander("Dataset")

# data_expander.header("Dataset")
# data_expander.write("""
#  **Rows:** """ + str(df.shape[0])+""" **Attributes:** """+str(df.shape[1]))

# data_expander.dataframe(df)

eda_expander=st.beta_expander("EDA")
eda_expander.write(df.describe())

c1, c2, c3 = eda_expander.beta_columns((1, 1, 1))

a=[ c1.write(df[v].unique()) if i%3==0 else ( c2.write(df[v].unique()) if i%3==1  else c3.write(df[v].unique())) for i,v in enumerate(cat_cols)]

## Plotting
plotting_expander=st.beta_expander("Plotting")
plot_col=plotting_expander.selectbox("Select Attributes to Plot",num_cols)
plt.figure(figsize=(15,3))
sns.kdeplot(data=df,x=plot_col)
plotting_expander.pyplot(plt)
plotting_expander.write(df[plot_col].describe())
plotting_expander.write(df[plot_col].skew())
plotting_expander.write(df[plot_col].kurtosis())

#
if data:
    data_expander=st.beta_expander("Dataset")
    data_expander.header("Dataset")
    data_expander.write("""
    **Rows:** """ + str(df.shape[0])+""" **Attributes:** """+str(df.shape[1]))
    data_expander.dataframe(df)