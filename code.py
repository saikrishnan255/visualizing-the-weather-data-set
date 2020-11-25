# --------------
# Import the required Libraries
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import calendar
import seaborn as sns
import warnings
import os
from os.path import dirname, abspath
warnings.filterwarnings("ignore")

weather_df = pd.read_csv(path, index_col='Date/Time')

# Generate a line chart that visualizes the readings in the months
def line_chart(df,period,col):
    """ A line chart that visualizes the readings in the months
    
    This function accepts the dataframe df ,period(day/month/year) and col(feature), which plots the aggregated value of the feature based on the periods. Ensure the period labels are properly named.
    
    Keyword arguments:
    df - Pandas dataframe which has the data.
    period - Period of time over which you want to aggregate the data
    col - Feature of the dataframe
    
    """ 
        # Converting date/time to datetime format
    df.reset_index(inplace = True)
    df['Month'] = (pd.to_datetime(df[period])).dt.month
            
        # Grouping the data for plotting
    data = df.groupby(['Month'])[[col]].mean()
        
        # Resetting index for ease of plotting data
    data.reset_index(inplace = True)
        
        # Plotting the data
    plt.figure(figsize = [6,6])
    plt.title(f'{col} trend over time')
    plt.xlabel('Month')
    plt.plot(data['Month'],data[col])
    return (plt.show())

print ('*'*100)

print(line_chart(weather_df, 'Date/Time', 'Temp (C)'))

weather_df.drop('Month', inplace = True, axis = 1)

weather_df.set_index('Date/Time', inplace = True)

# Function to perform univariate analysis of categorical columns
def plot_categorical_columns(df):
    """ Univariate analysis of categorical columns
    
    This function accepts the dataframe df which analyzes all the variable in the data and performs the univariate analysis using bar plot.
    
    Keyword arguments:
    df - Pandas dataframe which has the data.
    
    """
    categorical = df.select_dtypes(exclude = 'number')
    cat = list(categorical.columns)

    for i in cat:
        plt.figure(figsize = (16,14))
        sns.countplot(categorical[i])
        plt.xticks(rotation = 90)
        plt.show()
        
print('*'*100)

print(plot_categorical_columns(weather_df))
    
# Function to plot continous plots
def plot_cont(df,plt_typ):
    """ Univariate analysis of Numerical columns
    
    This function accepts the dataframe df, plt_type(boxplot/distplot) which analyzes all the variable in the data and performs the univariate analysis using boxplot or distplot plot.
    
    Keyword arguments:
    df - Pandas dataframe which has the data.
    plt_type - type of plot through which you want to visualize the data
    
    """
    # Storing numerical variables in numerical
    numerical = df.select_dtypes(include = 'number')
    
    num = list(numerical.columns)
    type = {'boxplot': sns.boxplot, 'distplot':sns.distplot}
    
    for i in num:
        type[plt_typ](numerical[i])
        plt.show()

print ('*'*100)

print(plot_cont(weather_df,'distplot'))

def group_values(df,col1,agg1,col2):
    
    agg = {'mean':np.mean,'max':np.max,'min':np.min,'sum':np.sum,'len':len}
    
    data = (df.groupby([col1])[[col2]].agg(agg1))
    data.reset_index(inplace = True)

    plt.figure(figsize = (20,10))
    plt.bar(data[col1], data[col2])
    plt.xticks(rotation = 90)
    plt.show()
    
(group_values(weather_df,'Weather','mean','Visibility (km)'))


