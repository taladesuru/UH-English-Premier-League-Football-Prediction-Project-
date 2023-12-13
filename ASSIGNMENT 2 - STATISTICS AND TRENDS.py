# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 23:10:09 2023

@author: Administrator
"""

#import PANDA, NUMPY, MATPLOTLIB, SEABORN AND STATS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import stats
import seaborn as sns

"""Defining a function that returns 2 dataframes after taking a filename as argument and reading a dataframe in World-
bank format; each using years and countries as columns
"""

def read_worldbank_data(filename):
    # Read CSV file into dataframe
    df = pd.read_csv(filename)

    # Extract relevant columns
    df = df[['Series Name', 'Country Name', '2000 [YR2000]', '2005 [YR2005]', '2010 [YR2010]', '2015 [YR2015]', '2020 [YR2020]']]

    # Pivot years as columns with aggregation
    df_yrs = df.pivot_table(index='Country Name', columns='Series Name', values=['2000 [YR2000]', '2005 [YR2005]', '2010 [YR2010]', '2015 [YR2015]', '2020 [YR2020]'], aggfunc='mean')

    # transposed dataframe
    df_yrs.columns = df_yrs.columns.droplevel(0)
    df_yrs.columns.name = None

    # Pivot countries as columns with aggregation
    df_coun = df.pivot_table(index='Series Name', columns='Country Name', values=['2000 [YR2000]', '2005 [YR2005]', '2010 [YR2010]', '2015 [YR2015]', '2020 [YR2020]'], aggfunc='mean')

    # transposed dataframe
    df_coun.columns = df_coun.columns.droplevel(0)
    df_coun.columns.name = None

    return df_yrs, df_coun

# Filename Path
filename = r'C:\Users\Administrator\Downloads\AssignmentADS\Total Population\World_Bank_Development_Indicators.csv'
df_yrs, df_coun = read_worldbank_data(filename)

print("Year Column as Dataframe :")
print(df_yrs.head())

print("\nCountries Column as DataFrame:")
print(df_coun.head())


# Exploration of Annual Population growth, GDP growth and current health expenditure per capita for analysis
series_interest = ['Population growth (annual %)', 'GDP growth (annual %)', 'Current health expenditure per capita (current US$)']

# Subset selected series
selec_series = df_yrs[series_interest]

# Applying the .describe() method for selected series
descr = selec_series.describe()

# Applying other statistical methods (Using the median and standard deviation statistics)
med_stats = selec_series.median()
std_dev_stats = selec_series.std()

# Print the results
print("Summary Statistics using .describe():")
print(descr)

print("\nMedian Statistics:")
print(med_stats)

print("\nStandard Deviation Statistics:")
print(std_dev_stats)


# Information inside years
df_yrs.head()



# Information inside countries
df_coun.head()


# Swap rows and columns in DataFrame using transpose method
df_trans = df_yrs.T

# Print tarnsposed dataframe to check the structure
print(df_trans.head())

# Applying .describe() method for summary statistics
descr_stats = df_trans.describe()

# Applying other statistical methods
median_stats = df_trans.median()
std_dev_stats = df_trans.std()

# Print the results
print("Summary Statistics using .describe():")
print(descr_stats)

print("\nMedian Statistics:")
print(median_stats)

print("\nStandard Deviation Statistics:")
print(std_dev_stats)


# Print df_yrs column dataframe to check the structure
print(df_yrs.columns)


# Print df_coun column dataframe to check the structure
print(df_coun.columns)


# Select series for analysis
series_interest = ['Population growth (annual %)', 'GDP growth (annual %)', 'Current health expenditure per capita (current US$)']

# Exploring the Correlations Between series
# Subset the selected series
selec_series = df_yrs[series_interest]

# Correlation matrix
corr_matrix = selec_series.corr()

# Correlation matrix visualization using a heatmap plot
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix for Selected Series')
plt.show()


print(df_coun.columns)


print(df_yrs.columns)


df_yrs.head()


# Plot Population growth and GDP growth using Scatter Plot
plt.figure(figsize=(10, 8))
plt.scatter(df_yrs['Population growth (annual %)'], df_yrs['GDP growth (annual %)'])
plt.title('Scatter Plot: Population growth vs. GDP growth')
plt.xlabel('Population growth (annual %)', color= 'green')
plt.ylabel('GDP growth (annual %)', color= 'blue')
plt.show()



# Reset Country Name Column
df_yrs_reset = df_yrs.reset_index()

# Select series for time-series analysis
series_interest = ['Population growth (annual %)', 'GDP growth (annual %)']

# Subset the selected series
time_series = df_yrs_reset[['Country Name'] + series_interest]

# Melt DataFrame for time-series analysis
time_series_melted = pd.melt(time_series, id_vars=['Country Name'], var_name='Indicator', value_name='Value')

# Plotting time-series for selected Series
plt.figure(figsize=(14, 8))

# Plotting Population growth (annual %) for histogram and density
plt.subplot(2, 1, 1)
sns.histplot(time_series_melted[time_series_melted['Indicator'] == 'Population growth (annual %)']['Value'], kde=True)
plt.title('Histogram and Density Plot - Population growth (annual %)')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Plotting GDP growth (annual %) for histogram and density
plt.subplot(2, 1, 2)
sns.histplot(time_series_melted[time_series_melted['Indicator'] == 'GDP growth (annual %)']['Value'], kde=True)
plt.title('Histogram and Density Plot - GDP growth (annual %)')
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# Reset Country Name Column
df_reset = df_yrs.reset_index()

# Select series for time-series analysis
series_interest = ['Population growth (annual %)', 'GDP growth (annual %)']

# Subset the selected Series using .loc
time_series = df_yrs_reset.loc[:, ['Country Name'] + series_interest]

# Applying Melt method for time-series analysis
time_series_melted = pd.melt(time_series, id_vars=['Country Name'], var_name='Indicator', value_name='Value')

# Plotting time-series for selected Series
plt.figure(figsize=(14, 8))
sns.lineplot(x='Country Name', y='Value', hue='Indicator', data=time_series_melted)
plt.title('Time-series Analysis of Selected Indicators')
plt.xlabel('Country')
plt.ylabel('Value')
plt.show()


# Plotting histograms and density plots for selected Series
plt.figure(figsize=(14, 8))

# Plot Population growth (annual %) Using Histogram and Density Plot
plt.subplot(2, 1, 1)
sns.histplot(time_series_melted[time_series_melted['Indicator'] == 'Population growth (annual %)']['Value'], kde=True)
plt.title('Histogram and Density Plot - Population growth (annual %)')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Plot GDP growth (annual %) histogram and density
plt.subplot(2, 1, 2)
sns.histplot(time_series_melted[time_series_melted['Indicator'] == 'GDP growth (annual %)']['Value'], kde=True)
plt.title('Histogram and Density Plot - GDP growth (annual %)')
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
