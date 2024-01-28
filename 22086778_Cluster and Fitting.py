# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 00:14:59 2024

@author: Administrator
"""

import pandas as pd
import numpy as np
import sklearn.cluster as cluster
import sklearn.metrics as skmet
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from scipy import stats
from scipy.stats import norm, t
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from lmfit import Model
import seaborn as sns
import matplotlib.pyplot as plt
import cluster_tools as ct
import scipy.optimize as opt
import itertools as iter
from prettytable import PrettyTable
def read_data(filename, skiprows_start=1, skiprows_end=0, **others):
    """
    A function that read GDP per person employed data and returns the dataset with the first rows skipped
    and the zero rows skipped from the end.

    Parameters:
        filename (str): The name of the world bank data file.
        skiprows_start (int): The number of rows to skip at the beginning.
        skiprows_end (int): The number of rows to skip at the end.
        **others: Additional arguments to pass into the function.

    Returns:
        pd.DataFrame: The dataset with the specified number of rows skipped.
    """
    world_data = pd.read_csv(filename, **others)
    
    # Determine the number of rows to keep from the end
    rows_to_keep = world_data.shape[0] - skiprows_end
    
    # Keep only the specified number of rows from the end
    world_data = world_data.iloc[:rows_to_keep]
    
    return world_data

# Read data with 4 rows skipped at the beginning and 5 rows skipped at the end
emp = read_data(r'C:\Users\Administrator\Downloads\P_Data_Extract_From_World_Development_Indicators\70ae2986-e744-4634-9ff2-bb69e39f3d47_GDP_per_person_employed.csv', skiprows_start=1, skiprows_end=5)
emp
emp.describe
emp.corr
# Dropping the columns not needed
emp = emp.drop(
    ['Country Code', 'Series Name', 'Series Code'], axis=1)

# reseting the index
emp.reset_index(drop=True, inplace=True)
# # Assuming net_mig is your DataFrame
# emp_corr_matrix = emp.head(14).corr()

# # Create a heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(emp_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
# plt.title('Correlation Matrix for the First 14 Rows')
# plt.show()
emp
# Assuming emp is your DataFrame
years = ['1995 [YR1995]', '2000 [YR2000]', '2005 [YR2005]', '2010 [YR2010]', '2015 [YR2015]', '2020 [YR2020]']

# Extracting the relevant columns
emp = emp[['Country Name'] + years]

# Plotting
plt.figure(figsize=(12, 8))

for index, row in emp.head(15).iterrows():
    plt.plot(years, row[years], label=row['Country Name'])

plt.title('GDP per person employed (constant 2017 PPP $) Trends (1995 - 2020)')
plt.xlabel('Year')
plt.ylabel('GDP per person employed (constant 2017 PPP $)')
plt.legend()
plt.show()
# Assuming emp is your DataFrame
years = ['1995 [YR1995]', '2000 [YR2000]', '2005 [YR2005]', '2010 [YR2010]', '2015 [YR2015]', '2020 [YR2020]']

# Extracting the relevant columns
emp = emp[['Country Name'] + years]

# Transpose the DataFrame for horizontal bar plotting
emp_T = emp.set_index('Country Name').T

# Plotting
plt.figure(figsize=(12, 8))

for country in emp_T.columns:
    plt.barh(years, emp_T[country], label=country)

plt.title('GDP per person employed (constant 2017 PPP $) Trends (1995 - 2020)')
plt.xlabel('GDP per person employed (constant 2017 PPP $)')
plt.ylabel('Year')
plt.legend()
plt.show()
# Assuming Employee is your DataFrame
years = ['1995 [YR1995]', '2000 [YR2000]', '2005 [YR2005]', '2010 [YR2010]', '2015 [YR2015]', '2020 [YR2020]']

# Extracting the relevant columns
emp = emp[['Country Name'] + years]

# Plotting a bar chart
plt.figure(figsize=(15, 8))

# Set up positions for the bars
bar_width = 0.2
bar_positions = np.arange(len(emp))

# Plot bars for each year
for i, year in enumerate(years):
    plt.bar(bar_positions + i * bar_width, emp[year], width=bar_width, label=f'{year[-5:-1]}')

plt.xlabel('Country')
plt.ylabel('GDP per person employed')
plt.title('GDP per person employed (constant 2017 PPP $) Trends (1995 - 2020) by Country')
plt.xticks(bar_positions + (len(years) - 1) * bar_width / 2, emp['Country Name'], rotation=45, ha='right')
plt.legend()
plt.show()
# Assuming emp is your DataFrame
years = ['1995 [YR1995]', '2000 [YR2000]', '2005 [YR2005]', '2010 [YR2010]', '2015 [YR2015]', '2020 [YR2020]']

# Extracting the relevant columns
emp = emp[['Country Name'] + years]

# Normalizing the data
scaler = StandardScaler()
emp_normalized = scaler.fit_transform(emp[years])

# Perform k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
emp['Cluster'] = kmeans.fit_predict(emp_normalized)

# Plotting the clusters and cluster centers
plt.figure(figsize=(12, 8))

for cluster in emp['Cluster'].unique():
    cluster_data = emp[emp['Cluster'] == cluster]
    plt.scatter(cluster_data.index, cluster_data[years[0]], label=f'Cluster {cluster}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', marker='d', label='Cluster Centers')

plt.title('K-Means Clustering of GDP per Person Employed (1995 - 2020)')
plt.xlabel('Country Index')
plt.ylabel('Normalized GDP per Person Employed')
plt.legend()
plt.grid(True)
plt.show()
# Normalizing the data
scaler = StandardScaler()
emp_normalized = scaler.fit_transform(emp[years])

# Perform k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
emp['Cluster'] = kmeans.fit_predict(emp_normalized)

# Calculate Silhouette Score
silhouette_avg = silhouette_score(emp_normalized, emp['Cluster'])
print(f'Silhouette Score: {silhouette_avg}')
emp
# Sample data (replace with your DataFrame)
data = {
    'Country Name': ['Australia', 'Belgium', 'Brazil', 'Cameroon', 'China', 'France', 'Germany', 'Ghana', 'Kenya', 'Mexico', 'Nigeria', 'Pakistan', 'United Kingdom', 'India', 'South Africa'],
    '1995 [YR1995]': [72141.689560, 100540.537487, 27836.163744, 6658.371000, 4361.126876, 89259.672021, 88658.630251, 6406.512169, 9249.627692, 46093.048435, 8896.340610, 12352.796678, 72380.185889, 6549.205613, 32604.708267],
    '2000 [YR2000]': [81896.719610, 106909.377131, 28563.016115, 7079.613234, 6154.548337, 97732.821627, 96141.452302, 7137.213310, 8684.297727, 50826.635996, 9038.228342, 11673.798183, 81658.982918, 7503.693224, 33624.772927],
    '2005 [YR2005]': [86790.171587, 114633.003455, 28859.991748, 7415.354888, 9467.675565, 99085.531009, 98939.692364, 7481.420757, 8573.277083, 48536.553225, 12119.862540, 12989.733117, 87887.941410, 9667.414049, 36621.289998],
    '2010 [YR2010]': [89456.170337, 116456.368831, 33189.413524, 8030.485897, 16082.516907, 100174.376950, 98485.623530, 9064.117160, 9199.205595, 47453.457978, 14999.525990, 13260.736896, 88714.988909, 12165.809235, 45419.970062],
    '2015 [YR2015]': [95996.584232, 122103.851146, 33253.581833, 9165.001761, 23378.569321, 105513.505129, 102705.094673, 11700.369297, 9865.568383, 48367.076426, 18023.143459, 14591.719210, 91332.697430, 16309.920500, 43513.663955],
    '2020 [YR2020]': [98186.647394, 115922.472123, 34726.800805, 9344.587867, 32213.118660, 102387.610131, 102961.620827, 12816.167906, 10663.231530, 46264.881242, 15710.173590, 16703.463953, 84947.633610, 19312.598568, 43627.616865],
}

df = pd.DataFrame(data)

# Extracting only the numeric columns
numeric_data = df.iloc[:, 1:]

# Normalize the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(numeric_data)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(normalized_data)

# Plotting
plt.figure(figsize=(12, 8))
for cluster in df['Cluster'].unique():
    plt.scatter(df[df['Cluster'] == cluster].index, df[df['Cluster'] == cluster]['2020 [YR2020]'], label=f'Cluster {cluster}')

# Plotting cluster centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)
for i, center in enumerate(centers):
    plt.scatter(len(df) + i, center[-1], marker='d', s=100, label=f'Cluster {i} Center', edgecolors='black')

plt.title('Cluster Membership and Cluster Centers')
plt.xlabel('Country Index')
plt.ylabel('GDP per person employed (constant 2017 PPP $)')
plt.grid(True)
plt.legend()
plt.show()
emp
# Sample data (replace with your DataFrame)
data = {
    'Year': [1995, 2000, 2005, 2010, 2015, 2020],
    'Australia': [72141.689560, 81896.719610, 86790.171587, 89456.170337, 95996.584232, 98186.647394],
    'Belgium': [100540.537487, 106909.377131, 114633.003455, 116456.368831, 122103.851146, 115922.472123],
    'Brazil': [27836.163744, 28563.016115, 28859.991748, 33189.413524, 33253.581833, 34726.800805],
    'Cameroon': [6658.371000, 7079.613234, 7415.354888, 8030.485897, 9165.001761, 9344.587867],
    'China': [4361.126876, 6154.548337, 9467.675565, 16082.516907, 23378.569321, 32213.118660],
    'France': [89259.672021, 97732.821627, 99085.531009, 100174.376950, 105513.505129, 102387.610131],
    'Germany': [88658.630251, 96141.452302, 98939.692364, 98485.623530, 102705.094673, 102961.620827],
    'Ghana': [6406.512169, 7137.213310, 7481.420757, 9064.117160, 11700.369297, 12816.167906],
    'Kenya': [9249.627692, 8684.297727, 8573.277083, 9199.205595, 9865.568383, 10663.231530],
    'Mexico': [46093.048435, 50826.635996, 48536.553225, 47453.457978, 48367.076426, 46264.881242],
    'Nigeria': [8896.340610, 9038.228342, 12119.862540, 14999.525990, 18023.143459, 15710.173590],
    'Pakistan': [12352.796678, 11673.798183, 12989.733117, 13260.736896, 14591.719210, 16703.463953],
    'United Kingdom': [72380.185889, 81658.982918, 87887.941410, 88714.988909, 91332.697430, 84947.633610],
    'India': [6549.205613, 7503.693224, 9667.414049, 12165.809235, 16309.920500, 19312.598568],
    'South Africa': [32604.708267, 33624.772927, 36621.289998, 45419.970062, 43513.663955, 43627.616865],
}

urban_data = pd.DataFrame(data)

# Fitting for Australia on Exponential Growth
def exponential(t, a, b):
    """Computes exponential growth of urban population

    Parameters:
        t: The current time
        a: The initial population
        b: The growth rate

    Returns:
        The population at the given time
    """
    f = a * np.exp(b * t)
    return f

def err_range(x, func, param, sigma):
    """Calculates the error range for a given function and its parameters

    Parameters:
        x: The input value for the function
        func: The function for which the error ranges will be calculated
        param: The parameters for the function
        sigma: The standard deviation of the data

    Returns:
        The lower and upper error ranges
    """
    lower = func(x, *param)
    upper = lower

    for i, p in enumerate(param):
        pmin = p - sigma[i]
        pmax = p + sigma[i]
        y = func(x, *param[:i], pmin, *param[i+1:])
        lower = np.minimum(lower, y)
        y = func(x, *param[:i], pmax, *param[i+1:])
        upper = np.maximum(upper, y)

    return lower, upper

def confidence_interval(data, mean, confidence=0.95):
    """Calculates the confidence interval for a dataset

    Parameters:
        data: The dataset
        mean: The mean of the dataset
        confidence: The desired confidence level (default is 0.95)

    Returns:
        The lower and upper bounds of the confidence interval
    """
    n = len(data)
    h = np.std(data) * t.ppf((1 + confidence) / 2, n - 1)
    return mean - h, mean + h

country_name = 'Australia'
years = urban_data['Year'].values
population = urban_data[country_name].values

# Check for NaN values
nan_indices = np.isnan(population)

# Remove NaN values
population = population[~nan_indices]
years = years[~nan_indices]

# Provide initial guess for exponential function
# You can adjust the initial guess if needed
initial_guess = [min(population), 0.01]

popt, pcov = curve_fit(exponential, years, population, p0=initial_guess, maxfev=5000)  # Increase maxfev

# Calculate confidence interval for the fitted parameters
alpha = 0.95  # desired confidence level
critical_value = t.ppf(1 - (1 - alpha) / 2, len(years) - len(popt))

# Calculate confidence interval for each parameter
param_confidence_interval = critical_value * np.sqrt(np.diag(pcov))

# Display the confidence intervals
for i, param in enumerate(popt):
    lower_bound = param - param_confidence_interval[i]
    upper_bound = param + param_confidence_interval[i]
    print(f'Parameter {i}: {param:.4f}, 95% Confidence Interval: ({lower_bound:.4f}, {upper_bound:.4f})')

prediction_2030 = exponential(2030, *popt)
prediction_2040 = exponential(2040, *popt)

print(f"GDP per person employed prediction for 2030 in {country_name}: {prediction_2030:.2f}")
print(f"GDP per person employed prediction for 2040 in {country_name}: {prediction_2040:.2f}")

# Generate points for the fitted curve
curve_years = np.linspace(min(years), max(years), 100)
predicted_years = np.array([2030, 2040])  # Add the predicted years here
curve_years_extended = np.concatenate((curve_years, predicted_years))

curve_population = exponential(curve_years_extended, *popt)

# Calculate error range
sigma = np.sqrt(np.diag(pcov))
lower, upper = err_range(curve_years_extended, exponential, popt, sigma)

# Plot the data, fitted curve, and confidence interval
plt.figure(dpi=100)
plt.plot(years, population, 'ro', label='Data')
plt.plot(curve_years_extended, curve_population, 'b-', label='Fitted Curve')
plt.fill_between(curve_years_extended, lower, upper, color='lightblue', label='95% Confidence Interval')
plt.fill_between(curve_years_extended, lower, upper, color='yellow', alpha=0.3, label='Error Range')
plt.xlabel('Year')
plt.ylabel('GDP per person employed')
plt.title(f'Exponential Growth Fit for GDP per person employed in {country_name}')
plt.legend()
plt.grid(True)
plt.show()
# Fitting for Australia on Exponential Growth
def exponential(t, a, b):
    """Computes exponential growth of urban population

    Parameters:
        t: The current time
        a: The initial population
        b: The growth rate

    Returns:
        The population at the given time
    """
    f = a * np.exp(b * t)
    return f

def err_range(x, func, param, sigma):
    """Calculates the error range for a given function and its parameters

    Parameters:
        x: The input value for the function
        func: The function for which the error ranges will be calculated
        param: The parameters for the function
        sigma: The standard deviation of the data

    Returns:
        The lower and upper error ranges
    """
    lower = func(x, *param)
    upper = lower

    for i, p in enumerate(param):
        pmin = p - sigma[i]
        pmax = p + sigma[i]
        y = func(x, *param[:i], pmin, *param[i+1:])
        lower = np.minimum(lower, y)
        y = func(x, *param[:i], pmax, *param[i+1:])
        upper = np.maximum(upper, y)

    return lower, upper

country_name = 'Australia'
years = urban_data['Year'].values
population = urban_data[country_name].values

# Check for NaN values
nan_indices = np.isnan(population)

# Remove NaN values
population = population[~nan_indices]
years = years[~nan_indices]

# Provide initial guess for exponential function
# You can adjust the initial guess if needed
initial_guess = [min(population), 0.01]

popt, pcov = curve_fit(exponential, years, population, p0=initial_guess, maxfev=5000)  # Increase maxfev

prediction_2030 = exponential(2030, *popt)
prediction_2040 = exponential(2040, *popt)

print(f"GDP per person employed prediction for 2030 in {country_name}: {prediction_2030:.2f}")
print(f"GDP per person employed prediction for 2040 in {country_name}: {prediction_2040:.2f}")

# Generate points for the fitted curve
curve_years = np.linspace(min(years), max(years), 100)
predicted_years = np.array([2030, 2040])  # Add the predicted years here
curve_years_extended = np.concatenate((curve_years, predicted_years))

curve_population = exponential(curve_years_extended, *popt)

# Calculate error range
sigma = np.sqrt(np.diag(pcov))
lower, upper = err_range(curve_years_extended, exponential, popt, sigma)

# Plot the data, fitted curve, and error range
plt.figure(dpi=100)
plt.plot(years, population, 'ro', label='Data')
plt.plot(curve_years_extended, curve_population, 'b-', label='Fitted Curve')
plt.fill_between(curve_years_extended, lower, upper, color='yellow', alpha=0.3, label='Error Range')
plt.xlabel('Year')
plt.ylabel('GDP per person employed')
plt.title(f'Exponential Growth Fit for GDP per person employed in {country_name}')
plt.legend()
plt.grid(True)
plt.show()
# Fitting for Nigeria on Exponential Growth
def exponential(t, a, b):
    """Computes exponential growth of urban population

    Parameters:
        t: The current time
        a: The initial population
        b: The growth rate

    Returns:
        The population at the given time
    """
    f = a * np.exp(b * t)
    return f

def err_range(x, func, param, sigma):
    """Calculates the error range for a given function and its parameters

    Parameters:
        x: The input value for the function
        func: The function for which the error ranges will be calculated
        param: The parameters for the function
        sigma: The standard deviation of the data

    Returns:
        The lower and upper error ranges
    """
    lower = func(x, *param)
    upper = lower

    for i, p in enumerate(param):
        pmin = p - sigma[i]
        pmax = p + sigma[i]
        y = func(x, *param[:i], pmin, *param[i+1:])
        lower = np.minimum(lower, y)
        y = func(x, *param[:i], pmax, *param[i+1:])
        upper = np.maximum(upper, y)

    return lower, upper

def confidence_interval(data, mean, confidence=0.95):
    """Calculates the confidence interval for a dataset

    Parameters:
        data: The dataset
        mean: The mean of the dataset
        confidence: The desired confidence level (default is 0.95)

    Returns:
        The lower and upper bounds of the confidence interval
    """
    n = len(data)
    h = np.std(data) * t.ppf((1 + confidence) / 2, n - 1)
    return mean - h, mean + h

country_name = 'Nigeria'
years = urban_data['Year'].values
population = urban_data[country_name].values

# Check for NaN values
nan_indices = np.isnan(population)

# Remove NaN values
population = population[~nan_indices]
years = years[~nan_indices]

# Provide initial guess for exponential function
# You can adjust the initial guess if needed
initial_guess = [min(population), 0.01]

popt, pcov = curve_fit(exponential, years, population, p0=initial_guess, maxfev=5000)  # Increase maxfev

# Calculate confidence interval for the fitted parameters
alpha = 0.95  # desired confidence level
critical_value = t.ppf(1 - (1 - alpha) / 2, len(years) - len(popt))

# Calculate confidence interval for each parameter
param_confidence_interval = critical_value * np.sqrt(np.diag(pcov))

# Display the confidence intervals
for i, param in enumerate(popt):
    lower_bound = param - param_confidence_interval[i]
    upper_bound = param + param_confidence_interval[i]
    print(f'Parameter {i}: {param:.4f}, 95% Confidence Interval: ({lower_bound:.4f}, {upper_bound:.4f})')

prediction_2030 = exponential(2030, *popt)
prediction_2040 = exponential(2040, *popt)

print(f"GDP per person employed prediction for 2030 in {country_name}: {prediction_2030:.2f}")
print(f"GDP per person employed prediction for 2040 in {country_name}: {prediction_2040:.2f}")

# Generate points for the fitted curve
curve_years = np.linspace(min(years), max(years), 100)
predicted_years = np.array([2030, 2040])  # Add the predicted years here
curve_years_extended = np.concatenate((curve_years, predicted_years))

curve_population = exponential(curve_years_extended, *popt)

# Calculate error range
sigma = np.sqrt(np.diag(pcov))
lower, upper = err_range(curve_years_extended, exponential, popt, sigma)

# Plot the data, fitted curve, and confidence interval
plt.figure(dpi=100)
plt.plot(years, population, 'ro', label='Data')
plt.plot(curve_years_extended, curve_population, 'b-', label='Fitted Curve')
plt.fill_between(curve_years_extended, lower, upper, color='lightblue', label='95% Confidence Interval')
plt.fill_between(curve_years_extended, lower, upper, color='yellow', alpha=0.3, label='Error Range')
plt.xlabel('Year')
plt.ylabel('GDP per person employed')
plt.title(f'Exponential Growth Fit for GDP per person employed in {country_name}')
plt.legend()
plt.grid(True)
plt.show()
# Extracting the years and population for the United Kingdom
years = urban_data['Year'].values
population_uk = urban_data['United Kingdom'].values

# Check for NaN values
nan_indices = np.isnan(population_uk)

# Remove NaN values
population_uk = population_uk[~nan_indices]
years = years[~nan_indices]

# Define the logistic growth function
def logistic_growth(t, a, b, c):
    """
    Computes logistic growth of urban population

    Parameters:
        t: The current time
        a: The carrying capacity
        b: The growth rate
        c: The time of maximum growth

    Returns:
        The population at the given time
    """
    x = -b * (t - c)
    exp_term = np.exp(np.clip(x, -700, 700))
    f = a / (1 + exp_term)
    return f

# Define the error range function
def err_range(x, func, param, sigma):
    """
    Calculates the error range for a given function and its parameters

    Parameters:
        x: The input value for the function
        func: The function for which the error ranges will be calculated
        param: The parameters for the function
        sigma: The standard deviation of the data

    Returns:
        The lower and upper error ranges
    """
    lower = func(x, *param)
    upper = lower

    for i, p in enumerate(param):
        pmin = p - sigma[i]
        pmax = p + sigma[i]
        y = func(x, *param[:i], pmin, *param[i+1:])
        lower = np.minimum(lower, y)
        y = func(x, *param[:i], pmax, *param[i+1:])
        upper = np.maximum(upper, y)

    return lower, upper

# Provide initial guess for logistic function
initial_guess_logistic = [max(population_uk), 1, np.median(years)]

# Curve fitting for logistic function
popt_logistic, pcov_logistic = curve_fit(logistic_growth, years, population_uk,
                                          p0=initial_guess_logistic, maxfev=7000)

# Generate points for the fitted curve
curve_years = np.linspace(min(years), max(years), 100)
curve_population_logistic = logistic_growth(curve_years, *popt_logistic)

# Add predictions for 2030 and 2040 to the extended curve_years
predicted_years = np.array([2030, 2040])
curve_years_extended = np.concatenate((curve_years, predicted_years))

# Calculate population values for the extended time period
curve_population_extended = logistic_growth(curve_years_extended, *popt_logistic)

# Calculate error range
sigma_logistic = np.sqrt(np.diag(pcov_logistic))
lower_logistic, upper_logistic = err_range(curve_years_extended, logistic_growth, popt_logistic, sigma_logistic)

# Print the fitted logistic growth parameters
print("Fitted Logistic Growth Parameters:")
print("Carrying Capacity:", popt_logistic[0])
print("Growth Rate:", popt_logistic[1])
print("Time of Maximum Growth:", popt_logistic[2])

# Print the predicted population for 2030 and 2040
prediction_2030_logistic = logistic_growth(2030, *popt_logistic)
prediction_2040_logistic = logistic_growth(2040, *popt_logistic)
print("GDP per person employed prediction for 2030:", prediction_2030_logistic)
print("GDP per person employed prediction for 2040:", prediction_2040_logistic)

# # Plot the data, fitted curve, and error range for logistic function
plt.figure(dpi=100)
plt.plot(years, population_uk, 'ro', label='Data')
plt.plot(curve_years_extended, curve_population_extended, 'b-', label='Fitted Logistic Curve')
plt.fill_between(curve_years_extended, lower_logistic, upper_logistic, color='lightblue', label='Confidence Interval (Logistic)')
plt.fill_between(curve_years_extended, lower_logistic, upper_logistic,color='yellow', alpha=0.3, label='Error Range (Logistic)')
plt.xlabel('Year')
plt.ylabel('GDP per person employed')
plt.title('Logistic Growth Fit for GDP per person employed in United Kingdom')
plt.legend()
plt.grid(True)
plt.show()
# Extracting the years and population for the China
years = urban_data['Year'].values
population_prc = urban_data['China'].values

# Check for NaN values
nan_indices = np.isnan(population_prc)

# Remove NaN values
population_prc = population_prc[~nan_indices]
years = years[~nan_indices]

# Define the logistic growth function
def logistic_growth(t, a, b, c):
    """
    Computes logistic growth of urban population

    Parameters:
        t: The current time
        a: The carrying capacity
        b: The growth rate
        c: The time of maximum growth

    Returns:
        The population at the given time
    """
    x = -b * (t - c)
    exp_term = np.exp(np.clip(x, -700, 700))
    f = a / (1 + exp_term)
    return f

# Define the error range function
def err_range(x, func, param, sigma):
    """
    Calculates the error range for a given function and its parameters

    Parameters:
        x: The input value for the function
        func: The function for which the error ranges will be calculated
        param: The parameters for the function
        sigma: The standard deviation of the data

    Returns:
        The lower and upper error ranges
    """
    lower = func(x, *param)
    upper = lower

    for i, p in enumerate(param):
        pmin = p - sigma[i]
        pmax = p + sigma[i]
        y = func(x, *param[:i], pmin, *param[i+1:])
        lower = np.minimum(lower, y)
        y = func(x, *param[:i], pmax, *param[i+1:])
        upper = np.maximum(upper, y)

    return lower, upper

# Provide initial guess for logistic function
initial_guess_logistic = [max(population_prc), 1, np.median(years)]

# Curve fitting for logistic function
popt_logistic, pcov_logistic = curve_fit(logistic_growth, years, population_prc,
                                          p0=initial_guess_logistic, maxfev=7000)

# Generate points for the fitted curve
curve_years = np.linspace(min(years), max(years), 100)
curve_population_logistic = logistic_growth(curve_years, *popt_logistic)

# Add predictions for 2030 and 2040 to the extended curve_years
predicted_years = np.array([2030, 2040])
curve_years_extended = np.concatenate((curve_years, predicted_years))

# Calculate population values for the extended time period
curve_population_extended = logistic_growth(curve_years_extended, *popt_logistic)

# Calculate error range
sigma_logistic = np.sqrt(np.diag(pcov_logistic))
lower_logistic, upper_logistic = err_range(curve_years_extended, logistic_growth, popt_logistic, sigma_logistic)

# Print the fitted logistic growth parameters
print("Fitted Logistic Growth Parameters:")
print("Carrying Capacity:", popt_logistic[0])
print("Growth Rate:", popt_logistic[1])
print("Time of Maximum Growth:", popt_logistic[2])

# Print the predicted population for 2030 and 2040
prediction_2030_logistic = logistic_growth(2030, *popt_logistic)
prediction_2040_logistic = logistic_growth(2040, *popt_logistic)
print("GDP per person employed prediction for 2030:", prediction_2030_logistic)
print("GDP per person employed prediction for 2040:", prediction_2040_logistic)

# # Plot the data, fitted curve, and error range for logistic function
plt.figure(dpi=100)
plt.plot(years, population_prc, 'ro', label='Data')
plt.plot(curve_years_extended, curve_population_extended, 'b-', label='Fitted Logistic Curve')
plt.fill_between(curve_years_extended, lower_logistic, upper_logistic, color='lightblue', label='Confidence Interval (Logistic)')
plt.fill_between(curve_years_extended, lower_logistic, upper_logistic,color='yellow', alpha=0.3, label='Error Range (Logistic)')
plt.xlabel('Year')
plt.ylabel('GDP per person employed')
plt.title('Logistic Growth Fit for GDP per person employed in China')
plt.legend()
plt.grid(True)
plt.show()
# Fitting for India on Low-order Polynomial
def polynomial(t, a, b, c):
    """Computes a low-order polynomial fit for GDP per person employed

    Parameters:
        t: The current time
        a, b, c: Coefficients of the polynomial

    Returns:
        The predicted GDP per person employed at the given time
    """
    f = a * t**2 + b * t + c
    return f

def confidence_interval(data, mean, confidence=0.95):
    """Calculates the confidence interval for a dataset

    Parameters:
        data: The dataset
        mean: The mean of the dataset
        confidence: The desired confidence level (default is 0.95)

    Returns:
        The lower and upper bounds of the confidence interval
    """
    n = len(data)
    h = np.std(data) * t.ppf((1 + confidence) / 2, n - 1)
    return mean - h, mean + h

# Select India data
country_name = 'India'
years = urban_data['Year'].values
population = urban_data[country_name].values

# Check for NaN values
nan_indices = np.isnan(population)

# Remove NaN values
population = population[~nan_indices]
years = years[~nan_indices]

# Provide initial guess for polynomial function
# You can adjust the initial guess if needed
initial_guess = [0, 0, 0]

# Fit the polynomial function to the data
popt, pcov = curve_fit(polynomial, years, population, p0=initial_guess)

# Generate points for the fitted curve
curve_years = np.linspace(min(years), max(years), 100)
predicted_years = np.array([2030, 2040])  # Add the predicted years here
curve_years_extended = np.concatenate((curve_years, predicted_years))

curve_population = polynomial(curve_years_extended, *popt)

# Calculate confidence interval
lower_ci, upper_ci = confidence_interval(population, np.mean(population))

# Plot the data, fitted curve, and confidence interval
plt.figure(dpi=100)
plt.plot(years, population, 'ro', label='Data')
plt.plot(curve_years_extended, curve_population, 'b-', label='Fitted Curve')
plt.fill_between(curve_years_extended, lower_ci, upper_ci, color='yellow', alpha=0.3, label='95% Confidence Interval')
plt.xlabel('Year')
plt.ylabel('GDP per person employed')
plt.title(f'Low-order Polynomial Fit with Confidence Interval for GDP per person employed in {country_name}')
plt.legend()
plt.grid(True)
plt.show()
# Fitting for Germany on Low-order Polynomial
def polynomial(t, a, b, c):
    """Computes a low-order polynomial fit for GDP per person employed

    Parameters:
        t: The current time
        a, b, c: Coefficients of the polynomial

    Returns:
        The predicted GDP per person employed at the given time
    """
    f = a * t**2 + b * t + c
    return f

def confidence_interval(data, mean, confidence=0.95):
    """Calculates the confidence interval for a dataset

    Parameters:
        data: The dataset
        mean: The mean of the dataset
        confidence: The desired confidence level (default is 0.95)

    Returns:
        The lower and upper bounds of the confidence interval
    """
    n = len(data)
    h = np.std(data) * t.ppf((1 + confidence) / 2, n - 1)
    return mean - h, mean + h

# Select India data
country_name = 'Germany'
years = urban_data['Year'].values
population = urban_data[country_name].values

# Check for NaN values
nan_indices = np.isnan(population)

# Remove NaN values
population = population[~nan_indices]
years = years[~nan_indices]

# Provide initial guess for polynomial function
# You can adjust the initial guess if needed
initial_guess = [0, 0, 0]

# Fit the polynomial function to the data
popt, pcov = curve_fit(polynomial, years, population, p0=initial_guess)

# Generate points for the fitted curve
curve_years = np.linspace(min(years), max(years), 100)
predicted_years = np.array([2030, 2040])  # Add the predicted years here
curve_years_extended = np.concatenate((curve_years, predicted_years))

curve_population = polynomial(curve_years_extended, *popt)

# Calculate confidence interval
lower_ci, upper_ci = confidence_interval(population, np.mean(population))

# Plot the data, fitted curve, and confidence interval
plt.figure(dpi=100)
plt.plot(years, population, 'ro', label='Data')
plt.plot(curve_years_extended, curve_population, 'b-', label='Fitted Curve')
plt.fill_between(curve_years_extended, lower_ci, upper_ci, color='yellow', alpha=0.3, label='95% Confidence Interval')
plt.xlabel('Year')
plt.ylabel('GDP per person employed')
plt.title(f'Low-order Polynomial Fit with Confidence Interval for GDP per person employed in {country_name}')
plt.legend()
plt.grid(True)
plt.show()
# # Sample data (replace with your DataFrame)
# data = {
#     'Year': [1995, 2000, 2005, 2010, 2015, 2020],
#     'Germany': [88658.630251, 96141.452302, 98939.692364, 98485.623530, 102705.094673, 102961.620827],
# }

# urban_data = pd.DataFrame(data)

# Fitting for Germany on Low-order Polynomial
def polynomial(t, a, b, c):
    """Computes a low-order polynomial fit for GDP per person employed

    Parameters:
        t: The current time
        a, b, c: Coefficients of the polynomial

    Returns:
        The predicted GDP per person employed at the given time
    """
    f = a * t**2 + b * t + c
    return f

def confidence_interval(data, mean, confidence=0.95):
    """Calculates the confidence interval for a dataset

    Parameters:
        data: The dataset
        mean: The mean of the dataset
        confidence: The desired confidence level (default is 0.95)

    Returns:
        The lower and upper bounds of the confidence interval
    """
    n = len(data)
    h = np.std(data) * t.ppf((1 + confidence) / 2, n - 1)
    return mean - h, mean + h

# Select Germany data
country_name = 'Germany'
years = urban_data['Year'].values
population_germany = urban_data[country_name].values

# Check for NaN values
nan_indices = np.isnan(population_germany)

# Remove NaN values
population_germany = population_germany[~nan_indices]
years = years[~nan_indices]

# Provide initial guess for polynomial function
# You can adjust the initial guess if needed
initial_guess = [0, 0, 0]

# Fit the polynomial function to the data
popt_germany, pcov_germany = curve_fit(polynomial, years, population_germany, p0=initial_guess)

# Generate points for the fitted curve
curve_years_germany = np.linspace(min(years), max(years), 100)
predicted_years_germany = np.array([2030, 2040])  # Add the predicted years here
curve_years_extended_germany = np.concatenate((curve_years_germany, predicted_years_germany))

curve_population_germany = polynomial(curve_years_extended_germany, *popt_germany)

# Calculate confidence interval
lower_ci_germany, upper_ci_germany = confidence_interval(population_germany, np.mean(population_germany))

# Plot the data, fitted curve, and confidence interval
plt.figure(dpi=100)
plt.plot(years, population_germany, 'ro', label='Data')
plt.plot(curve_years_extended_germany, curve_population_germany, 'b-', label='Fitted Curve')
plt.fill_between(curve_years_extended_germany, lower_ci_germany, upper_ci_germany,
                 color='yellow', alpha=0.3, label='95% Confidence Interval')
plt.xlabel('Year')
plt.ylabel('GDP per person employed')
plt.title(f'Low-order Polynomial Fit with Confidence Interval for GDP per person employed in {country_name}')
plt.legend()
plt.grid(True)
plt.show()
# Fitting for Germany on Low-order Polynomial
def polynomial(t, a, b, c):
    """Computes a low-order polynomial fit for GDP per person employed

    Parameters:
        t: The current time
        a, b, c: Coefficients of the polynomial

    Returns:
        The predicted GDP per person employed at the given time
    """
    f = a * t**2 + b * t + c
    return f

# Select Germany data
country_name = 'Germany'
years = urban_data['Year'].values
population_germany = urban_data[country_name].values

# Check for NaN values
nan_indices = np.isnan(population_germany)

# Remove NaN values
population_germany = population_germany[~nan_indices]
years = years[~nan_indices]

# Provide initial guess for polynomial function
# You can adjust the initial guess if needed
initial_guess = [0, 0, 0]

# Fit the polynomial function to the data
popt_germany, pcov_germany = curve_fit(polynomial, years, population_germany, p0=initial_guess)

# Generate points for the fitted curve
curve_years_germany = np.linspace(min(years), max(years), 100)
predicted_years_germany = np.array([2030, 2040])  # Add the predicted years here
curve_years_extended_germany = np.concatenate((curve_years_germany, predicted_years_germany))

# Calculate error range (let's assume a fixed percentage error for illustration)
error_percentage = 5
error_range_lower_germany = polynomial(curve_years_extended_germany, *popt_germany) * (1 - error_percentage / 100)
error_range_upper_germany = polynomial(curve_years_extended_germany, *popt_germany) * (1 + error_percentage / 100)

# Plot the data, fitted curve, and error range
plt.figure(dpi=100)
plt.plot(years, population_germany, 'ro', label='Data')
plt.plot(curve_years_extended_germany, polynomial(curve_years_extended_germany, *popt_germany), 'b-', label='Fitted Curve')
plt.fill_between(curve_years_extended_germany, error_range_lower_germany, error_range_upper_germany,
                 color='orange', alpha=0.3, label=f'{error_percentage}% Error Range')
plt.xlabel('Year')
plt.ylabel('GDP per person employed')
plt.title(f'Low-order Polynomial Fit with Error Range for GDP per person employed in {country_name}')
plt.legend()
plt.grid(True)
plt.show()
# Sample data DataFrame
data = {
    'Year': [1995, 2000, 2005, 2010, 2015, 2020],
    'Cameroon': [6658.371000, 7079.613234, 7415.354888, 8030.485897, 9165.001761, 9344.587867],
    'Pakistan': [12352.796678, 11673.798183, 12989.733117, 13260.736896, 14591.719210, 16703.463953],
}

urban_data = pd.DataFrame(data)

# Fitting for Exponential Growth
def exponential(t, a, b):
    """Computes exponential growth

    Parameters:
        t: The current time
        a: The initial population
        b: The growth rate

    Returns:
        The population at the given time
    """
    f = a * np.exp(b * t)
    return f

def calculate_growth_difference(country1, country2):
    years = urban_data['Year'].values
    population_country1 = urban_data[country1].values
    population_country2 = urban_data[country2].values

    # Check for NaN values
    nan_indices1 = np.isnan(population_country1)
    nan_indices2 = np.isnan(population_country2)

    # Remove NaN values
    population_country1 = population_country1[~nan_indices1]
    population_country2 = population_country2[~nan_indices2]
    years = years[~nan_indices1]

    # Provide initial guess for exponential function
    initial_guess = [min(population_country1), 0.01]

    # Fit the exponential function to the data for both countries
    popt_country1, _ = curve_fit(exponential, years, population_country1, p0=initial_guess, maxfev=5000)  # Increase maxfev
    popt_country2, _ = curve_fit(exponential, years, population_country2, p0=initial_guess, maxfev=5000)  # Increase maxfev

    # Calculate growth rate (b) for both countries
    growth_rate_country1 = popt_country1[1]
    growth_rate_country2 = popt_country2[1]

    # Calculate the difference in growth rates
    growth_difference = growth_rate_country2 - growth_rate_country1

    # Plot the exponential growth curves
    plt.figure(dpi=100)
    plt.plot(years, population_country1, 'ro', label=f'{country1} Data')
    plt.plot(years, population_country2, 'bo', label=f'{country2} Data')
    
    curve_years = np.linspace(min(years), max(years), 100)
    curve_country1 = exponential(curve_years, *popt_country1)
    curve_country2 = exponential(curve_years, *popt_country2)

    plt.plot(curve_years, curve_country1, 'r-', label=f'{country1} Exponential Fit')
    plt.plot(curve_years, curve_country2, 'b-', label=f'{country2} Exponential Fit')

    plt.xlabel('Year')
    plt.ylabel('Population')
    plt.title(f'Exponential Growth Fit for {country1} and {country2}')
    plt.legend()
    plt.grid(True)
    plt.show()

    return growth_difference

# Calculate the exponential growth difference between Cameroon and Pakistan and plot the curves
difference = calculate_growth_difference('Cameroon', 'Pakistan')
print(f"The exponential growth difference between Cameroon and Pakistan is: {difference:.5f}")
# Sample data DataFrame
data = {
    'Year': [1995, 2000, 2005, 2010, 2015, 2020],
    'Cameroon': [6658.371000, 7079.613234, 7415.354888, 8030.485897, 9165.001761, 9344.587867],
    'Pakistan': [12352.796678, 11673.798183, 12989.733117, 13260.736896, 14591.719210, 16703.463953],
}

urban_data = pd.DataFrame(data)

# Fitting for Exponential Growth
def exponential(t, a, b):
    """Computes exponential growth

    Parameters:
        t: The current time
        a: The initial population
        b: The growth rate

    Returns:
        The population at the given time
    """
    f = a * np.exp(b * t)
    return f

def err_range(x, func, param, sigma):
    """Calculates the error range for a given function and its parameters

    Parameters:
        x: The input value for the function
        func: The function for which the error ranges will be calculated
        param: The parameters for the function
        sigma: The standard deviation of the data

    Returns:
        The lower and upper error ranges
    """
    lower = func(x, *param)
    upper = lower

    for i, p in enumerate(param):
        pmin = p - sigma[i]
        pmax = p + sigma[i]
        y = func(x, *param[:i], pmin, *param[i+1:])
        lower = np.minimum(lower, y)
        y = func(x, *param[:i], pmax, *param[i+1:])
        upper = np.maximum(upper, y)

    return lower, upper

def calculate_growth_difference(country1, country2):
    years = urban_data['Year'].values
    population_country1 = urban_data[country1].values
    population_country2 = urban_data[country2].values

    # Check for NaN values
    nan_indices1 = np.isnan(population_country1)
    nan_indices2 = np.isnan(population_country2)

    # Remove NaN values
    population_country1 = population_country1[~nan_indices1]
    population_country2 = population_country2[~nan_indices2]
    years = years[~nan_indices1]

    # Provide initial guess for exponential function
    initial_guess = [min(population_country1), 0.01]

    # Fit the exponential function to the data for both countries
    popt_country1, pcov_country1 = curve_fit(exponential, years, population_country1, p0=initial_guess, maxfev=5000)  # Increase maxfev
    popt_country2, pcov_country2 = curve_fit(exponential, years, population_country2, p0=initial_guess, maxfev=5000)  # Increase maxfev

    # Calculate error range
    sigma_country1 = np.sqrt(np.diag(pcov_country1))
    sigma_country2 = np.sqrt(np.diag(pcov_country2))

    # Generate points for the fitted curve
    curve_years = np.linspace(min(years), max(years), 100)
    predicted_years = np.array([2030, 2040])  # Add the predicted years here
    curve_years_extended = np.concatenate((curve_years, predicted_years))

    curve_country1 = exponential(curve_years_extended, *popt_country1)
    curve_country2 = exponential(curve_years_extended, *popt_country2)

    # Calculate error range
    lower_country1, upper_country1 = err_range(curve_years_extended, exponential, popt_country1, sigma_country1)
    lower_country2, upper_country2 = err_range(curve_years_extended, exponential, popt_country2, sigma_country2)
    
    print('EXPONENTIAL GROWTH DIFFERENCE BETWEEN CAMEROON & PAKISTAN')

    # Plot the exponential growth curves with error range
    plt.figure(figsize=(12, 6))

    # Plot for Cameroon
    plt.subplot(1, 2, 1)
    plt.plot(years, population_country1, 'ro', label=f'{country1} Data')
    plt.plot(curve_years_extended, curve_country1, 'r-', label=f'{country1} Exponential Fit')
    plt.fill_between(curve_years_extended, lower_country1, upper_country1, color='yellow', alpha=0.3, label=f'{country1} Error Range')
    plt.xlabel('Year')
    plt.ylabel('Population')
    plt.title(f'Exponential Growth Fit for {country1}')
    plt.legend()
    plt.grid(True)

    # Plot for Pakistan
    plt.subplot(1, 2, 2)
    plt.plot(years, population_country2, 'bo', label=f'{country2} Data')
    plt.plot(curve_years_extended, curve_country2, 'b-', label=f'{country2} Exponential Fit')
    plt.fill_between(curve_years_extended, lower_country2, upper_country2, color='green', alpha=0.3, label=f'{country2} Error Range')
    plt.xlabel('Year')
    plt.ylabel('Population')
    plt.title(f'Exponential Growth Fit for {country2}')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

calculate_growth_difference('Cameroon', 'Pakistan')
# Sample data DataFrame
data = {
    'Year': [1995, 2000, 2005, 2010, 2015, 2020],
    'Belgium': [100540.537487, 106909.377131, 114633.003455, 116456.368831, 122103.851146, 115922.472123],
    'South Africa': [32604.708267, 33624.772927, 36621.289998, 45419.970062, 43513.663955, 43627.616865],
}

urban_data = pd.DataFrame(data)

# Fitting for Belgium and South Africa on Logistic Growth
def logistic_growth(t, a, b, c):
    x = -b * (t - c)
    exp_term = np.exp(np.clip(x, -700, 700))
    f = a / (1 + exp_term)
    return f

# Function to calculate error range
def err_range(x, func, param, sigma):
    lower = func(x, *param)
    upper = lower

    for i, p in enumerate(param):
        pmin = p - sigma[i]
        pmax = p + sigma[i]
        y = func(x, *param[:i], pmin, *param[i+1:])
        lower = np.minimum(lower, y)
        y = func(x, *param[:i], pmax, *param[i+1:])
        upper = np.maximum(upper, y)

    return lower, upper

# Select Belgium data
country_name_belgium = 'Belgium'
years_belgium = urban_data['Year'].values
population_belgium = urban_data[country_name_belgium].values

# Check for NaN values
nan_indices_belgium = np.isnan(population_belgium)

# Remove NaN values
population_belgium = population_belgium[~nan_indices_belgium]
years_belgium = years_belgium[~nan_indices_belgium]

# Provide initial guess for logistic function
initial_guess_logistic_belgium = [max(population_belgium), 1, np.median(years_belgium)]

# Curve fitting for logistic function for Belgium
popt_logistic_belgium, pcov_logistic_belgium = curve_fit(logistic_growth, years_belgium, population_belgium,
                                                        p0=initial_guess_logistic_belgium, maxfev=7000)

# Generate points for the fitted curve
curve_years_belgium = np.linspace(min(years_belgium), max(years_belgium), 100)
curve_population_logistic_belgium = logistic_growth(curve_years_belgium, *popt_logistic_belgium)

# Add predictions for 2030 and 2040 to the extended curve_years
predicted_years_belgium = np.array([2030, 2040])
curve_years_extended_belgium = np.concatenate((curve_years_belgium, predicted_years_belgium))

# Calculate population values for the extended time period
curve_population_extended_belgium = logistic_growth(curve_years_extended_belgium, *popt_logistic_belgium)

# Calculate error range
sigma_logistic_belgium = np.sqrt(np.diag(pcov_logistic_belgium))
lower_logistic_belgium, upper_logistic_belgium = err_range(curve_years_extended_belgium,
                                                           logistic_growth, popt_logistic_belgium, sigma_logistic_belgium)

print('LOGISTIC GROWTH DIFFERENCE BETWEEN BELGIUM & SOUTH AFRICA ')

# Print the fitted logistic growth parameters for Belgium
print("Fitted Logistic Growth Parameters for Belgium:")
print("Carrying Capacity:", popt_logistic_belgium[0])
print("Growth Rate:", popt_logistic_belgium[1])
print("Time of Maximum Growth:", popt_logistic_belgium[2])

# Plot the data, fitted curve, and error range for logistic function for Belgium
plt.figure(dpi=100)
plt.plot(years_belgium, population_belgium, 'ro', label=f'Data ({country_name_belgium})')
plt.plot(curve_years_extended_belgium, curve_population_extended_belgium, 'b-',
         label=f'Fitted Logistic Curve ({country_name_belgium})')
plt.fill_between(curve_years_extended_belgium, lower_logistic_belgium, upper_logistic_belgium,
                 color='yellow', alpha=0.3, label=f'Error Range (Logistic) - {country_name_belgium}')
plt.xlabel('Year')
plt.ylabel('GDP per person employed')
plt.title(f'Logistic Growth Fit for GDP per person employed in {country_name_belgium}')
plt.legend()
plt.grid(True)
plt.show()


# Select South Africa data
country_name_sa = 'South Africa'
years_sa = urban_data['Year'].values
population_sa = urban_data[country_name_sa].values

# Check for NaN values
nan_indices_sa = np.isnan(population_sa)

# Remove NaN values
population_sa = population_sa[~nan_indices_sa]
years_sa = years_sa[~nan_indices_sa]

# Provide initial guess for logistic function
initial_guess_logistic_sa = [max(population_sa), 1, np.median(years_sa)]

# Curve fitting for logistic function for South Africa
popt_logistic_sa, pcov_logistic_sa = curve_fit(logistic_growth, years_sa, population_sa,
                                                p0=initial_guess_logistic_sa, maxfev=7000)

# Generate points for the fitted curve
curve_years_sa = np.linspace(min(years_sa), max(years_sa), 100)
curve_population_logistic_sa = logistic_growth(curve_years_sa, *popt_logistic_sa)

# Add predictions for 2030 and 2040 to the extended curve_years
predicted_years_sa = np.array([2030, 2040])
curve_years_extended_sa = np.concatenate((curve_years_sa, predicted_years_sa))

# Calculate population values for the extended time period
curve_population_extended_sa = logistic_growth(curve_years_extended_sa, *popt_logistic_sa)

# Calculate error range
sigma_logistic_sa = np.sqrt(np.diag(pcov_logistic_sa))
lower_logistic_sa, upper_logistic_sa = err_range(curve_years_extended_sa,
                                                 logistic_growth, popt_logistic_sa, sigma_logistic_sa)

# Print the fitted logistic growth parameters for South Africa
print("\nFitted Logistic Growth Parameters for South Africa:")
print("Carrying Capacity:", popt_logistic_sa[0])
print("Growth Rate:", popt_logistic_sa[1])
print("Time of Maximum Growth:", popt_logistic_sa[2])

# Plot the data, fitted curve, and error range for logistic function for South Africa
plt.figure(dpi=100)
plt.plot(years_sa, population_sa, 'ro', label=f'Data ({country_name_sa})')
plt.plot(curve_years_extended_sa, curve_population_extended_sa, 'b-',
         label=f'Fitted Logistic Curve ({country_name_sa})')
plt.fill_between(curve_years_extended_sa, lower_logistic_sa, upper_logistic_sa,
                 color='green', alpha=0.3, label=f'Error Range (Logistic) - {country_name_sa}')
plt.xlabel('Year')
plt.ylabel('GDP per person employed')
plt.title(f'Logistic Growth Fit for GDP per person employed in {country_name_sa}')
plt.legend()
plt.grid(True)
plt.show()