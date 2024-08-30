#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import skew, kurtosis

import random




random.seed(4)

file_path = '/Users/vladelec/Desktop/Exeter /Summer Project/FTT_StandAlone-main 2/Final work /Disseration.xlsx'

Tech_Raw = pd.read_excel(file_path, sheet_name='All data')

print(Tech_Raw.head())

Tech_Raw = Tech_Raw.rename(columns={'Category': 'Cost Factor'})
Tech_Raw['Technology'] = Tech_Raw['Technology'].replace(
    'Utility-scale battery (4h)', 'LiB')


raw_summary = Tech_Raw.describe(include='all')

print(raw_summary)

# Group
category_group = Tech_Raw.groupby('Cost Factor').describe(include='all')
country_group = Tech_Raw.groupby('Country/region').describe(include='all')
technology_group = Tech_Raw.groupby('Technology').describe(include='all')


print("Grouped by Cost Factor:")
print(Tech_Raw)

print("\nGrouped by Country:")
print(Tech_Raw)

print("\nGrouped by Technology:")
print(Tech_Raw)


Tech_Raw['Average'] = Tech_Raw[['Mid', 'Low', 'High']].mean(axis=1)
summary_Tech_Raw = Tech_Raw.groupby(
    ['Country/region', 'Technology', 'Cost Factor']).agg({'Average': 'mean'}).reset_index()
summary_Tech_Raw = summary_Tech_Raw.pivot_table(
    index=['Country/region', 'Technology', 'Cost Factor'], values='Average')
print(summary_Tech_Raw)

inflation = pd.read_excel(file_path, sheet_name='inflation')
categories_to_adjust = [
    'CAPEX',
    'CAPEX(energy)',
    'OPEX',
    'OPEX(energy)',
    'Replacement Cost',
    'Replacement interval'
]

inflation = inflation.dropna(subset=['Inflation conversion'])
inflation = inflation[inflation['Inflation conversion'].str.contains(
    "Source") == False]


def adjust_for_inflation(row, inflation):
    # Identify the relevant inflation factor based on the year
    year = row['Year']

    if year in inflation.columns:
        try:
            inflation_factor = inflation.loc[inflation['Inflation conversion']
                                             == '$1 in 2024', year].values[0]
        except IndexError:
            inflation_factor = 1
    else:
        inflation_factor = 1  

    if row['Source'] != 1 and row['Cost Factor'] in categories_to_adjust:
        row['Mid'] *= inflation_factor
        row['Low'] *= inflation_factor
        row['High'] *= inflation_factor

    return row


Tech_Raw_inflation = Tech_Raw.apply(
    adjust_for_inflation, axis=1, inflation=inflation)

capex_original = Tech_Raw[(Tech_Raw['Source'] != 1)
                          & (Tech_Raw['Cost Factor'] == 'CAPEX')]
capex_inflation_adjusted = Tech_Raw_inflation[(Tech_Raw_inflation['Source'] != 1) & (
    Tech_Raw_inflation['Cost Factor'] == 'CAPEX')]

plt.figure(figsize=(12, 8))

capex_original['Type'] = 'Original'
capex_inflation_adjusted['Type'] = 'Inflation-adjusted'

# Combine the dataframes for plotting
capex_combined = pd.concat([capex_original, capex_inflation_adjusted])

plt.figure(figsize=(14, 8))
sns.barplot(data=capex_combined, x='Year', y='Mid',
            hue='Type', ci=None, palette='viridis', dodge=True)
plt.xlabel('Year')
plt.ylabel('CAPEX ($/kW)')
plt.title('Comparison of Original and Inflation-Adjusted CAPEX Over Years')
plt.legend(title='Type')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

plt.savefig('xx.jpeg', dpi=800)


# Function to clean up the year values from Bloomberg
def clean_year(row):
    if row['Source'] == 1:
        row['Year'] = row['Year'].replace(' (1H)', '').replace(' (2H)', '')
    return row


Tech_Raw_inflation = Tech_Raw_inflation.apply(clean_year, axis=1)



# Time Decay Factor

Tech_Raw_inflation['Year'] = Tech_Raw_inflation['Year'].astype(
    str).str.extract('(\d{4})').astype(int)

current_year = 2024

# A decay rate (lambda)
decay_rate = 0.1

Tech_Raw_inflation['Time Difference'] = current_year - \
    Tech_Raw_inflation['Year']

# Apply the time decay factor using the exponential decay formula
Tech_Raw_inflation['Decay Factor'] = np.exp(
    -decay_rate * Tech_Raw_inflation['Time Difference'])

Tech_Raw_inflation['Adjusted Low'] = Tech_Raw_inflation['Low'] * \
    Tech_Raw_inflation['Decay Factor']
Tech_Raw_inflation['Adjusted Mid'] = Tech_Raw_inflation['Mid'] * \
    Tech_Raw_inflation['Decay Factor']
Tech_Raw_inflation['Adjusted High'] = Tech_Raw_inflation['High'] * \
    Tech_Raw_inflation['Decay Factor']

print(Tech_Raw_inflation[['Country/region', 'Technology', 'Year', 'Low',
      'Mid', 'High', 'Adjusted Low', 'Adjusted Mid', 'Adjusted High']])

Tech_Raw_inflation['Weighted Value'] = np.where(
    Tech_Raw_inflation[['Adjusted Low', 'Adjusted High']].isnull().all(
        axis=1),  # Check if both Adjusted Low and Adjusted High are NaN
    Tech_Raw_inflation['Adjusted Mid'],  # If true, use Adjusted Mid
    (Tech_Raw_inflation['Adjusted Low'] + Tech_Raw_inflation['Adjusted Mid'] + \
     Tech_Raw_inflation['Adjusted High']) / 3  # Otherwise, calculate the average
)


capex_df = Tech_Raw_inflation[Tech_Raw['Cost Factor'] == 'CAPEX'].copy()

# Drop rows with NaN values in the 'Weighted CAPEX' column
capex_df = capex_df.dropna(subset=['Weighted Value'])

summary_results_list = []

for country in capex_df['Country/region'].unique():
    for tech in capex_df['Technology'].unique():
        # Filter the data for the specific country and technology
        data = capex_df[(capex_df['Country/region'] == country)
                        & (capex_df['Technology'] == tech)]

        # Ensure there is enough data to perform regression and no NaN values
        if len(data) > 1 and not data[['Year', 'Weighted Value']].isnull().values.any():
            # Prepare the regression inputs
            X = data[['Year']].values.reshape(-1, 1)
            y = data['Weighted Value'].values
            weights = data['Decay Factor'].values

            # Initialize and fit the linear regression model
            model = LinearRegression()
            model.fit(X, y, sample_weight=weights)

            # Predict the CAPEX value for the most recent year (or alternatively, you could use the average prediction)
            most_recent_year = data['Year'].max()
            predicted_capex = model.predict([[most_recent_year]])[0]

            # Store the summary result in a dictionary
            summary_results_list.append({
                'Country/region': country,
                'Technology': tech,
                'Predicted CAPEX': predicted_capex
            })

summary_results = pd.DataFrame(summary_results_list)

print(summary_results)


variables = [
    'Replacement Cost', 'Replacement interval', 'End-of-life cost',
    'Discount rate', 'Round-trip efficiency', 'Lifetime (100% DoD)',
    'Shelf life', 'Time degradation', 'Cycle degradation',
    'Construction time', 'Discharge time', 'CAGR'
]

summary_results_list = []

for variable in variables:
    for tech in Tech_Raw_inflation['Technology'].unique():
        # Filter the data for the specific variable within the 'Cost Factor' column and for the specific technology
        data = Tech_Raw_inflation[(Tech_Raw_inflation['Cost Factor'] == variable) &
                                  (Tech_Raw_inflation['Technology'] == tech)].dropna(subset=['Year', 'Decay Factor', 'Weighted Value'])

        if len(data) > 1:
            X = data[['Year']].values.reshape(-1, 1)
            # Using 'Mid' as the representative value
            y = data['Weighted Value'].values
            weights = data['Decay Factor'].values

            model = LinearRegression()
            model.fit(X, y, sample_weight=weights)

            # Predict the value for the most recent year
            most_recent_year = data['Year'].max()
            predicted_value = model.predict([[most_recent_year]])[0]
        else:
            print(
                f"No sufficient data for {variable} in {tech}, using average instead.")
            predicted_value = data['Weighted Value'].mean(
            ) if not data['Weighted Value'].isnull().all() else np.nan

        summary_results_list.append({
            'Technology': tech,
            'Variable': variable,
            'Predicted Value': predicted_value
        })

summary_results_variable = pd.DataFrame(summary_results_list)

print(summary_results_variable)


variables_new = [
    'Capacity factor', 'WACC', 'CAPEX', 'CAPEX(energy)', 'OPEX', 'OPEX(energy)'
]

n_bootstrap_samples = 1000

bootstrap_results_list = []

for variable in variables_new:
    for country in Tech_Raw_inflation['Country/region'].unique():
        for tech in Tech_Raw_inflation['Technology'].unique():
            # Filter the data for the specific variable, country, and technology
            data = Tech_Raw_inflation[(Tech_Raw_inflation['Cost Factor'] == variable) &
                                      (Tech_Raw_inflation['Country/region'] == country) &
                                      (Tech_Raw_inflation['Technology'] == tech)].dropna(subset=['Year', 'Decay Factor', 'Weighted Value'])

            if len(data) > 1:

                predictions_list = []


                for i in range(n_bootstrap_samples):
                    # Resample the data with replacement
                    bootstrap_sample = data.sample(
                        frac=1, replace=True, random_state=i)

                    # Prepare the regression inputs
                    X = bootstrap_sample[['Year']].values.reshape(-1, 1)
                    y = bootstrap_sample['Weighted Value'].values
                    weights = bootstrap_sample['Decay Factor'].values

                    # Initialize and fit the linear regression model
                    model = LinearRegression()
                    model.fit(X, y, sample_weight=weights)

                    # Predict the value for the most recent year
                    most_recent_year = data['Year'].max()
                    predicted_value = model.predict([[most_recent_year]])[0]

                    predictions_list.append(predicted_value)

                bootstrap_results_list.append({
                    'Country/region': country,
                    'Technology': tech,
                    'Variable': variable,
                    'Predicted Values': predictions_list
                })

bootstrap_results = pd.DataFrame(bootstrap_results_list)

bootstrap_results['Predicted Mean'] = bootstrap_results['Predicted Values'].apply(
    np.mean)
bootstrap_results['Predicted Lower'] = bootstrap_results['Predicted Values'].apply(
    lambda x: np.percentile(x, 2.5))
bootstrap_results['Predicted Upper'] = bootstrap_results['Predicted Values'].apply(
    lambda x: np.percentile(x, 97.5))

print(bootstrap_results[['Country/region', 'Technology', 'Variable',
      'Predicted Mean', 'Predicted Lower', 'Predicted Upper']])

bootstrap_results.to_csv('bootstrap_results.csv', index=False)

capacity = pd.read_excel(file_path, sheet_name='Capacity per year (MWh)')


capacity_melted = capacity.melt(
    id_vars=['Technology', 'Country'], var_name='Year', value_name='Capacity (MWh)')
capacity_melted['Year'] = pd.to_numeric(
    capacity_melted['Year'], errors='coerce')

capacity_imputed = capacity_melted.copy()
capacity_imputed['Capacity (MWh)'] = capacity_imputed.groupby(['Technology', 'Country'])[
    'Capacity (MWh)'].transform(lambda group: group.interpolate(method='linear'))

# Plotting the imputed time series data
plt.figure(figsize=(14, 8))

# Calculate the standard deviation for each row in the bootstrap results
bootstrap_results['Predicted StdDev'] = bootstrap_results['Predicted Values'].apply(lambda x: np.std(x) if len(x) > 0 else np.nan)

# Filter the data to include only the specified countries and factors
countries_to_include = ["Brazil", "United States", "United Kingdom", "China"]
factors_to_include = ['CAPEX', 'OPEX', 'CAPEX(energy)', 'OPEX(energy)']

filtered_data = bootstrap_results[
    (bootstrap_results['Country/region'].isin(countries_to_include)) &
    (bootstrap_results['Variable'].isin(factors_to_include))
].copy()

# Ensure UK and other countries are included
for country in countries_to_include:
    for tech in bootstrap_results['Technology'].unique():
        for variable in factors_to_include:
            if not ((filtered_data['Country/region'] == country) &
                    (filtered_data['Technology'] == tech) &
                    (filtered_data['Variable'] == variable)).any():
                # If the combination is missing, create an empty row and fill with appropriate data
                empty_row = pd.DataFrame({
                    'Country/region': [country],
                    'Technology': [tech],
                    'Variable': [variable],
                    'Predicted Values': [[]],
                    'Predicted Mean': [np.nan],
                    'Predicted StdDev': [np.nan]
                })
                filtered_data = pd.concat([filtered_data, empty_row], ignore_index=True)

def apply_custom_rules(row):
    if row['Country/region'] == 'United Kingdom' and row['Technology'] == 'PHS':
        if row['Variable'] in ['CAPEX(energy)', 'OPEX(energy)']:
            us_data = bootstrap_results[(bootstrap_results['Country/region'] == 'United States') &
                                        (bootstrap_results['Technology'] == 'PHS') &
                                        (bootstrap_results['Variable'] == row['Variable'])]
            if not us_data.empty:
                row['Predicted Mean'] = us_data['Predicted Mean'].values[0]
                row['Predicted StdDev'] = us_data['Predicted StdDev'].values[0]
    
    # Brazil and PHS technology
    elif row['Country/region'] == 'Brazil' and row['Technology'] == 'PHS':
        if row['Variable'] in ['CAPEX', 'OPEX']:
            global_data = bootstrap_results[(bootstrap_results['Country/region'] == 'Global') &
                                            (bootstrap_results['Technology'] == 'PHS') &
                                            (bootstrap_results['Variable'] == row['Variable'])]
            if not global_data.empty:
                row['Predicted Mean'] = global_data['Predicted Mean'].values[0]
                row['Predicted StdDev'] = global_data['Predicted StdDev'].values[0]
        elif row['Variable'] in ['CAPEX(energy)', 'OPEX(energy)']:
            us_data = bootstrap_results[(bootstrap_results['Country/region'] == 'United States') &
                                        (bootstrap_results['Technology'] == 'PHS') &
                                        (bootstrap_results['Variable'] == row['Variable'])]
            if not us_data.empty:
                row['Predicted Mean'] = us_data['Predicted Mean'].values[0]
                row['Predicted StdDev'] = us_data['Predicted StdDev'].values[0]

    # China and PHS technology
    elif row['Country/region'] == 'China' and row['Technology'] == 'PHS':
        if row['Variable'] in ['CAPEX(energy)', 'OPEX(energy)']:
            us_data = bootstrap_results[(bootstrap_results['Country/region'] == 'United States') &
                                        (bootstrap_results['Technology'] == 'PHS') &
                                        (bootstrap_results['Variable'] == row['Variable'])]
            if not us_data.empty:
                row['Predicted Mean'] = us_data['Predicted Mean'].values[0]
                row['Predicted StdDev'] = us_data['Predicted StdDev'].values[0]

    # LiB technology specific rules
    elif row['Technology'] == 'LiB':
        if row['Country/region'] == 'Brazil' and row['Variable'] == 'CAPEX':
            chile_data = bootstrap_results[(bootstrap_results['Country/region'] == 'Chile') &
                                           (bootstrap_results['Technology'] == 'LiB') &
                                           (bootstrap_results['Variable'] == 'CAPEX')]
            if not chile_data.empty:
                row['Predicted Mean'] = chile_data['Predicted Mean'].values[0]
                row['Predicted StdDev'] = chile_data['Predicted StdDev'].values[0]
        elif row['Country/region'] == 'Brazil' and row['Variable'] == 'CAPEX(energy)':
            global_data = bootstrap_results[(bootstrap_results['Country/region'] == 'Global') &
                                            (bootstrap_results['Technology'] == 'LiB') &
                                            (bootstrap_results['Variable'] == 'CAPEX(energy)')]
            if not global_data.empty:
                row['Predicted Mean'] = global_data['Predicted Mean'].values[0]
                row['Predicted StdDev'] = global_data['Predicted StdDev'].values[0]
        elif row['Variable'] in ['OPEX', 'OPEX(energy)']:
            us_data = bootstrap_results[(bootstrap_results['Country/region'] == 'United States') &
                                        (bootstrap_results['Technology'] == 'LiB') &
                                        (bootstrap_results['Variable'] == row['Variable'])]
            if not us_data.empty:
                row['Predicted Mean'] = us_data['Predicted Mean'].values[0]
                row['Predicted StdDev'] = us_data['Predicted StdDev'].values[0]

    # Fill missing data from Global if still NaN
    if pd.isna(row['Predicted Mean']):
        global_data = bootstrap_results[(bootstrap_results['Country/region'] == 'Global') &
                                        (bootstrap_results['Technology'] == row['Technology']) &
                                        (bootstrap_results['Variable'] == row['Variable'])]
        if not global_data.empty:
            row['Predicted Mean'] = global_data['Predicted Mean'].values[0]
            row['Predicted StdDev'] = global_data['Predicted StdDev'].values[0]

    return row

final_data = filtered_data.apply(apply_custom_rules, axis=1)

final_data = final_data.drop(columns=['Predicted Values','Predicted Lower', 'Predicted Upper'], errors='ignore')

csv_file_path = 'final_adjusted_data_with_stddev.csv'
final_data.to_csv(csv_file_path, index=False)

bootstrap_results['Predicted StdDev'] = bootstrap_results['Predicted Values'].apply(lambda x: np.std(x) if len(x) > 0 else np.nan)
bootstrap_results['Predicted Lower'] = bootstrap_results['Predicted Values'].apply(lambda x: np.percentile(x, 2.5))
bootstrap_results['Predicted Upper'] = bootstrap_results['Predicted Values'].apply(lambda x: np.percentile(x, 97.5))

costs = bootstrap_results['Variable'].unique()

for cost in costs:
    cost_data = bootstrap_results[bootstrap_results['Variable'] == cost]
    
    predicted_values = pd.to_numeric(cost_data['Predicted Values'].explode(), errors='coerce').dropna()

    mean_value = np.mean(predicted_values)
    std_dev = np.std(predicted_values)
    skewness_value = skew(predicted_values)
    kurtosis_value = kurtosis(predicted_values)
    lower_bound = np.percentile(predicted_values, 2.5)
    upper_bound = np.percentile(predicted_values, 97.5)

    print(f"Statistics for {cost}:")
    print(f"Mean: {mean_value}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Skewness: {skewness_value}")
    print(f"Kurtosis: {kurtosis_value}")
    print(f"95% Confidence Interval: [{lower_bound}, {upper_bound}]\n")

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(predicted_values, kde=True)
    plt.title(f'Distribution of Bootstrap Predictions for {cost}')
    plt.xlabel('Predicted Value')
    plt.ylabel('Frequency')
    plt.show()
    plt.savefig(f'distribution_bootstrap_{cost}.png', dpi=400)
