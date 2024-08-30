
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

file_path = '/Users/vladelec/Desktop/Exeter /Summer Project/FTT_StandAlone-main 2/Final work /Disseration.xlsx'

capacity_df = pd.read_excel(file_path, sheet_name='Capacity per year (MWh)')
cagr_df = pd.read_excel(file_path, sheet_name='CAGR')

df_long = pd.melt(capacity_df, id_vars=['Technology', 'Country'], var_name='Year', value_name='Generated Capacity')
df_long['Year'] = df_long['Year'].astype(int)

df_long.dropna(subset=['Country', 'Technology'], inplace=True)
cagr_df.dropna(subset=['Country', 'Technology', 'Scenario', 'CAGR'], inplace=True)

df_long['Generated Capacity'] = pd.to_numeric(df_long['Generated Capacity'], errors='coerce')
df_long.dropna(subset=['Generated Capacity'], inplace=True)

df_long = df_long[df_long['Country'] != 'Global']
cagr_df = cagr_df[cagr_df['Country'] != 'Global']

results = []

scenarios = ['Positive', 'Medium', 'Negative']
technologies = df_long['Technology'].unique()
countries = df_long['Country'].unique()

for scenario in scenarios:
    for technology in technologies:
        for country in countries:

            filtered_df = df_long[(df_long['Country'] == country) & (df_long['Technology'] == technology)]

            if filtered_df.empty:
                print(f"No data for {technology} in {country} under {scenario} scenario, skipping.")
                continue

            scenario_filter = (cagr_df['Technology'] == technology) & \
                              (cagr_df['Country'] == country) & \
                              (cagr_df['Scenario'] == scenario)
            if scenario_filter.sum() == 0:
                print(f"No CAGR found for {technology} in {country} under {scenario} scenario, skipping.")
                continue

            cagr = cagr_df.loc[scenario_filter, 'CAGR'].values[0]

            #Prepare the data for Bayesian Ridge Regression
            X = (filtered_df['Year'].values - filtered_df['Year'].min()).reshape(-1, 1)
            y = filtered_df['Generated Capacity'].values * (1 + cagr)

            #Filter out any rows where y is NaN
            valid_indices = ~np.isnan(y)
            X = X[valid_indices]
            y = y[valid_indices]

            #Split the data to check model performance (up to 2024) and validate (2025-2030)
            X_train = X[X.flatten() <= 2024]
            y_train = y[X.flatten() <= 2024]
            X_val = X[(X.flatten() >= 2025) & (X.flatten() <= 2030)]
            y_val = y[(X.flatten() >= 2025) & (X.flatten() <= 2030)]

            #Skip if training data is empty
            if X_train.size == 0 or y_train.size == 0:
                print(f"Skipping {technology} in {country} under {scenario} scenario due to lack of training data.")
                continue

            #Fit the Bayesian Ridge model
            model = BayesianRidge()
            model.fit(X_train, y_train)

            #Validate the model on 2025-2030 data
            if X_val.size > 0 and y_val.size > 0:
                y_val_pred = model.predict(X_val)
                mae = np.mean(np.abs(y_val - y_val_pred))
                print(f"Validation MAE for {technology} in {country} under {scenario} scenario: {mae}")
            else:
                mae = None

            #Predict the generated capacity based on the model
            y_pred = model.predict(X)

            #Predict future values (e.g., for 2030-2050)
            future_years = np.arange(2030, 2051)
            X_future = (future_years - filtered_df['Year'].min()).reshape(-1, 1)
            y_future_pred = model.predict(X_future)

            results.append({
                'Technology': technology,
                'Country': country,
                'Scenario': scenario,
                'MAE': mae,
                'Future Predictions': y_future_pred.tolist()  # Store as a list
            })

            plt.figure(figsize=(10, 6))
            plt.plot(filtered_df['Year'], filtered_df['Generated Capacity'], 'o', label='Observed Data')
            plt.plot(filtered_df['Year'], y_pred, '-', label=f'Prediction ({scenario})')
            plt.plot(future_years, y_future_pred, '--', label='Future Prediction')
            plt.xlabel('Year')
            plt.ylabel('Generated Capacity (MWh)')
            plt.title(f'{technology} in {country} ({scenario} Scenario)')
            plt.legend()
            plt.show()

results_df = pd.DataFrame(results)

flattened_results = []
for _, row in results_df.iterrows():
    for i, pred in enumerate(row['Future Predictions']):
        flattened_results.append({
            'Technology': row['Technology'],
            'Country': row['Country'],
            'Scenario': row['Scenario'],
            'Year': future_years[i],
            'Prediction': pred
        })

flattened_df = pd.DataFrame(flattened_results)

uncertainty_df = flattened_df.groupby(['Technology', 'Country', 'Year'])['Prediction'].agg([np.mean, np.std])
print(uncertainty_df)

results_df.to_csv('results_Baynesian.csv', index=False)

uncertainty_df.to_csv('uncertainty_Baynesian.csv', index=True)


num_samples = 1000

samples = {}

for index, row in uncertainty_df.iterrows():
    mean_val = row['mean']
    std_val = row['std']
    
    sampled_values = np.random.normal(loc=mean_val, scale=std_val, size=num_samples)
    
    samples[index] = sampled_values

samples_df = pd.DataFrame(samples)


num_samples = 5000

samples = []

for index, row in uncertainty_df.iterrows():
    mean_val = row['mean']
    std_val = row['std']
    
    sampled_values = np.random.normal(loc=mean_val, scale=std_val, size=num_samples)
    
    sample_df = pd.DataFrame({
        'Technology': [index[0]] * num_samples,
        'Country': [index[1]] * num_samples,
        'Year': [index[2]] * num_samples,
        'Sample': sampled_values
    })
    
    samples.append(sample_df)

samples_df = pd.concat(samples, ignore_index=True)

samples_df.to_csv('sampled_results_Bayesian.csv', index=False)

print(samples_df.head())


statistics = samples_df.groupby(['Technology', 'Country', 'Year']).agg(
    mean=('Sample', 'mean'),
    median=('Sample', 'median'),
    variance=('Sample', 'var'),
    std_dev=('Sample', 'std'),
    skewness=('Sample', lambda x: skew(x, bias=False)),
    kurtosis=('Sample', lambda x: kurtosis(x, bias=False)),
    percentile_5=('Sample', lambda x: np.percentile(x, 5)),
    percentile_95=('Sample', lambda x: np.percentile(x, 95))
).reset_index()

statistics.to_csv('sampled_statistics_Bayesian.csv', index=False)

print(statistics.head())

samples_df=pd.read_csv('/Users/vladelec/Desktop/Exeter /Summer Project/FTT_StandAlone-main 2/Final work /CSV results/sampled_results_Bayesian.csv')

averaged_samples = samples_df.groupby(['Technology', 'Country', 'Year'])['Sample'].mean().reset_index()

averaged_samples['Sample_GWh'] = averaged_samples['Sample'] / 1000

averaged_samples.to_csv('averaged_sampled_results_Bayesian.csv', index=False)

print(averaged_samples.head())


avg_capacity_df = averaged_samples.groupby(['Technology', 'Country', 'Year'])['Sample_GWh'].mean().unstack().fillna(0)
def plot_by_technology(avg_capacity_df):
    technologies = avg_capacity_df.index.get_level_values(0).unique()
    countries = avg_capacity_df.columns.get_level_values(0).unique()

    for tech in technologies:
        tech_data = avg_capacity_df.loc[tech]

        colors = sns.color_palette("bright", len(tech_data.columns))

        fig, ax1 = plt.subplots(figsize=(14, 8))
        tech_data.T.plot(kind='bar', stacked=True, ax=ax1, color=colors)

        ax1.set_xlabel('Year')
        ax1.set_ylabel('GWh')
        ax1.set_title(f'Capacity Additions for {tech} by Country')
        ax1.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))

        plt.tight_layout()

        plt.savefig(f'capacity_additions_{tech}_by_country.png', dpi=150)
        plt.show()

plot_by_technology(avg_capacity_df)

def plot_by_country(avg_capacity_df):
    countries = avg_capacity_df.columns.get_level_values(0).unique()
    technologies = avg_capacity_df.index.get_level_values(0).unique()

    for country in countries:
        country_data = avg_capacity_df.xs(country, axis=1, level=0)

        colors = sns.color_palette("bright", len(country_data.index))

        fig, ax1 = plt.subplots(figsize=(14, 8))
        country_data.T.plot(kind='bar', stacked=True, ax=ax1, color=colors)


        ax1.set_xlabel('Year')
        ax1.set_ylabel('GWh')
        ax1.set_title(f'Capacity Additions in {country} by Technology')
        ax1.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))

        plt.tight_layout()

        plt.savefig(f'capacity_additions_{country}_by_technology.png', dpi=150)
        plt.show()

plot_by_country(avg_capacity_df)
