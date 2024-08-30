
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sensitivity import SensitivityAnalyzer
import seaborn as sns
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol
from scipy.stats import norm



def calculate_lcoes_simple(data):
    """
    Simplified LCOE calculation with appropriate unit conversions.
    
    Parameters:
    - data: Dictionary containing necessary parameters for LCOE calculation.
    
    Returns:
    - LCOE value.
    """

    # Extract necessary variables
    investment_cost_power = data['Investment cost - Power ($/kW)']
    investment_cost_energy = data['Investment cost - Energy ($/kWh)']
    om_cost_power = data['O&M cost - Power ($/kW-yr)']
    om_cost_energy = data['O&M cost - Energy ($/kWh)']
    replacement_cost = data['Replacement cost ($/kW)']
    efficiency = data['Round-trip efficiency']
    price_of_electricity = data['Price of electricity ($/kWh)']
    discount_rate = data['Discount rate']
    cycle_life = data['Cycle life (cycles)']
    shelf_life = data['Shelf life (years)']
    
    # Time horizon (assuming 10 years for simplicity)
    time_horizon_years = 10
    
    # Capacity factor consideration
    capacity_factor = 0.245
    
    # Calculate initial capital expenditure (CapEx)
    capex_power = investment_cost_power * 7900 / (capacity_factor * 8760)  # Convert to total power-related CapEx over time
    capex_energy = investment_cost_energy * 14000  # Total energy-related CapEx
    
    capex = capex_power + capex_energy  

    # Operation and maintenance costs (O&M)
    om_total_power = om_cost_power * 7900 / (capacity_factor * 8760)
    om_total_energy = om_cost_energy * 24000
    om_total = om_total_power + om_total_energy

    # Replacement costs over the system's lifetime
    replacement_cost_total = (replacement_cost * 24000) / cycle_life

    # Calculate NPV of total costs
    discount_factors = np.array([(1 + discount_rate) ** (-t) for t in range(1, time_horizon_years + 1)])
    npv_costs = capex + om_total * np.sum(discount_factors) + replacement_cost_total

    # Assume general energy capacity of 10,000 MWh and consider capacity factor
    total_energy_generated = 24000* capacity_factor * efficiency * np.sum(discount_factors)  # MWh

    # Calculate NPV of energy produced (assuming constant production over time)
    npv_energy = total_energy_generated

    # Calculate LCOE
    lcoes_value = npv_costs / npv_energy

    return lcoes_value


def calculate_lcoes_with_npv(data):
    """
    LCOE calculation using a year-by-year NPV approach.
    
    Parameters:
    - data: Dictionary containing necessary parameters for LCOE calculation.
    
    Returns:
    - LCOE value.
    """

    # Extract necessary variables
    investment_cost_power = data['Investment cost - Power ($/kW)']
    investment_cost_energy = data['Investment cost - Energy ($/kWh)']
    om_cost_power = data['O&M cost - Power ($/kW-yr)']
    om_cost_energy = data['O&M cost - Energy ($/kWh)']
    replacement_cost = data['Replacement cost ($/kW)']
    efficiency = data['Round-trip efficiency']
    discount_rate = data['Discount rate']
    cycle_life = data['Cycle life (cycles)']
    shelf_life = data['Shelf life (years)']
    
    # Project details
    time_horizon_years = 10  # Project lifespan in years
    capacity_factor = 0.245  # Capacity factor
    
    # Annual energy production (assuming constant over time)
    total_energy_generated = 24000  # MWh, based on the 2023 figure

    # Initial capital expenditure (CapEx)
    capex_power = investment_cost_power * 7910  # Total power-related CapEx
    capex_energy = investment_cost_energy * 24000  # Total energy-related CapEx
    capex = capex_power + capex_energy  # Total CapEx

    npv_costs = 0  # NPV of costs
    npv_energy = 0  # NPV of energy produced

    # Initialize replacement cost for the first year
    replacement_cost_total = 0

    for year in range(1, time_horizon_years + 1):
        # Discount factor for the year
        discount_factor = (1 + discount_rate) ** (-year)
        
        # Annual O&M costs
        om_total_power = om_cost_power * 7910
        om_total_energy = om_cost_energy * 24000
        om_total = om_total_power + om_total_energy
        
        # Replacement costs (assuming it happens at the end of the cycle life)
        if year % cycle_life == 0:
            replacement_cost_total = replacement_cost * 7910
        else:
            replacement_cost_total = 0
        
        # Annual costs (CapEx is only added once, in the first year)
        if year == 1:
            annual_costs = capex + om_total + replacement_cost_total
        else:
            annual_costs = om_total + replacement_cost_total
        
        # Annual energy production
        annual_energy = total_energy_generated * capacity_factor * efficiency
        
        # NPV of costs and energy
        npv_costs += annual_costs * discount_factor
        npv_energy += annual_energy * discount_factor

    # LCOE calculation
    lcoes_value = npv_costs / npv_energy

    return lcoes_value

# Example data for the simplified calculation
data_example = {
    'Investment cost - Power ($/kW)': 1129,
    'Investment cost - Energy ($/kWh)': 29,
    'O&M cost - Power ($/kW-yr)': 12.21,
    'O&M cost - Energy ($/kWh)': 0.003,
    'Replacement cost ($/kW)': 61.77,
    'Round-trip efficiency': 0.93,
    'Price of electricity ($/kWh)': 0.05,
    'Discount rate': 0.08,
    'Cycle life (cycles)': 3250,
    'Shelf life (years)': 13

}

# Sensitivity Analysis Function
def sensitivity_analysis_simple(base_data, param_names, delta=0.2):
    base_lcoes = calculate_lcoes_with_npv(base_data)
    sensitivity_results = {}

    for param_name in param_names:
        base_value = base_data[param_name]
        relative_changes = np.linspace(-delta, delta, 5)  # ±20% change in 5 steps
        lcoes_values = []

        for change in relative_changes:
            varied_value = base_value * (1 + change)
            modified_data = base_data.copy()
            modified_data[param_name] = varied_value
            lcoes = calculate_lcoes_with_npv(modified_data)
            lcoes_values.append((lcoes - base_lcoes) / base_lcoes * 100)  
            
        sensitivity_results[param_name] = (relative_changes * 100, lcoes_values)

    return sensitivity_results


param_names = [
    'Investment cost - Power ($/kW)', 
    'Investment cost - Energy ($/kWh)', 
    'O&M cost - Power ($/kW-yr)', 
    'O&M cost - Energy ($/kWh)', 
    'Replacement cost ($/kW)', 
    'Round-trip efficiency',
    'Price of electricity ($/kWh)'
]

sensitivities = sensitivity_analysis_simple(data_example, param_names)

print("Sensitivities: ", sensitivities)

sensitivities = sensitivity_analysis_simple(data_example, param_names)

def plot_relative_change_lcoes(sensitivities):
    fig, ax = plt.subplots(figsize=(10, 6))

    for param_name, (relative_changes, lcoes_changes) in sensitivities.items():
        ax.plot(relative_changes, lcoes_changes, label=param_name, marker='o', linestyle='--')

    ax.set_xlabel('Relative change in input parameters [%]')
    ax.set_ylabel('Relative change in LCOES [%]')
    ax.legend(loc='best')
    plt.title("Sensitivity Analysis - Relative Change in LCOES")
    plt.grid(True)
    plt.show()

# Plot the results
plot_relative_change_lcoes(sensitivities)

from interpret import show
from interpret.blackbox import MorrisSensitivity
from sklearn.pipeline import Pipeline
from interpret import set_visualize_provider
from interpret.provider import InlineProvider
from sklearn.base import BaseEstimator, RegressorMixin



class LCOESModel(BaseEstimator, RegressorMixin):
    def __init__(self, include_efficiency=True):
        self.include_efficiency = include_efficiency
    
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        results = []
        for i in range(X.shape[0]):
            if self.include_efficiency:
                params = {
                    'Investment cost - Power ($/kW)': X[i, 0],
                    'Investment cost - Energy ($/kWh)': X[i, 1],
                    'O&M cost - Power ($/kW-yr)': X[i, 2],
                    'O&M cost - Energy ($/kWh)': X[i, 3],
                    'Replacement cost ($/kW)': X[i, 4],
                    'Round-trip efficiency': X[i, 5],
                    'Discount rate': 0.08,  # Fixed
                    'Cycle life (cycles)': 3250,  # Fixed
                    'Shelf life (years)': 13  # Fixed
                }
            else:
                params = {
                    'Investment cost - Power ($/kW)': X[i, 0],
                    'Investment cost - Energy ($/kWh)': X[i, 1],
                    'O&M cost - Power ($/kW-yr)': X[i, 2],
                    'O&M cost - Energy ($/kWh)': X[i, 3],
                    'Replacement cost ($/kW)': X[i, 4],
                    'Round-trip efficiency': 0.93,  # Default value when not included
                    'Discount rate': 0.08,  # Fixed
                    'Cycle life (cycles)': 3250,  # Fixed
                    'Shelf life (years)': 13  # Fixed
                }
            result = calculate_lcoes_with_npv(params)
            results.append(result)
        return np.array(results)

# Base data for common parameters
data_example = {
    'Investment cost - Power ($/kW)': 1129,
    'Investment cost - Energy ($/kWh)': 29,
    'O&M cost - Power ($/kW-yr)': 12.21,
    'O&M cost - Energy ($/kWh)': 0.003,
    'Replacement cost ($/kW)': 61.77,
    'Round-trip efficiency': 0.93,  # This will be varied in one plot and fixed in another
    'Discount rate': 0.08,
    'Cycle life (cycles)': 3250,
    'Shelf life (years)': 13
}

X_train_with_efficiency = np.array([
    [1129, 29, 12.21, 0.003, 61.77, 0.93]
])

X_train_without_efficiency = np.array([
    [1129, 29, 12.21, 0.003, 61.77]
])

# Initialize the LCOES model
lcoes_model_with_efficiency = LCOESModel(include_efficiency=True)
lcoes_model_without_efficiency = LCOESModel(include_efficiency=False)

# Function to run Morris Sensitivity Analysis and plot results
def run_and_plot_morris(model, X_train, feature_names, title, file_name):
    msa = MorrisSensitivity(model, X_train)
    explanation = msa.explain_global()

    importance_scores = explanation.data()['scores']

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importance_scores, color='skyblue')
    plt.xlabel('Morris Sensitivity Score')
    plt.title(title)
    plt.grid(True)

    plt.savefig(file_name, dpi=150)

    # Display the plot
    plt.show()

feature_names_with_efficiency = [
    'Investment cost - Power ($/kW)',
    'Investment cost - Energy ($/kWh)',
    'O&M cost - Power ($/kW-yr)',
    'O&M cost - Energy ($/kWh)',
    'Replacement cost ($/kW)',
    'Round-trip efficiency'
]

feature_names_without_efficiency = [
    'Investment cost - Power ($/kW)',
    'Investment cost - Energy ($/kWh)',
    'O&M cost - Power ($/kW-yr)',
    'O&M cost - Energy ($/kWh)',
    'Replacement cost ($/kW)'
]

# Run and plot Morris analysis with Round-trip efficiency
run_and_plot_morris(lcoes_model_with_efficiency, X_train_with_efficiency, feature_names_with_efficiency, 
                    'Morris Sensitivity Analysis for LCOES (Including Round-trip Efficiency)', 
                    'morris_sensitivity_with_efficiency.png')

# Run and plot Morris analysis without Round-trip efficiency
run_and_plot_morris(lcoes_model_without_efficiency, X_train_without_efficiency, feature_names_without_efficiency, 
                    'Morris Sensitivity Analysis for LCOES (Excluding Round-trip Efficiency)', 
                    'morris_sensitivity_without_efficiency.png')




msa_with_efficiency = MorrisSensitivity(lcoes_model_with_efficiency, X_train_with_efficiency)
explanation_with_efficiency = msa_with_efficiency.explain_global()

# Extract and print the scores
importance_scores_with_efficiency = explanation_with_efficiency.data()['scores']
print("Morris Sensitivity Scores (Including Round-trip Efficiency):")
for name, score in zip(feature_names_with_efficiency, importance_scores_with_efficiency):
    print(f"{name}: {score}")

msa_without_efficiency = MorrisSensitivity(lcoes_model_without_efficiency, X_train_without_efficiency)
explanation_without_efficiency = msa_without_efficiency.explain_global()

importance_scores_without_efficiency = explanation_without_efficiency.data()['scores']
print("\nMorris Sensitivity Scores (Excluding Round-trip Efficiency):")
for name, score in zip(feature_names_without_efficiency, importance_scores_without_efficiency):
    print(f"{name}: {score}")


param_ranges = {
    'Investment cost - Power ($/kW)': (1129 * 0.8, 1129 * 1.2),
    'Investment cost - Energy ($/kWh)': (29 * 0.8, 29 * 1.2),
    'O&M cost - Power ($/kW-yr)': (12.21 * 0.8, 12.21 * 1.2),
    'O&M cost - Energy ($/kWh)': (0.003 * 0.8, 0.003 * 1.2),
    'Replacement cost ($/kW)': (48.17 * 0.8, 48.17 * 1.2),
    'Round-trip efficiency': (0.75, 1),
    'Price of electricity ($/kWh)': (0.05 * 0.8, 0.05 * 1.2),
    'Discount rate': (0.08 * 0.8, 0.08 * 1.2)
}



#Sensitivity scenarios
sensitivity_dict = {
    'Investment cost - Power ($/kW)': [
        data_example['Investment cost - Power ($/kW)'] * 0.8,
        data_example['Investment cost - Power ($/kW)'] * 0.9,
        data_example['Investment cost - Power ($/kW)'],
        data_example['Investment cost - Power ($/kW)'] * 1.1,
        data_example['Investment cost - Power ($/kW)'] * 1.2,
    ],
    'Investment cost - Energy ($/kWh)': [
        data_example['Investment cost - Energy ($/kWh)'] * 0.8,
        data_example['Investment cost - Energy ($/kWh)'] * 0.9,
        data_example['Investment cost - Energy ($/kWh)'],
        data_example['Investment cost - Energy ($/kWh)'] * 1.1,
        data_example['Investment cost - Energy ($/kWh)'] * 1.2,
    ],
    'O&M cost - Power ($/kW-yr)': [
        data_example['O&M cost - Power ($/kW-yr)'] * 0.8,
        data_example['O&M cost - Power ($/kW-yr)'] * 0.9,
        data_example['O&M cost - Power ($/kW-yr)'],
        data_example['O&M cost - Power ($/kW-yr)'] * 1.1,
        data_example['O&M cost - Power ($/kW-yr)'] * 1.2,
    ],
    'O&M cost - Energy ($/kWh)': [
        data_example['O&M cost - Energy ($/kWh)'] * 0.8,
        data_example['O&M cost - Energy ($/kWh)'] * 0.9,
        data_example['O&M cost - Energy ($/kWh)'],
        data_example['O&M cost - Energy ($/kWh)'] * 1.1,
        data_example['O&M cost - Energy ($/kWh)'] * 1.2,
    ],
    'Replacement cost ($/kW)': [
        data_example['Replacement cost ($/kW)'] * 0.8,
        data_example['Replacement cost ($/kW)'] * 0.9,
        data_example['Replacement cost ($/kW)'],
        data_example['Replacement cost ($/kW)'] * 1.1,
        data_example['Replacement cost ($/kW)'] * 1.2,
    ],
    'Round-trip efficiency': [
        data_example['Round-trip efficiency'] * 0.93,
        data_example['Round-trip efficiency'] * 0.97,
        data_example['Round-trip efficiency'],
        data_example['Round-trip efficiency'] * 1.03,
        data_example['Round-trip efficiency'] * 1.08,
    ],
    'Discount rate': [
        data_example['Discount rate'] * 0.8,
        data_example['Discount rate'] * 0.9,
        data_example['Discount rate'],
        data_example['Discount rate'] * 1.1,
        data_example['Discount rate'] * 1.2,
    ]
}


def lcoe_model(**kwargs):
    data = data_example.copy()
    data.update(kwargs)
    return calculate_lcoes_with_npv(data)

# Create the sensitivity analyzer
sa = SensitivityAnalyzer(sensitivity_dict, lcoe_model)

# Generate the sensitivity analysis plot
plot = sa.plot()
styled_df = sa.styled_dfs()

# Define your model input ranges
problem = {
    'num_vars': 6,
    'names': [
        'Investment cost - Power ($/kW)', 
        'Investment cost - Energy ($/kWh)', 
        'O&M cost - Power ($/kW-yr)', 
        'O&M cost - Energy ($/kWh)', 
        'Replacement cost ($/kW)', 
        'Round-trip efficiency'
    ],
    'bounds': [
        [900, 1300],  # Investment cost - Power ($/kW)
        [20, 40],     # Investment cost - Energy ($/kWh)
        [10, 15],     # O&M cost - Power ($/kW-yr)
        [0.002, 0.004],  # O&M cost - Energy ($/kWh)
        [50, 70],     # Replacement cost ($/kW)
        [0.85, 0.96]  # Round-trip efficiency
    ]
}

def calculate_lcoes_with_npv(data):
    investment_cost_power = data['Investment cost - Power ($/kW)']
    investment_cost_energy = data['Investment cost - Energy ($/kWh)']
    om_cost_power = data['O&M cost - Power ($/kW-yr)']
    om_cost_energy = data['O&M cost - Energy ($/kWh)']
    replacement_cost = data['Replacement cost ($/kW)']
    efficiency = data['Round-trip efficiency']
    
    time_horizon_years = 10  
    capacity_factor = 0.245  
    
    total_energy_generated = 24000  

    capex_power = investment_cost_power * 7910  
    capex_energy = investment_cost_energy * 24000  
    capex = capex_power + capex_energy  

    npv_costs = 0  
    npv_energy = 0  

    replacement_cost_total = 0

    for year in range(1, time_horizon_years + 1):
        discount_factor = (1 + 0.08) ** (-year)  # Using a fixed discount rate of 0.08
        om_total_power = om_cost_power * 7910
        om_total_energy = om_cost_energy * 24000
        om_total = om_total_power + om_total_energy
        
        if year % 3250 == 0:  # Fixed cycle life
            replacement_cost_total = replacement_cost * 7910
        else:
            replacement_cost_total = 0
        
        if year == 1:
            annual_costs = capex + om_total + replacement_cost_total
        else:
            annual_costs = om_total + replacement_cost_total
        
        annual_energy = total_energy_generated * capacity_factor * efficiency
        npv_costs += annual_costs * discount_factor
        npv_energy += annual_energy * discount_factor

    lcoes_value = npv_costs / npv_energy
    return lcoes_value


param_values = sobol_sample.sample(problem, 1024)

Y = np.array([calculate_lcoes_with_npv(dict(zip(problem['names'], vals))) for vals in param_values])

sobol_indices = sobol.analyze(problem, Y, print_to_console=True)

param_df = pd.DataFrame(param_values, columns=problem['names'])
covariance_matrix = param_df.cov()

print("\nCovariance Matrix:")
print(covariance_matrix)

# Combine Sobol indices and covariance matrix in the analysis

cov_sobol_combined = covariance_matrix.multiply(sobol_indices['S1'], axis=0)
print("\nCovariance * First-order Sobol Indices:")
print(cov_sobol_combined)



plt.figure(figsize=(10, 8))
sns.heatmap(cov_sobol_combined, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Combined Covariance and First-order Sobol Indices")
plt.show()