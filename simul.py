import pandas as pd
import numpy as np

# Load the data from the Excel file
bcet_df = pd.read_excel('/Users/vladelec/Desktop/Exeter /Summer Project/FTT_StandAlone-main 2/Final work /simulation.xlsx', sheet_name='BCET')
mewg_df = pd.read_excel('/Users/vladelec/Desktop/Exeter /Summer Project/FTT_StandAlone-main 2/Final work /simulation.xlsx', sheet_name='MEWG')
meww_df = pd.read_excel('/Users/vladelec/Desktop/Exeter /Summer Project/FTT_StandAlone-main 2/Final work /simulation.xlsx', sheet_name='MEWW')

# Convert only the year columns to integers
def convert_year_columns_to_int(df):
    cols = df.columns.tolist()
    new_cols = []
    for col in cols:
        try:
            new_cols.append(int(col))  # Convert to integer if possible
        except ValueError:
            new_cols.append(col)  # Keep original if not a number
    df.columns = new_cols

# Apply conversion to the dataframes
convert_year_columns_to_int(mewg_df)
convert_year_columns_to_int(meww_df)

# Melt the DataFrames
mewg_df = pd.melt(mewg_df, id_vars=['Country', 'Technology'], var_name='Year', value_name='Capacity')
meww_df = pd.melt(meww_df, id_vars=['Country', 'Technology'], var_name='Year', value_name='Capacity')

# Convert the 'Year' column to numeric if it isn't already
mewg_df['Year'] = pd.to_numeric(mewg_df['Year'], errors='coerce')
meww_df['Year'] = pd.to_numeric(meww_df['Year'], errors='coerce')

# Drop any rows with NaN values in 'Year' or 'Capacity'
mewg_df = mewg_df.dropna(subset=['Year', 'Capacity'])
meww_df = meww_df.dropna(subset=['Year', 'Capacity'])

# Sort the DataFrame by Country, Technology, and Year for clarity
mewg_df = mewg_df.sort_values(by=['Country', 'Technology', 'Year'])
meww_df = meww_df.sort_values(by=['Country', 'Technology', 'Year'])

# Check and replace zero or NaN values in critical columns
mewg_df['Capacity'].replace(0, np.nan, inplace=True)
meww_df['Capacity'].replace(0, np.nan, inplace=True)

# Ensure BCET data is ready
bcet_melted = pd.melt(bcet_df, id_vars=['Country', 'Technology'], var_name='Variable', value_name='Value')
years = range(2026, 2030)
bcet_expanded = pd.DataFrame()

for year in years:
    temp_df = bcet_melted.copy()
    temp_df['Year'] = year
    bcet_expanded = pd.concat([bcet_expanded, temp_df], ignore_index=True)

bcet_final = pd.concat([bcet_melted.assign(Year=2025), bcet_expanded], ignore_index=True)
bcet_final = bcet_final.pivot_table(index=['Country', 'Technology', 'Year'], columns='Variable', values='Value').reset_index()

print(f"bcet_final.shape: {bcet_final.shape}")
print(f"mewg_df.shape: {mewg_df.shape}")
print(f"meww_df.shape: {meww_df.shape}")
print(bcet_final.head())
print(mewg_df.head())
print(meww_df.head())








def calculate_degradation(shelf_life, end_of_life):
    return 1 - end_of_life ** (1 / shelf_life)

def adjust_costs_for_learning(bcet, tech, dw, year):
    try:
        print(f"Adjusting costs for tech index {tech} in year {year}")
        adjustment_factor = 1.0 + bcet.loc[tech, '16 Learning exp'] * dw[tech] / max(bcet.loc[tech, 'MEWG'], 0.001)  # Avoid zero division
        print(f"Adjustment factor: {adjustment_factor}")
        
        # Ensure compatibility with float values
        bcet['3 Investment ($/MW)'] = bcet['3 Investment ($/MW)'].astype(float)
        bcet['4 std ($/MW)'] = bcet['4 std ($/MW)'].astype(float)
        bcet['7 O&M ($/MWh)'] = bcet['7 O&M ($/MWh)'].astype(float)
        bcet['8 std ($/MWh)'] = bcet['8 std ($/MWh)'].astype(float)

        # Apply adjustment factor
        bcet.loc[tech, '3 Investment ($/MW)'] *= adjustment_factor
        bcet.loc[tech, '4 std ($/MW)'] *= adjustment_factor
        bcet.loc[tech, '7 O&M ($/MWh)'] *= adjustment_factor
        bcet.loc[tech, '8 std ($/MWh)'] *= adjustment_factor

    except Exception as e:
        print(f"Error in adjust_costs_for_learning: {e}")


# Function to calculate LCOES
def get_lcoes(bcet, mewg, meww, year, histend, dw):
    print(f"Inside get_lcoes, processing year: {year}")
    
    if year > histend['BCET']:
        for tech in range(len(bcet)):
            if mewg.loc[tech, 'Capacity'] > 0.1:
                adjust_costs_for_learning(bcet, tech, dw, year)

    lt = bcet['9 Lifetime (years)'].values  # Convert to NumPy array
    bt = bcet['10 Lead Time (years)'].values  # Convert to NumPy array
    max_lt = int(np.max(bt + lt))
    
       
    full_lt_mat = np.linspace(np.zeros(3), max_lt - 1, num=max_lt, axis=1, endpoint=True)
    lt_max_mat = np.concatenate(max_lt * [(lt + bt - 1)[:, np.newaxis]], axis=1)
    bt_max_mat = np.concatenate(max_lt * [(bt - 1)[:, np.newaxis]], axis=1)

    bt_mask = full_lt_mat <= bt_max_mat
    lt_mask = full_lt_mat <= lt_max_mat
    
    print(f"bt_mask.shape: {bt_mask.shape}")
    print(f"lt_mask.shape: {lt_mask.shape}") 
    print(f"bt_mask: {bt_mask}") 
    print(f"lt_mask: {lt_mask}")




    cf_mu = bcet['11 Decision Load Factor'].values  # Convert to NumPy array
    cf_mu[cf_mu < 0.000001] = 0.000001
    conv_mu = 1 / np.maximum(bt[:, np.newaxis], 0.001) / np.maximum(cf_mu[:, np.newaxis], 0.001) / 8766 * 1000

    cf_av = np.maximum(mewg['Capacity'].values, 0.001)  # Convert to NumPy array
    conv_av = 1 / np.maximum(bt[:, np.newaxis], 0.001) / cf_av[:, np.newaxis] / 8766 * 1000

    shelf_life = lt
    end_of_life = bcet['25 End-of-life costs (%)'].values  # Convert to NumPy array

    deg_temporal = 1 - end_of_life ** (1 / shelf_life)
    dr = bcet['17 Discount Rate (%)'].values  # Convert to NumPy array

    try:
        it_mu = bcet['3 Investment ($/MW)'].values * conv_mu
        print(f"it_mu.shape: {it_mu.shape}")
    except Exception as e:
        print(f"Error in calculating it_mu: {e}")
    

    it_av = bcet['3 Investment ($/MW)'].values * conv_av
    it_av = np.where(bt_mask, it_av, 0)

    it_total_av = it_av + bcet['26 Investment ($/MWh)'].values
    it_total_mu = it_mu + bcet['26 Investment ($/MWh)'].values

    omt_fixed = bcet['28 O&M ($/MW-yr)'].values * conv_mu
    omt_fixed = np.where(lt_mask, omt_fixed, 0)

    omt_variable = bcet['7 O&M ($/MWh)'].values * (1 - deg_temporal)
    omt_variable = np.where(lt_mask, omt_variable, 0)

    omt_total = omt_fixed + omt_variable

    rte = bcet['13 Efficiency (%)'].values  # Convert to NumPy array
    chrg_cost = (bcet['5 Fuel ($/MWh)'].values) * 0.001 / np.maximum(rte, 0.001)
    chrg_cost = np.where(lt_mask, chrg_cost, 0)

    rep_power = bcet['21 Replacement Costs ($/kW)'].values * conv_mu
    rep_power = np.where(lt_mask, rep_power, 0)

    EoL = np.sum((bcet['21 Replacement Costs ($/kW)'].values * conv_mu + end_of_life * conv_av) / (1 + dr) ** (lt + 1), axis=1)
    EoL = EoL[:, np.newaxis]

    energy_prod = np.maximum(mewg['Capacity'].values, 0.001)

    denominator = (1 + dr) ** full_lt_mat
    npv_expenses_mu_no_policy = (it_total_mu + omt_total + chrg_cost + rep_power + EoL) / denominator
    npv_expenses_mu_all_policies = npv_expenses_mu_no_policy / denominator
    npv_expenses_no_policy = (it_total_av + omt_total + chrg_cost + rep_power + EoL) / denominator

    npv_utility = energy_prod / denominator
    utility_tot = np.sum(npv_utility, axis=1)

    lcoes_mu_no_policy = np.sum(npv_expenses_mu_no_policy, axis=1) / utility_tot
    lcoes_mu_all_policies = np.sum(npv_expenses_mu_all_policies, axis=1) / utility_tot
    lcoes_mu_gamma = lcoes_mu_all_policies  # Adjust this according to your additional policy costs
    lcoes_all = np.sum(npv_expenses_no_policy, axis=1) / utility_tot  # Adjust according to your additional policy costs

    npv_std = np.sqrt(rep_power ** 2) / denominator
    dlcoes = np.sum(npv_std, axis=1) / utility_tot

    results = {
        'LCOES_no_policy': lcoes_mu_no_policy,
        'LCOES_with_policies': lcoes_mu_all_policies,
        'LCOES_gamma': lcoes_mu_gamma,
        'LCOES_all': lcoes_all,
        'dlcoes': dlcoes
    }

    return results



def monte_carlo_simulation(bcet_df, mewg_df, meww_df, histend, num_simulations=1):
    results = []
    years = range(2026, 2030)
    
    for year in years:
        if year in mewg_df['Year'].unique():
            print(f"Processing year: {year}")
            if year - 1 in mewg_df['Year'].unique():
                dw = mewg_df[mewg_df['Year'] == year]['Capacity'].values - mewg_df[mewg_df['Year'] == year - 1]['Capacity'].values
            else:
                dw = np.zeros_like(mewg_df[mewg_df['Year'] == year]['Capacity'].values)
                

            for i in range(num_simulations):
                bcet_sample = bcet_df[bcet_df['Year'] == year].copy()  # Select data for the specific year
                
                try:
                    lcoes_result = get_lcoes(bcet_sample, mewg_df[mewg_df['Year'] == year], meww_df[meww_df['Year'] == year], year, histend, dw)
                    results.append({
                        'Year': year,
                        'Simulation': i,
                        'LCOES_no_policy': lcoes_result['LCOES_no_policy'].mean(),
                        'LCOES_with_policies': lcoes_result['LCOES_with_policies'].mean(),
                        'LCOES_gamma': lcoes_result['LCOES_gamma'].mean(),
                        'LCOES_all': lcoes_result['LCOES_all'].mean(),
                        'dlcoes': lcoes_result['dlcoes'].mean()
                    })
                except Exception as e:
                    print(f"Error processing simulation {i} for year {year}: {e}")
        else:
            print(f"Year {year} is not available in the data. Skipping.")
    
    results_df = pd.DataFrame(results)
    return results_df

# Setup for historical end and time lag
histend = {'BCET': 2025}

# Run the Monte Carlo simulation
lcoes_simulation_results = monte_carlo_simulation(
    bcet_final,
    mewg_df,
    meww_df,
    histend=histend,
    num_simulations=1
)

# Analyze the results
if not lcoes_simulation_results.empty:
    lcoes_summary = lcoes_simulation_results.groupby('Year').describe()
    print(lcoes_summary)
else:
    print("No simulation results were generated.")








