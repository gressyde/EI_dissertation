# -*- coding: utf-8 -*-
"""
=========================================
ftt_p_lcoe.py
=========================================
Power LCOE FTT module.


Functions included:
    - get_lcoe
        Calculate levelized costs

"""

# Third party imports
import numpy as np



# %% lcoe
# -----------------------------------------------------------------------------
# --------------------------- LCOE function -----------------------------------
# -----------------------------------------------------------------------------
def set_carbon_tax(data, c2ti, year):
    '''
    Convert the carbon price in REPP from euro / tC to $2013 dollars. 
    Apply the carbon price to power sector technologies based on their efficiencies

    REX --> EU local currency per euros rate (33 is US$)
    PRSC --> price index (local currency) consumer expenditure
    EX --> EU local currency per euro, 2005 == 1
    The 13 part of the variable denotes 2013. 
    
    Returns:
        Carbon costs per country and technology (2D), part of BCET
    '''
   
    carbon_costs = (
                    data['BCET'][:, :, c2ti['15 Emissions (tCO2/GWh)']]   # Emission per GWh
                    * data["REPPX"][:, :, 0]                              # Carbon price in euro / tC
                    * data["REX13"][33, 0, 0] / ( data["PRSCX"][:, :, 0] * data["EX13"][:, :, 0] / (data["PRSC13"][:, :, 0]  * data["EXX"][:, :, 0]) )
                    / 1000 / 3.666                                        # Conversion from GWh to MWh and from C to CO2. 
                    )
    
    
    if np.isnan(carbon_costs).any():
        print(f"Carbon price is nan in year {year}")
        print(f"The arguments of the nans are {np.argwhere(np.isnan(carbon_costs))}")
        print( ('Conversion factor:'
              f'{data["REX13"][33, 0, 0] / ( data["PRSCX"][:, :, 0] * data["EX13"][:, :, 0] / (data["PRSC13"][:, :, 0]  * data["EXX"][:, :, 0]) )}') )
        print(f"Emissions intensity {data['BCET'][:, :, c2ti['15 Emissions (tCO2/GWh)']]}")
        
        raise ValueError
                       
    return carbon_costs
    

def get_lcoe(data, titles, year):
    """
    Calculate levelized costs.

    The function calculates the levelised cost of electricity in $2013/MWh. It includes
    intangible costs (gamma values) and determines the investor preferences.

    Parameters
    -----------
    data: dictionary of NumPy arrays
        Model variables for the current year.
        Variable names are keys and the values are 3D NumPy arrays.
    titles: dictionary
        Titles is a container of all permissible dimension titles of the model.

    Returns
    ----------
    data: dictionary
        Updated values:
            The different LCOE variants (METC, MECW ..)
            The standard deviation of LCOE (MTCD)
            The components of LCOE (MCFC, MWIC)

    Notes
    ---------
    BCET = cost matrix 
    MEWL = Average capacity factor
    MEWT = Subsidies
    MTFT = Fuel tax
    
    """

    # Categories for the cost matrix (BCET)
    c2ti = {category: index for index, category in enumerate(titles['C2TI'])}

    for r in range(len(titles['RTI'])):

        # Cost matrix
        bcet = data['BCET'][r, :24, :]

        # Plant lifetime
        lt = bcet[:, c2ti['9 Lifetime (years)']]
        bt = bcet[:, c2ti['10 Lead Time (years)']]
        max_lt = int(np.max(bt+lt))
        
        # Define (matrix) masks to turn off cost components before or after contruction 
        full_lt_mat = np.linspace(np.zeros(24), max_lt-1,
                                  num=max_lt, axis=1, endpoint=True)
        lt_max_mat = np.concatenate(int(max_lt) * [(lt+bt-1)[:, np.newaxis]], axis=1)
        bt_max_mat = np.concatenate(int(max_lt) * [(bt-1)[:, np.newaxis]], axis=1)
        
        bt_mask = full_lt_mat <= bt_max_mat
        bt_mask_out = full_lt_mat > bt_max_mat
        lt_mask_in = full_lt_mat <= lt_max_mat
        lt_mask = np.where(lt_mask_in == bt_mask_out, True, False)
        
        # Capacity factor of marginal unit (for decision-making)
        cf_mu = bcet[:, c2ti['11 Decision Load Factor']].copy()
        # Trap for very low CF
        cf_mu[cf_mu<0.000001] = 0.000001
        # Factor to transfer cost components in terms of capacity to generation
        conv_mu = 1/bt / cf_mu/8766*1000
        
        # Average capacity factor (for electricity price)
        cf_av = data['MEWL'][r, :24, 0]
        # Trap for very low CF
        cf_av[cf_av<0.000001] = 0.000001
        # Factor to transfer cost components in terms of capacity to generation
        conv_av = 1/bt / cf_av/8766*1000        

        # Discount rate
        dr = bcet[:, c2ti['17 Discount Rate (%)'], np.newaxis]

        # Initialse the levelised cost components
        # Average investment cost of marginal unit (new investments)
        it_mu = np.ones([24, int(max_lt)])
        it_mu = it_mu * bcet[:, c2ti['3 Investment ($/MW)'], np.newaxis] * conv_mu[:, np.newaxis]
        it_mu = np.where(bt_mask, it_mu, 0)
        
        # Average investment costs of across all units (electricity price)
        it_av = np.ones([24, int(max_lt)])
        it_av = it_av * bcet[:, c2ti['3 Investment ($/MW)'], np.newaxis] * conv_av[:, np.newaxis]
        it_av = np.where(bt_mask, it_av, 0)       

        # Standard deviation of investment cost - marginal unit
        dit_mu = np.ones([24, int(max_lt)])
        dit_mu = dit_mu * bcet[:, c2ti['4 std ($/MW)'], np.newaxis] * conv_mu[:, np.newaxis]
        dit_mu = np.where(bt_mask, dit_mu, 0)

        # Standard deviation of investment cost - average of all units
        dit_av = np.ones([24, int(max_lt)])
        dit_av = dit_av * bcet[:, c2ti['4 std ($/MW)'], np.newaxis] * conv_av[:, np.newaxis]
        dit_av = np.where(bt_mask, dit_av, 0)

        # Subsidies - only valid for marginal unit
        st = np.ones([24, int(max_lt)])
        st = (st * bcet[:, c2ti['3 Investment ($/MW)'], np.newaxis]
              * data['MEWT'][r, :24, :] * conv_mu[:, np.newaxis])
        st = np.where(bt_mask, st, 0)

        # Average fuel costs
        ft = np.ones([24, int(max_lt)])
        ft = ft * bcet[:, c2ti['5 Fuel ($/MWh)'], np.newaxis]
        ft = np.where(lt_mask, ft, 0)

        # Standard deviation of fuel costs
        dft = np.ones([24, int(max_lt)])
        dft = dft * bcet[:, c2ti['6 std ($/MWh)'], np.newaxis]
        dft = np.where(lt_mask, dft, 0)

        # fuel tax/subsidies
        fft = np.ones([24, int(max_lt)])
        fft = fft * data['MTFT'][r, :24, 0, np.newaxis]
        fft = np.where(lt_mask, fft, 0)

        # Average operation & maintenance cost
        omt = np.ones([24, int(max_lt)])
        omt = omt * bcet[:, c2ti['7 O&M ($/MWh)'], np.newaxis]
        omt = np.where(lt_mask, omt, 0)

        # Standard deviation of operation & maintenance cost
        domt = np.ones([24, int(max_lt)])
        domt = domt * bcet[:, c2ti['8 std ($/MWh)'], np.newaxis]
        domt = np.where(lt_mask, domt, 0)

        # Carbon costs
        ct = np.ones([24, int(max_lt)])
        ct = ct * bcet[:, c2ti['1 Carbon Costs ($/MWh)'], np.newaxis]
        ct = np.where(lt_mask, ct, 0)
        
        # Standard deviation carbon costs (set to zero for now)
        dct = np.ones([24, int(max_lt)])
        dct = dct * bcet[:, c2ti['2 std ($/MWh)'], np.newaxis]
        dct = np.where(lt_mask, dct, 0)
        

        # Energy production over the lifetime (incl. buildtime)
        # No generation during the buildtime, so no benefits
        energy_prod = np.ones([24, int(max_lt)])
        energy_prod = np.where(lt_mask, energy_prod, 0)

        # Storage costs and marginal costs (lifetime only)
        stor_cost = np.ones([24, int(max_lt)])
        battery_cost = np.ones([24, int(max_lt)])
        marg_stor_cost = np.ones([24, int(max_lt)])

        if np.rint(data['MSAL'][r, 0, 0]) in [2]:
            stor_cost = stor_cost * (data['MSSP'][r, :24, 0, np.newaxis] +
                                     data['MLSP'][r, :24, 0, np.newaxis]) / 1000
            marg_stor_cost = marg_stor_cost * 0
            battery_cost = battery_cost * data['MSSP'][r, :24, 0, np.newaxis] / 1000
        elif np.rint(data['MSAL'][r, 0, 0]) in [3, 4, 5]:
            stor_cost = stor_cost * (data['MSSP'][r, :24, 0, np.newaxis] +
                                     data['MLSP'][r, :24, 0, np.newaxis]) / 1000
            marg_stor_cost = marg_stor_cost * (data['MSSM'][r,:24, 0, np.newaxis] +
                                          data['MLSM'][r, :24, 0, np.newaxis]) / 1000
            battery_cost = battery_cost * data['MSSP'][r, :24, 0, np.newaxis] / 1000
        else:
            stor_cost = stor_cost * 0
            marg_stor_cost = marg_stor_cost * 0
            battery_cost = battery_cost * 0
            
       
        stor_cost = np.where(lt_mask, stor_cost, 0)
        marg_stor_cost = np.where(lt_mask, marg_stor_cost, 0)
        battery_cost = np.where(lt_mask, battery_cost, 0)
        dstor_cost = 0.2 * stor_cost         # Assume a standard deviation of 20%

        # Net present value calculations
        
        # Discount rate
        denominator = (1+dr)**full_lt_mat
        
       
        # 1a – Expenses – marginal units
        npv_expenses_mu_no_policy      = (it_mu + ft + omt + stor_cost) / denominator 
        npv_expenses_mu_only_co2       = npv_expenses_mu_no_policy + ct / denominator
        npv_expenses_mu_all_policies   = npv_expenses_mu_no_policy + (ct + fft + st + marg_stor_cost) / denominator 
        npv_expenses_mu_no_policy_battery_only = (it_mu + ft + omt + battery_cost) / denominator 
        
        # 1b – Expenses – average LCOEs
        npv_expenses_no_policy        = (it_av + ft + omt + stor_cost) / denominator  
        npv_expenses_all_but_co2      = npv_expenses_no_policy + (fft + st) / denominator
        
        # 1c - Operation costs
        npv_operation                 = (ft + omt + stor_cost + marg_stor_cost + fft) / denominator
        
        # 2 – Utility
        npv_utility = energy_prod / denominator
        npv_utility[npv_utility==1] = 0 # Remove 1s for tech with smaller lifetime than max
        utility_tot = np.sum(npv_utility, axis=1) 
        
        # 3 – Standard deviation (propagation of error)
        npv_std = np.sqrt(dit_mu**2 + dft**2 + domt**2 + dct**2 + dstor_cost**2) / denominator  
        
        # 4a – levelised cost – marginal units 
        lcoe_mu_no_policy       = np.sum(npv_expenses_mu_no_policy, axis=1) / utility_tot        
        lcoe_mu_only_co2        = np.sum(npv_expenses_mu_only_co2, axis=1) / utility_tot 
        lcoe_mu_all_policies    = np.sum(npv_expenses_mu_all_policies, axis=1) / utility_tot - data['MEFI'][r, :24, 0]
        lcoe_mu_gamma           = lcoe_mu_all_policies + data['MGAM'][r, :24, 0]
        lcoe_mu_no_policy_battery_only = np.sum(npv_expenses_mu_no_policy_battery_only, axis=1) / utility_tot 

        # 4b levelised cost – average units 
        lcoe_all_but_co2        = np.sum(npv_expenses_all_but_co2, axis=1) / utility_tot - data['MEFI'][r, :24, 0]    
        
        # 4c - Operational costs
        lcoo                    = np.sum(npv_operation, axis=1) / utility_tot      # Levelised cost of operation
        
        # Standard deviation of LCOE
        dlcoe                   = np.sum(npv_std, axis=1) / utility_tot


        # Pass to variables that are stored outside.
        data['MEWC'][r, :24, 0] = lcoe_mu_no_policy       # The real bare LCOE without taxes
        data['MECW'][r, :24, 0] = lcoe_mu_only_co2        # Bare LCOE with CO2 costs
        data["MECC"][r, :24, 0] = lcoe_all_but_co2        # LCOE with policy, without CO2 costs
        data['METC'][r, :24, 0] = lcoe_mu_gamma           # As seen by consumer (generalised cost)
        data['MLCO'][r, :24, 0] = lcoo                    # Levelised cost of operation
        data['MTCD'][r, :24, 0] = dlcoe                   # Standard deviation LCOE 


        # Output variables
        data['MWIC'][r, :24, 0] = bcet[:, 2].copy()    # Investment cost component LCOE ($/kW)
        data["MECW battery only"][r, :24, 0] = lcoe_mu_no_policy_battery_only   # LCOE without policy with only short-term storage costs
        data['MWFC'][r, :24, 0] = bcet[:, 4].copy()    # Fuel cost component of the LCOE ($/MWh)
        data['MCOC'][r, :24, 0] = bcet[:, 0].copy()    # Carbon cost component of the LCOE ($/MWh)
        data['MCFC'][r, :24, 0] = bcet[:, c2ti['11 Decision Load Factor']].copy() # The (marginal) capacity factor 

        # MWMC: FTT Marginal costs power generation ($/MWh)
        if np.rint(data['MSAL'][r, 0, 0]) > 1: # rint rounds to nearest int
            data['MWMC'][r, :24, 0] = bcet[:, 0] + bcet[:, 4] + bcet[:, 6] + (data['MSSP'][r, :24, 0] + data['MLSP'][r, :24, 0])/1000
        else:
            data['MWMC'][r, :24, 0] = bcet[:, 0] + bcet[:, 4] + bcet[:, 6]


        data['MMCD'][r, :24, 0] = np.sqrt(bcet[:, 1] * bcet[:, 1] +
                                        bcet[:, 5] * bcet[:, 5] +
                                        bcet[:, 7] * bcet[:, 7])

    return data

def calculate_degradation(shelf_life, end_of_life):
    deg_temporal = 1 - end_of_life**(1/shelf_life)
    return deg_temporal

def get_lcoes(data, titles):
    c2ti = {category: index for index, category in enumerate(titles['C2TI'])}

    for r in range(len(titles['RTI'])):
        bcet = data['BCET'][r, 24:27, :]

        lt = bcet[:, c2ti['9 Lifetime (years)']]
        bt = bcet[:, c2ti['10 Lead Time (years)']]
        
        max_lt = int(np.max(bt + lt))

        full_lt_mat = np.linspace(np.zeros(3), max_lt-1,
                                  num=max_lt, axis=1, endpoint=True)
        lt_max_mat = np.concatenate(int(max_lt) * [(lt+bt-1)[:, np.newaxis]], axis=1)
        bt_max_mat = np.concatenate(int(max_lt) * [(bt-1)[:, np.newaxis]], axis=1)
        
        bt_mask = full_lt_mat <= bt_max_mat
        bt_mask_out = full_lt_mat > bt_max_mat
        lt_mask_in = full_lt_mat <= lt_max_mat
        lt_mask = np.where(lt_mask_in == bt_mask_out, True, False)
      
        bt_mask = full_lt_mat <= bt_max_mat
        lt_mask = full_lt_mat <= lt_max_mat

        cf_mu = bcet[:, c2ti['11 Decision Load Factor']].copy()
        cf_mu[cf_mu < 0.000001] = 0.000001
        conv_mu = 1 / bt[:, np.newaxis] / cf_mu[:, np.newaxis] / 8766 * 1000

        cf_av = data['MEWL'][r, 24:27, 0]
        cf_av[cf_av < 0.000001] = 0.000001
        conv_av = 1 / bt[:, np.newaxis] / cf_av[:, np.newaxis] / 8766 * 1000
        
        shelf_life = lt  
        end_of_life = bcet[:, c2ti['25 End-of-life costs (%)']]
       
        deg_temporal = calculate_degradation(shelf_life, end_of_life)
                
        dr = bcet[:, c2ti['17 Discount Rate (%)'], np.newaxis]
        
        it_mu = np.ones([3, int(max_lt)])
        it_mu = bcet[:, c2ti['3 Investment ($/MW)'], np.newaxis] * conv_mu
        it_mu = np.where(bt_mask, it_mu, 0)
        
        it_av = np.ones([3, int(max_lt)])
        it_av = bcet[:, c2ti['3 Investment ($/MW)'], np.newaxis] * conv_av
        it_av = np.where(bt_mask, it_av, 0)

        it_total_av = it_av + bcet[:, c2ti['26 Investment ($/MWh)'], np.newaxis]
        it_total_mu = it_mu + bcet[:, c2ti['26 Investment ($/MWh)'], np.newaxis]

        omt_fixed = bcet[:, c2ti['28 O&M ($/MW-yr)'], np.newaxis] * conv_mu
        omt_fixed = np.where(lt_mask, omt_fixed, 0)
        
        omt_variable = np.ones([3, int(max_lt)])
        omt_variable = omt_variable * bcet[:, c2ti['7 O&M ($/MWh)'], np.newaxis] * (1 - deg_temporal[:, np.newaxis])
        omt_variable = np.where(lt_mask, omt_variable, 0)

        omt_total = omt_fixed + omt_variable
       
        rte = bcet[:, c2ti['13 Efficiency (%)']]  # Round-trip efficiency
        chrg_cost = np.ones([3, int(max_lt)])
        chrg_cost = (bcet[:, c2ti['5 Fuel ($/MWh)'], np.newaxis]) * 0.001 / rte[:, np.newaxis]
        chrg_cost = np.where(lt_mask, chrg_cost, 0)

        rep_power = np.ones([3, int(max_lt)])
        rep_power = bcet[:, c2ti['21 Replacement Costs ($/kW)'], np.newaxis] * conv_mu 
        rep_power = np.where(lt_mask, rep_power, 0)
        
        EoL = np.sum((bcet[:, c2ti['21 Replacement Costs ($/kW)'], np.newaxis] * conv_mu + end_of_life[:, np.newaxis] * conv_av) / (1 + dr) ** (lt[:, np.newaxis] + 1), axis=1) 
        EoL = EoL[:, np.newaxis]
                
        energy_prod = np.ones([3, int(max_lt)])
        energy_prod = np.where(lt_mask, energy_prod, 0)

        denominator = (1 + dr) ** full_lt_mat
        npv_expenses_mu_no_policy = (it_total_mu + omt_total + chrg_cost + rep_power + EoL) / denominator
        npv_expenses_mu_all_policies = npv_expenses_mu_no_policy / denominator
        npv_expenses_no_policy = (it_total_av + omt_total + chrg_cost + rep_power + EoL) / denominator

        npv_utility = np.ones([3, int(max_lt)])
        npv_utility = energy_prod / denominator
        npv_utility = np.where(lt_mask, npv_utility, 0)
        utility_tot = np.sum(npv_utility, axis=1)

        lcoes_mu_no_policy = np.sum(npv_expenses_mu_no_policy, axis=1) / (utility_tot*1000)
        lcoes_mu_all_policies = np.sum(npv_expenses_mu_all_policies, axis=1) / (utility_tot*1000)
        lcoes_mu_gamma = lcoes_mu_all_policies + data['MGAM'][r, 24:27, 0]
        lcoes_all = np.sum(npv_expenses_no_policy, axis=1) / utility_tot - data['MEFI'][r, 24:27, 0]    

        npv_std = np.sqrt(rep_power ** 2) / denominator
        dlcoes = np.sum(npv_std, axis=1) / utility_tot

        data['MEWC'][r, 24:27, 0] = lcoes_mu_no_policy
        data['MECW'][r, 24:27, 0] = lcoes_mu_all_policies
        data['METC'][r, 24:27, 0] = lcoes_mu_gamma
        data["MECC"][r, 24:27, 0] = lcoes_all        # LCOE with policy, without CO2 costs
        data['MTCD'][r, 24:27, 0] = dlcoes
        
        data['MWIC'][r, 24:27, 0] = bcet[:, 2].copy() 
        data['MCOC'][r, 24:27, 0] = chrg_cost[:, c2ti['5 Fuel ($/MWh)']].copy()
        
        

    return data
