# -*- coding: utf-8 -*-
"""
=========================================
ftt_p_main.py
=========================================
Power generation FTT module.



This is the main file for the power module, FTT: Power. The power
module models technological replacement of electricity generation technologies due
to simulated investor decision making. Investors compare the **levelised cost of
electricity**, which leads to changes in the market shares of different technologies.

After market shares are determined, the rldc function is called, which calculates
**residual load duration curves**. This function estimates how much power needs to be
supplied by flexible or baseload technologies to meet electricity demand at all times.
This function also returns load band heights, curtailment, and storage information,
including storage costs and marginal costs for wind and solar.

FTT: Power also includes **dispatchers decisions**; dispatchers decide when different technologies
supply the power grid. Investor decisions and dispatcher decisions are matched up, which is an
example of a stable marriage problem.

Costs in the model change due to endogenous learning curves, costs for electricity
storage, as well as increasing marginal costs of resources calculated using cost-supply
curves. **Cost-supply curves** are recalculated at the end of the routine.

Local library imports:

    FTT: Core functions:
    - `get_sales <get_sales_or_investment.htlm>
        Generic investment function (new plus end-of-life replacement)
        
    FTT: Power functions:

    - `rldc <ftt_p_rldc.html>`__
        Residual load duration curves
    - `dspch <ftt_p_dspch.html>`__
        Dispatch of capacity
    - `get_lcoe <ftt_p_lcoe.html>`__
    - `get_lcoes <ftt_p_lcoes.html>`__
        Levelised cost calculation
    - `survival_function <ftt_p_surv.html>`__
        Calculation of scrappage, sales, tracking of age, and average efficiency.
    - `shares <ftt_p_shares.html>`__
        Market shares simulation (core of the model)
    - `cost_curves <ftt_p_costc.html>`__
        Calculates increasing marginal costs of resources

    Support functions:

    - `divide <divide.html>`__
        Element-wise divide which replaces divide-by-zeros with zeros

Functions included:
    - solve
        Main solution function for the module
"""

# Third-party imports
import numpy as np

# Local library imports
from SourceCode.support.divide import divide
from SourceCode.ftt_core.ftt_sales_or_investments import get_sales, get_sales_yearly
from SourceCode.Power.ftt_p_rldc import rldc
from SourceCode.Power.ftt_p_dspch import dspch
from SourceCode.Power.ftt_p_lcoe import get_lcoe, get_lcoes, set_carbon_tax, calculate_degradation
from SourceCode.Power.ftt_p_shares import shares
from SourceCode.Power.ftt_p_costc import cost_curves
from SourceCode.Power.ftt_p_mewp import get_marginal_fuel_prices_mewp

from SourceCode.Power.ftt_p_phase_out import set_linear_coal_phase_out

from SourceCode.sector_coupling.transport_batteries_to_power import second_hand_batteries
from SourceCode.sector_coupling.battery_lbd import quarterly_bat_add_power, battery_costs


# %% main function
# -----------------------------------------------------------------------------
# ----------------------------- Main ------------------------------------------
# -----------------------------------------------------------------------------
def solve(data, time_lag, iter_lag, titles, histend, year, domain):
    """
    Main solution function for the module.

    Add an extended description in the future.

    Parameters
    -----------
    data: dictionary of NumPy arrays
        Model variables for the current year
    time_lag: type
        Model variables in previous year
    iter_lag: type
        Description
    titles: dictionary of lists
        Dictionary containing all title classification
    histend: dict of integers
        Final year of historical data by variable
    year: int
        Current/active year of solution
    Domain: dictionary of lists
        Pairs variables to domains

    Returns
    ----------
    data: dictionary of NumPy arrays
        Model variables for the given year of solution

    Notes
    ---------
    survival_function is currently unused.
    """
    
    # Categories for the cost matrix (BCET)
    c2ti = {category: index for index, category in enumerate(titles['C2TI'])}
        
    # Conditional vector concerning technology properties (same for all regions)
    Svar = data['BCET'][:, :, c2ti['18 Variable (0 or 1)']]

    # TODO: This is a generic survival function
    HalfLife = data['BCET'][:, :, c2ti['9 Lifetime (years)']] / 2
    dLifeT = HalfLife / 10

    for age in range(len(titles['TYTI'])):

        age_matrix = np.ones_like(data['MSRV'][:, :, age]) * age
        data['MSRV'][:, :, age] = 1.0 - 0.5 * (1 + np.tanh(1.25 * (HalfLife - age_matrix) / dLifeT))

    # Copy over PRSC/EX values
    data['PRSC13'] = np.copy(time_lag['PRSC13'])
    data['EX13'] = np.copy(time_lag['EX13'])
    data['PRSC15'] = np.copy(time_lag['PRSC15'])
    data["REX13"] = np.copy(time_lag["REX13"])

    # %% First initialise if necessary
    T_Scal = 10  # Time scaling factor used in the share dynamics

    # Initialisation, which corresponds to lines 389 to 556 in fortran
    if year == 2013:
        data['PRSC13'] = np.copy(data['PRSCX'])
        data['EX13'] = np.copy(data['EXX'])
        data['REX13'] = np.copy(data['REXX'])

        data['MEWL'][:, :, 0] = data["MWLO"][:, :, 0]
        data['MEWK'][:, :, 0] = np.divide(data['MEWG'][:, :, 0], data['MEWL'][:, :, 0],
                                          where=data['MEWL'][:, :, 0] > 0.0) / 8766
        data['MEWS'][:, :, 0] = np.divide(data['MEWK'][:, :, 0], data['MEWK'][:, :, 0].sum(axis=1)[:, np.newaxis],
                                          where=data['MEWK'][:, :, 0].sum(axis=1)[:, np.newaxis] > 0.0)

        bcet, bcsc, mewl, mepd, merc, rery, mred, mres = cost_curves(
            data['BCET'], data['MCSC'], data['MEWDX'], data['MEWG'], data['MEWL'], data['MEPD'],
            data['MERC'], time_lag['MERC'], data['RERY'], data['MPTR'], data['MRED'], data['MRES'],
            titles['RTI'], titles['T2TI'], titles['ERTI'], year, 1.0
        )

        data['BCET'] = bcet
        data['MCSC'] = bcsc
        data['MEWL'] = mewl
        data['MEPD'] = mepd
        data['MERC'] = merc
        data['RERY'] = rery
        data['MRED'] = mred
        data['MRES'] = mres

        # Calculate LCOE for all technologies
        data = get_lcoe(data, titles, year) 
        # Get subset only of the subsets that are needed
        data = get_lcoes(data, titles)
               
        data = rldc(data, time_lag, iter_lag, year, titles)
        mslb, mllb, mes1, mes2 = dspch(data['MWDD'], data['MEWS'], data['MKLB'], data['MCRT'],
                                       data['MEWL'], data['MWMC'], data['MMCD'],
                                       len(titles['RTI']), len(titles['T2TI']), len(titles['LBTI']))
        data['MSLB'] = mslb
        data['MLLB'] = mllb
        data['MES1'] = mes1
        data['MES2'] = mes2

        # Total electricity demand
        tot_elec_dem = data['MEWDX'][:, 7, 0] * 1000 / 3.6

        earlysc = np.zeros([len(titles['RTI']), len(titles['T2TI'])])
        lifetsc = np.zeros([len(titles['RTI']), len(titles['T2TI'])])

        for r in range(len(titles['RTI'])):

            # Generation by tech x load band is share of total electricity demand
            glb3 = data['MSLB'][r, :, :] * data['MLLB'][r, :, :] * tot_elec_dem[r]
            # Capacity by tech x load band
            klb3 = glb3 / data['MLLB'][r, :, :]

            # Load factors
            data['MEWL'][r, :, 0] = np.zeros(len(titles['T2TI']))

            nonzero_cap = np.sum(klb3, axis=1) > 0
            data['MEWL'][r, nonzero_cap, 0] = np.sum(glb3[nonzero_cap, :], axis=1) / np.sum(klb3[nonzero_cap, :],
                                                                                            axis=1)

            # Generation by load band
            data['MWG1'][r, :, 0] = glb3[:, 0]
            data['MWG2'][r, :, 0] = glb3[:, 1]
            data['MWG3'][r, :, 0] = glb3[:, 2]
            data['MWG4'][r, :, 0] = glb3[:, 3]
            data['MWG5'][r, :, 0] = glb3[:, 4]
            data['MWG6'][r, :, 0] = glb3[:, 5]
            # To avoid division by 0 if 0 shares
            zero_lf = data['MEWL'][r, :, 0] == 0
            data['MEWL'][r, zero_lf, 0] = data['MWLO'][r, zero_lf, 0]

            # Capacities
            data['MEWK'][r, :, 0] = divide(data['MEWG'][r, :, 0], data['MEWL'][r, :, 0]) / 8766

            # Update market shares
            data["MEWS"][r, :, 0] = data['MEWK'][r, :, 0] / data['MEWK'][r, :, 0].sum()

            cap_diff = data['MEWK'][r, :, 0] - time_lag['MEWK'][r, :, 0]
            cap_drpctn = time_lag['MEWK'][r, :, 0] / time_lag['BCET'][r, :, c2ti['9 Lifetime (years)']]
            data['MEWI'][r, :, 0] = np.where(cap_diff > 0.0,
                                             cap_diff + cap_drpctn,
                                             cap_drpctn)

        data['MEWL'][:, :, 0] = data['MWLO'][:, :, 0].copy()
        data['MCFC'][:, :, 0] = data['MWLO'][:, :, 0].copy()
        data['BCET'][:, :, c2ti['11 Decision Load Factor']] = data['MCFC'][:, :, 0].copy()

        # Add standard deviation calculations
        investment_std = data['BCET'][:, :, c2ti['27 std of Investment ($/MWh)']]
        o_and_m_std = data['BCET'][:, :, c2ti['29 std of O&M ($/MW-yr)']]

        # Apply standard deviation to investment and O&M costs
        data['BCET'][:, :, c2ti['3 Investment ($/MW)']] *= (1 + investment_std)
        data['BCET'][:, :, c2ti['7 O&M ($/MWh)']] *= (1 + o_and_m_std)

        data = get_lcoe(data, titles, year)    
        data = get_lcoes(data, titles)

        
        data = get_marginal_fuel_prices_mewp(data, titles, Svar, glb3)

    # %%
    # Up to the last year of historical market share data
    elif year <= histend['MEWG']:
        if year == 2015:
            data['PRSC15'] = np.copy(data['PRSCX'])

        # Set starting values for MERC
        data['MERC'][:, 0, 0] = 0.255
        data['MERC'][:, 1, 0] = 5.689
        data['MERC'][:, 2, 0] = 0.4246
        data['MERC'][:, 3, 0] = 3.374
        data['MERC'][:, 4, 0] = 0.001
        data['MERC'][:, 7, 0] = 0.001

        if year > 2013:
            data['MEWL'][:, :, 0] = time_lag['MEWL'][:, :, 0].copy()

        cond = np.logical_and(data['MEWL'][:, :, 0] < 0.01, data['MWLO'][:, :, 0] > 0.0)
        data['MEWL'][:, :, 0] = np.where(cond,
                                         data['MWLO'][:, :, 0],
                                         data['MEWL'][:, :, 0])

        if year <= 2012:
            data['MEWK'][:, :, 0] = divide(data['MEWG'][:, :, 0], data['MWLO'][:, :, 0]) / 8766
        else:
            data['MEWK'][:, :, 0] = divide(data['MEWG'][:, :, 0], data['MEWL'][:, :, 0]) / 8766

        data['MEWS'][:, :, 0] = np.divide(data['MEWK'][:, :, 0], data['MEWK'][:, :, 0].sum(axis=1)[:, np.newaxis])

        if not time_lag['MMCD'][:, :, 0].any():
            time_lag = get_lcoe(data, titles, year)

        if year >= 2013:

            # 1 and 2 -- Estimate RLDC and storage parameters
            data = rldc(data, time_lag, iter_lag, year, titles)

            # 3--- Call dispatch routine to connect market shares to load bands
            if year == 2013:
                mslb, mllb, mes1, mes2 = dspch(data['MWDD'], data['MEWS'], data['MKLB'], data['MCRT'],
                                               data['MEWL'], data['MWMC'], data['MMCD'],
                                               len(titles['RTI']), len(titles['T2TI']), len(titles['LBTI']))
            else:
                mslb, mllb, mes1, mes2 = dspch(data['MWDD'], data['MEWS'], data['MKLB'], data['MCRT'],
                                               data['MEWL'], time_lag['MWMC'], time_lag['MMCD'],
                                               len(titles['RTI']), len(titles['T2TI']), len(titles['LBTI']))
            data['MSLB'] = mslb
            data['MLLB'] = mllb
            data['MES1'] = mes1
            data['MES2'] = mes2

            if year >= 2015:
                data['MSSP'][:, :, 0] = data['MSSP'][:, :, 0] * (data['PRSC13'][:, 0, 0, np.newaxis] / data['PRSC15'][:, 0, 0, np.newaxis]) / data['EX13'][33, 0, 0]
                data['MLSP'][:, :, 0] = data['MLSP'][:, :, 0] * (data['PRSC13'][:, 0, 0, np.newaxis] / data['PRSC15'][:, 0, 0, np.newaxis]) / data['EX13'][33, 0, 0]
                data['MSSM'][:, :, 0] = data['MSSM'][:, :, 0] * (data['PRSC13'][:, 0, 0, np.newaxis] / data['PRSC15'][:, 0, 0, np.newaxis]) / data['EX13'][33, 0, 0]
                data['MLSM'][:, :, 0] = data['MLSM'][:, :, 0] * (data['PRSC13'][:, 0, 0, np.newaxis] / data['PRSC15'][:, 0, 0, np.newaxis]) / data['EX13'][33, 0, 0]
            else:
                data['MSSP'][:, :, 0] = 0.0
                data['MLSP'][:, :, 0] = 0.0
                data['MSSM'][:, :, 0] = 0.0
                data['MLSM'][:, :, 0] = 0.0

            # Total electricity demand
            tot_elec_dem = data['MEWDX'][:, 7, 0] * 1000 / 3.6

            earlysc = np.zeros([len(titles['RTI']), len(titles['T2TI'])])
            lifetsc = np.zeros([len(titles['RTI']), len(titles['T2TI'])])

            for r in range(len(titles['RTI'])):

                # Generation by tech x load band is share of total electricity demand
                glb3 = data['MSLB'][r, :, :] * data['MLLB'][r, :, :] * tot_elec_dem[r]
                # Capacity by tech x load band
                klb3 = glb3 / data['MLLB'][r, :, :]

                # Load factors
                data['MEWL'][r, :, 0] = np.zeros(len(titles['T2TI']))

                nonzero_cap = np.sum(klb3, axis=1) > 0
                data['MEWL'][r, nonzero_cap, 0] = np.sum(glb3[nonzero_cap, :], axis=1) / np.sum(klb3[nonzero_cap, :], axis=1)

                # Generation by load band
                data['MWG1'][r, :, 0] = glb3[:, 0]
                data['MWG2'][r, :, 0] = glb3[:, 1]
                data['MWG3'][r, :, 0] = glb3[:, 2]
                data['MWG4'][r, :, 0] = glb3[:, 3]
                data['MWG5'][r, :, 0] = glb3[:, 4]
                data['MWG6'][r, :, 0] = glb3[:, 5]
                # To avoid division by 0 if 0 shares
                zero_lf = data['MEWL'][r, :, 0] == 0
                data['MEWL'][r, zero_lf, 0] = data['MWLO'][r, zero_lf, 0]

                # Adjust capacity factors for VRE due to curtailment, and to cover efficiency losses during
                data['MCGA'][r, 0, 0] = data['MCRT'][r, 0, 0] * np.sum(Svar[r, :] * data['MEWG'][r, :, 0])

                data['MCNA'][r, 0, 0] = np.maximum(data['MCGA'][r, 0, 0] - 0.45 * 2 * data['MLSG'][r, 0, 0], 0.55 * data['MCGA'][r, 0, 0])
                data['MCTN'][r, :, 0] = data['MCTG'][r, :, 0] * data['MCNA'][r, 0, 0] / data['MCGA'][r, 0, 0]

                data['MADG'][r, 0, 0] = data['MCGA'][r, 0, 0] - data['MCNA'][r, 0, 0] + data['MSSG'][r, 0, 0]

                data['MEWK'][:, :, 0] = divide(data['MEWG'][:, :, 0], data['MEWL'][:, :, 0]) / 8766

                data['MEWS'][:, :, 0] = np.divide(data['MEWK'][:, :, 0], data['MEWK'][:, :, 0].sum(axis=1)[:, np.newaxis],
                                                  where=data['MEWK'][:, :, 0].sum(axis=1)[:, np.newaxis] > 0.0)

                data['MEWE'][r, :, 0] = data['MEWG'][r, :, 0] * data['BCET'][r, :, c2ti['15 Emissions (tCO2/GWh)']] / 1e6

                cap_diff = data['MEWK'][r, :, 0] - time_lag['MEWK'][r, :, 0]
                cap_drpctn = time_lag['MEWK'][r, :, 0] / time_lag['BCET'][r, :, c2ti['9 Lifetime (years)']]
                data['MEWI'][r, :, 0] = np.where(cap_diff > 0.0,
                                                 cap_diff + cap_drpctn,
                                                 cap_drpctn)

            mewi0 = np.sum(data['MEWI'][:, :, 0], axis=0)
            dw = np.zeros(len(titles["T2TI"]))

            for i in range(len(titles["T2TI"])):
                dw_temp = np.copy(mewi0)
                dw_temp[dw_temp > dw_temp[i]] = dw_temp[i]
                dw[i] = np.dot(dw_temp, data['MEWB'][0, i, :])

            data["MEWW"][0, :, 0] = time_lag['MEWW'][0, :, 0] + dw

            data['BCET'][:, :, 1:17] = time_lag['BCET'][:, :, 1:17].copy()
            #data['BCET'][:, :, c2ti['21 Gamma ($/MWh)']] = data['MGAM'][:, :, 0]
            data['BCET'][:, :, c2ti['1 Carbon Costs ($/MWh)']] = set_carbon_tax(data, c2ti, year)

            if year > histend['BCET']:
                for tech in range(len(titles['T2TI'])):
                    if data['MEWW'][0, tech, 0] > 0.1:
                        data['BCET'][:, tech, c2ti['3 Investment ($/MW)']] = time_lag['BCET'][:, tech, c2ti['3 Investment ($/MW)']] * \
                                                                              (1.0 + data['BCET'][:, tech, c2ti['16 Learning exp']] * dw[tech] / data['MEWW'][0, tech, 0])
                        data['BCET'][:, tech, c2ti['4 std ($/MW)']] = time_lag['BCET'][:, tech, c2ti['4 std ($/MW)']] * \
                                                                       (1.0 + data['BCET'][:, tech, c2ti['16 Learning exp']] * dw[tech] / data['MEWW'][0, tech, 0])
                        data['BCET'][:, tech, c2ti['7 O&M ($/MWh)']] = time_lag['BCET'][:, tech, c2ti['7 O&M ($/MWh)']] * \
                                                                       (1.0 + data['BCET'][:, tech, c2ti['16 Learning exp']] * dw[tech] / data['MEWW'][0, tech, 0])
                        data['BCET'][:, tech, c2ti['8 std ($/MWh)']] = time_lag['BCET'][:, tech, c2ti['8 std ($/MWh)']] * \
                                                                       (1.0 + data['BCET'][:, tech, c2ti['16 Learning exp']] * dw[tech] / data['MEWW'][0, tech, 0])

            for r in range(len(titles['RTI'])):
                data['MWIY'][r, :, 0] = time_lag['MWIY'][r, :, 0] + data['MEWI'][r, :, 0] * data['BCET'][r, :, c2ti['3 Investment ($/MW)']] / 1.33

            bcet, bcsc, mewl, mepd, merc, rery, mred, mres = cost_curves(
                data['BCET'], data['MCSC'], data['MEWDX'], data['MEWG'], data['MEWL'], data['MEPD'],
                data['MERC'], time_lag['MERC'], data['RERY'], data['MPTR'], data['MRED'], data['MRES'],
                titles['RTI'], titles['T2TI'], titles['ERTI'], year, 1.0
            )
           
            data['BCET'] = bcet
            data['MCSC'] = bcsc
            data['MEWL'] = mewl
            data['MEPD'] = mepd
            data['MERC'] = merc
            data['RERY'] = rery
            data['MRED'] = mred
            data['MRES'] = mres

            data["MEWL"] = data["MEWL"] * (1 - data["MCTN"])
            data['BCET'][:, :, c2ti['11 Decision Load Factor']] *= (1 - data["MCTN"][:, :, 0])

            data = get_lcoe(data, titles, year)    
            data = get_lcoes(data, titles)

            data = get_marginal_fuel_prices_mewp(data, titles, Svar, glb3)

    return data


