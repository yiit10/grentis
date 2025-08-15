# one_time_simulation_fixed.py - Fixed version with df_full persistence and SOC continuity

from battery_brain_core import solve_ex_ante_model
import warnings

warnings.filterwarnings('ignore')
from pulp import *
import gurobipy

from epias_client import (
    get_tgt_token,
    get_mcp_data,
    get_id_avg_price_data,
    get_smf_data,
    get_injection_quantity,
    get_realtime_generation
)
from forecasts_utils import (
    generate_forecast_price_ar1,
    generate_forecast_price_id,
    generate_forecast_smf,
    generate_forecast_production_sto,  # Fixed: using _sto version
    generate_forecast_price_ar1_sto,
    generate_forecast_price_id_sto,
    generate_forecast_production_sto,
    generate_forecast_smf_sto,
    safe_normalize,
    predict_scenario_with_moving_window,
    generate_rank_preserving_matrix,
    compute_path_probabilities,
    mahalanobis_distance,
    find_closest_joint_scenario,
    generate_scenario_dt_for_window,
    generate_forecast_distribution_from_price_forecast,
    generate_forecast_distribution_from_id_price,
    generate_forecast_distribution_from_smf,
    generate_forecast_distribution_from_forecast,
    generate_rank_preserving_prod_scenarios,
    print_scenario_optimization_results,
    generate_scenario_dt_for_window_expost,
)
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np
import os

# Global df_full to persist across iterations
df_full = None
pd.set_option("display.max_columns", None)
istanbul = pytz.timezone("Europe/Istanbul")
now = datetime.now(istanbul).replace(minute=0, second=0, microsecond=0)
virtual_now = now - timedelta(days=3)


battery_params = {
    'capacity': 40.2,
    'power_limit': 61.8,
    'soc_min': 8,
    'soc_max': 40.2,
    'soc_target': 20,
    'efficiency': 0.97,
}

def initialize_static_inputs(virtual_now, START, END):
    USERNAME = "fyigitkavak@icloud.com"
    PASSWORD = "IDK1.35e"
    PLANT_ID = 2591

    battery_params = {
        'capacity': 40.2,
        'power_limit': 61.8,
        'soc_min': 8,
        'soc_max': 40.2,
        'soc_target': 20,
        'efficiency': 0.97,
    }

    # Set start date to 2024-01-01
    IST = pytz.timezone("Europe/Istanbul")
    current_hour = virtual_now.hour

    print(f"🚀 Starting simulation from {START} to {END}")

    # 🔐 Get token
    print("🔐 Getting authentication token...")
    tgt = get_tgt_token(USERNAME, PASSWORD)

    # === PRICE DATA ===
    print("📦 Fetching MCP data...")
    df_mcp = pd.DataFrame(get_mcp_data(tgt, START, END))
    df_mcp["date"] = pd.to_datetime(df_mcp["date"])

    print("📦 Fetching ID price data...")
    df_id = pd.DataFrame(get_id_avg_price_data(tgt, START, END))
    df_id["date"] = pd.to_datetime(df_id["date"])

    print("📦 Fetching SMF data...")
    df_smf = pd.DataFrame(get_smf_data(tgt, START, END))
    df_smf["date"] = pd.to_datetime(df_smf["date"])

    # === PRODUCTION DATA ===
    print("📦 Fetching production data (priority: realtime)...")
    prod_frames = []
    df_rt = pd.DataFrame()
    df_inj = pd.DataFrame()

    current_day = START.date()
    end_day = END.date()

    while current_day <= end_day:
        date_str = current_day.strftime("%Y-%m-%d")

        # Try realtime first
        df_daily_rt = get_realtime_generation(tgt, PLANT_ID, date_str)
        if df_daily_rt is not None and not df_daily_rt.empty:
            df_daily_rt["source"] = "realtime"
            prod_frames.append(df_daily_rt)
            df_rt = pd.concat([df_rt, df_daily_rt], ignore_index=True)
            print(f"✅ {date_str} → Realtime data retrieved")
        else:
            # Fallback to injection data
            df_daily_inj = get_injection_quantity(tgt, PLANT_ID, date_str)
            if df_daily_inj is not None and not df_daily_inj.empty:
                df_daily_inj["source"] = "injection"
                prod_frames.append(df_daily_inj)
                df_inj = pd.concat([df_inj, df_daily_inj], ignore_index=True)
                print(f"✅ {date_str} → Injection data used as fallback")
            else:
                print(f"🚫 {date_str} → No data available")

        current_day += timedelta(days=1)

    df_prod = pd.concat(prod_frames, ignore_index=True) if prod_frames else pd.DataFrame()

    return (df_mcp, df_id, df_smf, df_prod, battery_params, PLANT_ID)


def build_initial_df_full_complete(df_mcp, df_id, df_smf, df_prod, PLANT_ID, battery_params):
    """
    Build the complete df_full with all real values and pre-initialized decision variable columns
    """
    print("🏗️ Building complete df_full with all real values...")

    # Rename date columns to datetime
    df_mcp_copy = df_mcp.copy()
    df_id_copy = df_id.copy()
    df_smf_copy = df_smf.copy()

    df_mcp_copy.rename(columns={"date": "datetime"}, inplace=True)
    df_id_copy.rename(columns={"date": "datetime"}, inplace=True)
    df_smf_copy.rename(columns={"date": "datetime"}, inplace=True)

    # Rename real value columns
    df_mcp_copy = df_mcp_copy.rename(columns={
        "price": "DA_price_real"
    })
    df_id_copy = df_id_copy.rename(columns={
        "wap": "ID_price_real"
    })
    df_smf_copy = df_smf_copy.rename(columns={
        "systemMarginalPrice": "SMF_real"
    })

    # Process production data
    if not df_prod.empty:
        df_prod_copy = df_prod.copy()
        df_prod_copy["datetime"] = pd.to_datetime(df_prod_copy["date"])

        if "injectionQuantity" not in df_prod_copy.columns:
            if "total" in df_prod_copy.columns:
                df_prod_copy["injectionQuantity"] = pd.to_numeric(df_prod_copy["total"], errors="coerce")
            else:
                print("⚠️ Neither 'injectionQuantity' nor 'total' column found")
                df_prod_copy["injectionQuantity"] = 0

        df_prod_copy = df_prod_copy.rename(columns={
            "injectionQuantity": "real_production"
        })
        df_prod_copy["powerplantId"] = PLANT_ID
    else:
        df_prod_copy = pd.DataFrame()

    # Merge all real data
    df_full = df_mcp_copy.merge(df_id_copy, on="datetime", how="outer") \
        .merge(df_smf_copy, on="datetime", how="outer")

    if not df_prod_copy.empty:
        df_full = df_full.merge(df_prod_copy[["datetime", "real_production", "powerplantId"]],
                                on="datetime", how="left")
    else:
        df_full["real_production"] = 0
        df_full["powerplantId"] = PLANT_ID

    # Convert timezone
    df_full['datetime'] = df_full['datetime'].dt.tz_convert('UTC').dt.tz_convert('Europe/Istanbul')
    df_full['datetime'] = df_full['datetime'].dt.floor('H')

    # Initialize forecast columns (will be updated hourly)
    df_full["DA_price_forecasted"] = np.nan
    df_full["ID_price_forecasted"] = np.nan
    df_full["SMF_forecasted"] = np.nan
    df_full["forecasted_production_meteologica"] = np.nan

    # Initialize decision variable columns
    decision_vars = [
        'q_committed', 'q_committed_horizon', 'd_da_prod', 'd_da_bat', 'd_id_prod', 'd_id_bat',
        'x_id', 'x_ceza', 'is_charging', 'charge', 'soc',
        # Ex-post decision variables
        'soc_post', 'd_da_prod_extra', 'd_da_bat_extra', 'd_id_prod_extra', 'd_id_bat_extra',
        'd_da_prod_horizon', 'd_da_bat_horizon', 'charge_extra', 'x_id_adj', 'x_ceza_adj'
    ]

    for var in decision_vars:
        df_full[var] = np.nan

    # Initialize SOC with target value for first row
    df_full.loc[0, 'soc'] = battery_params['soc_target']

    print(f"✅ Complete df_full built with {len(df_full)} rows and {len(df_full.columns)} columns")
    print("🧩 Columns:", df_full.columns.tolist())

    return df_full


def update_forecasts_hourly(df_full, df_mcp, df_id, df_smf, df_prod, virtual_now, PLANT_ID):
    """
    Update forecast columns in df_full using *_sto functions
    """
    print(f"📈 Updating forecasts for {virtual_now}...")

    # Generate forecasts using _sto functions
    df_mcp_forecast = generate_forecast_price_ar1_sto(df_mcp.copy(), price_ceiling=3400, virtual_now=virtual_now)
    df_id_forecast = generate_forecast_price_id_sto(df_id.copy(), price_ceiling=3400, virtual_now=virtual_now)
    df_smf_forecast = generate_forecast_smf_sto(df_smf.copy(), price_ceiling=3400, virtual_now=virtual_now)

    # Production forecast
    df_prod_forecast = pd.DataFrame()
    if not df_prod.empty:
        df_prod_copy = df_prod.copy()
        df_prod_copy["datetime"] = pd.to_datetime(df_prod_copy["date"])
        if "injectionQuantity" not in df_prod_copy.columns:
            if "total" in df_prod_copy.columns:
                df_prod_copy["injectionQuantity"] = pd.to_numeric(df_prod_copy["total"], errors="coerce")
            else:
                df_prod_copy["injectionQuantity"] = 0

        if not df_prod_copy.empty:
            df_prod_forecast = generate_forecast_production_sto(df_prod_copy, virtual_now= virtual_now, power_limit= battery_params['power_limit'])
            df_prod_forecast = df_prod_forecast.rename(columns={
                "forecast_total": "forecasted_production_meteologica"
            })
            df_prod_forecast["powerplantId"] = PLANT_ID

    # Update df_full with new forecasts
    df_mcp_forecast.rename(columns={"date": "datetime"}, inplace=True)
    df_id_forecast.rename(columns={"date": "datetime"}, inplace=True)
    df_smf_forecast.rename(columns={"date": "datetime"}, inplace=True)

    # Convert timezone for forecasts
    df_mcp_forecast['datetime'] = df_mcp_forecast['datetime'].dt.tz_convert('UTC').dt.tz_convert(
        'Europe/Istanbul').dt.floor('H')
    df_id_forecast['datetime'] = df_id_forecast['datetime'].dt.tz_convert('UTC').dt.tz_convert(
        'Europe/Istanbul').dt.floor('H')
    df_smf_forecast['datetime'] = df_smf_forecast['datetime'].dt.tz_convert('UTC').dt.tz_convert(
        'Europe/Istanbul').dt.floor('H')

    # Update forecast columns in df_full
    for _, row in df_mcp_forecast.iterrows():
        mask = df_full['datetime'] == row['datetime']
        if mask.any():
            df_full.loc[mask, 'DA_price_forecasted'] = row['price_forecast']

    for _, row in df_id_forecast.iterrows():
        mask = df_full['datetime'] == row['datetime']
        if mask.any():
            df_full.loc[mask, 'ID_price_forecasted'] = row['price_forecast']

    for _, row in df_smf_forecast.iterrows():
        mask = df_full['datetime'] == row['datetime']
        if mask.any():
            df_full.loc[mask, 'SMF_forecasted'] = row['smf_forecast']

    if not df_prod_forecast.empty:
        df_prod_forecast['datetime'] = df_prod_forecast['datetime'].dt.tz_convert('UTC').dt.tz_convert(
            'Europe/Istanbul').dt.floor('H')
        for _, row in df_prod_forecast.iterrows():
            mask = df_full['datetime'] == row['datetime']
            if mask.any():
                df_full.loc[mask, 'forecasted_production_meteologica'] = row['forecasted_production_meteologica']

    return df_full


def apply_case_logic_flex(df, i, bp):
    # Get current row values
    d_da_prod = df.loc[i, 'd_da_prod']
    d_da_prod_extra = df.loc[i, 'd_da_prod_extra']
    d_da_bat = df.loc[i, 'd_da_bat']
    d_da_bat_extra = df.loc[i, 'd_da_bat_extra']
    charge = df.loc[i, 'charge']
    charge_extra = df.loc[i, 'charge_extra']
    d_id_prod = df.loc[i, 'd_id_prod']
    d_id_prod_extra = df.loc[i, 'd_id_prod_extra']
    d_id_bat = df.loc[i, 'd_id_bat']
    d_id_bat_extra = df.loc[i, 'd_id_bat_extra']
    real_prod = df.loc[i, 'real_production']
    forecasted_production = df.loc[i, 'forecasted_production_meteologica']
    # Battery parameters
    eff = bp['efficiency']
    soc_min = bp['soc_min']
    soc_max = bp['soc_max']

    if i == 0:
        prev_soc = bp['soc_target']
    else:
        prev_soc = df.loc[i - 1, 'soc']
        net_charge_prev = eff * df.loc[i - 1, 'charge'] - (1 / eff) * (
                df.loc[i - 1, 'd_id_bat'] + df.loc[i - 1, 'd_da_bat']
        )
        prev_soc = max(soc_min, min(soc_max, prev_soc + net_charge_prev))

    df.loc[i, 'soc'] = prev_soc  # saat başı SOC



    # Calculate delta (production difference)
    delta = real_prod - forecasted_production

    # Helper functions
    def bat_avail():
        """Available battery energy for discharge"""
        return max(0, prev_soc - soc_min) * eff

    def bat_room():
        """Available battery capacity for charging"""
        return max(0, soc_max - prev_soc) * eff


    # Determine case based on initial conditions
    case_id = f"{int((d_da_prod+d_da_prod_extra) > 0)}-{int((d_da_bat + d_da_bat_extra) > 0)}-{int((charge + charge_extra) > 0)}-{int((d_id_prod + d_id_prod_extra) > 0)}-{int((d_id_bat + d_id_bat_extra) > 0)}"

    # Process each case
    if case_id == "1-0-0-0-0":  # CASE 1: Only DA production
        if delta < 0:  # Need more energy
            need = abs(delta)
            avail = bat_avail()
            discharge = min(need, avail)
            grid = max(0, need - discharge)

            df.loc[i, 'd_da_bat'] += discharge
            df.loc[i, 'x_id'] += grid / 2
            df.loc[i, 'x_ceza'] += grid / 2

        elif delta > 0:  # Excess energy
            to_charge = min(delta, bat_room())
            df.loc[i, 'charge'] = to_charge
            df.loc[i, 'd_id_prod'] = delta - to_charge


    elif case_id == "1-1-0-0-0":  # CASE 2: DA production + DA battery
        if delta < 0:  # Need more energy
            need = abs(delta)
            avail = bat_avail()
            discharge = min(need, avail)
            grid = max(0, need - discharge)

            df.loc[i, 'd_da_bat'] += discharge
            df.loc[i, 'x_id'] += grid / 2
            df.loc[i, 'x_ceza'] += grid / 2

        elif delta > 0:  # Excess energy
            df.loc[i, 'd_id_prod'] += delta

    elif case_id == "1-1-0-0-1":  # CASE 3: DA production + DA battery + ID battery
        if delta < 0:  # Need more energy
            need = abs(delta)

            # First reduce ID battery usage
            cut = min((d_id_bat + d_id_bat_extra), need)
            df.loc[i, 'd_id_bat'] -= cut
            remain = need - cut

            # Then discharge from SOC if needed
            if remain > 0:
                avail = bat_avail()
                discharge = min(remain, avail)
                grid = max(0, remain - discharge)

                df.loc[i, 'x_id'] += grid / 2
                df.loc[i, 'x_ceza'] += grid / 2

            df.loc[i, 'd_da_bat'] += cut + (discharge if remain > 0 else 0)

        elif delta > 0:  # Excess energy
            df.loc[i, 'd_id_prod'] += delta

    elif case_id == "1-1-0-1-0":  # CASE 4: DA production + DA battery + ID production
        if delta < 0:  # Need more energy
            need = abs(delta)

            # First reduce ID production
            cut = min((d_id_prod + d_id_prod_extra), need)
            df.loc[i, 'd_id_prod'] -= cut
            df.loc[i, 'd_da_prod'] += cut
            remain = need - cut

            # Then discharge from SOC if needed
            if remain > 0:
                avail = bat_avail()
                discharge = min(remain, avail)
                grid = max(0, remain - discharge)

                df.loc[i, 'd_da_bat'] += discharge
                df.loc[i, 'x_id'] += grid / 2
                df.loc[i, 'x_ceza'] += grid / 2

        elif delta > 0:  # Excess energy
            df.loc[i, 'd_id_prod'] += delta

    elif case_id == "1-1-0-1-1":  # CASE 5: DA production + DA battery + ID production + ID battery
        if delta < 0:  # Need more energy
            need = abs(delta)

            # First reduce ID production
            dec_prod = min((d_id_prod + d_id_prod_extra), need)
            df.loc[i, 'd_id_prod'] -= dec_prod
            df.loc[i, 'd_da_prod'] += dec_prod
            remain = need - dec_prod

            # Then reduce ID battery
            if remain > 0:
                dec_bat_id = min((d_id_bat + d_id_bat_extra), remain)
                df.loc[i, 'd_id_bat'] -= dec_bat_id
                df.loc[i, 'd_da_bat'] += dec_bat_id
                remain -= dec_bat_id

            # Finally discharge from SOC if needed
            if remain > 0:
                avail = bat_avail()
                discharge = min(remain, avail)
                grid = max(0, remain - discharge)

                df.loc[i, 'd_da_bat'] += discharge
                df.loc[i, 'x_id'] += grid / 2
                df.loc[i, 'x_ceza'] += grid / 2

        elif delta > 0:  # Excess energy
            df.loc[i, 'd_id_prod'] += delta

    elif case_id == "1-0-0-0-1":  # CASE 6: DA production + ID battery
        if delta < 0:  # Need more energy
            need = abs(delta)

            # First reduce ID battery usage
            dec_bat_id = min((d_id_bat + d_id_bat_extra), need)
            df.loc[i, 'd_id_bat'] -= dec_bat_id
            df.loc[i, 'd_da_bat'] += dec_bat_id
            remain = need - dec_bat_id

            # Then discharge from SOC if needed
            if remain > 0:
                avail = bat_avail()
                discharge = min(remain, avail)
                grid = max(0, remain - discharge)

                df.loc[i, 'd_da_bat'] += discharge
                df.loc[i, 'x_id'] += grid / 2
                df.loc[i, 'x_ceza'] += grid / 2

        elif delta > 0:  # Excess energy
            df.loc[i, 'd_id_prod'] += delta

    elif case_id == "1-0-0-1-1":  # CASE 7: DA production + ID production + ID battery
        if delta < 0:  # Need more energy
            need = abs(delta)

            # First reduce ID production
            dec_prod = min((d_id_prod + d_id_prod_extra), need)
            df.loc[i, 'd_id_prod'] -= dec_prod
            df.loc[i, 'd_da_prod'] += dec_prod
            remain = need - dec_prod

            # Then reduce ID battery
            if remain > 0:
                dec_bat_id = min((d_id_bat + d_id_bat_extra), remain)
                df.loc[i, 'd_id_bat'] -= dec_bat_id
                df.loc[i, 'd_da_bat'] += dec_bat_id
                remain -= dec_bat_id

            # Finally discharge from SOC if needed
            if remain > 0:
                avail = bat_avail()
                discharge = min(remain, avail)
                grid = max(0, remain - discharge)

                df.loc[i, 'd_da_bat'] += discharge
                df.loc[i, 'x_id'] += grid / 2
                df.loc[i, 'x_ceza'] += grid / 2

        elif delta > 0:  # Excess energy
            df.loc[i, 'd_id_prod'] += delta

    elif case_id == "1-0-0-1-0":  # CASE 8: DA production + ID production
        if delta < 0:  # Need more energy
            need = abs(delta)

            # First reduce ID production
            dec_prod = min((d_id_prod + d_id_prod_extra), need)
            df.loc[i, 'd_id_prod'] -= dec_prod
            df.loc[i, 'd_da_prod'] += dec_prod
            remain = need - dec_prod

            # Then discharge from SOC if needed
            if remain > 0:
                avail = bat_avail()
                discharge = min(remain, avail)
                grid = max(0, remain - discharge)

                df.loc[i, 'd_da_bat'] += discharge
                df.loc[i, 'x_id'] += grid / 2
                df.loc[i, 'x_ceza'] += grid / 2

        elif delta > 0:  # Excess energy
            df.loc[i, 'd_id_prod'] += delta

    elif case_id == "1-0-1-0-0":  # CASE 9: DA production + charge
        if delta < 0:  # Need more energy
            need = abs(delta)

            # First reduce charging
            charge_dec = min((charge + charge_extra), need)
            df.loc[i, 'charge'] -= charge_dec
            remain = need - charge_dec

            # Then buy from grid if needed
            if remain > 0:
                df.loc[i, 'x_id'] += remain / 2
                df.loc[i, 'x_ceza'] += remain / 2

        elif delta > 0:  # Excess energy
            # Try to increase charging first
            room = bat_room()
            charge_inc = min(delta, room)
            df.loc[i, 'charge'] += charge_inc


            # Sell remaining to grid
            remaining = delta - charge_inc
            if remaining > 0:
                df.loc[i, 'd_id_prod'] += remaining

    # Store battery discharge for tracking
    df.loc[i, 'bat_to_discharge'] = df.loc[i, 'd_id_bat'] + df.loc[i, 'd_da_bat']

    return df


def run_full_simulation_fixed(virtual_now, df_mcp, df_id, df_smf, df_prod, battery_params, PLANT_ID, df_full = None):
    """
    Fixed simulation that maintains df_full persistence and SOC continuity
    """


    current_hour = virtual_now.hour

    # Initialize df_full only once
    if df_full is None:
        df_full = build_initial_df_full_complete(df_mcp, df_id, df_smf, df_prod, PLANT_ID, battery_params)
        print("✅ Initial df_full created with all real values.")

    # Update forecasts every hour
    df_full = update_forecasts_hourly(df_full, df_mcp, df_id, df_smf, df_prod, virtual_now, PLANT_ID)

    if current_hour == 10:
        print("⏰ Hour 10:00 - Running Ex-Ante model...")

        # Prepare data for ex-ante optimization
        forecast_dist_da = generate_forecast_distribution_from_price_forecast(
            df_full,
            price_ceiling=3400
        )
        forecast_dist_id = generate_forecast_distribution_from_id_price(
            df_full,
            price_ceiling=3400
        )
        forecast_dist_smf = generate_forecast_distribution_from_smf(
            df_full,
            price_ceiling=3400
        )
        forecast_dist_prod = generate_forecast_distribution_from_forecast(
            df_full,
            n_bins=100, error_sd=0.10
        )

        da_matrix = generate_rank_preserving_matrix(forecast_dist_da,
                                                    datetime_vec=forecast_dist_da['datetime'].unique(), n_scenarios=10)
        id_matrix = generate_rank_preserving_matrix(forecast_dist_id,
                                                    datetime_vec=forecast_dist_id['datetime'].unique(), n_scenarios=10)
        smf_matrix = generate_rank_preserving_matrix(forecast_dist_smf,
                                                     datetime_vec=forecast_dist_smf['datetime'].unique(),
                                                     n_scenarios=10)
        prod_matrix = generate_rank_preserving_prod_scenarios(forecast_dist_prod,
                                                              datetime_vec=forecast_dist_prod['datetime'].unique(),
                                                              n_scenarios=10)

        real_matrix = np.column_stack((df_full['DA_price_real'], df_full['ID_price_real'], df_full['SMF_real']))
        cov_est = np.cov(real_matrix, rowvar=False)

        ex_ante_time = virtual_now

        print("📆 Ex-Ante time window starting:", ex_ante_time)

        scenario_df, optimization_hours = generate_scenario_dt_for_window(
            da_matrix, id_matrix, smf_matrix, prod_matrix,
            real_da=df_full['DA_price_real'],
            real_id=df_full['ID_price_real'],
            real_smf=df_full['SMF_real'],
            cov_matrix=cov_est,
            ex_ante_time=ex_ante_time,
            db=df_full,
        )

        db_for_opt = scenario_df.rename(columns={
            'DA': 'DA_price_forecasted',
            'ID': 'ID_price_forecasted',
            'SMF': 'SMF_forecasted',
            'prod': 'forecasted_production'
        })

        db_for_opt = db_for_opt[db_for_opt['datetime'].isin(optimization_hours)].copy()
        db_for_opt['forecasted_production'] = db_for_opt['forecasted_production'].clip(
            upper=battery_params['power_limit'])
        print("db_for_opt columns:", db_for_opt.columns)
        print("\n🔍 db_for_opt İlk 25 satır ex-ante model öncesi:\n", db_for_opt.head(50))

        # Get current SOC from df_full
        current_soc_row = df_full[df_full['datetime'] == virtual_now]
        if not current_soc_row.empty and not pd.isna(current_soc_row['soc'].iloc[0]):
            initial_soc = current_soc_row['soc'].iloc[0]
        else:
            initial_soc = battery_params['soc_target']

        results = solve_ex_ante_model(db_for_opt, battery_params=battery_params, initial_soc=initial_soc)
        print("\n🧩 Optimization Results (Ex-Ante):", results)

        # Update df_full with ex-ante results
        for i, dt in enumerate(results['datetime']):
            mask = df_full['datetime'] == dt
            if mask.any():
                df_full.loc[mask, 'q_committed'] = results['q_committed'][i]
                df_full.loc[mask, 'd_da_prod'] = results['d_da_prod'][i]
                df_full.loc[mask, 'd_da_bat'] = results['d_da_bat'][i]
                df_full.loc[mask, 'd_id_prod'] = results['d_id_prod'][i]
                df_full.loc[mask, 'd_id_bat'] = results['d_id_bat'][i]
                df_full.loc[mask, 'x_id'] = results['x_id'][i]
                df_full.loc[mask, 'x_ceza'] = results['x_ceza'][i]
                df_full.loc[mask, 'is_charging'] = results['is_charging'][i]
                df_full.loc[mask, 'charge'] = results['charge'][i]
                df_full.loc[mask, 'soc'] = results['soc'][i]

        df_results = pd.DataFrame(results)
        print("\n🔍 df_full İlk 50 satır ex-ante model sonuçları:\n", df_full.head(50))



    else:
        print(f"⏰ {virtual_now} - Running Ex-Post model...")

        # Prepare data for ex-post optimization
        forecast_dist_da = generate_forecast_distribution_from_price_forecast(
            df_full,
            price_ceiling=3400
        )
        forecast_dist_id = generate_forecast_distribution_from_id_price(
            df_full,
            price_ceiling=3400
        )
        forecast_dist_smf = generate_forecast_distribution_from_smf(
            df_full,
            price_ceiling=3400
        )
        forecast_dist_prod = generate_forecast_distribution_from_forecast(
            df_full,
            n_bins=100, error_sd=0.10
        )

        da_matrix = generate_rank_preserving_matrix(forecast_dist_da,
                                                    datetime_vec=forecast_dist_da['datetime'].unique(), n_scenarios=10)
        id_matrix = generate_rank_preserving_matrix(forecast_dist_id,
                                                    datetime_vec=forecast_dist_id['datetime'].unique(), n_scenarios=10)
        smf_matrix = generate_rank_preserving_matrix(forecast_dist_smf,
                                                     datetime_vec=forecast_dist_smf['datetime'].unique(),
                                                     n_scenarios=10)
        prod_matrix = generate_rank_preserving_prod_scenarios(forecast_dist_prod,
                                                              datetime_vec=forecast_dist_prod['datetime'].unique(),
                                                              n_scenarios=10)

        real_matrix = np.column_stack((df_full['DA_price_real'], df_full['ID_price_real'], df_full['SMF_real']))
        cov_est = np.cov(real_matrix, rowvar=False)

        print("🛠️ Running Ex-Post model...")
        ex_post_time = virtual_now + timedelta(hours=1)

        ex_post_result = run_ex_post_model_scenario_based(
            db=df_full,
            ex_post_time=ex_post_time,
            battery_params=battery_params,
            prod_matrix=prod_matrix,
            da_matrix=da_matrix,
            id_matrix=id_matrix,
            smf_matrix=smf_matrix,
            real_da=df_full["DA_price_real"],
            real_id=df_full["ID_price_real"],
            real_smf=df_full["SMF_real"],
            window_size=10,
            sigma=2,
            cov_matrix=cov_est
        )


        if ex_post_result['status'] == 'success':
            df_full = ex_post_result['db']  # Update df_full with ex-post results

            # 👉 Eksik datetime ekle:
            dts = df_full[
                (df_full['datetime'] >= ex_post_time) &
                (df_full['datetime'] <= ex_post_time + timedelta(hours=47))
                ]['datetime'].reset_index(drop=True)

            df_results = pd.DataFrame(ex_post_result['decisions'])
            df_results['datetime'] = dts  # 🧠 Zamanı doldur
            df_results = df_results.set_index("datetime")
            df_full = df_full.set_index("datetime")

            for col in df_results.columns:
                df_full[col] = df_results[col]

            df_full = df_full.reset_index()

        else:
            print(f"❌ Ex-post optimization failed: {ex_post_result['message']}")
            df_results = pd.DataFrame(columns=["datetime"])  # Boş bir df döndür

    # Apply case logic for the current hour
    current_hour_idx = df_full[df_full['datetime'] == virtual_now].index
    if len(current_hour_idx) > 0:
        idx = current_hour_idx[0]
        df_full = apply_case_logic_flex(df_full, idx, battery_params)
        print(f"✅ Applied case logic for hour {virtual_now}")


    return df_results, df_full



def build_ex_post_optimization(db_window, battery_params, ex_post_time_index):
    """
    Build ex-post optimization model using PuLP
    """
    print(f"[{datetime.now()}] Building ex-post optimization model...")

    big_M = 1000
    horizon_hours = len(db_window)

    # Get current SOC
    current_soc = db_window.iloc[0].get('soc', battery_params['soc_target'])
    if pd.isna(current_soc):
        current_soc = battery_params['soc_target']


    # Extract scenario data
    price_DA_sce = db_window['DA_price_forecasted'].values
    price_ID_sce = db_window['ID_price_forecasted'].values
    price_SMF_sce = db_window['SMF_forecasted'].values
    production_sce = db_window['forecasted_production_meteologica'].values
    penalty_unit_price = np.maximum(price_DA_sce, price_SMF_sce) * 1.03

    # Extract existing decisions
    DA_price_real = db_window['DA_price_real'].values
    d_da_prod = db_window['d_da_prod'].fillna(0).values
    d_da_bat = db_window['d_da_bat'].fillna(0).values
    d_id_prod = db_window['d_id_prod'].fillna(0).values
    d_id_bat = db_window['d_id_bat'].fillna(0).values
    x_id = db_window['x_id'].fillna(0).values
    x_ceza = db_window['x_ceza'].fillna(0).values
    charge = db_window['charge'].fillna(0).values
    q_committed = db_window['q_committed'].fillna(0).values

    # Cost parameters
    unit_battery_cost = 1276

    eta = battery_params['efficiency']
    P_max = battery_params['power_limit']

    current_time = ex_post_time_index

    # Normalize current_time to tz-aware
    if current_time.tzinfo is None:
        current_time = pd.Timestamp(current_time).tz_localize('Europe/Istanbul')
    else:
        current_time = pd.Timestamp(current_time).tz_convert('Europe/Istanbul')

    # Get today's 00:00 and 23:00
    today_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = current_time.replace(hour=23, minute=0, second=0, microsecond=0)

    # Determine commitment hours
    if current_time.hour >= 10:
        # GÖP için yarının taahhüdü de eklenmiştir
        tomorrow_start = today_end + pd.Timedelta(hours=1)
        tomorrow_end = tomorrow_start + pd.Timedelta(hours=23)
        commitment_hours = pd.date_range(today_start, today_end, freq='H').union(
            pd.date_range(tomorrow_start, tomorrow_end, freq='H')
        )
    else:
        # Sadece bugünkü saatler kilit
        commitment_hours = pd.date_range(today_start, today_end, freq='H')
        commitment_hours = pd.DatetimeIndex(commitment_hours).tz_convert("Europe/Istanbul")



    # Flag: 0 = kilit, 1 = esnek
    is_gop_flexible = [0 if dt in commitment_hours else 1 for dt in db_window['datetime']]

    for i in range(horizon_hours):
        if is_gop_flexible[i] == 1:
            q_committed[i] = 0

    use_DA_forecast = []
    effective_DA_price = []
    for dt, real_price, forecast_price in zip(db_window['datetime'], DA_price_real, price_DA_sce):
        if dt.date() > current_time.date() and current_time.hour < 14:
            use_DA_forecast.append(1)
            effective_DA_price.append(forecast_price)
        else:
            use_DA_forecast.append(0)
            effective_DA_price.append(real_price)


    print(f"[{datetime.now()}] GOP and flexibility constraints defined, starting model definition...")
    print(f"[{datetime.now()}] Start time: {db_window.iloc[0]['datetime']}, SOC: {current_soc:.2f}")
    print("\n🧩 Columns:", db_window.columns.tolist())  # Tüm sütun adları listesi
    print("\n🔍 İlk 50 satır ex-post senaryo sonrası opt öncesi:\n", db_window.head(50))

    # Create the optimization model
    model = LpProblem("ExPostOptimization", LpMaximize)


    # Decision variables
    d_id_prod_extra = LpVariable.dicts("d_id_prod_extra", range(horizon_hours), -P_max, P_max)
    d_id_bat_extra = LpVariable.dicts("d_id_bat_extra", range(horizon_hours), -P_max, P_max)
    d_da_prod_extra = LpVariable.dicts("d_da_prod_extra", range(horizon_hours), -P_max, P_max)
    d_da_bat_extra = LpVariable.dicts("d_da_bat_extra", range(horizon_hours), -P_max, P_max)
    charge_extra = LpVariable.dicts("charge_extra", range(horizon_hours), -P_max, P_max)
    x_id_adj = LpVariable.dicts("x_id_adj", range(horizon_hours), -P_max, P_max)
    x_ceza_adj = LpVariable.dicts("x_ceza_adj", range(horizon_hours), -P_max, P_max)
    d_da_prod_horizon = LpVariable.dicts("d_da_prod_horizon", range(horizon_hours), 0, None)
    d_da_bat_horizon = LpVariable.dicts("d_da_bat_horizon", range(horizon_hours), 0, None)
    soc_post = LpVariable.dicts("soc_post", range(horizon_hours),
                                battery_params['soc_min'], battery_params['soc_max'])
    is_charging = LpVariable.dicts("is_charging", range(horizon_hours), cat='Binary')
    q_committed_horizon = LpVariable.dicts("q_committed_horizon", range(horizon_hours), 0, None)
    charge_horizon = LpVariable.dicts("charge_horizon", range(horizon_hours), 0, None)
    d_id_prod_horizon = LpVariable.dicts("d_id_prod_horizon", range(horizon_hours), 0, None)
    d_id_bat_horizon = LpVariable.dicts("d_id_bat_horizon", range(horizon_hours), 0, None)
    x_id_horizon = LpVariable.dicts("x_id_horizon", range(horizon_hours), 0, None)
    x_ceza_horizon = LpVariable.dicts("x_ceza_horizon", range(horizon_hours), 0, None)
    d_id_prod_adj = LpVariable.dicts("d_id_prod_adj", range(horizon_hours), 0, P_max)
    d_id_bat_adj = LpVariable.dicts("d_id_bat_adj", range(horizon_hours), 0, P_max)
    charge_adj = LpVariable.dicts("charge_adj", range(horizon_hours), 0, P_max)



    # Objective function
    objective = 0
    for t in range(horizon_hours):
        objective += (
                effective_DA_price[t] * (1 - is_gop_flexible[t]) * (d_da_prod[t] + d_da_bat[t]) +
                price_DA_sce[t] * is_gop_flexible[t] * q_committed_horizon[t] +
                price_ID_sce[t] * 0.97 * (1- is_gop_flexible[t]) * (
                            d_id_prod[t] + d_id_bat[t] + d_id_prod_extra[t] + d_id_bat_extra[t] - d_id_prod_adj[t] - d_id_bat_adj[t]) +
                price_ID_sce[t] * 0.97 *is_gop_flexible[t] * (d_id_prod_horizon[t] + d_id_bat_horizon[t]) -
                price_ID_sce[t] * 1.03 * (1 - is_gop_flexible[t]) *(x_id[t] + x_id_adj[t]) -
                price_ID_sce[t] * 1.03 * is_gop_flexible[t] * (x_id_horizon[t]) -
                penalty_unit_price[t] * (1- is_gop_flexible[t]) * (x_ceza[t] + x_ceza_adj[t]) -
                penalty_unit_price[t] * (is_gop_flexible[t]) * (x_ceza_horizon[t]) -
                (unit_battery_cost / 2) * (
                        ((1 - is_gop_flexible[t]) * (d_da_bat[t] + d_da_bat_extra[t])) +
                        (is_gop_flexible[t] * d_da_bat_horizon[t]) +
                        (1 - is_gop_flexible[t]) * (d_id_bat[t] + d_id_bat_extra[t] + charge[t] + charge_extra[t] - charge_adj[t]) +
                        (is_gop_flexible[t] * (d_id_bat_horizon[t] + charge_horizon[t])))
                )


    model += objective

    # Add constraints
    for t in range(horizon_hours):
        # Non-negativity constraints
        model += x_ceza[t] + x_ceza_adj[t] >= 0
        model += x_id[t] + x_id_adj[t] >= 0
        model += charge[t] + charge_extra[t] - charge_adj[t] >= 0
        model += d_id_bat[t] + d_id_bat_extra[t] - d_id_bat_adj[t] >= 0
        model += d_id_prod[t] + d_id_prod_extra[t] - d_id_prod_adj[t] >= 0
        model += d_da_prod[t] + d_da_prod_extra[t] >= 0
        model += d_da_bat[t] + d_da_bat_extra[t] >= 0


        # Commitment constraints
        model += (q_committed[t] ==
                  (1 - is_gop_flexible[t]) * (d_da_prod[t] + d_da_bat[t] + d_da_prod_extra[t] + d_da_bat_extra[t] + x_id_adj[t] + x_ceza_adj[t] + d_id_prod_adj[t] + d_id_bat_adj[t] + charge_adj[t]))

        model += (q_committed_horizon[t] ==
                  is_gop_flexible[t] * (d_da_prod_horizon[t] + d_da_bat_horizon[t]))

        model += battery_params['power_limit'] >= (1-is_gop_flexible[t]) * (d_da_prod[t] + d_da_bat[t] + d_da_prod_extra[t] + d_id_prod[t]) + is_gop_flexible[t] *  (d_da_prod_horizon[t] + d_da_bat_horizon[t] + d_id_prod_horizon[t] + d_id_bat_horizon[t])


        # Battery discharge constraint
        model += ((1 - is_gop_flexible[t]) * (d_da_bat[t] + d_da_bat_extra[t] + d_id_bat[t] + d_id_bat_extra[t]) +
                  is_gop_flexible[t] * (d_da_bat_horizon[t] + d_id_bat_horizon[t]))  <= P_max * (1 - is_charging[t])

        # Battery charge constraint
        model += ((charge[t]) <= P_max * is_charging[t])
        model += ((charge_extra[t]) <= P_max * is_charging[t])
        model += ((1 - is_gop_flexible[t]) * (charge_adj[t]) + is_gop_flexible[t] *
                  charge_horizon[t] <= P_max * is_charging[t])

        model += (((1 - is_gop_flexible[t]) * (production_sce[t]  + d_da_bat[t] + d_id_bat[t] + d_da_bat_extra[t] + d_id_bat_extra[t])
                                             + (is_gop_flexible[t] * (production_sce[t] + d_da_bat_horizon[t] + d_id_bat_horizon[t]))) == ((1 - is_gop_flexible[t]) * (q_committed[t] + d_id_prod[t] + d_id_prod_extra[t] - d_id_prod_adj[t] + d_id_bat[t] + d_id_bat_extra[t] - d_id_bat_adj[t] + charge[t] + charge_extra[t] - charge_adj[t] - x_id[t] - x_id_adj[t] - x_ceza[t] - x_ceza_adj[t])
                                                                                                                                                                               + ((is_gop_flexible[t]) * (q_committed_horizon[t] + d_id_prod_horizon[t] + d_id_bat_horizon[t] + charge_horizon[t] - x_id_horizon[t] - x_ceza_horizon[t]))))





    # SOC constraints
        model += soc_post[0] == current_soc
        model += d_da_bat_horizon[t] <= is_gop_flexible[t] * P_max
        model += d_da_prod_horizon[t] <= is_gop_flexible[t] * P_max

        model += battery_params['soc_min'] <= soc_post[t] - d_da_bat[t] - d_da_bat_extra[t] - d_id_bat[t] - d_id_bat_extra[t] - d_da_bat_horizon[t] + charge[t] + charge_extra[t] - charge_adj[t]
        model += battery_params['soc_max'] >= soc_post[t] - d_da_bat[t] - d_da_bat_extra[t] - d_id_bat[t] - d_id_bat_extra[t] - d_da_bat_horizon[t] + charge[t] + charge_extra[t] - charge_adj[t]

        for t in range(1, horizon_hours):
            model += (soc_post[t] == soc_post[t - 1] +
                      eta * ( (1- is_gop_flexible[t - 1]) * (charge[t - 1] + charge_extra[t - 1] - charge_adj[t - 1]) + is_gop_flexible[t - 1] * charge_horizon[t]) -
                      eta * (
                              (1 - is_gop_flexible[t - 1]) * (d_da_bat[t - 1] + d_da_bat_extra[t - 1] + d_id_bat[t - 1] + d_id_bat_extra[t - 1]) +
                              is_gop_flexible[t - 1] * (d_da_bat_horizon[t - 1] + d_id_bat_horizon[t -1])
                     ))

    print(f"[{datetime.now()}] Model setup completed.")

    return model, {
        'd_id_prod_extra': d_id_prod_extra,
        'd_id_bat_extra': d_id_bat_extra,
        'd_da_prod_extra': d_da_prod_extra,
        'd_da_bat_extra': d_da_bat_extra,
        'charge_extra': charge_extra,
        'd_da_prod_horizon': d_da_prod_horizon,
        'd_da_bat_horizon': d_da_bat_horizon,
        'x_id_adj': x_id_adj,
        'x_ceza_adj': x_ceza_adj,
        'soc_post': soc_post,
        'q_committed_horizon': q_committed_horizon,
        'd_id_prod_adj': d_id_prod_adj,
        'd_id_bat_adj': d_id_bat_adj,
        'charge_adj': charge_adj,
        'd_id_prod_horizon': d_id_prod_horizon,
        'd_id_bat_horizon': d_id_bat_horizon,
        'x_id_horizon': x_id_horizon,
        'x_ceza_horizon': x_ceza_horizon,
        'charge_horizon': charge_horizon,
        'forecasted_production': production_sce
    }


def run_ex_post_model_scenario_based(db, battery_params,
                                     da_matrix, id_matrix, smf_matrix, prod_matrix,
                                     real_da, real_id, real_smf, cov_matrix,
                                     ex_post_time,
                                     window_size=10, sigma=2):
    """
    Run ex-post optimization model with scenario-based approach
    """
    if isinstance(ex_post_time, str):
        ex_post_time = pd.Timestamp(ex_post_time, tz='Europe/Istanbul')

    print("\n🔍 db exporta gelen data:\n", db.head(5))
    ex_post_time = pd.Timestamp(ex_post_time).tz_convert("Europe/Istanbul").floor("H")
    horizon_hours = 47
    optimization_start_time = ex_post_time
    optimization_end_time = optimization_start_time + pd.Timedelta(hours=horizon_hours)
    print("🕒 ex_post_time:", ex_post_time)

    db['datetime'] = pd.to_datetime(db['datetime']).dt.tz_convert("Europe/Istanbul").dt.floor('H')

    # Filter data for the optimization window
    window_mask = (db['datetime'] >= ex_post_time) & (db['datetime'] <= optimization_end_time)
    db_window = db[window_mask].copy().reset_index(drop=True)
    print("🧮 db_window datetime values:", db['datetime'].dt.tz_convert("Europe/Istanbul").unique())
    cols_to_fill_expost = ['charge','is_charging','x_ceza','x_id','d_id_bat','d_id_prod','d_da_bat','d_da_prod','q_committed']
    fill_mask = db['q_committed'].isna()
    for col in cols_to_fill_expost:
        db.loc[fill_mask, col] = 0

    print("\n🧩 Columns:", db_window.columns.tolist())  # Tüm sütun adları listesi
    print("\n🔍 db_window Ex-post senaryo öncesi İlk 50 satır:\n", db_window.head(50))
    print("❓ db satır sayısı:", len(db))
    print("❓ ex_post_time var mı?", ex_post_time in db['datetime'].values)

    print(f"[{datetime.now()}] Ex-post scenario started...")

    # Get current SOC
    current_soc_row = db_window[db_window['datetime'] == ex_post_time - timedelta(hours=1)]
    if len(current_soc_row) == 0 or pd.isna(current_soc_row['soc'].iloc[0]):
        current_soc = battery_params['soc_target']
    else:
        current_soc = current_soc_row['soc'].iloc[0]

    print(f"Ex-post time: {ex_post_time}, current SOC: {current_soc:.2f}")

    # Generate ex-post scenarios
    print(f"[{datetime.now()}] Generating ex-post scenarios...")
    scenario_dt_expost, optimization_hours = generate_scenario_dt_for_window_expost(
        da_matrix, id_matrix, smf_matrix, prod_matrix,
        real_da, real_id, real_smf, cov_matrix,
        ex_post_time, db,
        window_size=window_size, sigma=sigma
    )

    print(f"[{datetime.now()}] Ex-post scenarios successfully generated.")
    scenario_dict = scenario_dt_expost.set_index('datetime').to_dict('index')

    for idx, row in db_window.iterrows():
        dt = row['datetime']
        if dt in scenario_dict:
            db_window.loc[idx, 'DA'] = scenario_dict[dt]['DA']
            db_window.loc[idx, 'ID'] = scenario_dict[dt]['ID']
            db_window.loc[idx, 'SMF'] = scenario_dict[dt]['SMF']
            db_window.loc[idx, 'prod'] = scenario_dict[dt]['prod']
            db_window.loc[idx, 'penalty'] = scenario_dict[dt]['penalty']

    print(f"[{datetime.now()}] Starting ex-post optimization model...")

    # Build and solve the optimization model
    ex_post_time_index = db_window[db_window['datetime'] == ex_post_time]['datetime'].iloc[0]
    model, variables = build_ex_post_optimization(db_window, battery_params, ex_post_time_index)

    print(f"[{datetime.now()}] Ex-post optimization model successfully built, starting solution...")

    # Solve the model
    try:
        model.solve(GUROBI(msg=1))

        if model.status != 1:  # LpStatusOptimal
            model.writeLP("expost_debug.lp")
            return {
                'status': 'failed',
                'message': f'Optimization failed with status: {LpStatus[model.status]}'
            }

        print(f"[{datetime.now()}] Solution successful, writing results...")

        # Extract results
        results = {}
        for var_name, var_dict in variables.items():
            results[var_name] = [
                getattr(var_dict[t], "varValue", var_dict[t]) for t in range(len(db_window))
            ]

        # Update the original database with results
        start_idx = db[db['datetime'] == optimization_start_time].index[0]
        end_idx = db[db['datetime'] == optimization_end_time].index[0]

        for var_name, values in results.items():
            if var_name in db.columns:
                db.loc[start_idx:end_idx, var_name] = values[0:]  # Skip first 2 hours
            else:
                # Create new column if it doesn't exist
                db[var_name] = np.nan
                db.loc[start_idx:end_idx, var_name] = values[0:]

        print(f"[{datetime.now()}] All values successfully transferred.")

        return {
            'status': 'success',
            'db': db,
            'model': model,
            'scenario_info': scenario_dt_expost,
            'decisions': results,
            'objective_value': model.objective.value()
        }

    except Exception as e:
        return {
            'status': 'failed',
            'message': str(e)
        }

def calculate_battery_trading_metrics_flex(df, battery_params):
    """
    Calculate comprehensive revenue and battery metrics for battery trading simulation

    Args:
        df: DataFrame with trading decisions and actual market data
        battery_params: Dict with battery specifications
        eur_to_tl: EUR to TL exchange rate

    Returns:
        DataFrame with all financial and battery metrics
    """

    # Create a copy to avoid modifying original data
    df_metrics = df.copy()

    # Initialize required columns if they don't exist
    required_cols = [
        'revenue', 'battery_cost', 'penalty', 'net_profit', 'bat_to_discharge'
        ,'id_costs',
        'total_battery_usage', 'degradation_cost', 'cycle_equivalent'
    ]

    for col in required_cols:
        if col not in df_metrics.columns:
            df_metrics[col] = 0.0

    # === COST CONSTANTS ===
    # Battery degradation cost per MWh (OPEX + CAPEX components)
    battery_cost_per_mwh = 1276

    # Process each hour
    for i in range(len(df_metrics)):
        row = df_metrics.iloc[i]

        # === 1. REVENUE CALCULATION ===
        # Revenue from Day-Ahead commitments
        da_revenue = row['q_committed'] * row['DA_price_real']

        # Revenue from Intraday sales (production and battery)
        id_revenue = (row['d_id_prod'] + row['d_id_bat'] + row['d_id_prod_extra'] + row['d_id_bat_extra'] - row['d_id_prod_adj'] - row['d_id_bat_adj']) * row['ID_price_real']

        total_revenue = da_revenue + id_revenue
        df_metrics.loc[i, 'DA_revenue'] = da_revenue
        df_metrics.loc[i, 'ID_revenue'] = id_revenue
        df_metrics.loc[i, 'revenue'] = total_revenue

        # === 2. BATTERY COSTS ===
        # Total battery usage (charge + discharge)
        total_battery_usage = row['d_da_bat'] + row['d_id_bat'] + row['charge'] + row['d_da_bat_extra'] + row['d_id_bat_extra'] - row['charge_adj'] + row['charge_extra']
        df_metrics.loc[i, 'total_battery_usage'] = total_battery_usage

        # Battery degradation cost
        battery_cost = total_battery_usage * battery_cost_per_mwh / 2
        df_metrics.loc[i, 'battery_cost'] = battery_cost

        # Battery cycle calculation (charge + discharge) / 2
        cycle_equivalent = total_battery_usage / 2
        df_metrics.loc[i, 'cycle_equivalent'] = cycle_equivalent

        # === 3. PRODUCTION DEVIATION ADJUSTMENTS ===
        forecasted_prod = row['forecasted_production']
        real_prod = row['real_production']
        production_diff = real_prod - forecasted_prod

        id_price = row['ID_price_real']
        penalty_price = max(row['DA_price_real'], row['SMF_real']) * 1.03

        # === 4. PENALTY COSTS ===
        # Standard penalty from imbalance (x_ceza)
        standard_penalty = (row['x_ceza'] + row['x_ceza_adj']) * penalty_price
        total_penalty = standard_penalty
        df_metrics.loc[i, 'penalty'] = total_penalty

        # === 5. INTRADAY MARKET COSTS ===
        # Cost of buying from intraday market (x_id)
        id_cost = (row['x_id'] + row['x_id_adj']) * id_price
        df_metrics.loc[i, 'id_costs'] = id_cost

        # === 6. NET PROFIT CALCULATION ===
        net_profit = (
                df_metrics.loc[i, 'DA_revenue'] +
                df_metrics.loc[i, 'ID_revenue'] -
                df_metrics.loc[i, 'battery_cost'] -
                df_metrics.loc[i, 'penalty'] -
                df_metrics.loc[i, 'id_costs']
        )

        df_metrics.loc[i, 'net_profit'] = net_profit

    return df_metrics


# if __name__ == "__main__":
    import pickle

  #  IST = pytz.timezone("Europe/Istanbul")
  #  START = IST.localize(datetime(2025, 1, 1, 0, 0))
  #  END = IST.localize(datetime(2025, 7, 24, 23, 0))

   # virtual_now = START + timedelta(hours=10)

   # df_mcp, df_id, df_smf, df_prod, battery_params, PLANT_ID = initialize_static_inputs(
   #     virtual_now=virtual_now, START=START, END=END
   # )

   # while virtual_now <= END:
   #     print(f"\n🔄 Simülasyon Zamanı: {virtual_now}")

    #    df_results, df_full = run_full_simulation_fixed(
    #        virtual_now=virtual_now,
     #       df_mcp=df_mcp,
     #       df_id=df_id,
      #      df_smf=df_smf,
      #      df_prod=df_prod,
       #     battery_params=battery_params,
        #    PLANT_ID=PLANT_ID
      #  )

      #  if virtual_now.hour == 0 and virtual_now.day % 5 == 0:
       #     date_str = virtual_now.strftime("%Y-%m-%d_%H-%M")
        #    file_name = f"batterybrain_simulation_{date_str}_plant{PLANT_ID}.csv"
        #    df_full.to_csv(file_name, index=False)
        #    print(f"💾 Kayıt yapıldı: {file_name}")

      #  virtual_now += timedelta(hours=1)

    # ➕ Simülasyon sonunda metrikleri hesapla
 #   df_metrics = calculate_battery_trading_metrics_flex(df_full, battery_params)

    # 🎯 Final veri kaydı
 #   final_file = f"batterybrain_final_full_plant{PLANT_ID}.csv"
 #   df_metrics.to_csv(final_file, index=False)
 #   print(f"✅ Tüm simülasyon tamamlandı. Sonuç kaydedildi: {final_file}")


def run_simulation_flex_incremental(csv_path, df_mcp, df_id, df_smf, df_prod, battery_params, PLANT_ID):
    # Load existing CSV
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path, parse_dates=['datetime'])
        df_existing['datetime'] = df_existing['datetime'].dt.tz_localize('Europe/Istanbul', ambiguous='NaT', nonexistent='NaT')
        df_existing = df_existing.dropna(subset=['datetime'])
    else:
        df_existing = pd.DataFrame()

    # En son optimize edilmiş saat: q_committed_horizon > 0 olan ilk saatin bir saat ÖNCESİ
    last_optimized_dt = None
    if not df_existing.empty and 'q_committed_horizon' in df_existing.columns:
        mask = df_existing['q_committed_horizon'] > 0
        committed_times = df_existing.loc[mask, 'datetime']
        if not committed_times.empty:
            first_committed = committed_times.min()
            last_optimized_dt = first_committed - timedelta(hours=1)

    # virtual_now'u belirle
    istanbul = pytz.timezone("Europe/Istanbul")
    now = datetime.now(istanbul).replace(minute=0, second=0, microsecond=0)
    virtual_now = now - timedelta(days=3)

    if last_optimized_dt is None:
        print("🔄 CSV'de daha önceden optimize edilmiş bir saat yok, baştan başlanacak.")
        current_time = pd.Timestamp("2025-06-30 10:00:00", tz=istanbul)
        all_results = pd.DataFrame()
    else:
        current_time = last_optimized_dt + timedelta(hours=1)
        all_results = df_existing.copy()
        print(f"🔄 Devam edilecek saat: {current_time} → Virtual Now: {virtual_now}")

    while current_time <= virtual_now:
        try:
            df_new, _ = run_full_simulation_fixed(current_time, df_mcp, df_id, df_smf, df_prod, battery_params, PLANT_ID)
            if not df_new.empty:
                all_results = pd.concat([all_results, df_new], ignore_index=True)
                all_results = all_results.drop_duplicates(subset='datetime').sort_values('datetime')
                print(f"✅ {current_time} saati optimize edildi.")
            else:
                print(f"⚠️ {current_time} için boş dönüş yapıldı")
        except Exception as e:
            print(f"❌ Hata: {e}")
        current_time += timedelta(hours=1)

    # Save
    all_results.to_csv(csv_path, index=False)
    print(f"💾 CSV güncellendi: {csv_path}")
    return all_results


def simulate_benchmark_flex_case(df, battery_params):
    """
    Flex benchmark: her gün saat 10:00'da stochastic forecast ile
    bir sonraki gün 00:00-23:00 arası için q_committed belirlenir.
    Gerçek üretimle fark ID/batarya/ceza ile kapatılır.
    """
    df = df.copy()
    df = df.sort_values("datetime")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.reset_index(drop=True)

    df["q_committed"] = 0.0
    df["SoC"] = 0.0
    df["charge_used"] = 0.0
    df["discharge_used"] = 0.0
    df["id_satis"] = 0.0
    df["id_alim"] = 0.0
    df["id_gelir"] = 0.0
    df["id_maliyet"] = 0.0
    df["ceza_miktar"] = 0.0
    df["ceza_maliyet"] = 0.0
    df["batarya_cost"] = 0.0
    df["DA_revenue"] = 0.0
    df["net_profit"] = 0.0

    eta = battery_params['efficiency']
    soc_min = battery_params['soc_min']
    soc_max = battery_params['soc_max']
    soc = battery_params['soc_target']
    power_limit = battery_params['power_limit']
    unit_bat_cost = 1276
    penalty_share = 0.50
    id_share = 1 - penalty_share

    istanbul = pytz.timezone("Europe/Istanbul")

    for i, row in df.iterrows():
        dt = row["datetime"]
        df.at[i, "SoC"] = soc

        # Her gün saat 10:00'da, bir sonraki gün için q_committed hesapla
        if dt.hour == 10:
            next_day_start = (dt + timedelta(days=1)).replace(hour=0)
            next_day_end = next_day_start + timedelta(hours=23)

            mask = (df["datetime"] >= next_day_start) & (df["datetime"] <= next_day_end)
            df_subset = (
                df[mask][["datetime", "real_production", "powerPlantId"]]
                .rename(columns={"datetime": "date", "real_production": "injectionQuantity"})
                .reset_index(drop=True)
            )

            df_prod_forecast = generate_forecast_production_sto(
                df_subset,
                virtual_now=dt,
                power_limit=power_limit
            )

            df.loc[mask, "q_committed"] = df_prod_forecast["forecast_total"].values


        # Uygulama: saatlik kâr hesaplama
        q = df.loc[i, "q_committed"]
        p_real = df.loc[i, "real_production"]
        da_price = df.loc[i, "DA_price_real"]
        id_price = df.loc[i, "ID_price_real"]
        smf_price = df.loc[i, "SMF_real"]

        if pd.isna(p_real) or pd.isna(q):
            continue

        delta = p_real - q
        df.at[i, "DA_revenue"] = q * da_price

        if delta > 0:
            # Fazla üretim varsa
            charge_amt = min(delta, soc_max - soc, power_limit)
            soc += charge_amt * eta
            soc = min(soc, soc_max)

            id_sale_amt = delta - charge_amt
            df.at[i, "charge_used"] = charge_amt
            df.at[i, "id_satis"] = id_sale_amt
            df.at[i, "id_gelir"] = id_sale_amt * id_price
            df.at[i, "batarya_cost"] = charge_amt * (unit_bat_cost / 2)

        elif delta < 0:
            deficit = -delta
            discharge_amt = min(deficit, soc - soc_min, power_limit)
            soc -= discharge_amt / eta
            soc = max(soc, soc_min)

            residual_deficit = deficit - discharge_amt
            id_buy_amt = residual_deficit * id_share
            ceza_amt = residual_deficit * penalty_share
            ceza_price = max(smf_price, da_price) * 1.03

            df.at[i, "discharge_used"] = discharge_amt
            df.at[i, "id_alim"] = id_buy_amt
            df.at[i, "id_maliyet"] = id_buy_amt * id_price
            df.at[i, "ceza_miktar"] = ceza_amt
            df.at[i, "ceza_maliyet"] = ceza_amt * ceza_price
            df.at[i, "batarya_cost"] = discharge_amt * (unit_bat_cost / 2)

        df.at[i, "net_profit"] = (
            df.at[i, "DA_revenue"] +
            df.at[i, "id_gelir"] -
            df.at[i, "id_maliyet"] -
            df.at[i, "ceza_maliyet"]
        )

    return df

def run_full_simulation_benchmark_flex():
    """
    Run benchmark simulation for flex plant using stochastic production forecasts and real price/production data.
    """
    import pytz
    from datetime import datetime, timedelta
    import pandas as pd
    from epias_client import get_tgt_token, get_mcp_data, get_id_avg_price_data, get_smf_data, get_realtime_generation, get_injection_quantity


    # === Parameters ===
    USERNAME = "fyigitkavak@icloud.com"
    PASSWORD = "IDK1.35e"
    PLANT_ID = 2591
    IST = pytz.timezone("Europe/Istanbul")
    battery_params = {
        'capacity': 61.8,
        'power_limit': 61.8,
        'soc_min': 8,
        'soc_max': 40.2,
        'soc_target': 20,
        'efficiency': 0.97,
    }

    START = datetime(2024, 1, 1, 0, 0, tzinfo=IST)
    END = datetime(2024, 12, 31, 0, 0, tzinfo=IST)

    print(f"🔐 Getting token...")
    tgt = get_tgt_token(USERNAME, PASSWORD)

    # === Price Data ===
    print("📦 Fetching real price data...")
    df_mcp = pd.DataFrame(get_mcp_data(tgt, START, END))
    df_mcp["datetime"] = pd.to_datetime(df_mcp["date"])
    df_mcp = df_mcp.rename(columns={"price": "DA_price_real"})

    df_id = pd.DataFrame(get_id_avg_price_data(tgt, START, END))
    df_id["datetime"] = pd.to_datetime(df_id["date"])
    df_id = df_id.rename(columns={"wap": "ID_price_real"})

    df_smf = pd.DataFrame(get_smf_data(tgt, START, END))
    df_smf["datetime"] = pd.to_datetime(df_smf["date"])
    df_smf = df_smf.rename(columns={"systemMarginalPrice": "SMF_real"})

    # === Production Data ===
    print("📦 Fetching real production data...")
    prod_frames = []
    current_day = START.date()
    while current_day <= END.date():
        date_str = current_day.strftime("%Y-%m-%d")
        df_rt = get_realtime_generation(tgt, PLANT_ID, date_str)
        if df_rt is not None and not df_rt.empty:
            df_rt["source"] = "realtime"
            prod_frames.append(df_rt)
        else:
            df_inj = get_injection_quantity(tgt, PLANT_ID, date_str)
            if df_inj is not None and not df_inj.empty:
                df_inj["source"] = "injection"
                prod_frames.append(df_inj)
        current_day += timedelta(days=1)
    df_prod = pd.concat(prod_frames, ignore_index=True)
    df_prod["datetime"] = pd.to_datetime(df_prod["date"])
    df_prod = df_prod.rename(columns={"total": "real_production"})
    df_prod["injectionQuantity"] = df_prod["real_production"]

    # === Merge All ===
    df = df_mcp.merge(df_id, on="datetime", how="outer")
    df = df.merge(df_smf, on="datetime", how="outer")
    df = df.merge(df_prod[["datetime", "real_production"]], on="datetime", how="outer")
    df = df.sort_values("datetime").reset_index(drop=True)
    df["powerPlantId"] = PLANT_ID

    print(df.head(5))

    # === Run Benchmark Simulation ===
    print("🚀 Running benchmark simulation for flex...")
    df_results = simulate_benchmark_flex_case(df, battery_params)

    # === Save to CSV ===
    output_file = f"benchmark_flex_results_2024_{PLANT_ID}.csv"
    df_results.to_csv(output_file, index=False)
    print(f"✅ Results saved to {output_file}")

if __name__ == "__main__":
   run_full_simulation_benchmark_flex()