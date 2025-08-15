# one_time_simulation.py - One-time simulation from 2024-01-01
from main import virtual_now
from performance_back import (
    apply_case_logic,
    battery_params, calculate_battery_trading_metrics, simulate_benchmark_case1, show_hourly_decisions, test_benchmark_simulation, validate_benchmark_simulation, prepare_benchmark_inputs
)
from battery_brain_core import solve_ex_ante_model, prepare_optimization_data
import warnings
warnings.filterwarnings('ignore')
from pulp import *


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
    generate_forecast_production,
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

df_full = None

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


    return(df_mcp, df_id, df_smf, df_prod, battery_params, PLANT_ID)

def run_full_simulation_from_2024_flex(
    virtual_now,
    df_id,
    df_smf,
    df_mcp,
    battery_params,
    df_prod,
    PLANT_ID,
    df_full = None
):
    current_hour = virtual_now.hour

    if df_full is None or df_full.empty:
        df_full = build_initial_df_full(df_id, df_smf, df_mcp, df_prod, PLANT_ID)
        print("✅ Yeni df_full oluşturuldu.")
    else:
        print("🔁 Mevcut df_full ile devam ediliyor.")

    if current_hour == 10:
        print("⏰ Saat 10:00 - Ex-Ante model çalıştırılıyor...")

        df_mcp = generate_forecast_price_ar1_sto(df_mcp, price_ceiling=3400, virtual_now=virtual_now)
        df_mcp = df_mcp.rename(columns={
            "price_real": "DA_price_real",
            "price_forecast": "DA_price_forecasted"
        })
        df_id = generate_forecast_price_id_sto(df_id, price_ceiling=3400, virtual_now=virtual_now)
        df_id = df_id.rename(columns={
            "price_real": "ID_price_real",
            "price_forecast": "ID_price_forecasted"
        })
        df_smf = generate_forecast_smf_sto(df_smf, price_ceiling=3400, virtual_now=virtual_now)
        df_smf = df_smf.rename(columns={
            "smf_real": "SMF_real",
            "smf_forecast": "SMF_forecasted"
        })
        # === PRODUCTION FORECAST ===
        print("📦 Generating production forecast...")
        if not df_prod.empty:
            df_prod["datetime"] = pd.to_datetime(df_prod["date"])
            if "injectionQuantity" not in df_prod.columns:
                if "total" in df_prod.columns:
                    df_prod["injectionQuantity"] = pd.to_numeric(df_prod["total"], errors="coerce")
                else:
                    print("⚠️ Neither 'injectionQuantity' nor 'total' column found → Cannot generate forecast.")
                    df_prod_forecast = pd.DataFrame()

            if not df_prod.empty:
                df_prod_forecast = generate_forecast_production(df_prod, error_sd=0.05)
                df_prod_forecast = df_prod_forecast.rename(columns={
                    "forecast_total": "forecasted_production_meteologica"
                })
                df_prod_forecast["powerplantId"] = PLANT_ID


        else:
            print("⚠ Cannot generate forecast: df_prod is empty.")
            df_prod_forecast = pd.DataFrame()

        df_mcp.rename(columns={"date": "datetime"}, inplace=True)
        df_id.rename(columns={"date": "datetime"}, inplace=True)
        df_smf.rename(columns={"date": "datetime"}, inplace=True)
        df_prices = df_mcp.merge(df_id, on="datetime", how="outer") \
            .merge(df_smf, on="datetime", how="outer")
        df_full = df_prices.merge(df_prod, on="datetime", how="left")
        df_full = df_full.merge(df_prod_forecast[["datetime", "forecasted_production_meteologica"]], on="datetime",
                                how="left")
        df_full['datetime'] = df_full['datetime'].dt.tz_convert('UTC').dt.tz_convert('Europe/Istanbul')

        # Eğer hala sorun yaşarsan (ki sanmıyorum), floor('H') işlemini bu dönüşümden sonra da yapabilirsin:
        df_full['datetime'] = df_full['datetime'].dt.floor('H')
        print(df_full.head(3))  # İlk 3 satır
        print("\n🧩 Columns:", df_full.columns.tolist())  # Tüm sütun adları listesi
        forecast_dist_da = generate_forecast_distribution_from_price_forecast(df_mcp, price_ceiling=3400)
        forecast_dist_id = generate_forecast_distribution_from_id_price(df_id, price_ceiling=3400)
        forecast_dist_smf = generate_forecast_distribution_from_smf(df_smf, price_ceiling=3400)
        forecast_dist_prod = generate_forecast_distribution_from_forecast(df_prod_forecast, n_bins=100,
                                                                          error_sd=0.10)



        da_matrix = generate_rank_preserving_matrix(forecast_dist_da,
                                                    datetime_vec=forecast_dist_da['datetime'].unique(), n_scenarios=100)
        id_matrix = generate_rank_preserving_matrix(forecast_dist_id,
                                                    datetime_vec=forecast_dist_id['datetime'].unique(), n_scenarios=100)
        smf_matrix = generate_rank_preserving_matrix(forecast_dist_smf,
                                                     datetime_vec=forecast_dist_smf['datetime'].unique(),
                                                     n_scenarios=100)
        prod_matrix = generate_rank_preserving_prod_scenarios(forecast_dist_prod,
                                                              datetime_vec=forecast_dist_prod['datetime'].unique(),
                                                              n_scenarios=100)

        real_matrix = np.column_stack((df_full['DA_price_real'], df_full['ID_price_real'], df_full['SMF_real']))
        cov_est = np.cov(real_matrix, rowvar=False)

        ex_ante_time = virtual_now

        print("📆 Ex-Ante zaman penceresi başlıyor:", ex_ante_time)

        scenario_df, optimization_hours = generate_scenario_dt_for_window(
            da_matrix, id_matrix, smf_matrix, prod_matrix,
            real_da=df_full['DA_price_real'],
            real_id=df_full['ID_price_real'],
            real_smf=df_full['SMF_real'],
            cov_matrix=cov_est,
            ex_ante_time=ex_ante_time,
            db=df_full,
        )

        print("scenario_df columns:", scenario_df.columns)

        db_for_opt = scenario_df.rename(columns={
            'DA': 'DA_price_forecasted',
            'ID': 'ID_price_forecasted',
            'SMF': 'SMF_forecasted',
            'prod': 'forecasted_production'
        })

        db_for_opt = db_for_opt[db_for_opt['datetime'].isin(optimization_hours)].copy()

        results = solve_ex_ante_model(db_for_opt, battery_params=battery_params, initial_soc=20)
        print("\n🧩 Optimization Results (Ex-Ante):", results)

        df_results = pd.DataFrame(results)

        # 🧩 Ex-Ante kararlarını ana veriye işle
        for i, dt in enumerate(results['datetime']):
            df_full.loc[df_full['datetime'] == dt, 'q_committed'] = results['q_committed'][i]
            df_full.loc[df_full['datetime'] == dt, 'd_da_prod'] = results['d_da_prod'][i]
            df_full.loc[df_full['datetime'] == dt, 'd_da_bat'] = results['d_da_bat'][i]
            df_full.loc[df_full['datetime'] == dt, 'd_id_prod'] = results['d_id_prod'][i]
            df_full.loc[df_full['datetime'] == dt, 'd_id_bat'] = results['d_id_bat'][i]
            df_full.loc[df_full['datetime'] == dt, 'x_id'] = results['x_id'][i]
            df_full.loc[df_full['datetime'] == dt, 'x_ceza'] = results['x_ceza'][i]
            df_full.loc[df_full['datetime'] == dt, 'is_charging'] = results['is_charging'][i]
            df_full.loc[df_full['datetime'] == dt, 'charge'] = results['charge'][i]
            df_full.loc[df_full['datetime'] == dt, 'soc'] = results['soc'][i]

    else:
        print(f"f\⏰ {virtual_now} - Ex-Post model çalıştırılıyor...")

        df_mcp = generate_forecast_price_ar1_sto(df_mcp, price_ceiling=3400, virtual_now=virtual_now)
        df_mcp = df_mcp.rename(columns={
            "price_real": "DA_price_real",
            "price_forecast": "DA_price_forecasted"
        })
        df_id = generate_forecast_price_id_sto(df_id, price_ceiling=3400, virtual_now=virtual_now)
        df_id = df_id.rename(columns={
            "price_real": "ID_price_real",
            "price_forecast": "ID_price_forecasted"
        })
        df_smf = generate_forecast_smf_sto(df_smf, price_ceiling=3400, virtual_now=virtual_now)
        df_smf = df_smf.rename(columns={
            "smf_real": "SMF_real",
            "smf_forecast": "SMF_forecasted"
        })
        # === PRODUCTION FORECAST ===
        print("📦 Generating production forecast...")
        if not df_prod.empty:
            df_prod["datetime"] = pd.to_datetime(df_prod["date"])
            if "injectionQuantity" not in df_prod.columns:
                if "total" in df_prod.columns:
                    df_prod["injectionQuantity"] = pd.to_numeric(df_prod["total"], errors="coerce")
                else:
                    print("⚠️ Neither 'injectionQuantity' nor 'total' column found → Cannot generate forecast.")
                    df_prod_forecast = pd.DataFrame()

            if not df_prod.empty:
                df_prod_forecast = generate_forecast_production_sto(df_prod, error_sd=0.05)
                df_prod_forecast = df_prod_forecast.rename(columns={
                    "forecast_total": "forecasted_production_meteologica"
                })
                df_prod_forecast["powerplantId"] = PLANT_ID


        else:
            print("⚠ Cannot generate forecast: df_prod is empty.")
            df_prod_forecast = pd.DataFrame()

        df_mcp.rename(columns={"date": "datetime"}, inplace=True)
        df_id.rename(columns={"date": "datetime"}, inplace=True)
        df_smf.rename(columns={"date": "datetime"}, inplace=True)
        df_prices = df_mcp.merge(df_id, on="datetime", how="outer") \
            .merge(df_smf, on="datetime", how="outer")
        df_full = df_prices.merge(df_prod, on="datetime", how="left")
        df_full = df_full.merge(df_prod_forecast[["datetime", "forecasted_production_meteologica"]], on="datetime",
                                how="left")
        df_full['datetime'] = df_full['datetime'].dt.tz_convert('UTC').dt.tz_convert('Europe/Istanbul')

        # Eğer hala sorun yaşarsan (ki sanmıyorum), floor('H') işlemini bu dönüşümden sonra da yapabilirsin:
        df_full['datetime'] = df_full['datetime'].dt.floor('H')
        print(df_full.head(3))  # İlk 3 satır
        print("\n🧩 Columns:", df_full.columns.tolist())  # Tüm sütun adları listesi
        forecast_dist_da = generate_forecast_distribution_from_price_forecast(df_mcp, price_ceiling=3400)
        forecast_dist_id = generate_forecast_distribution_from_id_price(df_id, price_ceiling=3400)
        forecast_dist_smf = generate_forecast_distribution_from_smf(df_smf, price_ceiling=3400)
        forecast_dist_prod = generate_forecast_distribution_from_forecast(df_prod_forecast, n_bins=100,
                                                                          error_sd=0.10)

        da_matrix = generate_rank_preserving_matrix(forecast_dist_da,
                                                    datetime_vec=forecast_dist_da['datetime'].unique(), n_scenarios=100)
        id_matrix = generate_rank_preserving_matrix(forecast_dist_id,
                                                    datetime_vec=forecast_dist_id['datetime'].unique(), n_scenarios=100)
        smf_matrix = generate_rank_preserving_matrix(forecast_dist_smf,
                                                     datetime_vec=forecast_dist_smf['datetime'].unique(),
                                                     n_scenarios=100)
        prod_matrix = generate_rank_preserving_prod_scenarios(forecast_dist_prod,
                                                              datetime_vec=forecast_dist_prod['datetime'].unique(),
                                                              n_scenarios=100)

        real_matrix = np.column_stack((df_full['DA_price_real'], df_full['ID_price_real'], df_full['SMF_real']))
        cov_est = np.cov(real_matrix, rowvar=False)
        print("🛠️ Ex-Post model çalıştırılıyor...")

        df_results = run_ex_post_model_scenario_based(
            db=df_full,
            ex_post_time= virtual_now,
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

        for i, dt in enumerate(df_results['datetime']):
            df_full.loc[df_full['datetime'] == dt, 'q_committed'] = df_results['q_committed'][i]
            df_full.loc[df_full['datetime'] == dt, 'q_committed_horizon'] = df_results['q_committed_horizon'][i]
            df_full.loc[df_full['datetime'] == dt, 'd_da_prod'] = df_results['d_da_prod'][i]
            df_full.loc[df_full['datetime'] == dt, 'd_da_bat'] = df_results['d_da_bat'][i]
            df_full.loc[df_full['datetime'] == dt, 'd_id_prod'] = df_results['d_id_prod'][i]
            df_full.loc[df_full['datetime'] == dt, 'd_id_bat'] = df_results['d_id_bat'][i]
            df_full.loc[df_full['datetime'] == dt, 'x_id'] = df_results['x_id'][i]
            df_full.loc[df_full['datetime'] == dt, 'x_ceza'] = df_results['x_ceza'][i]
            df_full.loc[df_full['datetime'] == dt, 'is_charging'] = df_results['is_charging'][i]
            df_full.loc[df_full['datetime'] == dt, 'charge'] = df_results['charge'][i]
            df_full.loc[df_full['datetime'] == dt, 'soc'] = df_results['soc'][i]
            df_full.loc[df_full['datetime'] == dt, 'soc_post'] = df_results['soc_post'][i]
            df_full.loc[df_full['datetime'] == dt, 'd_da_prod_extra'] = df_results['d_da_prod_extra'][i]
            df_full.loc[df_full['datetime'] == dt, 'd_da_bat_extra'] = df_results['d_da_bat_extra'][i]
            df_full.loc[df_full['datetime'] == dt, 'd_id_prod_extra'] = df_results['d_id_prod_extra'][i]
            df_full.loc[df_full['datetime'] == dt, 'd_id_bat_extra'] = df_results['d_id_bat_extra'][i]
            df_full.loc[df_full['datetime'] == dt, 'd_da_prod_horizon'] = df_results['d_da_prod_horizon'][i]
            df_full.loc[df_full['datetime'] == dt, 'd_id_prod_horizon'] = df_results['d_id_prod_horizon'][i]
            df_full.loc[df_full['datetime'] == dt, 'd_da_bat_horizon'] = df_results['d_da_bat_horizon'][i]
            df_full.loc[df_full['datetime'] == dt, 'x_id_adj'] = df_results['x_id_adj'][i]
            df_full.loc[df_full['datetime'] == dt, 'x_ceza_adj'] = df_results['x_ceza_adj'][i]
            df_full.loc[df_full['datetime'] == dt, 'x_id_horizon'] = df_results['x_id_horizon'][i]
            df_full.loc[df_full['datetime'] == dt, 'x_ceza_horizon'] = df_results['x_ceza_horizon'][i]
            df_full.loc[df_full['datetime'] == dt, 'charge_extra'] = df_results['charge_extra'][i]
            df_full.loc[df_full['datetime'] == dt, 'charge_horizon'] = df_results['charge_horizon'][i]

    return (df_results, df_full, cov_est, da_matrix, id_matrix, smf_matrix, prod_matrix)

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

    print(f"[{datetime.now()}] Start time: {db_window.iloc[0]['datetime']}, SOC: {current_soc:.2f}")

    # Extract scenario data
    price_DA_sce = db_window['DA'].values
    price_ID_sce = db_window['ID'].values
    price_SMF_sce = db_window['SMF'].values
    production_sce = db_window['prod'].values
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

    # Calculate flexibility flags
    current_time = ex_post_time_index

    # GOP flexibility calculation
    today_start = pd.Timestamp(current_time.date()).tz_localize('Europe/Istanbul')
    today_end = today_start + pd.Timedelta(hours=23)

    if current_time.hour >= 10:
        tomorrow_start = today_end + pd.Timedelta(hours=1)
        tomorrow_end = tomorrow_start + pd.Timedelta(hours=23)
        commitment_hours = pd.date_range(today_start, today_end, freq='H').union(
            pd.date_range(tomorrow_start, tomorrow_end, freq='H')
        )
    else:
        commitment_hours = pd.date_range(today_start, today_end, freq='H')

    is_gop_flexible = [0 if dt in commitment_hours else 1 for dt in db_window['datetime']]

    # ID flexibility (first 2 hours are fixed)
    is_id_flexible = [0 if i < 1 else 1 for i in range(horizon_hours)]

    print(f"[{datetime.now()}] GOP and flexibility constraints defined, starting model definition...")

    # Create the optimization model
    model = LpProblem("ExPostOptimization", LpMaximize)

    # Decision variables
    d_id_prod_extra = LpVariable.dicts("d_id_prod_extra", range(horizon_hours), -P_max, P_max)
    d_id_bat_extra = LpVariable.dicts("d_id_bat_extra", range(horizon_hours), -P_max, P_max)
    d_da_prod_extra = LpVariable.dicts("d_da_prod_extra", range(horizon_hours), -P_max, P_max)
    d_da_bat_extra = LpVariable.dicts("d_da_bat_extra", range(horizon_hours), -P_max, P_max)
    d_id_prod_horizon = LpVariable.dicts("d_id_prod_horizon", range(horizon_hours), 0, None)
    d_id_bat_horizon = LpVariable.dicts("d_id_bat_horizon", range(horizon_hours), 0, None)
    charge_extra = LpVariable.dicts("charge_extra", range(horizon_hours), -P_max, P_max)
    charge_horizon = LpVariable.dicts("charge_horizon", range(horizon_hours), 0, None)
    d_da_prod_horizon = LpVariable.dicts("d_da_prod_horizon", range(horizon_hours), 0, None)
    d_da_bat_horizon = LpVariable.dicts("d_da_bat_horizon", range(horizon_hours), 0, None)
    x_id_adj = LpVariable.dicts("x_id_adj", range(horizon_hours), -P_max, P_max)
    x_ceza_adj = LpVariable.dicts("x_ceza_adj", range(horizon_hours), -P_max, P_max)
    x_id_horizon = LpVariable.dicts("x_id_horizon", range(horizon_hours), 0, None)
    x_ceza_horizon = LpVariable.dicts("x_ceza_horizon", range(horizon_hours), 0, None)
    soc_post = LpVariable.dicts("soc_post", range(horizon_hours),
                                battery_params['soc_min'], battery_params['soc_max'])
    prod_surplus = LpVariable.dicts("prod_surplus", range(horizon_hours), 0, None)
    prod_deficit = LpVariable.dicts("prod_deficit", range(horizon_hours), 0, None)
    delta = LpVariable.dicts("delta", range(horizon_hours), cat='Binary')
    is_charging = LpVariable.dicts("is_charging", range(horizon_hours), cat='Binary')
    q_committed_horizon = LpVariable.dicts("q_committed_horizon", range(horizon_hours), 0, None)
    diff_abs = LpVariable.dicts("diff_abs", range(horizon_hours), 0, None)

    # Objective function
    objective = 0
    for t in range(horizon_hours):
        objective += (
                DA_price_real[t] * (1 - is_gop_flexible[t]) * q_committed[t] +
                price_DA_sce[t] * is_gop_flexible[t] * q_committed_horizon[t] +
                price_ID_sce[t] * 1.03 * (1 - is_id_flexible[t]) * (
                            d_id_prod[t] + d_id_bat[t] + d_id_prod_extra[t] + d_id_bat_extra[t]) +
                price_ID_sce[t] * 1.03 * is_id_flexible[t] * (d_id_prod_horizon[t] + d_id_bat_horizon[t]) -
                price_ID_sce[t] * 0.97 * (1 - is_id_flexible[t]) * (x_id[t] + x_id_adj[t]) -
                price_ID_sce[t] * 0.97 * is_id_flexible[t] * x_id_horizon[t] -
                penalty_unit_price[t] * (1 - is_id_flexible[t]) * (x_ceza[t] + x_ceza_adj[t]) -
                penalty_unit_price[t] * is_id_flexible[t] * x_ceza_horizon[t] -
                (unit_battery_cost / 2) * (
                        ((1 - is_gop_flexible[t]) * (d_da_bat[t])) +
                        (is_gop_flexible[t] * d_da_bat_horizon[t]) +
                        ((1 - is_id_flexible[t]) * (d_id_bat[t] + d_id_bat_extra[t] + charge[t] + charge_extra[t])) +
                        (is_id_flexible[t] * (d_id_bat_horizon[t] + charge_horizon[t]))
                )
        )

    model += objective

    # Add constraints
    for t in range(horizon_hours):
        # Non-negativity constraints
        model += x_ceza[t] + x_ceza_adj[t] >= 0
        model += x_id[t] + x_id_adj[t] >= 0
        model += charge[t] + charge_extra[t] >= 0

        # Commitment constraints
        model += (q_committed[t] ==
                  (1 - is_gop_flexible[t]) * (d_da_prod[t] + d_da_bat[t] + d_da_prod_extra[t] + d_da_bat_extra[t]))

        model += (q_committed_horizon[t] ==
                  is_gop_flexible[t] * (d_da_prod_horizon[t] + d_da_bat_horizon[t]))

        model += battery_params['power_limit'] >= d_da_prod[t] + d_da_bat[t] + d_da_prod_extra[t] + d_da_bat_extra[t] + d_id_prod[t] + d_id_bat[t] + d_id_bat_extra[t] + d_id_prod_extra[t] + d_da_prod_horizon[t] + d_da_bat_horizon[t] + d_id_prod_horizon[t] + d_id_bat_horizon[t]

        # Minimum production constraint
        model += ((1 - is_gop_flexible[t]) * (d_da_prod[t] + d_da_prod_extra[t]) +
                  is_gop_flexible[t] * d_da_prod_horizon[t] >= 0.25 * production_sce[t])

        # Production constraint
        model += (production_sce[t] >=
                  (1 - is_gop_flexible[t]) * (d_da_prod[t] + d_da_prod_extra[t]) +
                  is_gop_flexible[t] * d_da_prod_horizon[t] +
                  (1 - is_id_flexible[t]) * (d_id_prod[t] + d_id_prod_extra[t]) +
                  is_id_flexible[t] * d_id_prod_horizon[t])

        # Battery discharge constraint
        model += ((1 - is_gop_flexible[t]) * (d_da_bat[t] + d_da_bat_extra[t]) +
                  is_gop_flexible[t] * d_da_bat_horizon[t] +
                  (1 - is_id_flexible[t]) * (d_id_bat[t] + d_id_bat_extra[t]) +
                  is_id_flexible[t] * d_id_bat_horizon[t] <= P_max * (1 - is_charging[t]))

        # Battery charge constraint
        model += ((charge[t] + charge_extra[t]) * (1 - is_id_flexible[t]) +
                  charge_horizon[t] * is_id_flexible[t] <= P_max * is_charging[t])

        # Production surplus/deficit balance
        model += (prod_surplus[t] - prod_deficit[t] ==
                  production_sce[t] -
                  ((1 - is_gop_flexible[t]) * (d_da_prod[t] + d_da_bat[t] + d_da_prod_extra[t] + d_da_bat_extra[t])) -
                  is_gop_flexible[t] * (d_da_prod_horizon[t] + d_da_bat_horizon[t]))

        # Absolute difference constraints
        model += (diff_abs[t] >= production_sce[t] -
                  ((1 - is_gop_flexible[t]) * (d_da_prod[t] + d_da_bat[t] + d_da_prod_extra[t] + d_da_bat_extra[t])) -
                  is_gop_flexible[t] * (d_da_prod_horizon[t] + d_da_bat_horizon[t]))

        model += (diff_abs[t] >=
                  ((1 - is_gop_flexible[t]) * (d_da_prod[t] + d_da_bat[t] + d_da_prod_extra[t] + d_da_bat_extra[t])) +
                  is_gop_flexible[t] * (d_da_prod_horizon[t] + d_da_bat_horizon[t]) -
                  production_sce[t])

        model += prod_surplus[t] + prod_deficit[t] <= diff_abs[t] + 0.01

        # Big M constraints
        model += prod_surplus[t] <= big_M * delta[t]
        model += prod_deficit[t] <= big_M * (1 - delta[t])

        # Production surplus constraint
        model += (prod_surplus[t] ==
                  (1 - is_id_flexible[t]) * (charge[t] + charge_extra[t] + d_id_prod_extra[t] + d_id_prod[t]) +
                  is_id_flexible[t] * (charge_horizon[t] + d_id_prod_horizon[t]))

        # Production deficit constraint
        model += (prod_deficit[t] ==
                  (1 - is_id_flexible[t]) * (
                              d_id_bat[t] + d_id_bat_extra[t] + x_id[t] + x_id_adj[t] + x_ceza[t] + x_ceza_adj[t]) +
                  is_id_flexible[t] * (d_id_bat_horizon[t] + x_id_horizon[t] + x_ceza_horizon[t]))

    # SOC constraints
    model += soc_post[0] == current_soc

    for t in range(1, horizon_hours):
        model += (soc_post[t] == soc_post[t - 1] +
                  eta * (charge_horizon[t - 1] * is_id_flexible[t - 1] + (charge[t - 1] + charge_extra[t - 1]) * (
                            1 - is_id_flexible[t - 1])) -
                  (1 / eta) * (
                          (1 - is_gop_flexible[t - 1]) * (d_da_bat[t - 1] + d_da_bat_extra[t - 1]) +
                          is_gop_flexible[t - 1] * d_da_bat_horizon[t - 1] +
                          (1 - is_id_flexible[t - 1]) * (d_id_bat[t - 1] + d_id_bat_extra[t - 1]) +
                          is_id_flexible[t - 1] * d_id_bat_horizon[t - 1]
                  ))

    print(f"[{datetime.now()}] Model setup completed.")

    return model, {
        'd_id_prod_extra': d_id_prod_extra,
        'd_id_bat_extra': d_id_bat_extra,
        'd_da_prod_extra': d_da_prod_extra,
        'd_da_bat_extra': d_da_bat_extra,
        'd_id_prod_horizon': d_id_prod_horizon,
        'd_id_bat_horizon': d_id_bat_horizon,
        'charge_extra': charge_extra,
        'charge_horizon': charge_horizon,
        'd_da_prod_horizon': d_da_prod_horizon,
        'd_da_bat_horizon': d_da_bat_horizon,
        'x_id_adj': x_id_adj,
        'x_ceza_adj': x_ceza_adj,
        'x_id_horizon': x_id_horizon,
        'x_ceza_horizon': x_ceza_horizon,
        'soc_post': soc_post,
        'q_committed_horizon': q_committed_horizon
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

    horizon_hours = 48
    optimization_start_time = ex_post_time + pd.Timedelta(hours=1)
    optimization_end_time = optimization_start_time + pd.Timedelta(hours=horizon_hours)

    # Filter data for the optimization window
    mask = (db['datetime'] >= ex_post_time) & (db['datetime'] <= optimization_end_time)
    db_window = db[mask].copy().reset_index(drop=True)
    print("\n🧩 Columns:", db_window.columns.tolist())  # Tüm sütun adları listesi

    print(f"[{datetime.now()}] Ex-post scenario started...")

    # Get current SOC
    current_soc_row = db_window[db_window['datetime'] == ex_post_time]
    if len(current_soc_row) == 0 or pd.isna(current_soc_row['soc'].iloc[0]):
        current_soc = battery_params['soc_target']
    else:
        current_soc = current_soc_row['soc'].iloc[0]

    print(f"Ex-post time: {ex_post_time}, current SOC: {current_soc:.2f}")

    # Generate ex-post scenarios
    print(f"[{datetime.now()}] Generating ex-post scenarios...")
    scenario_dt_expost = generate_scenario_dt_for_window_expost(
        da_matrix, id_matrix, smf_matrix, prod_matrix,
        real_da, real_id, real_smf, cov_matrix,
        ex_post_time, db,
        window_size=window_size, sigma=sigma
    )

    print(f"[{datetime.now()}] Ex-post scenarios successfully generated.")

    # Update window data with scenarios
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
    ex_post_time_index = db_window[db_window['datetime'] == ex_post_time].index[0]
    model, variables = build_ex_post_optimization(db_window, battery_params, ex_post_time_index)

    print(f"[{datetime.now()}] Ex-post optimization model successfully built, starting solution...")

    # Solve the model
    try:
        model.solve(PULP_CBC_CMD(msg=0))

        if model.status != 1:  # LpStatusOptimal
            return {
                'status': 'failed',
                'message': f'Optimization failed with status: {LpStatus[model.status]}'
            }

        print(f"[{datetime.now()}] Solution successful, writing results...")

        # Extract results
        results = {}
        for var_name, var_dict in variables.items():
            results[var_name] = [var_dict[t].varValue for t in range(len(db_window))]

        # Update the original database with results
        start_idx = db[db['datetime'] == optimization_start_time].index[0]
        end_idx = db[db['datetime'] == optimization_end_time].index[0]

        for var_name, values in results.items():
            if var_name in db.columns:
                db.loc[start_idx:end_idx, var_name] = values[1:]  # Skip first 2 hours
            else:
                # Create new column if it doesn't exist
                db[var_name] = np.nan
                db.loc[start_idx:end_idx, var_name] = values[1:]

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




if __name__ == "__main__":
    import pandas as pd
    import pytz
    from datetime import datetime, timedelta

    # Başlangıç ve bitiş zamanı
    IST = pytz.timezone("Europe/Istanbul")
    START = IST.localize(datetime(2025, 1, 1, 0, 0))
    END = IST.localize(datetime(2025, 1, 15, 23, 0))
    virtual_now = START + timedelta(hours=10)

    print("🔧 Initializing static inputs...")
    (df_mcp, df_id, df_smf, df_prod, battery_params, PLANT_ID) = initialize_static_inputs(virtual_now, START, END)

    while virtual_now <= END:
        print(f"\n🕒 Simülasyon zamanı: {virtual_now.strftime('%Y-%m-%d %H:%M')}")
        current_hour = virtual_now

        df_results, df_full, *_ = run_full_simulation_from_2024_flex(
            virtual_now=virtual_now,
            df_prod=df_prod,
            df_id=df_id,
            df_smf=df_smf,
            df_mcp=df_mcp,
            battery_params=battery_params,
            PLANT_ID = PLANT_ID
        )

        # ⏳ Saatlik ilerleme
        virtual_now += timedelta(hours=1)

        # 💾 Her adımda (veya örneğin her 24 saatte bir) kaydet
        df_full.to_csv("df_full.csv", index=False)

    print("\n✅ Simülasyon tamamlandı ve df_full.csv'ye kaydedildi.")



