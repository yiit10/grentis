# one_time_simulation.py - One-time simulation from 2024-01-01
from main import virtual_now
from performance_back import (
    apply_case_logic,
    battery_params, calculate_battery_trading_metrics, simulate_benchmark_case1, show_hourly_decisions, test_benchmark_simulation, validate_benchmark_simulation, prepare_benchmark_inputs
)
from battery_brain_core import solve_ex_ante_model, prepare_optimization_data


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


def run_full_simulation_from_2024():
    """
    Run the complete simulation from 2024-01-01 to present and save results to CSV
    """
    USERNAME = "fyigitkavak@icloud.com"
    PASSWORD = "IDK1.35e"
    PLANT_ID = 2591
    IST = pytz.timezone("Europe/Istanbul")

    battery_params = {
        'capacity': 40.2,
        'power_limit': 61.8,
        'soc_min': 8,
        'soc_max': 40.2,
        'soc_target': 20,
        'efficiency': 0.97,
    }

    # Set start date to 2024-01-01
    START = datetime(2025, 1, 1, 0, 0, tzinfo=IST)
    END = datetime(2025, 6, 30, 0, 0, tzinfo=IST)

    print(f"🚀 Starting simulation from {START} to {END}")

    # 🔐 Get token
    print("🔐 Getting authentication token...")
    tgt = get_tgt_token(USERNAME, PASSWORD)

    # === PRICE DATA ===
    print("📦 Fetching MCP data...")
    df_mcp = pd.DataFrame(get_mcp_data(tgt, START, END))
    df_mcp["date"] = pd.to_datetime(df_mcp["date"])
    df_mcp = generate_forecast_price_ar1(df_mcp, price_ceiling= 3400)
    df_mcp = df_mcp.rename(columns={
        "price_real": "DA_price_real",
        "price_forecast": "DA_price_forecasted"
    })

    print("📦 Fetching ID price data...")
    df_id = pd.DataFrame(get_id_avg_price_data(tgt, START, END))
    df_id["date"] = pd.to_datetime(df_id["date"])
    df_id = generate_forecast_price_id(df_id, price_ceiling= 3400)
    df_id = df_id.rename(columns={
        "price_real": "ID_price_real",
        "price_forecast": "ID_price_forecasted"
    })

    print("📦 Fetching SMF data...")
    df_smf = pd.DataFrame(get_smf_data(tgt, START, END))
    df_smf["date"] = pd.to_datetime(df_smf["date"])
    df_smf = generate_forecast_smf(df_smf, price_ceiling= 3400)
    df_smf = df_smf.rename(columns={
        "smf_real": "SMF_real",
        "smf_forecast": "SMF_forecasted"
    })

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
    else:
        print("⚠ Cannot generate forecast: df_prod is empty.")
        df_prod_forecast = pd.DataFrame()

    # === MERGE DATA ===
    print("🧩 Merging all data...")
    dfs = [df_mcp, df_id, df_smf, df_prod_forecast]
    df_perf = dfs[0]

    for other_df in dfs[1:]:
        df_perf = pd.merge(df_perf, other_df, on="datetime", how="outer")

    # === ADD REAL PRODUCTION ===
    print("📦 Adding real production data...")
    real_production = pd.DataFrame()
    if not df_rt.empty:
        df_rt["datetime"] = pd.to_datetime(df_rt["date"])
        real_production = df_rt.groupby("datetime")["total"].sum().reset_index()
        real_production.rename(columns={"total": "real_production"}, inplace=True)
        print("✅ Realtime data processed successfully.")
    elif not df_inj.empty:
        df_inj["datetime"] = pd.to_datetime(df_inj["date"])
        real_production = df_inj[["datetime", "total"]].rename(columns={"total": "real_production"})
        print("✅ Injection data used as fallback.")

    if not real_production.empty:
        df_perf = pd.merge(df_perf, real_production, on="datetime", how="left")

    # Sort and clean
    df_perf = df_perf.sort_values("datetime").reset_index(drop=True)

    print(f"✅ All data prepared. Final df_perf shape: {df_perf.shape}")

    # === RUN SIMULATION ===
    print("🚀 Starting optimization simulation...")

    # Simulation parameters
    virtual_now = pd.Timestamp("2025-01-01 10:00:00", tz="Europe/Istanbul")
    simulation_end = df_perf['datetime'].max() - timedelta(days=2)

    # Storage for all decisions
    all_decisions = pd.DataFrame()

    optimization_count = 0

    while virtual_now <= simulation_end:
        if virtual_now.hour == 10:
            optimization_count += 1
            print(f"⚙️ Optimization #{optimization_count} starting: {virtual_now}")

            horizon_start = (virtual_now + timedelta(hours=14)).replace(minute=0, second=0)
            horizon_end = horizon_start + timedelta(hours=47)

            # Prepare forecast data
            df_gop_forecast = df_perf[['datetime', 'DA_price_forecasted']]
            df_id_forecast = df_perf[['datetime', 'ID_price_forecasted']]
            df_smf_forecast = df_perf[['datetime', 'SMF_forecasted']]

            production_data = {
                "meteologica": df_perf[['datetime', 'forecasted_production_meteologica']].rename(
                    columns={'datetime': 'x', 'forecasted_production_meteologica': 'y'})
            }

            df_input = prepare_optimization_data(df_gop_forecast, df_id_forecast, df_smf_forecast,
                                                 production_data, horizon_start, horizon_end)

            # Get previous SoC
            if not all_decisions.empty and any(all_decisions['datetime'] < virtual_now):
                last_soc = all_decisions.loc[all_decisions['datetime'] < virtual_now, 'soc'].iloc[-1]
            else:
                last_soc = battery_params['soc_target']

            # Solve optimization
            result = solve_ex_ante_model(df_input, initial_soc=last_soc)

            if result["status"] == "optimal":
                df_result = pd.DataFrame({
                    "datetime": result["datetime"],
                    "q_committed": result["q_committed"],
                    "soc": result["soc"],
                    "d_da_prod": result["d_da_prod"],
                    "d_id_prod": result["d_id_prod"],
                    "d_da_bat": result["d_da_bat"],
                    "d_id_bat": result["d_id_bat"],
                    "charge": result["charge"],
                    "is_charging": result["is_charging"],
                    "forecasted_production": result["forecasted_production"],
                    "forecasted_price_DA": result["forecasted_price_DA"],
                    "forecasted_price_ID": result["forecasted_price_ID"]
                })

                df_result["datetime"] = pd.to_datetime(df_result["datetime"])
                df_combined = df_result.merge(df_perf, on="datetime", how="left")

                all_decisions = pd.concat([all_decisions, df_combined], ignore_index=True)
                all_decisions = all_decisions.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)

                print(f"✅ Optimization #{optimization_count} completed: {virtual_now.strftime('%Y-%m-%d %H:%M')}")

                # Progress update every 100 optimizations
                if optimization_count % 100 == 0:
                    print(
                        f"📊 Progress: {optimization_count} optimizations completed, {len(all_decisions)} total decisions")

            else:
                print(f"❌ Optimization #{optimization_count} failed: {result['status']}")

        virtual_now += timedelta(hours=1)

    # Initialize missing columns
    for col in ['x_id', 'x_ceza', 'bat_to_discharge']:
        if col not in all_decisions.columns:
            all_decisions[col] = 0.0

    print(f"\n🧾 Simulation completed. Total decisions: {len(all_decisions)}")
    print(f"📊 Total optimizations run: {optimization_count}")

    # === APPLY CASE LOGIC ===
    print("\n⚙️ Applying dispatch case logic...")
    for i in range(len(all_decisions)):
        if i % 1000 == 0:
            print(f"Processing hour {i}/{len(all_decisions)}")
        all_decisions = apply_case_logic(all_decisions, i, battery_params)

    print("✅ Case logic applied successfully.")
    all_decisions = calculate_battery_trading_metrics(all_decisions, battery_params, eur_to_tl=35)

    # === SAVE RESULTS ===
    output_filename = f"simulation_results_2025H1_{PLANT_ID}.csv"

    print(f"\n💾 Saving results to {output_filename}...")
    all_decisions.to_csv(output_filename, index=False)

    print(f"✅ Results saved successfully!")
    print(f"📁 File: {output_filename}")
    print(f"📊 Total rows: {len(all_decisions)}")
    print(f"📅 Date range: {all_decisions['datetime'].min()} to {all_decisions['datetime'].max()}")

    # Show summary statistics
    print("\n📈 SUMMARY STATISTICS:")
    print(
        f"Total net profit: {all_decisions['net_profit'].sum():,.2f} TL" if 'net_profit' in all_decisions.columns else "Net profit not calculated")
    print(
        f"Average SOC: {all_decisions['soc'].mean():.2f} MWh" if 'soc' in all_decisions.columns else "SOC not available")
    print(
        f"Total battery cycles: {(all_decisions['charge'].sum() + all_decisions['d_da_bat'].sum() + all_decisions['d_id_bat'].sum()) / 2:.2f}")

    # 4. Prepare and run benchmark simulation
    df_perf = prepare_benchmark_inputs(df_perf)
    df_benchmark = simulate_benchmark_case1(df_perf, battery_params)

    # 5. Calculate benchmark daily metrics - FIXED: Group by date properly
    if 'date' not in df_benchmark.columns:
        df_benchmark['date'] = pd.to_datetime(df_benchmark['datetime']).dt.date

    output_filename = f"benchmark_results_2025H1_{PLANT_ID}.csv"

    print(f"\n💾 Saving results to {output_filename}...")
    df_benchmark.to_csv(output_filename, index=False)

    return all_decisions, output_filename, df_benchmark


if __name__ == "__main__":
    print("=" * 80)
    print("BATTERY TRADING SIMULATION - FULL 2024 RUN")
    print("=" * 80)

    try:
        results, filename = run_full_simulation_from_2024()
        print(f"\n🎉 Simulation completed successfully!")
        print(f"📁 Results saved to: {filename}")

    except Exception as e:
        print(f"❌ Simulation failed with error: {str(e)}")
        import traceback

        traceback.print_exc()




    # === RUN SIMULATION ===
    print("🚀 Starting optimization simulation...")

    # Simulation parameters
    virtual_now = pd.Timestamp("2025-01-01 10:00:00", tz="Europe/Istanbul")
    simulation_end = df_perf['datetime'].max() - timedelta(days=2)

    # Storage for all decisions
    all_decisions = pd.DataFrame()

    optimization_count = 0

    while virtual_now <= simulation_end:
        if virtual_now.hour == 10:
            optimization_count += 1
            print(f"⚙️ Optimization #{optimization_count} starting: {virtual_now}")

            horizon_start = (virtual_now + timedelta(hours=14)).replace(minute=0, second=0)
            horizon_end = horizon_start + timedelta(hours=47)

            # Prepare forecast data
            df_gop_forecast = df_perf[['datetime', 'DA_price_forecasted']]
            df_id_forecast = df_perf[['datetime', 'ID_price_forecasted']]
            df_smf_forecast = df_perf[['datetime', 'SMF_forecasted']]

            production_data = {
                "meteologica": df_perf[['datetime', 'forecasted_production_meteologica']].rename(
                    columns={'datetime': 'x', 'forecasted_production_meteologica': 'y'})
            }

            df_input = prepare_optimization_data(df_gop_forecast, df_id_forecast, df_smf_forecast,
                                                 production_data, horizon_start, horizon_end)

            # Get previous SoC
            if not all_decisions.empty and any(all_decisions['datetime'] < virtual_now):
                last_soc = all_decisions.loc[all_decisions['datetime'] < virtual_now, 'soc'].iloc[-1]
            else:
                last_soc = battery_params['soc_target']

            # Solve optimization
            result = solve_ex_ante_model(df_input, initial_soc=last_soc)

            if result["status"] == "optimal":
                df_result = pd.DataFrame({
                    "datetime": result["datetime"],
                    "q_committed": result["q_committed"],
                    "soc": result["soc"],
                    "d_da_prod": result["d_da_prod"],
                    "d_id_prod": result["d_id_prod"],
                    "d_da_bat": result["d_da_bat"],
                    "d_id_bat": result["d_id_bat"],
                    "charge": result["charge"],
                    "is_charging": result["is_charging"],
                    "forecasted_production": result["forecasted_production"],
                    "forecasted_price_DA": result["forecasted_price_DA"],
                    "forecasted_price_ID": result["forecasted_price_ID"]
                })

                df_result["datetime"] = pd.to_datetime(df_result["datetime"])
                df_combined = df_result.merge(df_perf, on="datetime", how="left")

                all_decisions = pd.concat([all_decisions, df_combined], ignore_index=True)
                all_decisions = all_decisions.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)

                print(f"✅ Optimization #{optimization_count} completed: {virtual_now.strftime('%Y-%m-%d %H:%M')}")

                # Progress update every 100 optimizations
                if optimization_count % 100 == 0:
                    print(
                        f"📊 Progress: {optimization_count} optimizations completed, {len(all_decisions)} total decisions")

            else:
                print(f"❌ Optimization #{optimization_count} failed: {result['status']}")

        virtual_now += timedelta(hours=1)

    # Initialize missing columns
    for col in ['x_id', 'x_ceza', 'bat_to_discharge']:
        if col not in all_decisions.columns:
            all_decisions[col] = 0.0

    print(f"\n🧾 Simulation completed. Total decisions: {len(all_decisions)}")
    print(f"📊 Total optimizations run: {optimization_count}")

    # === APPLY CASE LOGIC ===
    print("\n⚙️ Applying dispatch case logic...")
    for i in range(len(all_decisions)):
        if i % 1000 == 0:
            print(f"Processing hour {i}/{len(all_decisions)}")
        all_decisions = apply_case_logic(all_decisions, i, battery_params)

    print("✅ Case logic applied successfully.")
    all_decisions = calculate_battery_trading_metrics(all_decisions, battery_params, eur_to_tl=35)

    # === SAVE RESULTS ===
    output_filename = f"simulation_results_2025H1_{PLANT_ID}.csv"

    print(f"\n💾 Saving results to {output_filename}...")
    all_decisions.to_csv(output_filename, index=False)

    print(f"✅ Results saved successfully!")
    print(f"📁 File: {output_filename}")
    print(f"📊 Total rows: {len(all_decisions)}")
    print(f"📅 Date range: {all_decisions['datetime'].min()} to {all_decisions['datetime'].max()}")

    # Show summary statistics
    print("\n📈 SUMMARY STATISTICS:")
    print(
        f"Total net profit: {all_decisions['net_profit'].sum():,.2f} TL" if 'net_profit' in all_decisions.columns else "Net profit not calculated")
    print(
        f"Average SOC: {all_decisions['soc'].mean():.2f} MWh" if 'soc' in all_decisions.columns else "SOC not available")
    print(
        f"Total battery cycles: {(all_decisions['charge'].sum() + all_decisions['d_da_bat'].sum() + all_decisions['d_id_bat'].sum()) / 2:.2f}")

    # 4. Prepare and run benchmark simulation
    df_perf = prepare_benchmark_inputs(df_perf)
    df_benchmark = simulate_benchmark_case1(df_perf, battery_params)

    # 5. Calculate benchmark daily metrics - FIXED: Group by date properly
    if 'date' not in df_benchmark.columns:
        df_benchmark['date'] = pd.to_datetime(df_benchmark['datetime']).dt.date

    output_filename = f"benchmark_results_2025H1_{PLANT_ID}.csv"

    print(f"\n💾 Saving results to {output_filename}...")
    df_benchmark.to_csv(output_filename, index=False)

    return all_decisions, output_filename, df_benchmark


if __name__ == "__main__":
    print("=" * 80)
    print("BATTERY TRADING SIMULATION - FULL 2024 RUN")
    print("=" * 80)

    try:
        results, filename = run_full_simulation_from_2024()
        print(f"\n🎉 Simulation completed successfully!")
        print(f"📁 Results saved to: {filename}")

    except Exception as e:
        print(f"❌ Simulation failed with error: {str(e)}")
        import traceback

        traceback.print_exc()


