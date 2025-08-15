from datetime import datetime, timedelta
import pytz
import pandas as pd
import os
import json
from epias_client import get_tgt_token, get_mcp_data, get_id_avg_price_data, get_smf_data, get_realtime_generation
from forecasts_utils import (generate_forecast_price_ar1, generate_forecast_price_id, calculate_forecast_metrics,
                             generate_forecast_production, generate_forecast_smf, generate_forecast_price_ar1_sto, generate_forecast_price_id_sto,
                             generate_forecast_production_sto, generate_forecast_smf_sto)
from battery_brain_core import solve_ex_ante_model, prepare_optimization_data, calculate_flex_decisions
from bbflex_claude import run_full_simulation_fixed, initialize_static_inputs, build_initial_df_full_complete

users = {
    "yigit": "1234",
    "demo": "password",
    "suloglures": "bbcore123",
    "akyelres": "bbflex123"
}

user_to_plant_map = {
    "suloglures": 1920,
    "akyelres": 2591,
    "yigit": 1234,
    "demo": 5678
}

user_battery_params = {
    "suloglures": {
        'capacity': 40.2,
        'power_limit': 61.8,
        'soc_min': 8,
        'soc_max': 40.2,
        'soc_target': 20,
        'efficiency': 0.97,
    },
    "akyelres": {
        'capacity': 62,
        'power_limit': 62,
        'soc_min': 10,
        'soc_max': 62,
        'soc_target': 31,
        'efficiency': 0.95,
    }
}



def calculate_default_strategy_revenue(forecasted_production, forecasted_price_DA):
    """
    Basit stratejide KGÜP = tahmini üretim olarak alınır ve tüm üretim DA'ya satılır.
    Bu fonksiyon bu stratejideki 48 saatlik toplam geliri hesaplar.
    """
    if len(forecasted_production) != len(forecasted_price_DA):
        raise ValueError("forecasted_production ve forecasted_price_DA uzunlukları eşleşmeli.")

    return round(sum(p * pr for p, pr in zip(forecasted_production, forecasted_price_DA)), 2)

def get_dashboard_data(user, active_forecast="meteomatics"):
    production_data = {"real": [], "meteomatics": [], "meteologica": [], "custom": []}
    price_data, price_forecast_data = {"labels": [], "values": []}, {"labels": [], "values": []}
    price_id_data, price_id_forecast_data = {"labels": [], "values": []}, {"labels": [], "values": []}
    q_committed_stacked_data = {"labels": [], "da_prod": [], "da_battery_use": []}
    production_usage_data = {"labels": [], "da_prod": [], "id_prod": [], "charge": [], "forecast": []}
    battery_usage_data = {"labels": [], "charge": [], "d_da_bat": [], "d_id_bat": [], "soc": []}
    default_strategy_df = pd.DataFrame(),

    optimization_results = None
    flex_decisions = []
    expected_revenue = 0
    q_committed_data = {"labels": [], "values": []}
    virtual_now = datetime.now(pytz.timezone("Europe/Istanbul")).replace(minute=0, second=0, microsecond=0) - timedelta(days=3)

    battery_params = user_battery_params[user]

    if user == "suloglures":
        try:
            default_strategy_revenue = 0
            istanbul = pytz.timezone("Europe/Istanbul")
            now = datetime.now(istanbul)
            virtual_now = now - timedelta(days=3)
            start = virtual_now - timedelta(days=3)
            end = virtual_now + timedelta(days=2)
            graph_start = virtual_now - timedelta(days=1)
            graph_end = virtual_now + timedelta(days=2)
            decision_time = virtual_now.replace(hour=10, minute=0, second=0, microsecond=0)
            horizon_start = (decision_time + timedelta(days=1)).replace(hour=0)
            horizon_end = horizon_start + timedelta(hours=47)
            tgt = get_tgt_token("fyigitkavak@icloud.com", "IDK1.35e")

            df_gop = pd.DataFrame(get_mcp_data(tgt, start, end))
            df_gop_forecast = generate_forecast_price_ar1(df_gop)
            df_gop_forecast = df_gop_forecast[(df_gop_forecast["datetime"] >= graph_start) & (df_gop_forecast["datetime"] <= graph_end.replace(hour=23))]

            cutoff = virtual_now.replace(hour=23) if virtual_now.hour < 14 else (virtual_now + timedelta(days=1)).replace(hour=23)
            df_real = df_gop_forecast[df_gop_forecast["datetime"] <= cutoff]
            df_forecast = df_gop_forecast

            price_data = {
                "labels": [dt.strftime("%d %b %H:%M") for dt in df_real["datetime"]],
                "values": df_real["price_real"].round(2).tolist()
            }
            price_forecast_data = {
                "labels": [dt.strftime("%d %b %H:%M") for dt in df_forecast["datetime"]],
                "values": df_forecast["price_forecast"].round(2).tolist()
            }

            df_id = pd.DataFrame(get_id_avg_price_data(tgt, start, end))
            df_id_forecast = generate_forecast_price_id(df_id)
            df_id_forecast = df_id_forecast[(df_id_forecast["datetime"] >= graph_start) & (df_id_forecast["datetime"] <= graph_end.replace(hour=23))]
            df_id_real = df_id_forecast[df_id_forecast["datetime"] <= virtual_now]

            price_id_data = {
                "labels": [dt.strftime("%d %b %H:%M") for dt in df_id_real["datetime"]],
                "values": df_id_real["price_real"].round(2).tolist()
            }
            price_id_forecast_data = {
                "labels": [dt.strftime("%d %b %H:%M") for dt in df_id_forecast["datetime"]],
                "values": df_id_forecast["price_forecast"].round(2).tolist()
            }

            df_smf = pd.DataFrame(get_smf_data(tgt, start, end))
            df_smf_forecast = generate_forecast_smf(df_smf)
            df_smf_forecast = df_smf_forecast[(df_smf_forecast["datetime"] >= graph_start) & (df_smf_forecast["datetime"] <= graph_end.replace(hour=23))]

            plant_id = 1920
            max_allowed_date = now - timedelta(days=1)
            date_range = pd.date_range(start=start, end=min(end, max_allowed_date), freq="D")
            prod_dfs = []
            for single_date in date_range:
                date_str = single_date.strftime("%Y-%m-%d")
                daily_data = get_realtime_generation(tgt, plant_id, date_str)
                if daily_data is not None and not daily_data.empty:
                    prod_dfs.append(daily_data)

            if prod_dfs:
                df_prod = pd.concat(prod_dfs, ignore_index=True)
                df_prod["datetime"] = pd.to_datetime(df_prod.iloc[:, 0])
                df_prod = df_prod.sort_values("datetime")
                df_prod["injectionQuantity"] = pd.to_numeric(df_prod["total"], errors="coerce")


                print("\n🔍 Production Debug Log Başlangıcı")
                print("📆 graph_start:", graph_start)
                print("📆 graph_end:", graph_end)
                print("🕒 virtual_now:", virtual_now)
                print("📊 df_prod datetime aralığı:", df_prod['datetime'].min(), "→", df_prod['datetime'].max())
                print("📊 df_prod satır sayısı:", len(df_prod))

                print("🧪 İlk 5 satır:")
                print(df_prod.head())

                print("📊 injectionQuantity örnek:", df_prod['injectionQuantity'].dropna().head(5).tolist())

                forecast_dict = {
                    "meteomatics": generate_forecast_production(df_prod, error_sd=0.05, power_limit= 61.8),
                    "meteologica": generate_forecast_production(df_prod, error_sd=0.15, power_limit= 61.8),
                    "custom": None
                }

                if os.path.exists("user_forecast.json"):
                    with open("user_forecast.json", "r", encoding="utf-8") as f:
                        user_forecast = json.load(f)
                    forecast_dict["custom"] = pd.DataFrame({
                        "datetime": pd.to_datetime(list(user_forecast.keys())),
                        "forecast_total": list(map(float, user_forecast.values()))
                    })

                df_real_prod = df_prod[df_prod["datetime"] <= virtual_now]
                all_datetimes = df_prod["datetime"]
                real_series = df_real_prod.set_index("datetime").reindex(all_datetimes)["injectionQuantity"].round(2)

                production_data = {
                    "real": [{"x": dt.isoformat(), "y": y} for dt, y in zip(all_datetimes, real_series)],
                    "meteomatics": [],
                    "meteologica": [],
                    "custom": []
                }
                for key, df in forecast_dict.items():
                    if df is not None:
                        filled = df.set_index("datetime").reindex(all_datetimes)["forecast_total"].fillna(0).round(2)
                        production_data[key] = [{"x": dt.isoformat(), "y": y} for dt, y in zip(all_datetimes, filled)]

                df_gop_forecast = df_gop_forecast.rename(columns={"price_forecast": "DA_price_forecasted"})
                df_id_forecast = df_id_forecast.rename(columns={"price_forecast": "ID_price_forecasted"})
                df_smf_forecast = df_smf_forecast.rename(columns={"smf_forecast": "SMF_forecasted"})

                df_gop_forecast = df_gop_forecast[(df_gop_forecast["datetime"] >= horizon_start) & (df_gop_forecast["datetime"] <= horizon_end)]
                df_id_forecast = df_id_forecast[(df_id_forecast["datetime"] >= horizon_start) & (df_id_forecast["datetime"] <= horizon_end)]

                selected_forecast_df = forecast_dict.get(active_forecast)
                if selected_forecast_df is not None:
                    selected_forecast_df = selected_forecast_df[(selected_forecast_df["datetime"] >= horizon_start) & (selected_forecast_df["datetime"] <= horizon_end)]

                    optimization_data = prepare_optimization_data(
                        df_gop_forecast, df_id_forecast, df_smf_forecast,
                        {active_forecast: [{"x": dt.isoformat(), "y": y} for dt, y in zip(selected_forecast_df["datetime"], selected_forecast_df["forecast_total"])]},
                        horizon_start, horizon_end
                    )

                    if not optimization_data.empty:
                        optimization_results = solve_ex_ante_model(optimization_data)
                        if optimization_results['status'] == 'optimal':
                            expected_revenue = optimization_results['expected_revenue_total']
                            flex_decisions = calculate_flex_decisions(optimization_results)
                            optimization_results['forecast_total'] = selected_forecast_df['forecast_total'].round(2).tolist()

                            q_committed_stacked_data = {
                                "labels": [pd.to_datetime(dt).strftime("%d %b %H:%M") for dt in optimization_results['datetime']],
                                "da_prod": [round(x, 2) for x in optimization_results['d_da_prod']],
                                "da_battery_use": [round(x, 2) for x in optimization_results['d_da_bat']]
                            }
                            production_usage_data = {
                                "labels": [pd.to_datetime(dt).strftime("%d %b %H:%M") for dt in optimization_results['datetime']],
                                "da_prod": [round(x, 2) for x in optimization_results['d_da_prod']],
                                "id_prod": [round(x, 2) for x in optimization_results['d_id_prod']],
                                "charge": [round(x, 2) for x in optimization_results['charge']],
                                "forecast": optimization_results['forecast_total']
                            }

                            battery_usage_data = {
                                "labels": [pd.to_datetime(dt).strftime("%d %b %H:%M") for dt in
                                           optimization_results['datetime']],
                                "charge": [round(x, 2) for x in optimization_results['charge']],
                                "d_da_bat": [round(x, 2) for x in optimization_results['d_da_bat']],
                                "d_id_bat": [round(x, 2) for x in optimization_results['d_id_bat']],
                                "soc": [round(x, 2) for x in optimization_results['soc']]
                            }

                            default_strategy_revenue = calculate_default_strategy_revenue(
                                forecasted_production=optimization_results['forecast_total'],
                                forecasted_price_DA=optimization_results['forecasted_price_DA']
                            )

                            common_timeline = selected_forecast_df["datetime"]
                            df_gop_forecast = df_gop_forecast.set_index("datetime").reindex(
                                common_timeline).reset_index()
                            df_id_forecast = df_id_forecast.set_index("datetime").reindex(common_timeline).reset_index()
                            df_smf_forecast = df_smf_forecast.set_index("datetime").reindex(
                                common_timeline).reset_index()

                            default_strategy_df = pd.DataFrame({
                                "datetime": selected_forecast_df["datetime"].tolist(),
                                "forecast": selected_forecast_df["forecast_total"].round(2).tolist(),
                                "da_price": df_gop_forecast["DA_price_forecasted"].round(2).fillna(0).tolist(),
                                "id_price": df_id_forecast["ID_price_forecasted"].round(2).fillna(0).tolist(),
                                "smf": df_smf_forecast["SMF_forecasted"].round(2).fillna(0).tolist(),
                                "kgup_prod": selected_forecast_df["forecast_total"].round(2).tolist(),
                                "gip_prod": [0.0] * len(selected_forecast_df),
                                "charge": [0.0] * len(selected_forecast_df),
                                "kgup_bat": [0.0] * len(selected_forecast_df),
                                "gip_bat": [0.0] * len(selected_forecast_df),
                                "soc": [battery_params["soc_target"]] * len(selected_forecast_df),
                            })

        except Exception as e:
            print("HATA:", e)



    elif user == "akyelres":

        try:
            default_strategy_revenue = 0
            istanbul = pytz.timezone("Europe/Istanbul")
            now = datetime.now(istanbul)
            virtual_now = now - timedelta(days=3)
            start = virtual_now - timedelta(days=3)
            end = virtual_now + timedelta(days=2)
            graph_start = virtual_now - timedelta(days=1)
            graph_end = virtual_now + timedelta(days=2)
            decision_time = (virtual_now - timedelta(days=1)).replace(hour=10, minute=0, second=0, microsecond=0)
            horizon_start = (decision_time + timedelta(days=1)).replace(hour=0)
            horizon_end = horizon_start + timedelta(hours=47)
            tgt = get_tgt_token("fyigitkavak@icloud.com", "IDK1.35e")
            plant_id = 2591
            just_before = virtual_now.replace(minute=0, second=0, microsecond=0)

            df_mcp_ante, df_id_ex_ante, df_smf_ante, df_prod_ante, battery_params, PLANT_ID = initialize_static_inputs(decision_time, start, end)
            df_mcp_post, df_id_ex_post, df_smf_post, df_prod_post, battery_params, PLANT_ID = initialize_static_inputs(
                just_before, start, end)


            df_results, df_full = run_full_simulation_fixed(decision_time,df_mcp_ante, df_id_ex_ante, df_smf_ante, df_prod_ante, battery_params, plant_id)
            df_results, df_full = run_full_simulation_fixed(just_before, df_mcp_post, df_id_ex_post, df_smf_post, df_prod_post, battery_params, plant_id, df_full = df_full)
            mask_q = (df_full['datetime'] >= horizon_start) & (df_full['datetime'] <= horizon_end)
            labels = df_full.loc[mask_q, 'datetime'].dt.strftime("%d %b %H:%M").tolist()
            q_committed = []
            da_prod = []
            da_battery_use = []

            for dt in df_full.loc[mask_q, 'datetime']:
                if dt <= virtual_now.replace(hour=23):
                    q_committed.append(df_full.loc[df_full['datetime'] == dt, 'q_committed'].values[0])
                    da_prod.append(
                        df_full.loc[df_full['datetime'] == dt, ['d_da_prod', 'd_da_prod_extra', 'd_id_prod_adj']]
                        .fillna(0).sum(axis=1).values[0]
                    )
                    da_battery_use.append(
                        df_full.loc[df_full['datetime'] == dt, ['d_da_bat', 'd_da_bat_extra', 'd_id_bat_adj']]
                        .fillna(0).sum(axis=1).values[0]
                    )
                else:
                    q_committed.append(df_full.loc[df_full['datetime'] == dt, 'q_committed_horizon'].values[0])
                    da_prod.append(df_full.loc[df_full['datetime'] == dt, 'd_da_prod_horizon'].fillna(0).values[0])
                    da_battery_use.append(
                        df_full.loc[df_full['datetime'] == dt, 'd_da_bat_horizon'].fillna(0).values[0])

            id_prod = []
            charge = []
            forecast = []

            for dt in df_full.loc[mask_q, 'datetime']:
                if dt <= virtual_now.replace(hour=23):
                    id_prod.append(
                        df_full.loc[df_full['datetime'] == dt, ['d_id_prod', 'd_id_prod_extra']]
                        .fillna(0).sum(axis=1).values[0] -
                        df_full.loc[df_full['datetime'] == dt, 'd_id_prod_adj'].fillna(0).values[0]
                    )
                    charge.append(
                        df_full.loc[df_full['datetime'] == dt, ['charge', 'charge_extra']]
                        .fillna(0).sum(axis=1).values[0] -
                        df_full.loc[df_full['datetime'] == dt, 'charge_adj'].fillna(0).values[0]
                    )
                else:
                    id_prod.append(df_full.loc[df_full['datetime'] == dt, 'd_id_prod_horizon'].fillna(0).values[0])
                    charge.append(df_full.loc[df_full['datetime'] == dt, 'charge_horizon'].fillna(0).values[0])

                forecast.append(df_full.loc[df_full['datetime'] == dt, 'forecasted_production_meteologica'].fillna(0).values[0])

            production_usage_data = {
                "labels": labels,
                "da_prod": [round(x, 2) for x in da_prod],  # bu zaten yukarıda oluşturuldu
                "id_prod": [round(x, 2) for x in id_prod],
                "charge": [round(x, 2) for x in charge],
                "forecast": [round(x, 2) for x in forecast],
            }

            d_da_bat = da_battery_use  # daha önce zaten oluşturuldu
            d_id_bat = []
            soc = []

            for dt in df_full.loc[mask_q, 'datetime']:
                if dt <= virtual_now.replace(hour=23):
                    d_id_bat.append(
                        df_full.loc[df_full['datetime'] == dt, ['d_id_bat', 'd_id_bat_extra']]
                        .fillna(0).sum(axis=1).values[0] -
                        df_full.loc[df_full['datetime'] == dt, 'd_id_bat_adj'].fillna(0).values[0]
                    )
                else:
                    d_id_bat.append(df_full.loc[df_full['datetime'] == dt, 'd_id_bat_horizon'].fillna(0).values[0])

                soc.append(df_full.loc[df_full['datetime'] == dt, 'soc_post'].fillna(0).values[0])

            battery_usage_data = {
                "labels": labels,
                "charge": [round(x, 2) for x in charge],  # daha önce oluşturulan charge
                "d_da_bat": [round(x, 2) for x in d_da_bat],
                "d_id_bat": [round(x, 2) for x in d_id_bat],
                "soc": [round(x, 2) for x in soc]
            }

            # Decision time'daki forecast ve fiyatları al
            print("\n📊 [DEBUG] df_prod_ante:")
            print("🟡", df_prod_ante)
            df_prod_ante["injectionQuantity"] = pd.to_numeric(df_prod_ante["total"], errors="coerce")
            df_forecast_ante = generate_forecast_production_sto(df_prod_ante, decision_time, error_sd_min= 0.03, error_sd_max= 0.15, error_sd_past= 0.03, power_limit= 62)
            df_forecast_ante["datetime"] = pd.to_datetime(df_forecast_ante["datetime"])
            df_forecast_ante = df_forecast_ante[
                (df_forecast_ante["datetime"] >= horizon_start) & (df_forecast_ante["datetime"] <= horizon_end)]

            print("\n📊 [DEBUG] df_mcp_ante:")
            print("🟡", df_mcp_ante)


            df_price_ante = df_mcp_ante.copy()
            df_price_ante["datetime"] = pd.to_datetime(df_price_ante["date"])
            df_price_ante = df_price_ante[
                (df_price_ante["datetime"] >= horizon_start) & (df_price_ante["datetime"] <= horizon_end)]

            # Forecastler ile fiyatları reindex ile hizala
            df_price_ante = df_price_ante.set_index("datetime").reindex(df_forecast_ante["datetime"]).reset_index()
            df_forecast_price_ante = generate_forecast_price_ar1_sto(df_price_ante, decision_time)
            forecasted_production = df_forecast_ante["forecast_total"].round(2).tolist()
            forecasted_price_DA = df_forecast_price_ante["price_forecast"].round(2).fillna(0).tolist()

            print("\n📊 [DEBUG] Forecasted Production:")
            print("🟡", forecasted_production)

            print("\n📈 [DEBUG] Forecasted DA Prices:")
            print("🔵", forecasted_price_DA)
            # Strateji geliri hesapla
            default_strategy_revenue = calculate_default_strategy_revenue(
                forecasted_production=forecasted_production,
                forecasted_price_DA=forecasted_price_DA
            )

            # sadece horizon aralığını al
            df_default = df_full.loc[mask_q].copy()
            print("\n📈 [DEBUG] df_default:")
            print(df_default)
            default_strategy_df = 0
            default_strategy_df = pd.DataFrame({
                "datetime": df_default["datetime"].tolist(),
                "forecast": df_default["forecasted_production"].round(2).tolist(),
                "da_price": df_default["DA_price_real"].round(2).fillna(0).tolist(),
                "id_price": df_default["ID_price_real"].round(2).fillna(0).tolist(),
                "smf": df_default["SMF_real"].round(2).fillna(0).tolist(),
                "kgup_prod": df_default["forecasted_production"].round(2).tolist(),
                "gip_prod": [0.0] * len(df_default),
                "charge": [0.0] * len(df_default),
                "kgup_bat": [0.0] * len(df_default),
                "gip_bat": [0.0] * len(df_default),
                "soc": [battery_params["soc_target"]] * len(df_default),
            })



        except Exception as e:

            print("AKYELRES HATA:", e)

    return {
        "user": user,
        "virtual_now": virtual_now,
        "price_data": price_data,
        "price_forecast_data": price_forecast_data,
        "price_id_data": price_id_data,
        "price_id_forecast_data": price_id_forecast_data,
        "flex_decisions": flex_decisions,
        "expected_revenue": expected_revenue,
        "production_data": production_data,
        "q_committed_data": q_committed_data,
        "q_committed_stacked_data": q_committed_stacked_data,
        "production_usage_data": production_usage_data,
        "optimization_results": optimization_results,
        "active_forecast": active_forecast,
        "initial_soc": "%50",
        "plant_status": "Aktif",
        "capacity_utilization": "%95",
        "graph_start": graph_start.isoformat(),
        "graph_end": graph_end.isoformat(),
        "horizon_start": horizon_start,
        "horizon_end": horizon_end,
        "battery_usage_data": battery_usage_data,
        "default_strategy_revenue": default_strategy_revenue,
        "default_strategy_df": default_strategy_df

    }
