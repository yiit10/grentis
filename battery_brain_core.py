import pandas as pd
import numpy as np
from pulp import *
from datetime import datetime, timedelta
import pytz


def solve_ex_ante_model(db, battery_params=None, initial_soc = 31):


    # Default battery parameters
    if battery_params is None:
        battery_params = {
            'capacity': 40.2,
            'power_limit': 61.8,
            'soc_min': 8,
            'soc_max': 40.2,
            'soc_target': 20,
            'efficiency': 0.97,
        }

    # Extract parameters
    horizon_hours = len(db)
    forecasted_price_DA = db['DA_price_forecasted'].values
    forecasted_price_ID = db['ID_price_forecasted'].values
    forecasted_price_SMF = db['SMF_forecasted'].values
    forecasted_production = db['forecasted_production'].values

    # Cost parameters
    unit_battery_cost = 1276

    # Battery parameters
    eta = battery_params['efficiency']
    P_max = battery_params['power_limit']
    big_M = 1000
    penalty_unit_price = np.maximum(forecasted_price_DA, forecasted_price_SMF) * 1.03

    # Create the optimization problem
    model = LpProblem("Battery_Optimization", LpMaximize)

    # Decision variables
    d_da_prod = LpVariable.dicts("d_da_prod", range(horizon_hours), lowBound=0, cat='Continuous')
    d_id_prod = LpVariable.dicts("d_id_prod", range(horizon_hours), lowBound=0, cat='Continuous')
    d_da_bat = LpVariable.dicts("d_da_bat", range(horizon_hours), lowBound=0, cat='Continuous')
    d_id_bat = LpVariable.dicts("d_id_bat", range(horizon_hours), lowBound=0, cat='Continuous')
    charge = LpVariable.dicts("charge", range(horizon_hours), lowBound=0, cat='Continuous')
    x_id = LpVariable.dicts("x_id", range(horizon_hours), lowBound=0, cat='Continuous')
    x_ceza = LpVariable.dicts("x_ceza", range(horizon_hours), lowBound=0, cat='Continuous')
    is_charging = LpVariable.dicts("is_charging", range(horizon_hours), cat='Binary')
    soc_final = LpVariable("soc_final", lowBound=battery_params['soc_min'], upBound=battery_params['soc_max'])
    soc = LpVariable.dicts("soc", range(horizon_hours),
                           lowBound=battery_params['soc_min'],
                           upBound=battery_params['soc_max'],
                           cat='Continuous')

    # Additional variables
    prod_surplus = LpVariable.dicts("prod_surplus", range(horizon_hours), lowBound=0, cat='Continuous')
    prod_deficit = LpVariable.dicts("prod_deficit", range(horizon_hours), lowBound=0, cat='Continuous')
    delta = LpVariable.dicts("delta", range(horizon_hours), cat='Binary')
    q_committed = LpVariable.dicts("q_committed", range(horizon_hours), lowBound=0, cat='Continuous')
    diff_abs = LpVariable.dicts("diff_abs", range(horizon_hours), lowBound=0, cat='Continuous')

    # Objective function
    revenue = 0
    for t in range(horizon_hours):
        revenue += (forecasted_price_DA[t] * (d_da_prod[t] + d_da_bat[t]) +
                    min(forecasted_price_ID[t], 3000) * (d_id_prod[t] + d_id_bat[t]) -
                    forecasted_price_ID[t] * x_id[t] -
                    penalty_unit_price[t] * x_ceza[t] -
                    (unit_battery_cost / 2) * (charge[t] + d_da_bat[t] + d_id_bat[t]))

    model += revenue

    # Constraints
    for t in range(horizon_hours):
        # Production commitment constraint
        model += q_committed[t] == d_da_prod[t] + d_da_bat[t]
        model += battery_params['power_limit'] >= d_da_prod[t] + d_id_bat[t] + d_da_bat[t] + d_id_prod[t]

        # Minimum production constraint
        model += d_da_prod[t] >= 0.25 * forecasted_production[t]

        # Production capacity constraint
        model += forecasted_production[t] >= d_da_prod[t] + d_id_prod[t] + charge[t]

        model += forecasted_production[t] + x_id[t] + x_ceza[t] + d_da_bat[t] + d_id_bat[t] == q_committed[t] + d_id_prod[t] + d_id_bat[t] + charge[t]

        # Battery power constraints
        model += d_da_bat[t] + d_id_bat[t] <= P_max * (1 - is_charging[t])
        model += charge[t] <= P_max * is_charging[t]

    # SOC constraints
    model += soc[0] == initial_soc
    model += q_committed[0] <= forecasted_production[0] + initial_soc

    # Final SOC eşitliğini ekle
    model += soc_final == soc[horizon_hours - 1] + eta * charge[horizon_hours - 1] - (1 / eta) * (
                d_da_bat[horizon_hours - 1] + d_id_bat[horizon_hours - 1])

    # Finalde SOC batarya limitleri içinde olsun
    model += soc_final >= battery_params['soc_min']
    model += soc_final <= battery_params['soc_max']

    for t in range(1, horizon_hours):
        model += soc[t] == soc[t - 1] + eta * charge[t - 1] - (1 / eta) * (d_da_bat[t - 1] + d_id_bat[t - 1])
        model += q_committed[t] <= forecasted_production[t] + soc[t]

    # Solve the model
    try:
        model.solve(GUROBI(msg=1))


        if model.status == 1:  # Optimal solution found
            # Battery cost'u post-processing aşamasında hesapla
            battery_cost = (unit_battery_cost / 2) * sum(
                value(charge[t]) + value(d_da_bat[t]) + value(d_id_bat[t])
                for t in range(horizon_hours)
            )

            expected_revenue = value(model.objective) + battery_cost

            results = {
                'status': 'optimal',
                'objective_value': value(model.objective),
                'datetime': db['datetime'].tolist(),
                'q_committed': [value(q_committed[t]) for t in range(horizon_hours)],
                'd_da_prod': [value(d_da_prod[t]) for t in range(horizon_hours)],
                'd_id_prod': [value(d_id_prod[t]) for t in range(horizon_hours)],
                'd_da_bat': [value(d_da_bat[t]) for t in range(horizon_hours)],
                'd_id_bat': [value(d_id_bat[t]) for t in range(horizon_hours)],
                'charge': [value(charge[t]) for t in range(horizon_hours)],
                'soc': [value(soc[t]) for t in range(horizon_hours)],
                'x_id': [value(x_id[t]) for t in range(horizon_hours)],
                'x_ceza': [value(x_ceza[t]) for t in range(horizon_hours)],
                'is_charging': [value(is_charging[t]) for t in range(horizon_hours)],
                'forecasted_production': forecasted_production.tolist(),
                'forecasted_price_DA': forecasted_price_DA.tolist(),
                'forecasted_price_ID': forecasted_price_ID.tolist(),
                'battery_cost_total': battery_cost,
                'expected_revenue_total': expected_revenue,
            }

            print(print_optimization_decisions(results))

            return results
        else:
            if model.status != 1:  # LpStatusOptimal
                model.writeLP("exante_debug.lp")
                return {
                    'status': 'failed',
                    'message': f'Optimization failed with status: {LpStatus[model.status]}'
                }

    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error during optimization: {str(e)}'
        }


def prepare_optimization_data(df_gop_forecast, df_id_forecast, df_smf_forecast,
                              production_data,
                              horizon_start, horizon_end):
    try:
        # --- 1. Forecast datasını dataframe'e çevir ---
        meteologica_prod = pd.DataFrame(production_data['meteologica'])

        if meteologica_prod.empty:
            print("⚠️ Meteologica forecast datası boş, optimizasyon iptal.")
            return pd.DataFrame()

        meteologica_prod['datetime'] = pd.to_datetime(meteologica_prod['x'])
        meteologica_prod = meteologica_prod.rename(columns={'y': 'forecasted_production'})

        # --- 2. Fiyat verilerini merge et ---
        df_combined = (
            df_gop_forecast
            .merge(df_id_forecast, on='datetime', how='outer')
            .merge(df_smf_forecast, on='datetime', how='outer')
        )

        # --- 3. Üretim forecast'ini merge et ---
        df_combined = df_combined.merge(meteologica_prod[['datetime', 'forecasted_production']],
                                        on='datetime', how='left')
        print(df_combined)

        # --- 4. Kolon isimlerini netleştir ---
        df_combined['DA_price_forecasted'] = df_combined['DA_price_forecasted']
        df_combined['ID_price_forecasted'] = df_combined['ID_price_forecasted']  # İki tahmin ayrıysa değiştir
        df_combined['SMF_forecasted'] = df_combined['SMF_forecasted']

        # --- 5. Sadece gerekli kolonları bırak ---
        df_combined = df_combined[['datetime', 'DA_price_forecasted', 'ID_price_forecasted',
                                   'SMF_forecasted', 'forecasted_production']]

        # --- 6. NaN varsa 0'a çevir ---
        df_combined = df_combined.fillna(0)

        # --- 7. Sadece geleceğe dönük 48 saatlik veriyle çalış ---
        future_data = df_combined[
            (df_combined['datetime'] >= horizon_start) &
            (df_combined['datetime'] <= horizon_end)
        ].copy()

        pd.set_option('display.max_rows', None)  # Satır limiti yok
        pd.set_option('display.max_columns', None)  # Sütun limiti yok
        pd.set_option('display.width', None)  # Satırlar tek satırda kesilmeden gelsin
        pd.set_option('display.max_colwidth', None)  # Hücrelerde metin kesilmesin
        print(future_data)

        print("✅ prepare_optimization_data tamamlandı:", len(future_data), "saatlik veri hazır.")
        return future_data

    except Exception as e:
        print(f"❌ prepare_optimization_data hatası: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def print_optimization_decisions(opt_results):
    if opt_results['status'] != 'optimal':
        print("⚠️ Optimum çözüm bulunamadı.")
        return

    print("\n🔍 48 Saatlik Karar Tablosu:\n" + "-"*80)
    for i, dt in enumerate(opt_results["datetime"]):
        print(f"""
🕒 {pd.to_datetime(dt).strftime("%d %b %H:%M")}
    📌 Q Committed       : {opt_results['q_committed'][i]:>6.2f} MWh
    🛠️  DA Prod           : {opt_results['d_da_prod'][i]:>6.2f}
    🛠️  ID Prod           : {opt_results['d_id_prod'][i]:>6.2f}
    🔋 DA Battery Use    : {opt_results['d_da_bat'][i]:>6.2f}
    🔋 ID Battery Use    : {opt_results['d_id_bat'][i]:>6.2f}
    ⚡ Charge            : {opt_results['charge'][i]:>6.2f}
    🪫 SoC               : {opt_results['soc'][i]:>6.2f}
    ⚙️  Charging Mode     : {"Yes" if opt_results['is_charging'][i] > 0.5 else "No"}
""")
    print("-"*80)


def calculate_flex_decisions(optimization_results):

    if optimization_results['status'] != 'optimal':
        return []

    flex_decisions = []

    for i in range(len(optimization_results['datetime'])):
        dt = pd.to_datetime(optimization_results['datetime'][i])
        is_charging = optimization_results['is_charging'][i]
        charge_power = optimization_results['charge'][i]
        discharge_power = optimization_results['d_da_bat'][i] + optimization_results['d_id_bat'][i]
        soc = optimization_results['soc'][i]

        if is_charging > 0.5:  # Charging
            action = "Şarj"
            power = charge_power
        elif discharge_power > 0.1:  # Discharging
            action = "Deşarj"
            power = discharge_power
        else:
            action = "Bekle"
            power = 0

        flex_decisions.append({
            "hour": dt.strftime("%H:%M"),
            "datetime": dt.strftime("%d %b %H:%M"),
            "action": action,
            "power": round(power, 2),
            "soc": round(soc, 2),
            "q_committed": round(optimization_results['q_committed'][i], 2)
        })

    return flex_decisions