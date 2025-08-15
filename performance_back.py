# performance_back.py - Updated with revenue calculation
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
)
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np
import os
from bbflex_claude import calculate_battery_trading_metrics_flex

battery_params = {
    'efficiency': 0.95,
    'soc_min': 10,
    'soc_max': 62,
    'soc_target': 31
}

def apply_case_logic(df, i, bp):
    """
    Apply battery case logic and update SOC properly

    Args:
        df: DataFrame with energy data
        i: Current row index
        bp: Battery parameters dict with keys: efficiency, soc_min, soc_max, soc_target
    """
    # Get current row values
    d_da_prod = df.loc[i, 'd_da_prod']
    d_da_bat = df.loc[i, 'd_da_bat']
    charge = df.loc[i, 'charge']
    d_id_prod = df.loc[i, 'd_id_prod']
    d_id_bat = df.loc[i, 'd_id_bat']
    real_prod = df.loc[i, 'real_production']
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
    delta = real_prod - d_da_prod - d_id_prod - charge

    # Helper functions
    def bat_avail():
        """Available battery energy for discharge"""
        return max(0, prev_soc - soc_min) * eff

    def bat_room():
        """Available battery capacity for charging"""
        return max(0, soc_max - prev_soc) * eff


    # Determine case based on initial conditions
    case_id = f"{int(d_da_prod > 0)}-{int(d_da_bat > 0)}-{int(charge > 0)}-{int(d_id_prod > 0)}-{int(d_id_bat > 0)}"

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
            cut = min(d_id_bat, need)
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
            cut = min(d_id_prod, need)
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
            dec_prod = min(d_id_prod, need)
            df.loc[i, 'd_id_prod'] -= dec_prod
            df.loc[i, 'd_da_prod'] += dec_prod
            remain = need - dec_prod

            # Then reduce ID battery
            if remain > 0:
                dec_bat_id = min(d_id_bat, remain)
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
            dec_bat_id = min(d_id_bat, need)
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
            dec_prod = min(d_id_prod, need)
            df.loc[i, 'd_id_prod'] -= dec_prod
            df.loc[i, 'd_da_prod'] += dec_prod
            remain = need - dec_prod

            # Then reduce ID battery
            if remain > 0:
                dec_bat_id = min(d_id_bat, remain)
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
            dec_prod = min(d_id_prod, need)
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
            charge_dec = min(charge, need)
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


def calculate_battery_trading_metrics(df, battery_params):
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
        id_revenue = (row['d_id_prod'] + row['d_id_bat']) * row['ID_price_real']

        total_revenue = da_revenue + id_revenue
        df_metrics.loc[i, 'DA_revenue'] = da_revenue
        df_metrics.loc[i, 'ID_revenue'] = id_revenue
        df_metrics.loc[i, 'revenue'] = total_revenue

        # === 2. BATTERY COSTS ===
        # Total battery usage (charge + discharge)
        total_battery_usage = row['d_da_bat'] + row['d_id_bat'] + row['charge']
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
        standard_penalty = row['x_ceza'] * penalty_price
        total_penalty = standard_penalty
        df_metrics.loc[i, 'penalty'] = total_penalty

        # === 5. INTRADAY MARKET COSTS ===
        # Cost of buying from intraday market (x_id)
        id_cost = row['x_id'] * id_price
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


def calculate_battery_metrics(df_metrics, battery_params):
    """
    Calculate detailed battery performance metrics

    Args:
        df_metrics: DataFrame with hourly metrics
        battery_params: Battery specifications

    Returns:
        Dict with battery performance metrics
    """

    # Battery capacity and efficiency
    battery_capacity = battery_params['soc_max']
    efficiency = battery_params['efficiency']

    # Calculate battery metrics
    metrics = {
        # === ENERGY METRICS ===
        'total_energy_charged': df_metrics['charge'].sum(),
        'total_energy_discharged': df_metrics['d_da_bat'].sum() + df_metrics['d_id_bat'].sum(),
        'total_battery_throughput': df_metrics['total_battery_usage'].sum(),

        # === CYCLE METRICS ===
        'total_cycles': df_metrics['cycle_equivalent'].sum(),
        'total_cycles_normalized': df_metrics['cycle_equivalent'].sum() / battery_capacity,
        'avg_daily_cycles': df_metrics['cycle_equivalent'].sum() / (len(df_metrics) / 24),

        # === UTILIZATION METRICS ===
        'avg_soc': df_metrics['soc'].mean(),
        'min_soc': df_metrics['soc'].min(),
        'max_soc': df_metrics['soc'].max(),
        'soc_variance': df_metrics['soc'].var(),

        # === EFFICIENCY METRICS ===
        'round_trip_efficiency': efficiency,
        'energy_efficiency': (df_metrics['charge'].sum() * efficiency) / (
                    df_metrics['d_da_bat'].sum() + df_metrics['d_id_bat'].sum()) if (df_metrics['d_da_bat'].sum() +
                                                                                     df_metrics[
                                                                                         'd_id_bat'].sum()) > 0 else 0,

        # === OPERATIONAL METRICS ===
        'charging_hours': (df_metrics['charge'] > 0).sum(),
        'discharging_hours': ((df_metrics['d_da_bat'] + df_metrics['d_id_bat']) > 0).sum(),
        'idle_hours': ((df_metrics['charge'] == 0) & (df_metrics['d_da_bat'] + df_metrics['d_id_bat'] == 0)).sum(),

        # === COST METRICS ===
        'total_degradation_cost': df_metrics['battery_cost'].sum(),
        'cost_per_cycle': df_metrics['battery_cost'].sum() / df_metrics['cycle_equivalent'].sum() if df_metrics[
                                                                                                         'cycle_equivalent'].sum() > 0 else 0,
        'cost_per_mwh': df_metrics['battery_cost'].sum() / df_metrics['total_battery_usage'].sum() if df_metrics[
                                                                                                          'total_battery_usage'].sum() > 0 else 0,
    }

    return metrics


def calculate_daily_metrics(df_metrics, battery_params):
    """
    Calculate daily aggregated metrics

    Args:
        df_metrics: DataFrame with hourly metrics
        battery_params: Battery specifications

    Returns:
        DataFrame with daily metrics
    """

    # Create date column
    df_metrics['date'] = pd.to_datetime(df_metrics['datetime']).dt.date

    # Daily aggregations
    daily_metrics = df_metrics.groupby('date').agg({
        # Financial metrics
        'net_profit': 'sum',
        'revenue': 'sum',
        'DA_revenue': 'sum',
        'ID_revenue': 'sum',
        'battery_cost': 'sum',
        'penalty': 'sum',
        'id_costs': 'sum',
        'extra_income': 'sum',

        # Battery metrics
        'soc': ['mean', 'min', 'max', 'std'],
        'charge': 'sum',
        'd_da_bat': 'sum',
        'd_id_bat': 'sum',
        'total_battery_usage': 'sum',
        'cycle_equivalent': 'sum',

        # Market metrics
        'DA_price_real': 'mean',
        'ID_price_real': 'mean',
        'SMF_real': 'mean',
        'real_production': 'sum',
        'forecasted_production': 'sum',

        # Operational metrics
        'x_id': 'sum',
        'x_ceza': 'sum'
    }).reset_index()

    # Flatten column names
    daily_metrics.columns = ['_'.join(col).strip() if col[1] else col[0] for col in daily_metrics.columns.values]
    daily_metrics.rename(columns={'date_': 'date'}, inplace=True)

    # Calculate additional daily metrics
    battery_capacity = battery_params['soc_max']

    daily_metrics['total_discharge'] = daily_metrics['d_da_bat_sum'] + daily_metrics['d_id_bat_sum']
    daily_metrics['total_cycles_normalized'] = daily_metrics['cycle_equivalent_sum'] / battery_capacity
    daily_metrics['profit_per_cycle'] = daily_metrics['net_profit_sum'] / daily_metrics['cycle_equivalent_sum']
    daily_metrics['battery_utilization'] = daily_metrics['total_battery_usage_sum'] / (
                battery_capacity * 2)  # Max possible daily usage
    daily_metrics['production_forecast_accuracy'] = 1 - abs(
        daily_metrics['real_production_sum'] - daily_metrics['forecasted_production_sum']) / daily_metrics[
                                                        'forecasted_production_sum']

    return daily_metrics


def print_comprehensive_summary(df_metrics, daily_metrics, battery_metrics, battery_params):
    """
    Print detailed simulation summary

    Args:
        df_metrics: Hourly metrics DataFrame
        daily_metrics: Daily metrics DataFrame
        battery_metrics: Battery performance metrics dict
        battery_params: Battery specifications
    """

    print("=" * 80)
    print("COMPREHENSIVE BATTERY TRADING SIMULATION SUMMARY")
    print("=" * 80)

    # === FINANCIAL PERFORMANCE ===
    print("\n📊 FINANCIAL PERFORMANCE")
    print("-" * 40)
    print(f"Total Net Profit: {daily_metrics['net_profit_sum'].sum():,.2f} TL")
    print(f"Total Revenue: {daily_metrics['revenue_sum'].sum():,.2f} TL")
    print(f"Total Battery Cost: {daily_metrics['battery_cost_sum'].sum():,.2f} TL")
    print(f"Total Penalties: {daily_metrics['penalty_sum'].sum():,.2f} TL")
    print(f"Total ID Costs: {daily_metrics['id_costs_sum'].sum():,.2f} TL")
    print(f"Average Daily Profit: {daily_metrics['net_profit_sum'].mean():,.2f} TL")
    print(f"Best Day Profit: {daily_metrics['net_profit_sum'].max():,.2f} TL")
    print(f"Worst Day Profit: {daily_metrics['net_profit_sum'].min():,.2f} TL")
    print(f"Profit Volatility (Std): {daily_metrics['net_profit_sum'].std():,.2f} TL")

    # === BATTERY PERFORMANCE ===
    print("\n🔋 BATTERY PERFORMANCE")
    print("-" * 40)
    print(f"Total Battery Cycles: {battery_metrics['total_cycles']:,.2f}")
    print(f"Total Cycles Normalized: {battery_metrics['total_cycles_normalized']:,.2f}")
    print(f"Average Daily Cycles: {battery_metrics['avg_daily_cycles']:,.2f}")
    print(f"Total Energy Charged: {battery_metrics['total_energy_charged']:,.2f} MWh")
    print(f"Total Energy Discharged: {battery_metrics['total_energy_discharged']:,.2f} MWh")
    print(f"Round-trip Efficiency: {battery_metrics['round_trip_efficiency']:.1%}")
    print(f"Average SOC: {battery_metrics['avg_soc']:.2f} MWh")
    print(f"SOC Range: {battery_metrics['min_soc']:.2f} - {battery_metrics['max_soc']:.2f} MWh")

    # === OPERATIONAL METRICS ===
    print("\n⚙️ OPERATIONAL METRICS")
    print("-" * 40)
    print(f"Charging Hours: {battery_metrics['charging_hours']:,}")
    print(f"Discharging Hours: {battery_metrics['discharging_hours']:,}")
    print(f"Idle Hours: {battery_metrics['idle_hours']:,}")
    print(
        f"Battery Utilization: {battery_metrics['charging_hours'] + battery_metrics['discharging_hours']:,}/{len(df_metrics):,} hours ({(battery_metrics['charging_hours'] + battery_metrics['discharging_hours']) / len(df_metrics):.1%})")

    # === COST ANALYSIS ===
    print("\n💰 COST ANALYSIS")
    print("-" * 40)
    print(f"Total Degradation Cost: {battery_metrics['total_degradation_cost']:,.2f} TL")
    print(f"Cost per Cycle: {battery_metrics['cost_per_cycle']:,.2f} TL")
    print(f"Cost per MWh Throughput: {battery_metrics['cost_per_mwh']:,.2f} TL")

    # === MARKET PERFORMANCE ===
    print("\n📈 MARKET PERFORMANCE")
    print("-" * 40)
    print(f"Average DA Price: {daily_metrics['DA_price_real_mean'].mean():,.2f} TL/MWh")
    print(f"Average ID Price: {daily_metrics['ID_price_real_mean'].mean():,.2f} TL/MWh")
    print(f"Average SMF Price: {daily_metrics['SMF_real_mean'].mean():,.2f} TL/MWh")
    print(f"Average Production Forecast Accuracy: {daily_metrics['production_forecast_accuracy'].mean():.1%}")

    # === SIMULATION DETAILS ===
    print("\n📋 SIMULATION DETAILS")
    print("-" * 40)
    print(f"Simulation Period: {daily_metrics['date'].min()} to {daily_metrics['date'].max()}")
    print(f"Number of Days: {len(daily_metrics)}")
    print(f"Total Hours: {len(df_metrics):,}")
    print(f"Battery Capacity: {battery_params['soc_max']:.1f} MWh")
    print(f"Battery Efficiency: {battery_params['efficiency']:.1%}")
    print(f"SOC Range: {battery_params['soc_min']:.1f} - {battery_params['soc_max']:.1f} MWh")

    print("=" * 80)


def analyze_simulation_results(all_decisions, battery_params, save_results=False):
    # Flex mi değil mi ona göre karar ver
    is_flex = "d_id_prod_extra" in all_decisions.columns

    # Hesapla
    if is_flex:
        df_hourly = calculate_battery_trading_metrics_flex(all_decisions, battery_params)
    else:
        df_hourly = calculate_battery_trading_metrics(all_decisions, battery_params)

    df_hourly['date'] = df_hourly['datetime'].dt.date

    # Ortak metrikler
    agg_dict = {
        'net_profit': 'sum',
        'battery_cost': 'sum',
        'penalty': 'sum',
        'revenue': 'sum',
        'DA_revenue': 'sum',
        'ID_revenue': 'sum',
        'id_costs': 'sum'
    }

    # Ortalama SOC (her iki versiyonda da kullanılabilir)
    if "soc" in df_hourly.columns:
        agg_dict['soc'] = 'mean'

    # Toplam çevrim sayısı
    if "total_battery_usage" in df_hourly.columns:
        df_hourly['total_cycle'] = df_hourly['total_battery_usage'] / 2
        agg_dict['total_cycle'] = 'sum'
    elif set(['charge', 'd_da_bat', 'd_id_bat']).issubset(df_hourly.columns):
        df_hourly['total_cycle'] = (df_hourly['charge'] + df_hourly['d_da_bat'] + df_hourly['d_id_bat']) / 2
        agg_dict['total_cycle'] = 'sum'

    df_daily = df_hourly.groupby('date').agg(agg_dict).reset_index()

    return df_hourly, df_daily



# Example usage
if __name__ == "__main__":
    # Example battery parameters
    battery_params = {
        'efficiency': 0.95,
        'soc_min': 10,
        'soc_max': 62,
        'soc_target': 31
    }


def run_simulation(plant_id):
    USERNAME = "fyigitkavak@icloud.com"
    PASSWORD = "IDK1.35e"
    PLANT_ID = plant_id
    IST = pytz.timezone("Europe/Istanbul")

    START = datetime(2025, 6, 29, 0, 0, tzinfo=IST)
    END = datetime.now(IST).replace(minute=0, second=0, microsecond=0) - timedelta(days=1)

    # 🔐 Token al
    tgt = get_tgt_token(USERNAME, PASSWORD)

    # === FİYAT VERİLERİ ===
    print("📦 MCP verisi çekiliyor...")
    df_mcp = pd.DataFrame(get_mcp_data(tgt, START, END))
    df_mcp["date"] = pd.to_datetime(df_mcp["date"])
    df_mcp = generate_forecast_price_ar1(df_mcp)
    df_mcp = df_mcp.rename(columns={
        "price_real": "DA_price_real",
        "price_forecast": "DA_price_forecasted"
    })

    print("📦 GİP verisi çekiliyor...")
    df_id = pd.DataFrame(get_id_avg_price_data(tgt, START, END))
    df_id["date"] = pd.to_datetime(df_id["date"])
    df_id = generate_forecast_price_id(df_id)
    df_id = df_id.rename(columns={
        "price_real": "ID_price_real",
        "price_forecast": "ID_price_forecasted"
    })

    print("📦 SMF verisi çekiliyor...")
    df_smf = pd.DataFrame(get_smf_data(tgt, START, END))
    df_smf["date"] = pd.to_datetime(df_smf["date"])
    df_smf = generate_forecast_smf(df_smf)
    df_smf = df_smf.rename(columns={
        "smf_real": "SMF_real",
        "smf_forecast": "SMF_forecasted"
    })

    # === ÜRETİM VERİSİ ===
    print("📦 Üretim verileri (öncelik: realtime) çekiliyor...")
    prod_frames = []
    df_rt = pd.DataFrame()
    df_inj = pd.DataFrame()

    current_day = START.date()
    end_day = END.date()

    while current_day <= end_day:
        date_str = current_day.strftime("%Y-%m-%d")

        df_daily_rt = get_realtime_generation(tgt, PLANT_ID, date_str)
        if df_daily_rt is not None and not df_daily_rt.empty:
            df_daily_rt["source"] = "realtime"
            prod_frames.append(df_daily_rt)
            df_rt = pd.concat([df_rt, df_daily_rt], ignore_index=True)
            print(f"✅ {date_str} → Realtime verisi alındı")
        else:
            df_daily_inj = get_injection_quantity(tgt, PLANT_ID, date_str)
            if df_daily_inj is not None and not df_daily_inj.empty:
                df_daily_inj["source"] = "injection"
                prod_frames.append(df_daily_inj)
                df_inj = pd.concat([df_inj, df_daily_inj], ignore_index=True)
                print(f"✅ {date_str} → Injection verisi fallback olarak alındı")
            else:
                print(f"🚫 {date_str} → Hiçbir veri alınamadı")

        current_day += timedelta(days=1)

    df_prod = pd.concat(prod_frames, ignore_index=True) if prod_frames else pd.DataFrame()

    # === ÜRETİM TAHMİNİ ===
    print("📦 Üretim tahmini üretiliyor...")
    if not df_prod.empty:
        df_prod["datetime"] = pd.to_datetime(df_prod["date"])
        if "injectionQuantity" not in df_prod.columns:
            if "total" in df_prod.columns:
                df_prod["injectionQuantity"] = pd.to_numeric(df_prod["total"], errors="coerce")
            else:
                print("⚠️ Ne 'injectionQuantity' ne de 'total' kolonu var → Forecast üretilemedi.")
                df_prod_forecast = pd.DataFrame()
        df_prod_forecast = generate_forecast_production(df_prod, error_sd=0.05)
        df_prod_forecast = df_prod_forecast.rename(columns={
            "forecast_total": "forecasted_production_meteologica"
        })
    else:
        print("⚠ Forecast üretilemedi: df_prod tamamen boş.")
        df_prod_forecast = pd.DataFrame()

    # === VERİLERİ BİRLEŞTİR ===
    print("🧩 Veriler birleştiriliyor...")
    dfs = [df_mcp, df_id, df_smf, df_prod_forecast]
    df_perf = dfs[0]

    for other_df in dfs[1:]:
        df_perf = pd.merge(df_perf, other_df, on="datetime", how="outer")

    # === GERÇEKLEŞEN ÜRETİMİ EKLE ===
    print("📦 Gerçekleşen üretim verisi birleştiriliyor...")
    real_production = pd.DataFrame()
    if not df_rt.empty:
        df_rt["datetime"] = pd.to_datetime(df_rt["date"])
        real_production = df_rt.groupby("datetime")["total"].sum().reset_index()
        real_production.rename(columns={"total": "real_production"}, inplace=True)
        print("✅ Realtime verisi başarıyla işlendi.")
    elif not df_inj.empty:
        df_inj["datetime"] = pd.to_datetime(df_inj["date"])
        real_production = df_inj[["datetime", "total"]].rename(columns={"total": "real_production"})
        print("✅ Injection verisi fallback olarak kullanıldı.")

    if not real_production.empty:
        df_perf = pd.merge(df_perf, real_production, on="datetime", how="left")

    # Sıralayıp eksikleri at
    df_perf = df_perf.sort_values("datetime").reset_index(drop=True)

    # 🔍 Terminalde göster
    pd.set_option('display.max_rows', 200)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)

    print("✅ Tüm veriler hazır. Final df_perf DataFrame:")
    print(df_perf)
    print("✅ Tüm veriler hazır. Final df_perf DataFrame:", df_perf.shape)

    # Başlangıç zamanı ve horizon
    virtual_now = pd.Timestamp("2025-06-30 10:00:00", tz="Europe/Istanbul")
    simulation_end = df_perf['datetime'].max() - timedelta(days=2)

 # Birleştirilmiş tüm kararlar burada birikecek
    all_decisions = pd.DataFrame()

    all_decisions = pd.DataFrame()

    while virtual_now <= simulation_end:
        if virtual_now.hour == 10:
            print(f"⚙️ Optimizasyon başlatılıyor: {virtual_now}")

            horizon_start = (virtual_now + timedelta(hours=14)).replace(minute=0, second=0)
            horizon_end = horizon_start + timedelta(hours=47)

            df_gop_forecast = df_perf[['datetime', 'DA_price_forecasted']]
            df_id_forecast = df_perf[['datetime', 'ID_price_forecasted']]
            df_smf_forecast = df_perf[['datetime', 'SMF_forecasted']]

            production_data = {
                "meteologica": df_perf[['datetime', 'forecasted_production_meteologica']].rename(
                    columns={'datetime': 'x', 'forecasted_production_meteologica': 'y'})
            }

            df_input = prepare_optimization_data(df_gop_forecast, df_id_forecast, df_smf_forecast,
                                                 production_data, horizon_start, horizon_end)

            # 🔋 Get previous SoC
            if not all_decisions.empty and any(all_decisions['datetime'] < virtual_now):
                last_soc = all_decisions.loc[all_decisions['datetime'] < virtual_now, 'soc'].iloc[-1]
            else:
                last_soc = battery_params['soc_target']

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

                print(f"✅ {virtual_now.strftime('%Y-%m-%d %H:%M')} optimizasyonu tamamlandı.")
                print(all_decisions)
            else:
                print(f"❌ Optimizasyon başarısız: {result['status']}")

        virtual_now += timedelta(hours=1)

    # Init missing columns
    for col in ['x_id', 'x_ceza', 'bat_to_discharge']:
        if col not in all_decisions.columns:
            all_decisions[col] = 0.0

    print("\n🧾 Simülasyon tamamlandı. Toplam karar sayısı:", len(all_decisions))

    print("\n⚙️ Dispatch case logic uygulanıyor...")
    for i in range(len(all_decisions)):
        all_decisions = apply_case_logic(all_decisions, i, battery_params)

    return all_decisions


def prepare_benchmark_inputs(df_perf):
    """
    Prepare necessary columns for benchmark simulation:
    - forecasted_production
    - diff (real - forecasted)
    - kgup_total (min(real, forecasted))
    """
    df = df_perf.copy()

    if "forecasted_production_meteologica" in df.columns:
        df["forecasted_production"] = df["forecasted_production_meteologica"]
    else:
        raise ValueError("'forecasted_production_meteologica' column is missing.")

    if "real_production" not in df.columns:
        raise ValueError("'real_production' column is missing.")

    df["diff"] = df["real_production"] - df["forecasted_production"]
    df["kgup_total"] = df["forecasted_production"]

    return df


def simulate_benchmark_case1(df, battery_params, penalty_share=0.50):
    """
    Simulate benchmark strategy (Case 1) with SOC logic updated:
    - SOC reflects value at the **start** of the hour
    """
    df = df.copy()

    # Initialize output columns
    new_cols = [
        "SoC", "charge_used", "discharge_used",
        "id_satis", "id_alim", "id_gelir", "id_maliyet",
        "ceza_miktar", "ceza_maliyet",
        "batarya_cost", "DA_revenue", "net_profit"
    ]
    for col in new_cols:
        df[col] = 0.0

    # Extract parameters
    soc = battery_params['soc_target']
    eta = battery_params['efficiency']
    soc_min = battery_params['soc_min']
    soc_max = battery_params['soc_max']
    power_limit = battery_params.get('power_limit', soc_max)

    # Cost parameters
    unit_bat_cost = 1276
    bat_cost_half = unit_bat_cost / 2
    id_share = 1 - penalty_share

    for i in range(len(df)):
        # Saat başındaki SOC’yi yaz
        df.loc[i, "SoC"] = soc

        # Skip NaNs
        real_prod = df.loc[i, "real_production"]
        forecasted_prod = df.loc[i, "forecasted_production"]
        id_price = df.loc[i, "ID_price_real"]
        da_price = df.loc[i, "DA_price_real"]
        smf_price = df.loc[i, "SMF_real"]

        if pd.isna(real_prod) or pd.isna(forecasted_prod) or pd.isna(id_price) or pd.isna(da_price) or pd.isna(smf_price):
            continue

        # DA gelirini yaz
        df.loc[i, "DA_revenue"] = da_price * forecasted_prod

        # Farkı bul
        delta = real_prod - forecasted_prod

        # Default values
        charge_amt = discharge_amt = 0
        id_sale_amt = id_buy_amt = ceza_amt = 0

        if delta > 0:
            # Fazla üretim varsa
            max_charge_capacity = soc_max - soc
            max_charge_power = power_limit
            charge_amt = min(delta, max_charge_capacity, max_charge_power)

            soc_temp = soc + charge_amt * eta
            soc_temp = min(soc_temp, soc_max)

            id_sale_amt = delta - charge_amt

            df.loc[i, "charge_used"] = charge_amt
            df.loc[i, "id_satis"] = id_sale_amt
            df.loc[i, "id_gelir"] = id_sale_amt * id_price
            df.loc[i, "batarya_cost"] = charge_amt * bat_cost_half

        elif delta < 0:
            # Üretim açığı varsa
            deficit = abs(delta)
            max_discharge_capacity = soc - soc_min
            max_discharge_power = power_limit
            discharge_amt = min(deficit, max_discharge_capacity, max_discharge_power)

            soc_temp = soc - discharge_amt / eta
            soc_temp = max(soc_temp, soc_min)

            residual_deficit = deficit - discharge_amt
            id_buy_amt = residual_deficit * id_share
            ceza_amt = residual_deficit * penalty_share
            ceza_fiyat = max(smf_price, da_price) * 1.03

            df.loc[i, "discharge_used"] = discharge_amt
            df.loc[i, "id_alim"] = id_buy_amt
            df.loc[i, "id_maliyet"] = id_buy_amt * id_price
            df.loc[i, "ceza_miktar"] = ceza_amt
            df.loc[i, "ceza_maliyet"] = ceza_amt * ceza_fiyat
            df.loc[i, "batarya_cost"] = discharge_amt * bat_cost_half

        # Saat sonundaki SOC bir sonraki saat için kullanılacak
        soc = soc_temp

        # Net kâr
        df.loc[i, "net_profit"] = (
            df.loc[i, "DA_revenue"] +
            df.loc[i, "id_gelir"] -
            df.loc[i, "id_maliyet"] -
            df.loc[i, "ceza_maliyet"]
        )

    return df


def show_hourly_decisions(df_hourly, df_benchmark, hours=24):
    """
    Compare decisions between your model and benchmark model for the first 'hours' hours.
    """
    # Ensure required columns exist for hourly model
    for col in ["charge_used", "discharge_used", "id_alim", "id_satis", "penalty", "battery_cost", "net_profit"]:
        if col not in df_hourly.columns:
            df_hourly[col] = 0.0

    # Ensure required columns exist for benchmark model
    for col in ["charge_used", "discharge_used", "id_alim", "id_satis", "ceza_maliyet", "batarya_cost", "net_profit"]:
        if col not in df_benchmark.columns:
            df_benchmark[col] = 0.0

    # Select relevant columns for comparison
    df_h = df_hourly[["datetime", "charge_used", "discharge_used", "id_alim", "id_satis", "penalty", "battery_cost",
                      "net_profit"]].copy()
    df_b = df_benchmark[
        ["datetime", "charge_used", "discharge_used", "id_alim", "id_satis", "ceza_maliyet", "batarya_cost",
         "net_profit"]].copy()

    # Rename columns for clarity
    df_h.columns = ["datetime", "bb_charge", "bb_discharge", "bb_id_buy", "bb_id_sell", "bb_penalty", "bb_bat_cost",
                    "bb_profit"]
    df_b.columns = ["datetime", "bm_charge", "bm_discharge", "bm_id_buy", "bm_id_sell", "bm_penalty", "bm_bat_cost",
                    "bm_profit"]

    # Merge öncesi timezone normalize et
    df_h["datetime"] = pd.to_datetime(df_h["datetime"]).dt.tz_localize(None)
    df_b["datetime"] = pd.to_datetime(df_b["datetime"]).dt.tz_localize(None)

    # Merge the dataframes
    df_merge = pd.merge(df_h, df_b, on="datetime", how="inner")

    # Set pandas options for better display
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", '{:.2f}'.format)

    print(f"Showing first {hours} hours of comparison:")
    print(df_merge.head(hours))


def test_benchmark_simulation():
    """
    Test the benchmark simulation with sample data
    """
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=48, freq='H')
    np.random.seed(42)

    df_test = pd.DataFrame({
        'datetime': dates,
        'real_production': np.random.normal(100, 20, 48),  # Real production with some variation
        'forecasted_production': np.random.normal(100, 15, 48),  # Forecasted production
        'ID_price_real': np.random.normal(50, 10, 48),  # ID market price
        'DA_price_real': np.random.normal(45, 8, 48),  # DA market price
        'SMF_real': np.random.normal(55, 12, 48)  # SMF price
    })

    # Battery parameters - Fixed to include power_limit
    battery_params = {
        'soc_target': 31,  # Starting SoC (MWh)
        'efficiency': 0.95,  # Round-trip efficiency
        'soc_max': 62,  # Maximum capacity (MWh)
        'soc_min': 10,  # Minimum capacity (MWh)
        'power_limit': 62  # Power limit (MW) - This is important!
    }

    # Run simulation
    result = simulate_benchmark_case1(df_test, battery_params)

    print("Sample results:")
    print(result[['datetime', 'real_production', 'forecasted_production', 'SoC', 'charge_used', 'discharge_used',
                  'id_satis', 'id_alim', 'net_profit']].head(10))

    return result


def validate_benchmark_simulation(df_result):
    """
    Validate the benchmark simulation results
    """
    print("Validation Results:")
    print(f"Total rows: {len(df_result)}")
    print(f"Non-zero charging events: {(df_result['charge_used'] > 0).sum()}")
    print(f"Non-zero discharging events: {(df_result['discharge_used'] > 0).sum()}")
    print(f"SoC range: {df_result['SoC'].min():.2f} - {df_result['SoC'].max():.2f}")
    print(f"Total net profit: {df_result['net_profit'].sum():.2f}")

    # Check for any violations
    if df_result['SoC'].min() < 0:
        print("WARNING: SoC went below minimum!")
    if df_result['SoC'].max() > 100:
        print("WARNING: SoC exceeded maximum!")

    return True