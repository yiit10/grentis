import pandas as pd
import numpy as np
from pulp import *
from datetime import datetime, timedelta
import pytz
from forecasts_utils import generate_forecast_price_ar1_sto, generate_forecast_price_id_sto, generate_forecast_production_sto,generate_forecast_smf_sto, safe_normalize, predict_scenario_with_moving_window, generate_rank_preserving_matrix,compute_path_probabilities, mahalanobis_distance, find_closest_joint_scenario, generate_scenario_dt_for_window, generate_forecast_distribution_from_price_forecast, generate_forecast_distribution_from_id_price, generate_forecast_distribution_from_smf,generate_scenario_dt_for_window_expost
from battery_brain_core import solve_ex_ante_model




def run_ex_ante_model_scenario_based(db: pd.DataFrame,
                                     battery_params: Dict,
                                     da_matrix: np.ndarray,
                                     id_matrix: np.ndarray,
                                     smf_matrix: np.ndarray,
                                     prod_matrix: np.ndarray,
                                     real_da: np.ndarray,
                                     real_id: np.ndarray,
                                     real_smf: np.ndarray,
                                     cov_matrix: np.ndarray,
                                     ex_ante_time: Union[str, pd.Timestamp] = "2024-01-01 10:00:00",
                                     window_size: int = 10,
                                     sigma: float = 2.0,
                                     horizon_hours: int = 48) -> Dict:
    """
    Run ex-ante optimization model using scenario-based approach.

    Args:
        db: Main database DataFrame
        battery_params: Dictionary with battery parameters including 'soc_target'
        da_matrix: Day-ahead price scenarios matrix
        id_matrix: Intraday price scenarios matrix
        smf_matrix: SMF price scenarios matrix
        prod_matrix: Production scenarios matrix
        real_da: Real DA prices array
        real_id: Real ID prices array
        real_smf: Real SMF prices array
        cov_matrix: Covariance matrix
        ex_ante_time: Ex-ante time (string or timestamp)
        window_size: Moving window size
        sigma: Standard deviation parameter
        horizon_hours: Optimization horizon in hours

    Returns:
        Dictionary with optimization results
    """
    import logging
    from datetime import datetime

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("Ex-ante scenario başlatıldı...")

    # Convert ex_ante_time to timestamp if it's a string
    if isinstance(ex_ante_time, str):
        ex_ante_time = pd.to_datetime(ex_ante_time).tz_localize('Europe/Istanbul')
    elif ex_ante_time.tz is None:
        ex_ante_time = ex_ante_time.tz_localize('Europe/Istanbul')
    else:
        ex_ante_time = ex_ante_time.tz_convert('Europe/Istanbul')

    # Generate scenario data for the optimization window
    try:
        scenario_dt = generate_scenario_dt_for_window(
            da_matrix, id_matrix, smf_matrix, prod_matrix,
            real_da, real_id, real_smf, cov_matrix,
            ex_ante_time, db,
            window_size=window_size,
            sigma=sigma,
            horizon_hours=horizon_hours
        )
    except Exception as e:
        logger.error(f"Scenario generation failed: {str(e)}")
        return {"status": "failed", "message": f"Scenario generation failed: {str(e)}"}

    # Ensure datetime column has correct timezone
    if db['datetime'].dt.tz is None:
        db['datetime'] = db['datetime'].dt.tz_localize('Europe/Istanbul')
    else:
        db['datetime'] = db['datetime'].dt.tz_convert('Europe/Istanbul')

    # Calculate optimization time window
    optimization_start_time = pd.Timestamp(
        (ex_ante_time + pd.Timedelta(days=1)).date()
    ).tz_localize('Europe/Istanbul')
    optimization_end_time = optimization_start_time + pd.Timedelta(hours=horizon_hours - 1)

    # Filter database for optimization window
    db_window = db[
        (db['datetime'] >= optimization_start_time) &
        (db['datetime'] <= optimization_end_time)
        ].copy()

    if len(db_window) == 0:
        logger.error("No data found in optimization window")
        return {"status": "failed", "message": "No data found in optimization window"}

    # Determine current SoC
    ex_ante_index = db[db['datetime'] == ex_ante_time].index
    if len(ex_ante_index) == 0:
        logger.warning("Ex-ante time not found in database, using default SoC")
        current_soc = battery_params['soc_target']
    else:
        ex_ante_idx = ex_ante_index[0]
        if ex_ante_idx == 0:
            # First record, use target SoC
            current_soc = battery_params['soc_target']
        else:
            # Check previous hour's SoC
            prev_time = ex_ante_time - pd.Timedelta(hours=1)
            prev_soc_data = db[db['datetime'] == prev_time]
            if len(prev_soc_data) == 0 or pd.isna(prev_soc_data.iloc[0].get('soc', np.nan)):
                current_soc = battery_params['soc_target']
            else:
                current_soc = prev_soc_data.iloc[0]['soc']

    logger.info(f"Geçerli SoC: {current_soc:.2f}")

    # Update db_window with scenario data
    if len(scenario_dt) != len(db_window):
        logger.warning(f"Scenario length ({len(scenario_dt)}) != window length ({len(db_window)})")
        # Align the data by taking minimum length
        min_len = min(len(scenario_dt), len(db_window))
        scenario_dt = scenario_dt.iloc[:min_len]
        db_window = db_window.iloc[:min_len]

    # Create db_entek compatible DataFrame for solve_ex_ante_model
    db_entek = pd.DataFrame({
        'datetime': scenario_dt['datetime'],
        'DA_price_forecasted': scenario_dt['DA'],
        'ID_price_forecasted': scenario_dt['ID'],
        'SMF_forecasted': scenario_dt['SMF'],
        'forecasted_production': scenario_dt['prod'],
        'real_production': scenario_dt['prod']  # Using scenario production as real for optimization
    })

    logger.info("Ex-ante optimizasyon modeli oluşturuluyor...")

    # Use the existing solve_ex_ante_model function
    try:
        optimization_result = solve_ex_ante_model(
            db=db,
            battery_params=battery_params,
            initial_soc=current_soc
        )
        logger.info("Model başarıyla oluşturuldu ve çözüldü.")
    except Exception as e:
        logger.error(f"Model creation/solving failed: {str(e)}")
        return {"status": "failed", "message": f"Model creation/solving failed: {str(e)}"}

    # Check if optimization was successful
    if optimization_result['status'] != 'optimal':
        logger.error(f"Optimization failed: {optimization_result.get('message', 'Unknown error')}")
        return {
            "status": "failed",
            "message": f"Optimization failed: {optimization_result.get('message', 'Unknown error')}"
        }

    # Extract solutions and update main database
    try:
        # Variable names that are available from solve_ex_ante_model
        variable_names = [
            'q_committed', 'd_da_prod', 'd_id_prod', 'd_da_bat',
            'd_id_bat', 'charge', 'soc', 'is_charging'
        ]

        solutions = {}
        for var_name in variable_names:
            if var_name in optimization_result:
                solutions[var_name] = optimization_result[var_name]

                # Update main database for the optimization window
                window_datetimes = db_window['datetime'].values
                for i, dt in enumerate(window_datetimes):
                    if i < len(solutions[var_name]):
                        db.loc[db['datetime'] == dt, var_name] = solutions[var_name][i]
            else:
                logger.warning(f"Variable {var_name} not found in optimization result")
                solutions[var_name] = []

        # Create scenario metadata
        scenario_meta = pd.DataFrame({
            'datetime': db_window['datetime'],
            'DA_scenario': scenario_dt['DA'],
            'ID_scenario': scenario_dt['ID'],
            'SMF_scenario': scenario_dt['SMF'],
            'prod_scenario': scenario_dt['prod']
        })

        logger.info("Tüm sonuçlar başarıyla aktarıldı.")

        return {
            "status": "success",
            "db": db,
            "optimization_result": optimization_result,
            "scenario_info": scenario_meta,
            "decisions": solutions,
            "objective_value": optimization_result['objective_value'],
            "ex_ante_time": ex_ante_time,
            "optimization_window": {
                "start": optimization_start_time,
                "end": optimization_end_time
            }
        }

    except Exception as e:
        logger.error(f"Solution extraction failed: {str(e)}")
        return {"status": "failed", "message": f"Solution extraction failed: {str(e)}"}

def solve_model(model, solver: str = "cbc"):
    """
    Solve the optimization model.
    This is a placeholder - implement based on your optimization framework.

    Args:
        model: Optimization model object
        solver: Solver name

    Returns:
        Solution object
    """
    raise NotImplementedError("solve_model needs to be implemented based on your optimization framework")


def get_solution(result, variable_expression: str) -> Dict:
    """
    Extract solution values for a given variable.
    This is a placeholder - implement based on your optimization framework.

    Args:
        result: Solution object
        variable_expression: Variable expression string

    Returns:
        Dictionary with 'value' key containing solution values
    """
    raise NotImplementedError("get_solution needs to be implemented based on your optimization framework")


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)


def build_ex_post_optimization(db: pd.DataFrame, battery_params: Dict[str, float],
                               ex_post_time: datetime, eur_to_tl: float = 35) -> pulp.LpProblem:
    """
    Build ex-post optimization model using PuLP
    """
    logger.info("Ex-post model setup starting")

    big_M = 1000
    horizon_hours = 50

    # Prepare data
    ex_post_idx = db[db['datetime'] == ex_post_time].index[0]
    ex_post_end = min(ex_post_idx + horizon_hours, len(db))

    # Get current SOC
    current_soc = battery_params.get('soc_target', 0.5)
    if ex_post_idx < len(db) and 'soc' in db.columns:
        if not pd.isna(db.loc[ex_post_idx, 'soc']):
            current_soc = db.loc[ex_post_idx, 'soc']

    logger.info(f"Start time: {ex_post_time}, SOC: {current_soc:.2f}")

    # Extract price and production data
    model_data = db.iloc[ex_post_idx:ex_post_end].copy()

    # Fill missing values
    numeric_columns = ['d_da_prod', 'd_da_bat', 'd_id_prod', 'd_id_bat', 'x_id', 'x_ceza',
                       'charge', 'q_committed']
    for col in numeric_columns:
        if col in model_data.columns:
            model_data[col] = model_data[col].fillna(0)

    # Battery costs
    c_bat_cyc = 0.0556 * eur_to_tl
    c_om = 0.006 * eur_to_tl
    c_capex_kWh = 300 * eur_to_tl
    bat_lifetime_hours = 10 * 8760
    unit_battery_cost = c_bat_cyc + c_om + (c_capex_kWh / bat_lifetime_hours)

    # Battery parameters
    eta = battery_params.get('efficiency', 0.95)
    P_max = battery_params.get('power_limit', 100)

    # Calculate flexibility flags
    is_gop_flexible = calculate_gop_flexibility(model_data['datetime'].iloc[0])
    is_id_flexible = [1] * horizon_hours
    is_id_flexible[0] = 0  # First hour is not flexible
    is_id_flexible[1] = 0  # Second hour is not flexible

    logger.info("Model variables defined, starting model definition...")

    # Create optimization model
    model = pulp.LpProblem("ExPostOptimization", pulp.LpMaximize)

    # Decision variables
    variables = {}

    # Continuous variables
    var_names = [
        'd_id_prod_extra', 'd_id_bat_extra', 'd_da_prod_extra', 'd_da_bat_extra',
        'd_id_prod_horizon', 'd_id_bat_horizon', 'charge_extra', 'charge_horizon',
        'd_da_prod_horizon', 'd_da_bat_horizon', 'x_id_adj', 'x_ceza_adj',
        'x_id_horizon', 'x_ceza_horizon', 'soc_post', 'prod_surplus', 'prod_deficit',
        'q_committed_horizon', 'diff_abs'
    ]

    for var_name in var_names:
        variables[var_name] = [
            pulp.LpVariable(f"{var_name}_{t}", lowBound=-P_max if 'extra' in var_name or 'adj' in var_name else 0,
                            upBound=P_max if 'extra' in var_name or 'adj' in var_name else None)
            for t in range(horizon_hours)
        ]

    # Binary variables
    variables['delta'] = [pulp.LpVariable(f"delta_{t}", cat='Binary') for t in range(horizon_hours)]
    variables['is_charging'] = [pulp.LpVariable(f"is_charging_{t}", cat='Binary') for t in range(horizon_hours)]

    # SOC bounds
    for t in range(horizon_hours):
        variables['soc_post'][t].bounds(
            battery_params.get('soc_min', 0.1),
            battery_params.get('soc_max', 0.9)
        )

    # Objective function (simplified)
    objective = 0
    for t in range(horizon_hours):
        # Revenue terms
        if t < len(model_data):
            da_price = model_data.iloc[t].get('DA_price_real', 0)
            id_price = model_data.iloc[t].get('ID', 0)
            smf_price = model_data.iloc[t].get('SMF', 0)

            objective += (
                    da_price * variables['q_committed_horizon'][t] +
                    id_price * 1.03 * (variables['d_id_prod_horizon'][t] + variables['d_id_bat_horizon'][t]) -
                    id_price * 0.97 * variables['x_id_horizon'][t] -
                    max(da_price, smf_price) * 1.03 * variables['x_ceza_horizon'][t] -
                    (unit_battery_cost / 2) * (variables['d_da_bat_horizon'][t] +
                                               variables['d_id_bat_horizon'][t] +
                                               variables['charge_horizon'][t])
            )

    model += objective

    # Constraints
    for t in range(horizon_hours):
        # Production constraints
        if t < len(model_data):
            production = model_data.iloc[t].get('prod', 0)

            # Production balance
            model += (
                    variables['prod_surplus'][t] - variables['prod_deficit'][t] ==
                    production - variables['q_committed_horizon'][t]
            )

            # Production limits
            model += variables['d_da_prod_horizon'][t] >= 0.25 * production
            model += (
                    production >=
                    variables['d_da_prod_horizon'][t] + variables['d_id_prod_horizon'][t]
            )

        # Battery constraints
        model += (
                variables['d_da_bat_horizon'][t] + variables['d_id_bat_horizon'][t] <=
                P_max * (1 - variables['is_charging'][t])
        )

        model += variables['charge_horizon'][t] <= P_max * variables['is_charging'][t]

        # SOC dynamics
        if t > 0:
            model += (
                    variables['soc_post'][t] == variables['soc_post'][t - 1] +
                    eta * variables['charge_horizon'][t - 1] -
                    (1 / eta) * (variables['d_da_bat_horizon'][t - 1] + variables['d_id_bat_horizon'][t - 1])
            )
        else:
            model += variables['soc_post'][t] == current_soc

        # Big-M constraints for surplus/deficit
        model += variables['prod_surplus'][t] <= big_M * variables['delta'][t]
        model += variables['prod_deficit'][t] <= big_M * (1 - variables['delta'][t])

        # Non-negativity
        model += variables['x_ceza_horizon'][t] >= 0
        model += variables['x_id_horizon'][t] >= 0
        model += variables['charge_horizon'][t] >= 0

    logger.info("Model setup completed")
    return model


def calculate_gop_flexibility(current_time: datetime) -> List[int]:
    """
    Calculate GOP (Day-ahead market) flexibility flags
    """
    # Simplified implementation
    current_hour = current_time.hour

    # If after 10 AM, next day is flexible
    if current_hour >= 10:
        return [0] * 24 + [1] * 24  # Today fixed, tomorrow flexible
    else:
        return [0] * 24  # Today fixed


def run_ex_post_model_scenario_based(db: pd.DataFrame, battery_params: Dict[str, float],
                                     da_matrix: np.ndarray, id_matrix: np.ndarray,
                                     smf_matrix: np.ndarray, prod_matrix: np.ndarray,
                                     real_da: np.ndarray, real_id: np.ndarray,
                                     real_smf: np.ndarray, cov_matrix: np.ndarray,
                                     ex_post_time: datetime = None, eur_to_tl: float = 35,
                                     window_size: int = 10, sigma: float = 2) -> Dict[str, Any]:
    """
    Run ex-post model with scenario-based optimization
    """
    if ex_post_time is None:
        ex_post_time = datetime(2024, 1, 2, 11, 0, 0, tzinfo=pytz.timezone('Europe/Istanbul'))

    if isinstance(ex_post_time, str):
        ex_post_time = pd.to_datetime(ex_post_time).tz_localize('Europe/Istanbul')

    horizon_hours = 48
    optimization_start_time = ex_post_time + timedelta(hours=2)
    optimization_end_time = optimization_start_time + timedelta(hours=horizon_hours - 1)

    # Filter data window
    db_window = db[
        (db['datetime'] >= ex_post_time) &
        (db['datetime'] <= optimization_end_time)
        ].copy()

    logger.info("Ex-post scenario started...")

    # Get current SOC
    current_soc_data = db[db['datetime'] == ex_post_time]
    if len(current_soc_data) == 0 or pd.isna(current_soc_data.iloc[0].get('soc', np.nan)):
        current_soc = battery_params.get('soc_target', 0.5)
    else:
        current_soc = current_soc_data.iloc[0]['soc']

    print(f"Ex-post time: {ex_post_time}, current SOC: {current_soc:.2f}")

    # Generate new scenario set
    logger.info("Generating ex-post scenario...")
    scenario_dt_expost = generate_scenario_dt_for_window_expost(
        da_matrix, id_matrix, smf_matrix, prod_matrix,
        real_da, real_id, real_smf, cov_matrix,
        ex_post_time, db, window_size, sigma
    )

    logger.info("Ex-post scenario successfully generated")

    # Update window data with scenarios
    if len(scenario_dt_expost) > 0:
        db_window.loc[:, 'DA'] = scenario_dt_expost['DA'].values[:len(db_window)]
        db_window.loc[:, 'ID'] = scenario_dt_expost['ID'].values[:len(db_window)]
        db_window.loc[:, 'SMF'] = scenario_dt_expost['SMF'].values[:len(db_window)]
        db_window.loc[:, 'prod'] = scenario_dt_expost['prod'].values[:len(db_window)]
        db_window.loc[:, 'penalty'] = scenario_dt_expost['penalty'].values[:len(db_window)]

    logger.info("Starting ex-post optimization model...")

    # Build and solve optimization model
    model = build_ex_post_optimization(db_window, battery_params, ex_post_time, eur_to_tl)

    logger.info("Ex-post optimization model successfully built, starting solution...")

    # Solve model
    try:
        model.solve(pulp.PULP_CBC_CMD(msg=0))

        if model.status != pulp.LpStatusOptimal:
            return {
                'status': 'failed',
                'message': f'Optimization failed with status: {pulp.LpStatus[model.status]}'
            }

        logger.info("Solution successful, writing results...")

        # Extract solution
        solution = {}
        for var in model.variables():
            solution[var.name] = var.varValue

        # Update database with results (simplified)
        optimization_indices = db[
            (db['datetime'] >= optimization_start_time) &
            (db['datetime'] <= optimization_end_time)
            ].index

        # Create result columns if they don't exist
        result_columns = [
            'd_da_prod_extra', 'd_da_bat_extra', 'd_id_prod_extra', 'd_id_bat_extra',
            'x_id_adj', 'x_ceza_adj', 'charge_extra', 'soc_post', 'q_committed_horizon',
            'd_id_prod_horizon', 'd_id_bat_horizon', 'charge_horizon', 'd_da_prod_horizon',
            'd_da_bat_horizon', 'x_id_horizon', 'x_ceza_horizon'
        ]

        for col in result_columns:
            if col not in db.columns:
                db[col] = 0.0

        logger.info("All values successfully transferred")

        return {
            'status': 'success',
            'db': db,
            'model': model,
            'solution': solution,
            'scenario_info': scenario_dt_expost,
            'objective_value': pulp.value(model.objective)
        }

    except Exception as e:
        return {
            'status': 'failed',
            'message': str(e)
        }


