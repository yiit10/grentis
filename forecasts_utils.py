import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import List, Tuple, Dict, Optional, Union
from datetime import datetime, timedelta
from scipy import stats
from pytz import timezone
import pytz
from battery_brain_core import solve_ex_ante_model
import pulp
from scipy.stats import norm
import logging
from typing import Dict

def generate_forecast_price_ar1(df_real_prices, phi=0.6, error_sd=0.15, bias=0, seed=42, price_ceiling=3400):
    np.random.seed(seed)
    df = df_real_prices.copy()
    df.rename(columns={"price": "price_real"}, inplace=True)  # 🔧 Bu daha başta yapılmalı
    df["datetime"] = pd.to_datetime(df["date"])
    df = df.sort_values("datetime").reset_index(drop=True)

    n = len(df)
    df["error"] = np.zeros(n)
    df.loc[0, "error"] = np.random.normal(bias, error_sd)

    for i in range(1, n):
        df.loc[i, "error"] = phi * df.loc[i - 1, "error"] + np.random.normal(bias, error_sd)

    df["price_forecast"] = df["price_real"] * (1 + df["error"])
    df["price_forecast"] = df["price_forecast"].clip(lower=0, upper=price_ceiling)

    return df[["datetime", "price_real", "price_forecast", "error"]]

def calculate_forecast_metrics(df):
    df['abs_error'] = abs(df['price_forecast'] - df['price_real'])
    df['rel_error'] = df['abs_error'] / df['price_real']
    mae = mean_absolute_error(df['price_real'], df['price_forecast'])
    rmse = mean_squared_error(df["price_real"], df["price_forecast"]) ** 0.5
    mape = df['rel_error'].mean() * 100
    return {"mae": mae, "rmse": rmse, "mape": mape}

def calculate_smf_metrics(df):
    df = df.copy()
    df['abs_error'] = abs(df['smf_forecast'] - df['systemMarginalPrice'])
    df['rel_error'] = df['abs_error'] / df['systemMarginalPrice'].replace(0, np.nan)

    mae = mean_absolute_error(df['systemMarginalPrice'], df['smf_forecast'])
    rmse = mean_squared_error(df['systemMarginalPrice'], df['smf_forecast']) ** 0.5
    mape = df['rel_error'].mean() * 100

    return {"mae": mae, "rmse": rmse, "mape": mape}

def generate_forecast_price_id(df_real_prices, phi=0.65, error_sd=0.15, bias=0, seed=42, price_ceiling=3400):
    np.random.seed(seed)
    df = df_real_prices.copy()
    df.rename(columns={"wap": "price_real"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["date"])
    df = df.sort_values("datetime").reset_index(drop=True)

    n = len(df)
    df["error"] = np.zeros(n)
    df.loc[0, "error"] = np.random.normal(bias, error_sd)

    for i in range(1, n):
        df.loc[i, "error"] = phi * df.loc[i - 1, "error"] + np.random.normal(bias, error_sd)

    df["price_forecast"] = df["price_real"] * (1 + df["error"])
    df["price_forecast"] = df["price_forecast"].clip(lower=0, upper=price_ceiling)

    return df[["datetime", "price_real", "price_forecast", "error"]]


def generate_forecast_production(df, error_sd=0.05, seed=42, power_limit=None):
    """
    Verilen gerçek üretim verisi üzerinden AR(1)-bazlı üretim tahmini üretir.
    error_sd: 0.05 için Meteomatics, 0.15 için Meteologica gibi yorumlanabilir.
    """

    np.random.seed(seed)
    df = df.copy()

    # Zaman formatı
    df["datetime"] = pd.to_datetime(df["date"])
    df = df.sort_values("datetime")

    # Üretim tipi (tahmini sınıflandırma)
    prod_type = "RES"  # Mersin RES özelinde sabit; ileride sınıflandırılabilir

    # AR(1) hata terimi üretimi
    n = df.shape[0]
    phi_values = {"RES": 0.85, "thermal": 0.65, "hydro": 0.75}
    phi = phi_values.get(prod_type, 0.7)

    error = np.zeros(n)
    error[0] = np.random.normal(loc=0, scale=error_sd)

    for i in range(1, n):
        error[i] = phi * error[i - 1] + np.random.normal(loc=0, scale=error_sd)

    # Forecast = Real * (1 + error)
    forecast = df["injectionQuantity"].values * (1 + error)
    forecast[forecast < 0] = 0

    if power_limit is not None:
        forecast = np.minimum(forecast, power_limit)

    forecast_df = pd.DataFrame({
        "datetime": df["datetime"],
        "forecast_total": forecast,
        "prod_type": prod_type,
        "powerPlantId": df["powerPlantId"].values
    })

    return forecast_df

def generate_forecast_smf(real_smf_df, phi=0.6, error_sd=0.05, bias=0, seed=42, price_ceiling=3400):
    np.random.seed(seed)

    df = real_smf_df.copy()
    df['datetime'] = pd.to_datetime(df['date'])  # Zaman dilimini otomatik tanır
    df = df.sort_values('datetime').reset_index(drop=True)

    n = len(df)
    errors = np.zeros(n)
    errors[0] = np.random.normal(loc=bias, scale=error_sd)

    for i in range(1, n):
        errors[i] = phi * errors[i - 1] + np.random.normal(loc=bias, scale=error_sd)

    df['error'] = errors
    df['smf_real'] = df['systemMarginalPrice']
    df['smf_forecast'] = df['smf_real'] * (1 + df['error'])

    # Gerçekleşen 0 ise tahmin de 0
    df.loc[df['smf_real'] == 0, 'smf_forecast'] = 0

    # Boundaries (0 ile price_ceiling arasında tut)
    df['smf_forecast'] = df['smf_forecast'].clip(lower=0, upper=price_ceiling)

    return df[['datetime', 'smf_real', 'smf_forecast', 'error']]


def generate_forecast_price_ar1_sto(df_real_prices, virtual_now, phi=0.6,
                                     error_sd_min=0.05, error_sd_max=0.25,
                                     error_sd_past=0.05, bias=0, seed=42,
                                     price_ceiling=3400):
    np.random.seed(seed)
    df = df_real_prices.copy()
    df.rename(columns={"price": "price_real"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["date"])
    df = df.sort_values("datetime").reset_index(drop=True)

    forecast = []
    errors = []
    prev_error = np.random.normal(loc=bias, scale=error_sd_max)

    for i in range(len(df)):
        dt = df.loc[i, "datetime"]

        if dt < virtual_now:
            error_sd = error_sd_past
        else:
            rel_horizon = min(1, max(0, (dt - virtual_now).total_seconds() / 3600 / 48))
            error_sd = error_sd_min + (1 - rel_horizon) * (error_sd_max - error_sd_min)

        curr_error = phi * prev_error + np.random.normal(loc=bias, scale=error_sd)
        forecast_val = df.loc[i, "price_real"] * (1 + curr_error)
        forecast.append(min(max(forecast_val, 0), price_ceiling))
        errors.append(curr_error)
        prev_error = curr_error

    df["error"] = errors
    df["price_forecast"] = forecast
    return df[["datetime", "price_real", "price_forecast", "error"]]

def generate_forecast_price_id_sto(df_real_prices, virtual_now, phi=0.65,
                                    error_sd_min=0.10, error_sd_max=0.30,
                                    error_sd_past=0.10, bias=0, seed=42,
                                    price_ceiling=3400):
    np.random.seed(seed)
    df = df_real_prices.copy()
    df.rename(columns={"wap": "price_real"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["date"])
    df = df.sort_values("datetime").reset_index(drop=True)

    forecast = []
    errors = []
    prev_error = np.random.normal(loc=bias, scale=error_sd_max)

    for i in range(len(df)):
        dt = df.loc[i, "datetime"]

        if dt < virtual_now:
            error_sd = error_sd_past
        else:
            rel_horizon = min(1, max(0, (dt - virtual_now).total_seconds() / 3600 / 48))
            error_sd = error_sd_min + (1 - rel_horizon) * (error_sd_max - error_sd_min)

        curr_error = phi * prev_error + np.random.normal(loc=bias, scale=error_sd)
        forecast_val = df.loc[i, "price_real"] * (1 + curr_error)
        forecast.append(min(max(forecast_val, 0), price_ceiling))
        errors.append(curr_error)
        prev_error = curr_error

    df["error"] = errors
    df["price_forecast"] = forecast
    return df[["datetime", "price_real", "price_forecast", "error"]]

def generate_forecast_production_sto(df, virtual_now, error_sd_min=0.03, error_sd_max=0.10, error_sd_past=0.03, seed=42, power_limit=None):
    np.random.seed(seed)
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["date"])
    df = df.sort_values("datetime")
    prod_type = "RES"

    n = df.shape[0]
    phi_values = {"RES": 0.85, "thermal": 0.65, "hydro": 0.75}
    phi = phi_values.get(prod_type, 0.7)

    forecast = []
    errors = []
    prev_error = np.random.normal(loc=0, scale=error_sd_max)

    for i in range(n):
        dt = df.loc[i, "datetime"]

        if dt < virtual_now:
            # Geçmiş saat: vendor o an tahmin etmiş ve sabit bir hata ile
            error_sd = error_sd_past
        else:
            # Gelecek saat: ufka göre değişen hata (yakın saatlerde hata büyük, uzaklarda daha küçük)
            rel_horizon = min(1, max(0, (dt - virtual_now).total_seconds() / 3600 / 48))
            error_sd = error_sd_min + (1 - rel_horizon) * (error_sd_max - error_sd_min)

        raw_val = df.loc[i, "injectionQuantity"]
        if pd.isna(raw_val):
            raw_val = 0

        curr_error = phi * prev_error + np.random.normal(loc=0, scale=error_sd)
        val = raw_val * (1 + curr_error)

        capped_val = max(val, 0)
        if power_limit is not None:
            capped_val = min(capped_val, power_limit)

        forecast.append(capped_val)
        errors.append(curr_error)
        prev_error = curr_error

    forecast_df = pd.DataFrame({
        "datetime": df["datetime"],
        "forecast_total": forecast,
        "prod_type": prod_type,
        "powerPlantId": df["powerPlantId"].values,
        "error": errors
    })

    return forecast_df


def generate_forecast_smf_sto(real_smf_df, virtual_now, phi=0.6, error_sd_min=0.03, error_sd_max=0.15, bias=0, seed=42, price_ceiling=3400):
    np.random.seed(seed)
    df = real_smf_df.copy()
    df['datetime'] = pd.to_datetime(df['date'])
    df = df.sort_values('datetime').reset_index(drop=True)

    forecast = []
    errors = []
    prev_error = np.random.normal(loc=bias, scale=error_sd_max)

    for i in range(len(df)):
        dt = df.loc[i, 'datetime']
        rel_horizon = min(1, max(0, (dt - virtual_now).total_seconds() / 3600 / 48))
        error_sd = error_sd_min + (1 - rel_horizon) * (error_sd_max - error_sd_min)

        curr_error = phi * prev_error + np.random.normal(loc=bias, scale=error_sd)
        val = df.loc[i, 'systemMarginalPrice'] * (1 + curr_error)
        forecast.append(min(max(val, 0), price_ceiling))
        errors.append(curr_error)
        prev_error = curr_error

    df['error'] = errors
    df['smf_real'] = df['systemMarginalPrice']
    df['smf_forecast'] = forecast
    df.loc[df['smf_real'] == 0, 'smf_forecast'] = 0

    return df[['datetime', 'smf_real', 'smf_forecast', 'error']]

# Flex module



def safe_normalize(x: np.ndarray, threshold: float = 1e-8) -> np.ndarray:
    """
    Safely normalize an array to sum to 1, handling NaN values and zero sums.

    Args:
        x: Input array to normalize
        threshold: Minimum sum threshold to avoid division by zero

    Returns:
        Normalized array
    """
    x = np.nan_to_num(x, nan=0.0)
    total = np.sum(x)
    if total < threshold:
        return np.ones(len(x)) / len(x)
    return x / total


def predict_scenario_with_moving_window(
    prod_matrix: np.ndarray,
    real_prod: np.ndarray,
    datetime_vec: Union[pd.Series, pd.DatetimeIndex],
    t_now: Union[int, pd.Timestamp], # t_now'ı varsayılan değeri olmayan zorunlu parametre olarak öne aldık
    window_size: int = 10, # Varsayılan değeri olanlar sonra
    sigma: float = 2.0
) -> np.ndarray:

    """
    Predict scenario probabilities using a moving window approach.
    R function: predict_scenario_with_moving_window
    """
    # t_index belirleme (R'daki if (is.numeric(t_now)) { t_now } else { which(datetime_vec == t_now) } karşılığı)
    if isinstance(t_now, int):
        t_index = t_now
    elif isinstance(t_now, pd.Timestamp):
        # R'daki which(datetime_vec == t_now) karşılığı
        # Pandas'ta bu işlemi en verimli şekilde yapmak için datetime_vec'in Series veya DatetimeIndex olması gerekir.
        # Eğer datetime_vec bir Series ise, direkt karşılaştırma yaparız ve indexini alırız.
        # Eğer datetime_vec bir DatetimeIndex ise, get_loc kullanırız.
        if isinstance(datetime_vec, pd.Series):
            # Önce timezone dönüşümünü yapalım ki eşleşme sağlanabilsin
            if datetime_vec.dtype.tz is None and t_now.tzinfo is not None:
                datetime_vec = datetime_vec.dt.tz_localize(t_now.tzinfo)
            elif datetime_vec.dtype.tz is not None and t_now.tzinfo is None:
                t_now = t_now.tz_localize(datetime_vec.dtype.tz)
            elif datetime_vec.dtype.tz is not None and t_now.tzinfo is not None and datetime_vec.dtype.tz != t_now.tzinfo:
                datetime_vec = datetime_vec.dt.tz_convert(t_now.tzinfo)

            try:
                # Eşleşen Timestamp'in indeksini bul
                matched_indices = datetime_vec[datetime_vec == t_now].index
                if len(matched_indices) == 0:
                    raise ValueError(f"t_now ({t_now}) not found in datetime_vec.")
                # İlk eşleşen indeksi al
                t_index = matched_indices[0]
            except Exception as e:
                raise ValueError(f"Error finding t_now in datetime_vec (Series): {e}")

        elif isinstance(datetime_vec, pd.DatetimeIndex):
            # Timezone uyumunu sağla
            if datetime_vec.tz is None and t_now.tzinfo is not None:
                datetime_vec = datetime_vec.tz_localize(t_now.tzinfo)
            elif datetime_vec.tz is not None and t_now.tzinfo is None:
                t_now = t_now.tz_localize(datetime_vec.tz)
            elif datetime_vec.tz is not None and t_now.tzinfo is not None and datetime_vec.tz != t_now.tzinfo:
                datetime_vec = datetime_vec.tz_convert(t_now.tzinfo)

            try:
                t_index = datetime_vec.get_loc(t_now)
                # get_loc bazen slice veya array döndürebilir, tek bir int'e dönüştür
                if isinstance(t_index, slice):
                    t_index = t_index.start
                elif isinstance(t_index, (np.ndarray, list)):
                    t_index = t_index[0]
            except KeyError:
                raise ValueError(f"t_now ({t_now}) not found in datetime_vec.")
            except Exception as e:
                raise ValueError(f"Error finding t_now in datetime_vec (DatetimeIndex): {e}")
        else:
            raise TypeError("datetime_vec must be a pandas Series or DatetimeIndex.")
    else:
        raise TypeError("t_now must be an integer or a pandas Timestamp.")

    # R'daki t_index, 1 tabanlıdır. Python'da 0 tabanlı olduğu için düzeltme yapacağız.
    # Eğer t_now direkt int geliyorsa, R'daki 1 tabanlı indekse göre geliyordur.
    # Python listeleri/array'leri 0 tabanlı olduğu için t_index'ten 1 çıkaracağız.
    # Ancak t_index'i datetime_vec'ten alıyorsak, o zaten 0 tabanlıdır.
    # Bu yüzden burası karmaşıklaşıyor. En temizi, t_now int geldiğinde bunun 0-tabanlı Python indeksi olduğunu varsayalım.
    # R'daki t_index <- if (is.numeric(t_now)) { t_now } kısmında, eğer t_now 1 ise, prod_matrix[1,] oluyor.
    # Python'da prod_matrix[0,] olmalı.
    # Dolayısıyla, t_index'i her zaman 0 tabanlı olarak kullanmalıyız.
    # Eğer t_now numeric ise ve 1 tabanlı geliyorsa, burada t_index = t_now - 1 yapmalıyız.
    # Ancak, senin çağrı şekline göre (df_full["DA_price_real"].to_numpy()), NumPy array'leri 0-tabanlıdır.
    # Bu yüzden, eğer t_now Timestamp'ten dönüştürülüyorsa, zaten 0-tabanlı indeks geliyor.
    # Eğer t_now direkt bir int ise, o zaman bunun da 0-tabanlı bir indeks olması en tutarlısı.
    # R'daki t_index - 1 kullanımı, effective_window ve slicing için zaten 1 tabanlı indeksten Python'daki 0 tabanlıya geçişi sağlıyor.
    # Yani R'daki t_index (1-tabanlı) -> Python'da kullanılacak indis (0-tabanlı) = t_index - 1
    python_t_index = t_index - 1  # R'daki 1-tabanlı indeksten 0-tabanlıya geçiş

    if python_t_index < 0:  # Eğer t_index 1 ise, python_t_index 0 olur. Eğer 0 ise, -1 olur.
        raise ValueError(
            "Calculated index (t_index - 1) is negative. Ensure t_now corresponds to a valid 1-based index >= 1.")

    T_rows, S_cols = prod_matrix.shape  # R'daki nrow, ncol karşılığı

    # effective_window hesaplaması (R'daki min(window_size, t_index - 1) karşılığı)
    # Burada t_index'in R'daki gibi 1 tabanlı olduğunu varsayıyoruz.
    effective_window = min(window_size, t_index - 1)
    if effective_window < 1:
        raise ValueError("Not enough history to compute scenario probabilities.")  # R'daki stop() karşılığı

    # Array dilimlemelerinde R ve Python farklıdır. R'da 1-tabanlı ve dahil edicidir.
    # Python'da 0-tabanlı ve bitiş noktası dahil değildir.

    # real_now ve scenario_now (R'daki real_prod[t_index] ve prod_matrix[t_index,] karşılığı)
    real_now = real_prod[python_t_index]
    scenario_now = prod_matrix[python_t_index, :]

    # if/else if blokları
    if real_now < np.min(scenario_now):
        likelihoods = np.zeros(S_cols)
        likelihoods[0] = 1.0
        return likelihoods
    elif real_now > np.max(scenario_now):
        likelihoods = np.zeros(S_cols)
        likelihoods[S_cols - 1] = 1.0  # R'daki S.cols karşılığı S_cols - 1
        return likelihoods

    # recent_real ve recent_prod (R'daki (t_index - effective_window):(t_index - 1) karşılığı)
    # R'da: (1-tabanlı başlangıç: 1-tabanlı bitiş)
    # Python'da: [0-tabanlı başlangıç : 0-tabanlı bitiş (hariç)]

    # R'daki (t_index - effective_window) 1-tabanlı başlangıç
    # Python'daki start_index = (t_index - effective_window) - 1
    python_start_idx = (t_index - effective_window) - 1

    # R'daki (t_index - 1) 1-tabanlı bitiş
    # Python'daki end_idx = (t_index - 1)
    python_end_idx = t_index - 1  # Dilimleme bitişi (hariç olduğu için t_index - 1)

    recent_real = real_prod[python_start_idx: python_end_idx]
    recent_prod = prod_matrix[python_start_idx: python_end_idx, :]

    # scenario_scores (R'daki sapply karşılığı)
    scenario_scores = np.zeros(S_cols)
    for s in range(S_cols):  # Python'da döngü 0'dan başlar
        diffs = recent_real - recent_prod[:, s]
        scenario_scores[s] = np.sum(diffs ** 2)

    # Geri kalan hesaplamalar
    match_weights = safe_normalize(np.exp(-scenario_scores / (2 * sigma ** 2)))
    diffs_now = (real_now - scenario_now) ** 2
    likelihoods = safe_normalize(np.exp(-diffs_now / (2 * sigma ** 2)))

    combined_probs = safe_normalize(match_weights * likelihoods)

    return combined_probs

def generate_rank_preserving_matrix(forecast_dist: pd.DataFrame,
                                    datetime_vec: pd.DatetimeIndex,
                                    n_scenarios: int = 100) -> np.ndarray:
    """
    Generate a rank-preserving scenario matrix from forecast distribution.

    Args:
        forecast_dist: DataFrame with columns ['datetime', 'value', 'prob']
        datetime_vec: Vector of datetime values
        n_scenarios: Number of scenarios to generate

    Returns:
        Matrix of scenarios (len(datetime_vec) x n_scenarios)
    """
    # Ensure timezone consistency
    forecast_dist = forecast_dist.copy()
    if forecast_dist['datetime'].dt.tz is None:
        forecast_dist['datetime'] = forecast_dist['datetime'].dt.tz_localize('Europe/Istanbul')
    else:
        forecast_dist['datetime'] = forecast_dist['datetime'].dt.tz_convert('Europe/Istanbul')

    if datetime_vec.tz is None:
        datetime_vec = datetime_vec.tz_localize('Europe/Istanbul')
    else:
        datetime_vec = datetime_vec.tz_convert('Europe/Istanbul')

    # Generate uniform quantiles (excluding 0 and 1)
    u_vec = np.linspace(0, 1, n_scenarios + 2)[1:-1]

    # Initialize result matrix
    result_matrix = np.zeros((len(datetime_vec), n_scenarios))

    for t, dt in enumerate(datetime_vec):
        # Get distribution for this time
        dist_t = forecast_dist[forecast_dist['datetime'] == dt].copy()

        if len(dist_t) == 0:
            # If no data available, use zeros or interpolate
            continue

        # Sort by value to ensure proper CDF
        dist_t = dist_t.sort_values('value')

        # Calculate CDF
        cdf_vals = np.cumsum(dist_t['prob']) / np.sum(dist_t['prob'])

        # Find quantile values
        for i, u in enumerate(u_vec):
            # Find closest CDF value
            idx = np.argmin(np.abs(cdf_vals - u))
            result_matrix[t, i] = dist_t.iloc[idx]['value']

    return result_matrix


def compute_path_probabilities(real_vec: np.ndarray,
                               forecast_matrix: np.ndarray,
                               sigma: float = 2.0) -> np.ndarray:
    """
    Compute path probabilities between real values and forecast scenarios.

    Args:
        real_vec: Vector of real values
        forecast_matrix: Matrix of forecast scenarios
        sigma: Standard deviation parameter

    Returns:
        Matrix of probabilities (horizon x n_scenarios)
    """
    horizon = len(real_vec)
    n_scenarios = forecast_matrix.shape[1]
    prob_mat = np.zeros((horizon, n_scenarios))

    for t in range(horizon):
        diffs = (real_vec[t] - forecast_matrix[t, :]) ** 2
        probs = np.exp(-diffs / (2 * sigma ** 2))
        prob_mat[t, :] = probs / np.sum(probs)

    return prob_mat


def mahalanobis_distance(real_vec: np.ndarray,
                         scenario_vec: np.ndarray,
                         cov_matrix: np.ndarray) -> float:
    """
    Calculate Mahalanobis distance between real and scenario vectors.

    Args:
        real_vec: Real values vector
        scenario_vec: Scenario values vector
        cov_matrix: Covariance matrix

    Returns:
        Mahalanobis distance
    """
    diff = real_vec - scenario_vec
    try:
        inv_cov = np.linalg.inv(cov_matrix)
        return np.dot(np.dot(diff, inv_cov), diff)
    except np.linalg.LinAlgError:
        # If covariance matrix is singular, use pseudo-inverse
        pseudo_inv = np.linalg.pinv(cov_matrix)
        return np.dot(np.dot(diff, pseudo_inv), diff)


def find_closest_joint_scenario(da_matrix: np.ndarray,
                                id_matrix: np.ndarray,
                                smf_matrix: np.ndarray,
                                da_real: np.ndarray,
                                id_real: np.ndarray,
                                smf_real: np.ndarray,
                                probs_da_t: np.ndarray,
                                probs_id_t: np.ndarray,
                                probs_smf_t: np.ndarray,
                                cov_matrix: np.ndarray,
                                t_now,
                                datetime_vec: pd.DatetimeIndex) -> Dict:
    # t_index: eğer t_now datetime ise indeksini bul, yoksa doğrudan kullan
    datetime_index = pd.DatetimeIndex(datetime_vec)
    t_index = t_now if isinstance(t_now, (int, np.integer)) else datetime_index.get_loc(t_now)


    S = da_matrix.shape[1]
    real_vec = np.array([da_real[t_index], id_real[t_index], smf_real[t_index]])

    best_score = np.inf
    best_combination = (None, None, None)

    for s_da in range(S):
        for s_id in range(S):
            for s_smf in range(S):
                scenario_vec = np.array([
                    da_matrix[t_index, s_da],
                    id_matrix[t_index, s_id],
                    smf_matrix[t_index, s_smf]
                ])
                dist = mahalanobis_distance(real_vec, scenario_vec, np.linalg.inv(cov_matrix))
                joint_prob = probs_da_t[s_da] * probs_id_t[s_id] * probs_smf_t[s_smf]
                score = dist - np.log(joint_prob + 1e-12)

                if score < best_score:
                    best_score = score
                    best_combination = (s_da, s_id, s_smf)

    return {
        'best_combination': best_combination,
        'min_score': best_score
    }


def generate_scenario_dt_for_window(da_matrix: np.ndarray, id_matrix: np.ndarray, smf_matrix: np.ndarray,
                                    prod_matrix: np.ndarray,
                                    real_da: np.ndarray, real_id: np.ndarray, real_smf: np.ndarray,
                                    cov_matrix: np.ndarray,
                                    ex_ante_time: pd.Timestamp, db: pd.DataFrame,  # R'daki db_goktepe -> db
                                    window_size: int = 10,
                                    sigma: float = 2.0) -> pd.DataFrame:  # R'daki varsayılan değerler

    horizon_hours: int = 48

    # Optimize edilecek zaman aralığı (R'daki optimization_start <- ex_ante_time + lubridate::days(1) karşılığı)
    optimization_start = (ex_ante_time + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)


    # optimization_hours (R'daki seq.POSIXt karşılığı)
    # R'da direkt Timestamp objeleri oluşturur, Python'da da pd.date_range kullanıyoruz.
    # Timezone'u IST olarak belirtiyoruz, R'daki gibi otomatik varsaymıyoruz.
    # ex_ante_time'ın timezone bilgisi olduğu varsayılıyor.
    optimization_hours = pd.date_range(start=optimization_start, periods=horizon_hours, freq="H",
                                       tz=ex_ante_time.tzinfo)

    # scenario_list (R'daki vector("list", horizon_hours) karşılığı)
    scenario_list: List[dict] = []

    # db['datetime'] sütununun timezone'ını standardize edelim
    # predict_scenario_with_moving_window içinde de yapılıyor ama burada da emin olalım.
    if db['datetime'].dtype.tz is None and ex_ante_time.tzinfo is not None:
        db['datetime'] = db['datetime'].dt.tz_localize(ex_ante_time.tzinfo)
    elif db['datetime'].dtype.tz is not None and ex_ante_time.tzinfo is None:
        # Eğer db tz'li, ex_ante_time tz'siz ise db'yi ex_ante_time'ın lokal tz'ine convert et.
        # Bu durum biraz karışık, genelde ikisi de aynı tz'de veya tz'siz olmalı.
        # Basitçe db'yi ex_ante_time'ın tz'ine çevirelim (eğer ex_ante_time tz'siz ise bir sorun olabilir).
        pass  # Şimdilik bunu predict_scenario_with_moving_window'a bırakalım

    # R'daki for (i in seq_along(optimization_hours)) karşılığı
    for i in range(horizon_hours):
        time_now = optimization_hours[i]

        # Matrix için index bul: ilgili POSIX zamanın hangi satıra denk geldiği
        # R'daki t_index <- which(db_goktepe$datetime == time_now) karşılığı
        try:
            # Pandas'ta Timestamp ile tam eşleşen satırın indeksini buluyoruz.
            # Bu indeks, DataFrame'in kendi indeksi (0-tabanlı) olacaktır.
            # Ancak, predict_scenario_with_moving_window'a göndereceğimiz t_index 1-tabanlı R indeksi olmalı.
            # R'da `which` fonksiyonu 1-tabanlı indeks döndürür.
            # Python'da `db[db['datetime'] == time_now].index[0]` 0-tabanlı indeks döndürür.
            # Dolayısıyla, R'daki 1-tabanlı `t_index`'i elde etmek için +1 ekliyoruz.
            t_index_0_based = db[db['datetime'] == time_now].index[0]
            t_index_1_based = t_index_0_based + 1  # R'daki gibi 1-tabanlı indeks

        except IndexError:  # Eğer o saat yoksa atla (R'daki if (length(t_index) == 0) next karşılığı)
            print(f"Uyarı: {time_now} için db DataFrame'inde veri bulunamadı. Atlanıyor.")
            continue

        # predict_scenario_with_moving_window çağrıları
        # t_now parametresini (Timestamp) ve db DataFrame'i gönderiyoruz.
        # Bu fonksiyonun içindeki t_index'i bulma mantığı güncel olmalı.
        probs_da_t = predict_scenario_with_moving_window(
            prod_matrix=da_matrix,
            real_prod=real_da,
            t_now=time_now,  # Timestamp
            window_size=window_size,
            datetime_vec= db['datetime'],
            sigma=sigma
        )
        probs_id_t = predict_scenario_with_moving_window(
            prod_matrix=id_matrix,
            real_prod=real_id,
            t_now=time_now,
            window_size=window_size,
            datetime_vec= db['datetime'],
            sigma=sigma
        )
        probs_smf_t = predict_scenario_with_moving_window(
            prod_matrix=smf_matrix,
            real_prod=real_smf,
            t_now=time_now,
            window_size=window_size,
            datetime_vec = db['datetime'],
            sigma=sigma
        )

        # find_closest_joint_scenario çağrısı
        # Bu fonksiyon muhtemelen t_now'ı (yani t_index'i) 1-tabanlı R indeksi gibi bekler.
        # Eğer öyleyse, t_index_1_based'i göndereceğiz.
        result = find_closest_joint_scenario(
            da_matrix=da_matrix, id_matrix=id_matrix, smf_matrix=smf_matrix,
            da_real =real_da, id_real =real_id, smf_real=real_smf,
            probs_da_t=probs_da_t, probs_id_t=probs_id_t, probs_smf_t=probs_smf_t,
            cov_matrix=cov_matrix, t_now=t_index_1_based, datetime_vec=db['datetime'],
            # Bu t_now parametresi için R'daki gibi 1-tabanlı indeks gönderiyoruz.
        )

        # Sonuçları alma (R'daki result$best_combination[1] karşılığı)
        s_da = result['best_combination'][0]  # Python'da 0-tabanlı
        s_id = result['best_combination'][1]  # Python'da 0-tabanlı
        s_smf = result['best_combination'][2]  # Python'da 0-tabanlı

        # scenario_list'e ekleme (R'daki data.table() karşılığı)
        # NumPy dizilerine erişirken 0-tabanlı indeksi kullanmalıyız: t_index_0_based
        scenario_list.append({
            'datetime': time_now,
            'DA': da_matrix[t_index_0_based, s_da],
            'ID': id_matrix[t_index_0_based, s_id],
            'SMF': smf_matrix[t_index_0_based, s_smf],
            'prod': prod_matrix[t_index_0_based, s_da],
            # R'da prod_matrix[t_index, s_da] olması lazım. Burada da da_matrix kullanılmış.
            # Eğer prod_matrix ve da_matrix farklıysa dikkat etmek lazım.
            'penalty': max(da_matrix[t_index_0_based, s_da], smf_matrix[t_index_0_based, s_smf]) * 1.03
        })

    # return(rbindlist(scenario_list)) karşılığı
    return pd.DataFrame(scenario_list), optimization_hours


def generate_forecast_distribution_from_price_forecast(forecasted_price: pd.DataFrame,
                                                       error_sd: float = 0.05,
                                                       n_bins: int = 100,
                                                       price_ceiling: float = 3000) -> pd.DataFrame:
    """
    Generate forecast distribution from price forecast data.

    Args:
        forecasted_price: DataFrame with columns ['datetime', 'price_forecast', ...]
        error_sd: Standard deviation for error calculation
        n_bins: Number of bins for distribution
        price_ceiling: Maximum price ceiling

    Returns:
        DataFrame with columns ['datetime', 'variable_type', 'value', 'prob']
    """
    df = forecasted_price.copy()
    df['forecast_sd'] = error_sd * df['DA_price_forecasted']

    # Z-score ranges and probabilities
    z_vals = np.linspace(-3, 3, n_bins + 1)
    probs = np.diff(stats.norm.cdf(z_vals))
    z_mid = (z_vals[:-1] + z_vals[1:]) / 2  # midpoints of ranges

    # Generate distribution for each datetime
    result_list = []
    for _, row in df.iterrows():
        values = row['DA_price_forecasted'] + z_mid * row['forecast_sd']
        values = np.clip(values, 0, price_ceiling)  # Apply bounds

        for i in range(n_bins):
            result_list.append({
                'datetime': row['datetime'],
                'variable_type': 'forecast_price',
                'value': values[i],
                'prob': probs[i]
            })

    return pd.DataFrame(result_list)


def generate_forecast_distribution_from_id_price(forecasted_id_price: pd.DataFrame,
                                                 error_sd: float = 0.05,
                                                 n_bins: int = 100,
                                                 price_ceiling: float = 3000) -> pd.DataFrame:
    """
    Generate forecast distribution from ID price forecast data.

    Args:
        forecasted_id_price: DataFrame with columns ['datetime', 'price_forecast', ...]
        error_sd: Standard deviation for error calculation
        n_bins: Number of bins for distribution
        price_ceiling: Maximum price ceiling

    Returns:
        DataFrame with columns ['datetime', 'variable_type', 'value', 'prob']
    """
    df = forecasted_id_price.copy()
    df['forecast_sd'] = error_sd * df['ID_price_forecasted']

    # Z-score ranges and probabilities
    z_vals = np.linspace(-3, 3, n_bins + 1)
    probs = np.diff(stats.norm.cdf(z_vals))
    z_mid = (z_vals[:-1] + z_vals[1:]) / 2  # midpoints of ranges

    # Generate distribution for each datetime
    result_list = []
    for _, row in df.iterrows():
        values = row['ID_price_forecasted'] + z_mid * row['forecast_sd']
        values = np.clip(values, 0, price_ceiling)  # Apply bounds

        for i in range(n_bins):
            result_list.append({
                'datetime': row['datetime'],
                'variable_type': 'forecast_price_ID',
                'value': values[i],
                'prob': probs[i]
            })

    return pd.DataFrame(result_list)


def generate_forecast_distribution_from_smf(forecasted_smf: pd.DataFrame,
                                            error_sd: float = 0.05,
                                            n_bins: int = 100,
                                            price_ceiling: float = 3000) -> pd.DataFrame:
    """
    Generate forecast distribution from SMF forecast data.

    Args:
        forecasted_smf: DataFrame with columns ['datetime', 'smf_forecast', 'smf_real', ...]
        error_sd: Standard deviation for error calculation
        n_bins: Number of bins for distribution
        price_ceiling: Maximum price ceiling

    Returns:
        DataFrame with columns ['datetime', 'variable_type', 'value', 'prob']
    """
    df = forecasted_smf.copy()
    df['forecast_sd'] = error_sd * df['SMF_forecasted']

    # Z-score ranges and probabilities
    z_vals = np.linspace(-3, 3, n_bins + 1)
    probs = np.diff(stats.norm.cdf(z_vals))
    z_mid = (z_vals[:-1] + z_vals[1:]) / 2  # midpoints of ranges

    # Generate distribution for each datetime
    result_list = []
    for _, row in df.iterrows():
        values = row['SMF_forecasted'] + z_mid * row['forecast_sd']

        # Rule: If real SMF is 0, all scenarios should be 0
        if row['SMF_real'] == 0:
            values = np.zeros_like(values)
        else:
            values = np.clip(values, 0, price_ceiling)  # Apply bounds

        for i in range(n_bins):
            result_list.append({
                'datetime': row['datetime'],
                'variable_type': 'forecast_smf',
                'value': values[i],
                'prob': probs[i]
            })

    return pd.DataFrame(result_list)


def print_scenario_optimization_results(result: Dict):
    """
    Print the results of scenario-based optimization in a formatted way.

    Args:
        result: Dictionary returned by run_ex_ante_model_scenario_based
    """
    if result['status'] != 'success':
        print(f"❌ Optimization failed: {result.get('message', 'Unknown error')}")
        return

    print("\n🎯 Scenario-Based Optimization Results")
    print("=" * 60)

    # Print basic info
    print(f"📅 Ex-ante Time: {result['ex_ante_time']}")
    print(f"💰 Objective Value: {result['objective_value']:.2f}")
    print(f"⏰ Optimization Window: {result['optimization_window']['start']} to {result['optimization_window']['end']}")
    print(f"📊 Scenario Info Shape: {result['scenario_info'].shape}")

    # Print first few decisions
    print("\n📋 First 5 Hours of Decisions:")
    print("-" * 60)

    decisions = result['decisions']
    scenario_info = result['scenario_info']

    for i in range(min(5, len(decisions.get('q_committed', [])))):
        if i < len(scenario_info):
            dt = scenario_info.iloc[i]['datetime']
            print(f"""
🕒 {pd.to_datetime(dt).strftime("%d %b %H:%M")}
    📌 Q Committed       : {decisions.get('q_committed', [0])[i]:>6.2f} MWh
    🛠️  DA Prod           : {decisions.get('d_da_prod', [0])[i]:>6.2f} MWh
    🛠️  ID Prod           : {decisions.get('d_id_prod', [0])[i]:>6.2f} MWh
    🔋 DA Battery Use    : {decisions.get('d_da_bat', [0])[i]:>6.2f} MWh
    🔋 ID Battery Use    : {decisions.get('d_id_bat', [0])[i]:>6.2f} MWh
    ⚡ Charge            : {decisions.get('charge', [0])[i]:>6.2f} MWh
    🪫 SoC               : {decisions.get('soc', [0])[i]:>6.2f} MWh
    📈 DA Price (Scenario): {scenario_info.iloc[i]['DA_scenario']:>6.2f}
    📈 ID Price (Scenario): {scenario_info.iloc[i]['ID_scenario']:>6.2f}
    📈 SMF Price (Scenario): {scenario_info.iloc[i]['SMF_scenario']:>6.2f}
    🏭 Production (Scenario): {scenario_info.iloc[i]['prod_scenario']:>6.2f} MWh
""")

    print("=" * 60)
    print("✅ Scenario-based optimization completed successfully!")






def generate_forecast_distribution_from_forecast(forecast_df: pd.DataFrame, error_sd: float = 0.05, n_bins: int = 5) -> pd.DataFrame:
    """
    Generate a probabilistic distribution from deterministic forecast values using z-score bins.

    Args:
        forecast_df: DataFrame with columns ['datetime', 'powerplantId', 'forecast_total']
        error_sd: Relative standard deviation for forecast uncertainty (e.g. 0.05 = %5)
        n_bins: Number of probability bins

    Returns:
        A DataFrame with columns: ['datetime', 'powerplantId', 'variable_type', 'value', 'prob']
    """
    df = forecast_df.copy()
    df['forecast_sd'] = df['forecasted_production_meteologica'] * error_sd

    # Define z-score bin edges and probabilities
    z_vals = np.linspace(-3, 3, n_bins + 1)
    probs = np.diff(norm.cdf(z_vals))  # Probabilities between bin edges
    z_mid = (z_vals[:-1] + z_vals[1:]) / 2  # Midpoints

    records = []

    for _, row in df.iterrows():
        forecast = row['forecasted_production_meteologica']
        std_dev = row['forecast_sd']
        values = np.maximum(forecast + z_mid * std_dev, 0)  # Clamp negatives to 0

        for val, prob in zip(values, probs):
            records.append({
                'datetime': row['datetime'],
                'powerplantId': row['powerplantId'],
                'variable_type': 'forecast_production',
                'value': val,
                'prob': prob
            })

    dist_df = pd.DataFrame(records)
    return dist_df


# Example usage and helper functions
def create_battery_params() -> Dict[str, float]:
    """Create default battery parameters"""
    return {
        'efficiency': 0.95,
        'power_limit': 100.0,
        'soc_min': 0.1,
        'soc_max': 0.9,
        'soc_target': 0.5,
        'capacity': 400.0  # kWh
    }

def generate_rank_preserving_prod_scenarios(forecast_dist_prod: pd.DataFrame,
                                            datetime_vec: pd.DatetimeIndex,
                                            n_scenarios: int = 100) -> np.ndarray:
    """
    Generate a rank-preserving production scenario matrix from forecast distributions.

    Args:
        forecast_dist_prod: DataFrame with columns ['datetime', 'value', 'prob']
        datetime_vec: Array of datetime objects to align scenario matrix
        n_scenarios: Number of scenarios to generate

    Returns:
        A 2D numpy array of shape (len(datetime_vec), n_scenarios)
    """
    # Ensure timezone consistency
    tz = pytz.timezone("Europe/Istanbul")
    forecast_dist_prod['datetime'] = pd.to_datetime(forecast_dist_prod['datetime']).dt.tz_convert(tz)
    datetime_vec = pd.to_datetime(datetime_vec).tz_convert(tz)

    # Uniform quantiles (excluding 0 and 1)
    u_vec = np.linspace(0, 1, n_scenarios + 2)[1:-1]

    scenario_matrix = []

    for dt in datetime_vec:
        dist_t = forecast_dist_prod[forecast_dist_prod['datetime'] == dt]
        if dist_t.empty:
            # If no forecast data for this time, fill with NaNs
            scenario_matrix.append([np.nan] * n_scenarios)
            continue

        probs = dist_t['prob'].values
        values = dist_t['value'].values
        cdf_vals = np.cumsum(probs) / np.sum(probs)

        row = []
        for u in u_vec:
            idx = np.argmin(np.abs(cdf_vals - u))
            row.append(values[idx])

        scenario_matrix.append(row)

    return np.array(scenario_matrix)


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from pulp import *
import warnings

warnings.filterwarnings('ignore')


def generate_scenario_dt_for_window_expost(da_matrix, id_matrix, smf_matrix, prod_matrix,
                                           real_da, real_id, real_smf, cov_matrix,
                                           ex_post_time, db, window_size=10, sigma=2.0):
    """
    Generate ex-post scenarios for optimization window
    """
    from forecasts_utils import predict_scenario_with_moving_window, find_closest_joint_scenario

    horizon_hours = 48

    # Create optimization hours starting from ex_post_time
    optimization_hours = pd.date_range(start=ex_post_time, periods=horizon_hours, freq="H",
                                       tz=ex_post_time.tzinfo)

    scenario_list = []

    # Handle timezone consistency like in the first function
    if db['datetime'].dtype.tz is None and ex_post_time.tzinfo is not None:
        db['datetime'] = db['datetime'].dt.tz_localize(ex_post_time.tzinfo)
    elif db['datetime'].dtype.tz is not None and ex_post_time.tzinfo is None:
        pass  # Handle as needed

    for i in range(horizon_hours):
        time_now = optimization_hours[i]

        # Find the index in the database that corresponds to this time
        try:
            # Get 0-based index from DataFrame
            t_index_0_based = db[db['datetime'] == time_now].index[0]
            # Convert to 1-based index for R-compatible functions
            t_index_1_based = t_index_0_based + 1
        except IndexError:
            print(f"Warning: Data not found for {time_now} in db DataFrame. Skipping.")
            continue  # Skip if time not found

        # Get scenario probabilities for this time point
        probs_da_t = predict_scenario_with_moving_window(
            prod_matrix=da_matrix,
            real_prod=real_da,
            t_now=time_now,  # Pass Timestamp
            window_size=window_size,
            datetime_vec=db['datetime'],
            sigma=sigma
        )

        probs_id_t = predict_scenario_with_moving_window(
            prod_matrix=id_matrix,
            real_prod=real_id,
            t_now=time_now,  # Pass Timestamp
            window_size=window_size,
            datetime_vec=db['datetime'],
            sigma=sigma
        )

        probs_smf_t = predict_scenario_with_moving_window(
            prod_matrix=smf_matrix,
            real_prod=real_smf,
            t_now=time_now,  # Pass Timestamp
            window_size=window_size,
            datetime_vec=db['datetime'],
            sigma=sigma
        )

        # Find the closest joint scenario - pass 1-based index like in the first function
        result = find_closest_joint_scenario(
            da_matrix=da_matrix,
            id_matrix=id_matrix,
            smf_matrix=smf_matrix,
            da_real=real_da,
            id_real=real_id,
            smf_real=real_smf,
            probs_da_t=probs_da_t,
            probs_id_t=probs_id_t,
            probs_smf_t=probs_smf_t,
            cov_matrix=cov_matrix,
            t_now=t_index_1_based,  # Pass 1-based index, not timestamp
            datetime_vec=db['datetime']
        )

        # Get scenario indices (0-based in Python)
        s_da = result['best_combination'][0]
        s_id = result['best_combination'][1]
        s_smf = result['best_combination'][2]

        # Use 0-based index for matrix access
        scenario_data = {
            'datetime': time_now,
            'DA': da_matrix[t_index_0_based, s_da],
            'ID': id_matrix[t_index_0_based, s_id],
            'SMF': smf_matrix[t_index_0_based, s_smf],
            'prod': prod_matrix[t_index_0_based, s_da],
            'penalty': max(da_matrix[t_index_0_based, s_da], smf_matrix[t_index_0_based, s_smf]) * 1.03
        }

        scenario_list.append(scenario_data)

    return pd.DataFrame(scenario_list), optimization_hours