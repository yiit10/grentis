from fastapi import FastAPI, Form, Request, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from epias_client import get_tgt_token, get_mcp_data, get_id_avg_price_data, get_injection_quantity, get_realtime_generation, get_smf_data
from forecasts_utils import generate_forecast_price_ar1, generate_forecast_price_id, calculate_forecast_metrics, calculate_smf_metrics, generate_forecast_production, generate_forecast_smf
from datetime import date, datetime, timedelta
import pytz
import pandas as pd
import json
from fastapi import Body
import os
from battery_brain_core import solve_ex_ante_model, prepare_optimization_data, calculate_flex_decisions
from dashboard_data_logic import get_dashboard_data
from performance_back import run_simulation, prepare_benchmark_inputs
from pathlib import Path
import numpy as np
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from bbflex_claude import run_simulation_flex_incremental




app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="supersecret")


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
        'capacity': 65,
        'power_limit': 80.4,
        'soc_min': 13,
        'soc_max': 65,
        'soc_target': 31,
        'efficiency': 0.97,
    },
    "akyelres": {
        'capacity': 61.8,
        'power_limit': 61.8,
        'soc_min': 8,
        'soc_max': 40.2,
        'soc_target': 20,
        'efficiency': 0.97,
    }
}


istanbul = pytz.timezone("Europe/Istanbul")
now = datetime.now(istanbul).replace(minute=0, second=0, microsecond=0)
virtual_now = now - timedelta(days=3)
start = virtual_now - timedelta(days=3)
end = virtual_now + timedelta(days=2)
graph_start = virtual_now - timedelta(days=1)
graph_end = virtual_now + timedelta(days=2)
decision_time = virtual_now.replace(hour=10, minute=0, second=0, microsecond=0)
horizon_start = (decision_time + timedelta(days=1)).replace(hour=0)
horizon_end = horizon_start + timedelta(hours=47)

# statik dosyaları serve etmek için bu satırı ekle:
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
@app.get("/")
def home(request: Request):
    user = request.session.get("user")
    return templates.TemplateResponse("index.html", {"request": request, "user": user})


@app.post("/contact")
async def contact_form(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    company: str = Form(...),
    message: str = Form(...)
):
    print(f"Yeni iletişim formu:\nAd: {name}\nE-posta: {email}\nŞirket: {company}\nMesaj: {message}")
    # İleride burayı DB kayıt, email gönderimi, webhook vb. için kullanabiliriz
    return RedirectResponse(url="/", status_code=303)

# 🔥 Tüm tarihsel veri tiplerini string'e çevir
def convert_dates_to_str(df):
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].apply(lambda x: x.isoformat() if pd.notnull(x) else x)
    return df


def currencyformat(value, symbol='₺'):
    return f"{symbol}{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

templates.env.filters["currencyformat"] = currencyformat

def render_dashboard(request, user, virtual_now, price_data, price_forecast_data,
                     price_id_data, price_id_forecast_data, flex_decisions,
                     expected_revenue, production_data, q_committed_data,
                     q_committed_stacked_data, production_usage_data, optimization_results,
                     active_forecast,
                     initial_soc, plant_status, capacity_utilization, battery_params = None,
                     graph_start=None, graph_end=None,
                     horizon_start = None, horizon_end = None, battery_usage_data=None, default_strategy_revenue = None,
                     custom_strategy_json = None, default_strategy_df= None,# <-- 💥 BURAYI EKLE
                     template_name="dashboard.html"):
    print("📦 default_strategy_df örneği:")


    if battery_params is None:
        battery_params = user_battery_params.get(user, {})



    if default_strategy_df is not None:
        default_strategy_df = convert_dates_to_str(default_strategy_df)

    if custom_strategy_json is None and default_strategy_df is not None:
        custom_strategy_json = json.dumps(default_strategy_df.to_dict(orient="records"))

    return templates.TemplateResponse(template_name, {
        "request": request,
        "price_data": json.dumps(price_data),
        "price_forecast_data": json.dumps(price_forecast_data),
        "price_id_data": json.dumps(price_id_data),
        "price_id_forecast_data": json.dumps(price_id_forecast_data),
        "flex_decisions": flex_decisions,
        "expected_revenue": expected_revenue,
        "production_data": json.dumps(production_data),
        "q_committed_data": json.dumps(q_committed_data),
        "battery_usage_data": json.dumps(battery_usage_data),
        "default_strategy_revenue": default_strategy_revenue,
        "default_strategy_df": json.dumps(
            default_strategy_df.to_dict(orient="records")) if default_strategy_df is not None else "[]",
        "custom_strategy_json": custom_strategy_json or "[]",
        "virtual_now": virtual_now.isoformat(),
        "user": user,
        "q_committed_stacked_data": json.dumps(q_committed_stacked_data),
        "production_usage_data": json.dumps(production_usage_data),
        "optimization_status": optimization_results['status'] if optimization_results else 'not_run',
        "active_forecast": active_forecast,
        "initial_soc": initial_soc,
        "plant_status": plant_status,
        "capacity_utilization": capacity_utilization,
        "graph_start": graph_start,
        "graph_end": graph_end,
        "battery_params": battery_params,
        "horizon_start": horizon_start,
        "horizon_end": horizon_end
    })


@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    user = request.session.get("user")
    return templates.TemplateResponse("login.html", {"request": request, "user": user})


@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if username in users and users[username] == password:
        request.session["user"] = username
        return RedirectResponse(url="/dashboard", status_code=302)
    return HTMLResponse("<h3>Geçersiz kullanıcı adı veya şifre.</h3>", status_code=401)

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, active_forecast: str = Query("meteologica")):
    user = request.session.get("user")
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    data = get_dashboard_data(user, active_forecast)
    return render_dashboard(request, **data, template_name="dashboard.html")


@app.get("/optimization", response_class=HTMLResponse)
def optimization_view(request: Request, active_forecast: str = Query("meteologica")):
    user = request.session.get("user")
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    data = get_dashboard_data(user, active_forecast)

    # 🔧 JSON'a çevirmeden önce datetime'ları string'e çevir
    if "default_strategy_df" in data and isinstance(data["default_strategy_df"], pd.DataFrame):
        df_clean = convert_dates_to_str(data["default_strategy_df"])
        data["custom_strategy_json"] = json.dumps(df_clean.to_dict(orient="records"))
    else:
        data["custom_strategy_json"] = json.dumps([])

    return render_dashboard(request, **data, template_name="optimization.html")
@app.post("/custom-forecast")
async def custom_forecast(data: dict = Body(...)):
    print("📅 Kullanıcıdan gelen üretim tahmini:", data)

    try:
        with open("user_forecast.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/clear-custom-forecast")
def clear_custom_forecast():
    try:
        if os.path.exists("user_forecast.json"):
            os.remove("user_forecast.json")
            return {"status": "ok", "message": "Tahmin silindi."}
        return {"status": "empty", "message": "Zaten boş."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/performance", response_class=HTMLResponse)

def performance_view(request: Request):
    user = request.session.get("user")
    plant_id = user_to_plant_map.get(user)
    flex_plant_ids = ["2591"]
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    from performance_back import run_simulation, analyze_simulation_results, prepare_benchmark_inputs, \
        simulate_benchmark_case1, show_hourly_decisions

    try:
        # 1. Run your battery brain simulation
        if plant_id in flex_plant_ids:
            all_decisions = run_simulation_flex_incremental(plant_id)
        else:
            all_decisions = run_simulation(plant_id)

        all_decisions["datetime"] = pd.to_datetime(all_decisions["datetime"]).dt.tz_localize(None)
        all_decisions = all_decisions[all_decisions["datetime"] < virtual_now.replace(tzinfo=None)]

        # 2. Battery parameters - FIXED: Added missing power_limit
        battery_params=user_battery_params[user]

        # 3. Analyze your simulation results
        df_hourly, df_daily = analyze_simulation_results(
            all_decisions,
            battery_params,
            save_results=False
        )

        df_hourly["net_profit"] += df_hourly["battery_cost"]



        # 4. Prepare and run benchmark simulation
        df_perf = prepare_benchmark_inputs(df_hourly)
        df_benchmark = simulate_benchmark_case1(df_perf, battery_params)



        # 5. Calculate benchmark daily metrics - FIXED: Group by date properly
        if 'date' not in df_benchmark.columns:
            df_benchmark['date'] = pd.to_datetime(df_benchmark['datetime']).dt.date

        df_benchmark_daily_detailed = df_benchmark.groupby('date').agg({
            'net_profit': 'sum',
            'DA_revenue': 'sum',
            'id_gelir': 'sum',
            'id_maliyet': 'sum',
            'batarya_cost': 'sum',
            'ceza_maliyet': 'sum'
        }).reset_index()

        # Rename kolonlar frontend'e uygun olacak şekilde
        df_benchmark_daily_detailed = df_benchmark_daily_detailed.rename(columns={
            'net_profit': 'benchmark_profit',
            'DA_revenue': 'bm_DA_revenue',
            'id_gelir': 'bm_ID_revenue',
            'id_maliyet': 'bm_id_costs',
            'batarya_cost': 'bm_battery_cost',
            'ceza_maliyet': 'bm_penalty'
        })

        df_benchmark_daily = df_benchmark.groupby('date')['net_profit'].sum().reset_index(name='benchmark_profit')

        # 6. Calculate benchmark totals - FIXED: Use correct column names
        total_benchmark_battery_cost = df_benchmark["batarya_cost"].sum()  # Changed from "battery_cost"
        total_benchmark_penalty = df_benchmark["ceza_maliyet"].sum()  # Changed from "penalty"
        total_benchmark_idrevenue = df_benchmark["id_gelir"].sum()
        total_benchmark_idcost = df_benchmark["id_maliyet"].sum()

        # 7. Merge daily data
        df_daily['date'] = pd.to_datetime(df_daily['date'])
        df_benchmark_daily['date'] = pd.to_datetime(df_benchmark_daily['date'])
        df_daily["date"] = pd.to_datetime(df_daily["date"]).dt.date
        df_benchmark_daily_detailed["date"] = pd.to_datetime(df_benchmark_daily_detailed["date"]).dt.date
        df_combined = pd.merge(df_daily, df_benchmark_daily_detailed, on='date', how='left')
        virtual_now_naive = virtual_now.replace(tzinfo=None)
        df_daily = df_daily[pd.to_datetime(df_daily["date"]) < virtual_now_naive]
        df_combined = df_combined[pd.to_datetime(df_combined["date"]) < virtual_now_naive]
        print("🔍 df_combined örneği:")
        print(df_combined.head(3).to_dict())


        # 8. Prepare data for frontend
        performance_data_serialized = deep_convert_numpy_to_python(df_hourly.to_dict(orient="records"))
        daily_metrics_serialized = deep_convert_numpy_to_python(df_combined.to_dict(orient="records"))
        print("🔍 daily_metrics_serialized örneği:")
        print(daily_metrics_serialized[:3])
        monthly_metrics = prepare_monthly_historical_metrics(battery_params,plant_id)
        monthly_metrics_benchmark = prepare_monthly_historical_metrics_benchmark(battery_params, plant_id)
        monthly_metrics = [m for m in monthly_metrics if pd.to_datetime(m["month"] + "-01") < virtual_now_naive]
        monthly_metrics_benchmark = [m for m in monthly_metrics_benchmark if
                                     pd.to_datetime(m["month"] + "-01") < virtual_now_naive]


        for i, row in enumerate(monthly_metrics):
            for k, v in row.items():
                if isinstance(v, (datetime, date)):
                    print(f"🔴 Found datetime in monthly_metrics[{i}]['{k}']: {v} ({type(v)})")



        # 9. Calculate totals for display - FIXED: Use correct column names
        total_bb_bat_cost = df_hourly["battery_cost"].sum()
        total_bb_penalty = df_hourly["penalty"].sum()
        total_bb_DA_revenue = df_hourly['DA_revenue'].sum()
        total_bb_ID_revenue = df_hourly['ID_revenue'].sum()
        total_bb_ID_costs = df_hourly['id_costs'].sum()
        total_bm_bat_cost = df_benchmark["batarya_cost"].sum()
        total_bm_penalty = df_benchmark["ceza_maliyet"].sum()
        total_bm_ID_revenue = df_benchmark["id_gelir"].sum()
        total_bm_ID_costs = df_benchmark["id_maliyet"].sum()
        total_bm_DA_revenue = df_benchmark["DA_revenue"].sum()

        # 10. Debug output - FIXED: Use correct column names
        print("=== HOURLY DECISIONS COMPARISON ===")
        show_hourly_decisions(df_hourly, df_benchmark, hours=24)

        print("\n=== BATTERY BRAIN TOTALS ===")
        print(df_hourly[["net_profit", "battery_cost", "penalty"]].sum())

        print("\n=== BENCHMARK TOTALS ===")
        print(df_benchmark[["net_profit", "batarya_cost", "ceza_maliyet"]].sum())

        # 11. Save combined results - IMPROVED: Better column alignment
        try:
            # Ensure datetime columns are properly handled
            df_hourly_save = df_hourly.copy()
            df_benchmark_save = df_benchmark.copy()

            # Standardize column names for comparison
            column_mapping = {
                'batarya_cost': 'battery_cost',
                'ceza_maliyet': 'penalty_cost',
                'id_satis': 'id_sell',
                'id_alim': 'id_buy',
                'id_gelir': 'id_revenue',
                'id_maliyet': 'id_cost',
                'ceza_miktar': 'penalty_amount'
            }

            if 'battery_cost' in df_benchmark_save.columns:
                print("⚠️ battery_cost zaten var, siliniyor")
                df_benchmark_save = df_benchmark_save.drop(columns=['battery_cost'])

            df_benchmark_save = df_benchmark_save.rename(columns=column_mapping)



            # Sadece ortak, eşsiz kolonları al
            common_cols = pd.Index(df_hourly_save.columns).intersection(df_benchmark_save.columns).unique()



            # Kolonlar tekrar etmeyecek şekilde slice et
            df_hourly_aligned = df_hourly_save.loc[:, common_cols].copy()
            df_benchmark_aligned = df_benchmark_save.loc[:, common_cols].copy()

            print("🎯 Common columns:", common_cols)
            print("🎯 Duplicated cols in hourly:", df_hourly_save.columns[df_hourly_save.columns.duplicated()])
            print("🎯 Duplicated cols in benchmark:", df_benchmark_save.columns[df_benchmark_save.columns.duplicated()])

            # Add strategy identifier
            df_hourly_aligned['strategy'] = 'battery_brain'
            df_benchmark_aligned['strategy'] = 'benchmark'

            # Save both strategy outputs separately
            csv_hourly_path = r"C:\Users\HanEE\Documents\BatteryBrain\batterybrain_web\bb_hourly_decisions.csv"
            csv_benchmark_path = r"C:\Users\HanEE\Documents\BatteryBrain\batterybrain_web\benchmark_decisions.csv"

            df_hourly_save.to_csv(csv_hourly_path, index=False)
            df_benchmark_save.to_csv(csv_benchmark_path, index=False)

            print(f"✅ Battery Brain hourly saved to: {csv_hourly_path}")
            print(f"✅ Benchmark decisions saved to: {csv_benchmark_path}")
            print(monthly_metrics)

        except Exception as e:
            print(f"❌ Error saving CSV: {str(e)}")


        # 12. Return template with data
        return templates.TemplateResponse("performance.html", {
            "request": request,
            "performance_data": performance_data_serialized,
            "daily_metrics": daily_metrics_serialized,
            "benchmark_battery_cost": total_benchmark_battery_cost,
            "benchmark_penalty": total_benchmark_penalty,
            "total_bb_profit": df_daily["net_profit"].sum() + df_daily["battery_cost"].sum(),
            "total_bm_profit": df_benchmark["net_profit"].sum() + df_benchmark["batarya_cost"].sum(),
            "total_bb_bat_cost": total_bb_bat_cost,
            "total_bb_penalty": total_bb_penalty,
            "total_bb_DA_revenue": total_bb_DA_revenue,
            "total_bb_ID_revenue": total_bb_ID_revenue,
            "total_bb_ID_costs": total_bb_ID_costs,
            "total_bm_bat_cost": total_bm_bat_cost,
            "total_bm_penalty": total_bm_penalty,
            "total_bm_ID_revenue": total_bm_ID_revenue,
            "total_bm_ID_costs": total_bm_ID_costs,
            "total_bm_DA_revenue": total_bm_DA_revenue,
            "monthly_metrics": monthly_metrics,
            "monthly_metrics_benchmark": monthly_metrics_benchmark
        })


    except Exception as e:
        print(f"❌ Error in performance endpoint: {str(e)}")
        import traceback
        traceback.print_exc()

        # Return error page or redirect
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_message": f"Performance calculation failed: {str(e)}"
        })



# === Load Historical CSVs ===
def load_historical_simulation_data(plant_id):
    path_2024 = Path(f"C:/Users/HanEE/Documents/BatteryBrain/batterybrain_web/simulation_results_2024_{plant_id}.csv")
    path_2025h1 = Path(f"C:/Users/HanEE/Documents/BatteryBrain/batterybrain_web/simulation_results_2025H1_{plant_id}.csv")


    df_2024 = pd.read_csv(path_2024, parse_dates=["datetime"]) if path_2024.exists() else pd.DataFrame()
    df_2025 = pd.read_csv(path_2025h1, parse_dates=["datetime"]) if path_2025h1.exists() else pd.DataFrame()


    df_all = pd.concat([df_2024, df_2025], ignore_index=True)


    # ✅ Timezone'dan temizle
    df_all["datetime"] = pd.to_datetime(df_all["datetime"])
    if df_all["datetime"].dt.tz is None:
        df_all["datetime"] = df_all["datetime"].dt.tz_localize("Europe/Istanbul")
    else:
        df_all["datetime"] = df_all["datetime"].dt.tz_convert("Europe/Istanbul")


    return df_all.sort_values("datetime")


def load_historical_benchmark_data(plant_id):
    path_2024_sim = Path(f"C:/Users/HanEE/Documents/BatteryBrain/batterybrain_web/benchmark_results_2024_{plant_id}.csv")
    path_2025h1_sim = Path(
        f"C:/Users/HanEE/Documents/BatteryBrain/batterybrain_web/benchmark_results_2025H1_{plant_id}.csv")

    df_2024_sim = pd.read_csv(path_2024_sim, parse_dates=["datetime"]) if path_2024_sim.exists() else pd.DataFrame()
    df_2025h1_sim = pd.read_csv(path_2025h1_sim,
                                parse_dates=[
                                    "datetime"]) if path_2025h1_sim.exists() else pd.DataFrame()  # parse_dates burada olmalıydı

    df_all_sim = pd.concat([df_2024_sim, df_2025h1_sim], ignore_index=True)
    df_all_sim["datetime"] = pd.to_datetime(df_all_sim["datetime"])
    if df_all_sim["datetime"].dt.tz is None:
        df_all_sim["datetime"] = df_all_sim["datetime"].dt.tz_localize("Europe/Istanbul")
    else:
        df_all_sim["datetime"] = df_all_sim["datetime"].dt.tz_convert("Europe/Istanbul")

    # --- BURADAN İTİBAREN EKLEYİN ---
    print("\n--- load_historical_benchmark_data() Çıktısı Kontrolü ---")
    print(f"df_2024_sim yüklendi mi? {not df_2024_sim.empty}. Satır sayısı: {len(df_2024_sim)}")
    if not df_2024_sim.empty:
        print(f"df_2024_sim Tarih Aralığı: {df_2024_sim['datetime'].min()} - {df_2024_sim['datetime'].max()}")
        print(f"df_2024_sim 2024 H2 için veri var mı? (Temmuz-Aralık):")
        print(df_2024_sim[(df_2024_sim['datetime'].dt.year == 2024) & (df_2024_sim['datetime'].dt.month >= 7)].shape)

    print(f"df_2025h1_sim yüklendi mi? {not df_2025h1_sim.empty}. Satır sayısı: {len(df_2025h1_sim)}")
    if not df_2025h1_sim.empty:
        print(f"df_2025h1_sim Tarih Aralığı: {df_2025h1_sim['datetime'].min()} - {df_2025h1_sim['datetime'].max()}")

    print(f"df_all_sim (birleştirilmiş) toplam satır sayısı: {len(df_all_sim)}")
    if not df_all_sim.empty:
        print(f"df_all_sim Tarih Aralığı: {df_all_sim['datetime'].min()} - {df_all_sim['datetime'].max()}")
        print(
            f"df_all_sim 2024 H2 için satır sayısı (Temmuz-Aralık): {df_all_sim[(df_all_sim['datetime'].dt.year == 2024) & (df_all_sim['datetime'].dt.month >= 7)].shape[0]}")
        print(f"df_all_sim 2025 için satır sayısı: {df_all_sim[df_all_sim['datetime'].dt.year == 2025].shape[0]}")

        # Kritik sütunlarda NaN kontrolü, özellikle 2024 H2 için
        critical_cols = ["real_production", "forecasted_production_meteologica", "ID_price_real", "DA_price_real",
                         "SMF_real"]
        df_2024_h2 = df_all_sim[(df_all_sim['datetime'].dt.year == 2024) & (df_all_sim['datetime'].dt.month >= 7)]
        print("\n2024 H2 için Kritik Sütunlarda NaN Sayıları:")
        print(df_2024_h2[critical_cols].isnull().sum())
    # --- YUKARIDAKİ KOD PARÇASINI EKLEYİN ---

    return df_all_sim.sort_values("datetime")

# === Analyze + Aylık Gruplama ===
def prepare_monthly_historical_metrics(battery_params,plant_id):
    df_hist = load_historical_simulation_data(plant_id)

    if df_hist.empty:
        print("⚠️ No historical simulation data found.")
        return []

    from performance_back import analyze_simulation_results  # Ensure it's imported here if used externally

    df_hourly, df_daily = analyze_simulation_results(
        df_hist,
        battery_params,
        save_results=False
    )

    df_hourly["net_profit"] += df_hourly["battery_cost"]
    df_daily["net_profit"] += df_daily["battery_cost"]

    df_daily["month"] = pd.to_datetime(df_daily["date"]).dt.to_period("M").astype(str)
    df_monthly = df_daily.groupby("month").agg({
        "net_profit": "sum",
        "battery_cost": "sum",
        "penalty": "sum",
        "DA_revenue": "sum",
        "ID_revenue": "sum",
        "id_costs": "sum"
    }).reset_index()

    return deep_convert_numpy_to_python(df_monthly.to_dict(orient="records"))

# main.py veya ilgili yardımcı dosyanızdaki düzeltilmiş fonksiyon
def prepare_monthly_historical_metrics_benchmark(battery_params,plant_id):
    df_hist = load_historical_benchmark_data(plant_id) # Tarihsel benchmark verisini yüklüyor

    flex_plant_ids = ["2591"]
    is_flex = str(plant_id) in flex_plant_ids


    if df_hist.empty:
        print("⚠️ Tarihsel benchmark simülasyon verisi bulunamadı.")
        return []

    # performance_back modülünden gerekli fonksiyonları çağırın
    from performance_back import prepare_benchmark_inputs, simulate_benchmark_case1
    from bbflex_claude import run_full_simulation_benchmark_flex

    # prepare_benchmark_inputs, tarihsel veriyi simülasyon için hazırlar
    df_perf = prepare_benchmark_inputs(df_hist)
    # Doğru benchmark fonksiyonunu seç
    if is_flex:
        df_benchmark = run_full_simulation_benchmark_flex(df_perf, battery_params)
        df_benchmark["net_profit_with_battery"] = df_benchmark["net_profit"] + df_benchmark["batarya_cost"]
    else:
        df_benchmark = simulate_benchmark_case1(df_perf, battery_params)
        df_benchmark["net_profit_with_battery"] = df_benchmark["net_profit"] + df_benchmark["batarya_cost"]

    if 'date' not in df_benchmark.columns:
        df_benchmark['date'] = pd.to_datetime(df_benchmark['datetime']).dt.date

    df_benchmark["month"] = pd.to_datetime(df_benchmark["date"]).dt.to_period("M").astype(str)

    df_monthly_benchmark = df_benchmark.groupby("month").agg({
        "net_profit_with_battery": "sum",
        "batarya_cost": "sum",
        "ceza_maliyet": "sum",
        "DA_revenue": "sum",
        "id_gelir": "sum",
        "id_maliyet": "sum"
    }).reset_index()

    df_monthly_benchmark = df_monthly_benchmark.rename(columns={
        "net_profit_with_battery": "net_profit",
        "batarya_cost": "battery_cost",
        "ceza_maliyet": "penalty",
        "id_gelir": "ID_revenue",
        "id_maliyet": "id_costs"
    })

    return deep_convert_numpy_to_python(df_monthly_benchmark.to_dict(orient="records"))


def deep_convert_numpy_to_python(obj):
    if isinstance(obj, dict):
        return {k: deep_convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, (datetime, date)):  # ← bu şart!
        return obj.isoformat()
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


# Helper function to ensure date columns exist
def ensure_date_column(df, datetime_col='datetime'):
    """
    Ensure the dataframe has a 'date' column derived from datetime
    """
    if 'date' not in df.columns and datetime_col in df.columns:
        df['date'] = pd.to_datetime(df[datetime_col]).dt.date
    return df


# Validation function for debugging
def validate_performance_data(df_hourly, df_benchmark):
    """
    Validate the performance data before sending to frontend
    """
    print("=== VALIDATION RESULTS ===")

    # Check data shapes
    print(f"Hourly data shape: {df_hourly.shape}")
    print(f"Benchmark data shape: {df_benchmark.shape}")

    # Check required columns
    required_bb_cols = ['net_profit', 'battery_cost', 'penalty']
    required_bm_cols = ['net_profit', 'batarya_cost', 'ceza_maliyet']

    missing_bb = [col for col in required_bb_cols if col not in df_hourly.columns]
    missing_bm = [col for col in required_bm_cols if col not in df_benchmark.columns]

    if missing_bb:
        print(f"❌ Missing Battery Brain columns: {missing_bb}")
    if missing_bm:
        print(f"❌ Missing Benchmark columns: {missing_bm}")

    # Check for null values
    bb_nulls = df_hourly[required_bb_cols].isnull().sum()
    bm_nulls = df_benchmark[required_bm_cols].isnull().sum()

    if bb_nulls.any():
        print(f"⚠️  Battery Brain null values: {bb_nulls}")
    if bm_nulls.any():
        print(f"⚠️  Benchmark null values: {bm_nulls}")

    # Summary statistics
    print(f"BB Total Profit: {df_hourly['net_profit'].sum():.2f}")
    print(f"BM Total Profit: {df_benchmark['net_profit'].sum():.2f}")


    return True

@app.get("/logout")
def logout(request: Request):
    request.session.pop("user", None)
    return RedirectResponse(url="/", status_code=302)

from pydantic import BaseModel
from typing import List

class CustomStrategyItem(BaseModel):
    datetime: str
    da_price: float
    id_price: float
    smf: float
    forecast: float
    kgup_prod: float
    gip_prod: float
    charge: float
    kgup_bat: float
    gip_bat: float
    soc: float

@app.post("/custom-strategy")
async def custom_strategy(request: Request, items: List[CustomStrategyItem]):
    try:
        user = request.session.get("user")
        if not user:
            return {"error": "User not authenticated"}

        revenue = 0.0
        for item in items:
            revenue += (
                item.kgup_prod * item.da_price +
                item.kgup_bat * item.da_price +
                item.gip_prod * item.id_price +
                item.gip_bat * item.id_price -
                1276/2 * (item.gip_bat + item.kgup_bat + item.charge)
            )
        return {"status": "ok", "revenue": round(revenue, 2)}
    except Exception as e:
        return {"error": str(e)}

