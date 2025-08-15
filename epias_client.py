import json
import pandas as pd
import requests
import time
import pytz
import pandas as pd
from datetime import datetime, timedelta

def get_tgt_token(username, password):
    url = "https://giris.epias.com.tr/cas/v1/tickets"
    headers = {
        "Accept": "text/plain",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "username": username,
        "password": password
    }
    response = requests.post(url, data=data, headers=headers)
    if response.status_code != 201:
        raise Exception("❌ TGT alınamadı!")
    tgt_url = response.headers['Location']
    return tgt_url.split("/")[-1]

def get_mcp_data(tgt, start_dt, end_dt):
    url = "https://seffaflik.epias.com.tr/electricity-service/v1/markets/dam/data/mcp"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "TGT": tgt
    }

    payload = {
        "startDate": start_dt.strftime("%Y-%m-%dT%H:%M:%S") + "+03:00",
        "endDate": end_dt.strftime("%Y-%m-%dT%H:%M:%S") + "+03:00"
    }

    print("📤 Gönderilen payload:")
    print(json.dumps(payload, indent=2))
    print("📤 Header:")
    print(headers)

    res = requests.post(url, json=payload, headers=headers)

    print("🧾 API yanıt:", res.status_code)
    print("🧾 Yanıt içeriği:", res.text)

    if res.status_code != 200:
        raise Exception(f"❌ API hatası: {res.status_code}")
    return res.json().get("items", [])


def get_id_avg_price_data(tgt, start_dt, end_dt):
    import json  # JSON çıktısı için
    url = "https://seffaflik.epias.com.tr/electricity-service/v1/markets/idm/data/weighted-average-price"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "TGT": tgt
    }

    payload = {
        "startDate": start_dt.strftime("%Y-%m-%dT%H:%M:%S") + "+03:00",
        "endDate": end_dt.strftime("%Y-%m-%dT%H:%M:%S") + "+03:00"
    }

    print("📤 [GİP] Payload:", json.dumps(payload, indent=2))
    res = requests.post(url, json=payload, headers=headers)
    print("📩 [GİP] Yanıt Kodu:", res.status_code)

    if res.status_code != 200:
        raise Exception(f"❌ GİP API hatası: {res.status_code}")

    return res.json().get("items", [])

def get_smf_data(tgt, start_dt, end_dt):
    url = "https://seffaflik.epias.com.tr/electricity-service/v1/markets/bpm/data/system-marginal-price"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "TGT": tgt
    }

    payload = {
        "startDate": start_dt.strftime("%Y-%m-%dT%H:%M:%S") + "+03:00",
        "endDate": end_dt.strftime("%Y-%m-%dT%H:%M:%S") + "+03:00"
    }

    print("📤 [SMF] Payload:", json.dumps(payload, indent=2))
    res = requests.post(url, json=payload, headers=headers)
    print("📩 [SMF] Yanıt Kodu:", res.status_code)

    if res.status_code != 200:
        raise Exception(f"❌ SMF API hatası: {res.status_code}")

    return res.json().get("items", [])


def get_injection_quantity(tgt_token, plant_id, date_str, max_retries=3):
    url = "https://seffaflik.epias.com.tr/electricity-service/v1/generation/data/injection-quantity"

    start_date = f"{date_str}T00:00:00+03:00"

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "TGT": tgt_token
    }

    body = {
        "powerplantId": plant_id,
        "startDate": start_date,
        "endDate": start_date
    }

    for attempt in range(1, max_retries + 1):
        try:
            # Use json parameter instead of data + json.dumps
            res = requests.post(url, headers=headers, json=body, timeout=10)
        except Exception as e:
            print(f"⚠ Hata (deneme {attempt}):", e)
            continue

        print(f"📡 {date_str} - Plant: {plant_id} | HTTP: {res.status_code}")

        if res.status_code == 200:
            try:
                parsed = res.json()
                print(f"🔍 Full API Response: {parsed}")
                items = parsed.get("items", [])
                if not items:
                    print(f"⚠ Veri yok: {date_str}")
                    return None
                df = pd.DataFrame(items)
                df["powerplantId"] = plant_id
                df["date_day"] = date_str
                return df
            except Exception as e:
                print(f"❌ JSON parse hatası: {e}")
                print(f"Response text: {res.text[:200]}")
                return None

        elif res.status_code == 429:
            print(f"⏳ 429 Too Many Requests → 60 sn bekleniyor [{attempt}/{max_retries}]")
            time.sleep(60)
        else:
            print(f"❌ HTTP Hatası: {res.status_code}")
            print(f"Response: {res.text[:200]}")
            break

    return None

def get_realtime_generation(tgt_token, plant_id, date_str, max_retries=3):
    istanbul = pytz.timezone("Europe/Istanbul")
    date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    today_minus_1 = (datetime.now(istanbul) - timedelta(days=1)).date()

    if date_obj > today_minus_1:
        print(f"🚫 {date_str} → Bu tarih için veri henüz EPİAŞ'ta yok (today - 1 constraint)")
        return None

    url = "https://seffaflik.epias.com.tr/electricity-service/v1/generation/data/realtime-generation"

    start_date = f"{date_str}T00:00:00+03:00"
    end_date   = f"{date_str}T23:59:59+03:00"

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "TGT": tgt_token
    }

    body = {
        "powerPlantId": plant_id,
        "startDate": start_date,
        "endDate": end_date
    }

    for attempt in range(1, max_retries + 1):
        try:
            res = requests.post(url, headers=headers, json=body, timeout=10)
        except Exception as e:
            print(f"⚠ Hata (deneme {attempt}):", e)
            continue

        print(f"📡 {date_str} - Plant: {plant_id} | HTTP: {res.status_code}")

        if res.status_code == 200:
            try:
                parsed = res.json()
                print(f"🔍 Response keys: {list(parsed.keys())}")
                items = parsed.get("items", [])
                if not items:
                    print(f"⚠ Veri yok: {date_str}")
                    return None
                df = pd.DataFrame(items)
                df["powerPlantId"] = plant_id
                df["date_day"] = date_str
                print(f"✅ Veri alındı: {date_str}, satır: {len(df)}")
                return df
            except Exception as e:
                print(f"❌ JSON parse hatası: {e}")
                print(f"Response text: {res.text[:200]}")
                return None

        elif res.status_code == 429:
            print(f"⏳ 429 Too Many Requests → 60 sn bekleniyor [{attempt}/{max_retries}]")
            time.sleep(60)
        else:
            print(f"❌ HTTP Hatası: {res.status_code}")
            print(f"Response: {res.text[:200]}")
            break

    return None



