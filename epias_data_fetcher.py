import requests
import pandas as pd
from datetime import datetime, timedelta

def get_tgt_token(username: str, password: str) -> str:
    url = "https://giris.epias.com.tr/cas/v1/tickets"
    res = requests.post(url, data={"username": username, "password": password})

    if res.status_code != 201:
        raise Exception("❌ TGT alınamadı!")

    return res.headers["location"].split("/")[-1]

def fetch_mcp_data(start_date: str, end_date: str, tgt: str) -> pd.DataFrame:
    url = "https://seffaflik.epias.com.tr/electricity-service/v1/markets/dam/data/mcp"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "TGT": tgt
    }

    payload = {
        "startDate": f"{start_date}T00:00:00+03:00",
        "endDate": f"{end_date}T23:00:00+03:00"
    }

    res = requests.post(url, json=payload, headers=headers)

    if res.status_code != 200:
        raise Exception(f"❌ API hatası: {res.status_code}")

    data = res.json()["items"]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df

def get_recent_mcp_data(username: str, password: str) -> pd.DataFrame:
    now = datetime.now()
    reference_time = now - timedelta(days=2)
    end_date = (reference_time).strftime("%Y-%m-%d")
    start_date = (reference_time - timedelta(days=3)).strftime("%Y-%m-%d")

    tgt = get_tgt_token(username, password)
    df = fetch_mcp_data(start_date, end_date, tgt)
    return df
