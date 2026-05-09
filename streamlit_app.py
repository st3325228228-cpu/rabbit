"""
台股技術分析儀表板 ── Streamlit 完整進階版 v3.0
新增：多來源股市爬蟲（即時報價 / 融資融券 / 新聞 / 法人 / 每日收盤）
依賴：pip install streamlit yfinance plotly pandas numpy requests beautifulsoup4 lxml
執行：streamlit run stock_dashboard_streamlit.py
"""

import warnings, math, io, re, time
import requests
import xml.etree.ElementTree as ET
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

# ══════════════════════════════════════════════════════════════════
#  頁面設定
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="台股技術分析儀表板 v3",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
#  全域色彩
# ══════════════════════════════════════════════════════════════════
BG    = "#050A14"
PANEL = "#08101E"
CARD  = "#0D1728"
BORD  = "#18283C"
CYAN  = "#00C8FF"
RED   = "#FF3030"
GRN   = "#00C864"
GOLD  = "#FFD700"
ORNG  = "#FF8C00"
TEXT  = "#C0CCD8"
MUTED = "#607080"
PURP  = "#A050FF"

# ══════════════════════════════════════════════════════════════════
#  共用 HTTP Headers
# ══════════════════════════════════════════════════════════════════
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    "Accept": "application/json, text/html, */*",
    "Referer": "https://www.twse.com.tw/",
}

# ══════════════════════════════════════════════════════════════════
#  全域 CSS
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
  .stApp, .main, [data-testid="stAppViewContainer"] {{
      background-color: {BG} !important; color: {TEXT} !important;
  }}
  [data-testid="stHeader"], [data-testid="stSidebar"] {{
      background-color: {PANEL} !important;
  }}
  .stTextInput input, .stNumberInput input {{
      background-color: {CARD} !important;
      border: 1px solid {BORD} !important;
      color: #E0EAF4 !important; border-radius: 5px !important;
  }}
  .stSlider > div > div {{ background-color: {CYAN} !important; }}
  .stButton button {{
      background: linear-gradient(90deg, #0A1E38, #122840) !important;
      color: {CYAN} !important; border: 1px solid {CYAN} !important;
      font-weight: 700 !important; border-radius: 5px !important;
      transition: all 0.2s ease !important;
  }}
  .stButton button:hover {{
      background: linear-gradient(90deg, #122840, #1A3050) !important;
      color: #fff !important; box-shadow: 0 0 8px rgba(0,200,255,0.3) !important;
  }}
  .stCheckbox label {{ color: {TEXT} !important; }}
  .js-plotly-plot, .plotly-graph-div {{ background: transparent !important; }}
  #MainMenu, footer, [data-testid="stToolbar"] {{ visibility: hidden; }}
  [data-testid="column"] {{ padding: 0 4px !important; }}
  ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
  ::-webkit-scrollbar-track {{ background: {BG}; }}
  ::-webkit-scrollbar-thumb {{ background: {BORD}; border-radius: 3px; }}
  label[data-testid="stWidgetLabel"] > p {{
      color: {MUTED} !important; font-size: .78rem !important;
  }}
  hr {{ border-color: {BORD} !important; }}
  [data-testid="stSidebar"] .stButton button {{
      font-size: .75rem !important; padding: 3px 8px !important;
  }}
  .stTabs [data-baseweb="tab-list"] {{
      background-color: {PANEL} !important;
      border-bottom: 1px solid {BORD} !important;
  }}
  .stTabs [data-baseweb="tab"] {{
      color: {MUTED} !important; font-size: .78rem !important;
  }}
  .stTabs [aria-selected="true"] {{
      color: {CYAN} !important; border-bottom: 2px solid {CYAN} !important;
  }}
  [data-testid="stMetric"] {{
      background: {CARD} !important; border: 1px solid {BORD} !important;
      border-radius: 6px !important; padding: 8px !important;
  }}
  [data-testid="stMetricValue"] {{ color: {TEXT} !important; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  ███  爬蟲模組  ███
# ══════════════════════════════════════════════════════════════════

def _get(url: str, params: dict = None, timeout: int = 8,
         is_json: bool = True):
    """統一 GET 請求，失敗回傳 None"""
    try:
        resp = requests.get(url, headers=HEADERS,
                            params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json() if is_json else resp
    except Exception:
        return None


# ── 1. TWSE 即時報價（盤中）─────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def crawl_realtime_twse(stock_id: str) -> dict:
    """
    證交所即時報價
    回傳 dict: price, change, change_pct, volume, high, low, open, time
    """
    url  = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp"
    data = _get(url, params={"ex_ch": f"tse_{stock_id}.tw", "json": "1",
                              "delay": "0"})
    if not data or not data.get("msgArray"):
        return {}
    m = data["msgArray"][0]
    def _f(k, default=0.0):
        try: return float(m.get(k, default))
        except: return default
    z = _f("z"); y = _f("y")
    return {
        "price":      z,
        "change":     round(z - y, 2),
        "change_pct": round((z - y) / (y + 1e-9) * 100, 2),
        "volume":     int(_f("v") * 1000),
        "high":       _f("h"),
        "low":        _f("l"),
        "open":       _f("o"),
        "prev_close": y,
        "time":       m.get("t", "—"),
        "name":       m.get("n", stock_id),
        "source":     "TWSE即時",
    }


# ── 2. TPEX 即時報價（上櫃）────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def crawl_realtime_tpex(stock_id: str) -> dict:
    """
    櫃買中心即時報價
    """
    url  = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes"
    data = _get(url)
    if not data:
        return {}
    for row in data:
        if row.get("SecuritiesCompanyCode") == stock_id:
            def _f(k):
                try: return float(str(row.get(k, "0")).replace(",", ""))
                except: return 0.0
            z = _f("Close"); y = _f("PreviousClose")
            return {
                "price":      z,
                "change":     round(z - y, 2),
                "change_pct": round((z - y) / (y + 1e-9) * 100, 2),
                "volume":     int(_f("TradingShares")),
                "high":       _f("High"),
                "low":        _f("Low"),
                "open":       _f("Open"),
                "prev_close": y,
                "time":       datetime.now().strftime("%H:%M"),
                "name":       row.get("CompanyName", stock_id),
                "source":     "TPEX即時",
            }
    return {}


# ── 3. 自動判斷上市/上櫃並抓即時報價 ───────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def crawl_realtime(stock_id: str) -> dict:
    """先試 TWSE，失敗改試 TPEX"""
    result = crawl_realtime_twse(stock_id)
    if result and result.get("price", 0) > 0:
        return result
    result = crawl_realtime_tpex(stock_id)
    if result and result.get("price", 0) > 0:
        return result
    return {}


# ── 4. TWSE 每日收盤資料（歷史）────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def crawl_daily_twse(stock_id: str, months: int = 3) -> pd.DataFrame:
    """
    抓取最近 N 個月的每日收盤資料（TWSE 歷史）
    """
    frames = []
    for i in range(months):
        d = datetime.today().replace(day=1) - timedelta(days=i * 28)
        date_str = d.strftime("%Y%m%d")
        url  = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"
        data = _get(url, params={
            "response": "json", "date": date_str, "stockNo": stock_id
        })
        if not data or data.get("stat") != "OK":
            continue
        for row in data.get("data", []):
            try:
                def _clean_num(s):
                    return float(str(s).replace(",", ""))
                frames.append({
                    "Date":   pd.to_datetime(
                        str(int(row[0].split("/")[0]) + 1911) +
                        "-" + row[0].split("/")[1] +
                        "-" + row[0].split("/")[2]
                    ),
                    "Volume": int(str(row[1]).replace(",", "")),
                    "Open":   _clean_num(row[3]),
                    "High":   _clean_num(row[4]),
                    "Low":    _clean_num(row[5]),
                    "Close":  _clean_num(row[6]),
                })
            except Exception:
                continue
        time.sleep(0.3)   # 避免過快請求

    if not frames:
        return pd.DataFrame()
    df = pd.DataFrame(frames).sort_values("Date").reset_index(drop=True)
    df = df.drop_duplicates("Date")
    return df


# ── 5. 融資融券 ─────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def crawl_margin(stock_id: str) -> dict:
    """
    TWSE 融資融券餘額
    回傳 dict: margin_buy, margin_sell, short_sell, short_cover
    """
    for delta in range(0, 5):
        date_str = (datetime.today() - timedelta(days=delta)).strftime("%Y%m%d")
        url  = "https://www.twse.com.tw/exchangeReport/MI_MARGN"
        data = _get(url, params={
            "response": "json", "date": date_str,
            "selectType": "ALL"
        })
        if not data or data.get("stat") != "OK":
            continue
        for row in data.get("data", []):
            if row[0] == stock_id:
                def _p(x):
                    try: return int(str(x).replace(",", ""))
                    except: return 0
                return {
                    "margin_buy":   _p(row[2]),   # 融資買進
                    "margin_sell":  _p(row[3]),   # 融資賣出
                    "margin_bal":   _p(row[4]),   # 融資餘額
                    "short_sell":   _p(row[8]),   # 融券賣出
                    "short_cover":  _p(row[9]),   # 融券買進
                    "short_bal":    _p(row[10]),  # 融券餘額
                    "date":         date_str,
                }
    return {}


# ── 6. 三大法人（升級版）────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def crawl_institutional(stock_id: str) -> dict:
    """
    三大法人買賣超（外資、投信、自營商）
    """
    def _p(x):
        try: return int(str(x).replace(",", "").replace(" ", ""))
        except: return 0

    for delta in range(0, 5):
        date_str = (datetime.today() - timedelta(days=delta)).strftime("%Y%m%d")
        url  = "https://www.twse.com.tw/fund/T86"
        data = _get(url, params={
            "response": "json", "date": date_str,
            "selectType": "ALLBUT0999"
        })
        if not data or data.get("stat") != "OK":
            continue
        for row in data.get("data", []):
            if row[0] == stock_id:
                return {
                    "foreign":  _p(row[4]),   # 外資買賣超
                    "invest":   _p(row[10]),  # 投信買賣超
                    "dealer":   _p(row[14]),  # 自營商買賣超
                    "total":    _p(row[18]) if len(row) > 18 else 0,
                    "date":     date_str,
                }
    return {}


# ── 7. 股票新聞（Yahoo RSS）─────────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def crawl_news(stock_id: str, name: str = "", limit: int = 8) -> list:
    """
    從 Yahoo 財經 RSS 抓取股票相關新聞
    回傳 list of dict: title, link, pubDate, source
    """
    queries = [f"{stock_id}", name[:4] if name else stock_id]
    results = []
    seen    = set()

    for q in queries:
        if len(results) >= limit:
            break
        url  = f"https://tw.news.yahoo.com/rss/search?p={q}+股票"
        resp = _get(url, is_json=False, timeout=6)
        if resp is None:
            continue
        try:
            root = ET.fromstring(resp.content)
            for item in root.iter("item"):
                title   = item.findtext("title", "").strip()
                link    = item.findtext("link",  "").strip()
                pubdate = item.findtext("pubDate", "").strip()
                source  = item.findtext("source", "Yahoo新聞").strip()
                if title and title not in seen:
                    seen.add(title)
                    results.append({
                        "title":   title,
                        "link":    link,
                        "pubDate": pubdate[:16] if pubdate else "—",
                        "source":  source,
                    })
                if len(results) >= limit:
                    break
        except Exception:
            continue

    return results[:limit]


# ── 8. 個股基本面（TWSE 公開資訊觀測站）────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def crawl_fundamental(stock_id: str) -> dict:
    """
    抓取 EPS、殖利率、本益比（TWSE 公開資訊觀測站）
    """
    url  = "https://www.twse.com.tw/exchangeReport/BWIBBU_d"
    data = _get(url, params={
        "response": "json", "stockNo": stock_id,
        "selectType": "ALL"
    })
    if not data or data.get("stat") != "OK":
        return {}
    rows = data.get("data", [])
    if not rows:
        return {}
    # 取最新一筆
    r = rows[-1]
    def _f(x):
        try: return float(str(x).replace(",", ""))
        except: return None
    return {
        "yield_pct":  _f(r[2]),   # 殖利率
        "pe_ratio":   _f(r[4]),   # 本益比
        "pb_ratio":   _f(r[5]),   # 股價淨值比
        "date":       r[0],
    }


# ── 9. 股東持股分級（TWSE）──────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def crawl_shareholder(stock_id: str) -> dict:
    """
    大股東持股比例（前 10 大股東）
    """
    url  = "https://www.twse.com.tw/fund/持股分級"
    # 改用公開資訊觀測站 API
    url2 = (f"https://mops.twse.com.tw/mops/web/ajax_t51sb07"
            f"?encodeURIComponent=1&step=1&firstin=1"
            f"&off=1&keyword4=&code1=&TYPEK=all&isnew=false"
            f"&co_id={stock_id}&year=113&season=4")
    resp = _get(url2, is_json=False, timeout=10)
    if resp is None:
        return {}
    try:
        soup   = BeautifulSoup(resp.content, "lxml")
        tables = soup.find_all("table")
        if not tables:
            return {}
        rows = tables[0].find_all("tr")
        holders = []
        for row in rows[1:6]:   # 取前 5 大
            cols = [c.get_text(strip=True) for c in row.find_all("td")]
            if len(cols) >= 3:
                holders.append({
                    "name":  cols[1] if len(cols) > 1 else "—",
                    "pct":   cols[2] if len(cols) > 2 else "—",
                })
        return {"holders": holders}
    except Exception:
        return {}


# ── 10. 每週籌碼（集保戶股權分散）──────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def crawl_chip_distribution(stock_id: str) -> list:
    """
    集保戶股權分散表（散戶 vs 大戶比例）
    """
    url  = "https://www.tdcc.com.tw/smWeb/QryStockAjax.do"
    data = _get(url, params={
        "SCA_DATE": "latest",
        "SqlMethod": "StockNo",
        "StockNo": stock_id,
        "StockName": "",
    })
    if not data:
        return []
    try:
        rows = data if isinstance(data, list) else data.get("data", [])
        result = []
        for row in rows[:8]:
            result.append({
                "level": row.get("LEVEL", "—"),
                "count": row.get("HOLDER_CNT", 0),
                "pct":   row.get("HOLDER_PCT", "0"),
            })
        return result
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════
#  技術指標
# ══════════════════════════════════════════════════════════════════

def safe_float(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return default if math.isnan(v) or math.isinf(v) else v
    except Exception:
        return default

def ma(s, n):   return s.rolling(n).mean()
def ema(s, n):  return s.ewm(span=n, adjust=False).mean()

def kd(df, n=9):
    lo  = df["Low"].rolling(n).min()
    hi  = df["High"].rolling(n).max()
    rsv = (df["Close"] - lo) / (hi - lo + 1e-9) * 100
    k   = rsv.ewm(com=2, adjust=False).mean()
    return k, k.ewm(com=2, adjust=False).mean()

def macd(s, f=12, sl=26, sig=9):
    m  = ema(s, f) - ema(s, sl)
    sg = ema(m, sig)
    return m, sg, m - sg

def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / (l + 1e-9))

def bb(s, n=20, k=2):
    m   = s.rolling(n).mean()
    std = s.rolling(n).std()
    u, l = m + k*std, m - k*std
    return u, m, l, (u - l) / (m + 1e-9) * 100

def atr(df, n=14):
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"]  - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).cumsum() / (df["Volume"].cumsum() + 1e-9)

def obv(df):
    return (np.sign(df["Close"].diff().fillna(0)) * df["Volume"]).cumsum()

def williams_r(df, n=14):
    hi = df["High"].rolling(n).max()
    lo = df["Low"].rolling(n).min()
    return -100 * (hi - df["Close"]) / (hi - lo + 1e-9)

def cci(df, n=20):
    tp  = (df["High"] + df["Low"] + df["Close"]) / 3
    ma_ = tp.rolling(n).mean()
    md  = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma_) / (0.015 * md + 1e-9)

def pivot_points(df):
    prev = df.iloc[-2]
    H, L, C = float(prev["High"]), float(prev["Low"]), float(prev["Close"])
    P = (H + L + C) / 3
    return {"P": P, "R1": 2*P-L, "R2": P+(H-L),
            "S1": 2*P-H, "S2": P-(H-L)}


# ══════════════════════════════════════════════════════════════════
#  資料抓取
# ══════════════════════════════════════════════════════════════════

def _clean(df):
    df = df.reset_index()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    dc = "Datetime" if "Datetime" in df.columns else "Date"
    df["Date"] = pd.to_datetime(df[dc])
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_all(ticker: str, days: int):
    end   = datetime.today()
    start = end - timedelta(days=max(int(days) + 30, 260))

    def get(iv):
        try:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=True, progress=False, interval=iv)
            return _clean(df) if not df.empty else None
        except Exception:
            return None

    d = get("1d")
    if d is None or d.empty:
        return None, None, None

    d["MA5"]  = ma(d["Close"], 5)
    d["MA10"] = ma(d["Close"], 10)
    d["MA20"] = ma(d["Close"], 20)
    d["MA60"] = ma(d["Close"], 60)
    d["K"], d["D"]                 = kd(d)
    d["MACD"], d["SIG"], d["HIST"] = macd(d["Close"])
    d["RSI"]                       = rsi(d["Close"])
    d["BBU"], d["BBM"], d["BBL"], d["BBW"] = bb(d["Close"])
    d["ATR"]   = atr(d)
    d["VWAP"]  = vwap(d)
    d["OBV"]   = obv(d)
    d["WR"]    = williams_r(d)
    d["CCI"]   = cci(d)

    w = get("1wk")
    if w is not None and len(w) >= 5:
        w["MA5"], w["MA20"]            = ma(w["Close"], 5), ma(w["Close"], 20)
        w["K"], w["D"]                 = kd(w)
        w["MACD"], w["SIG"], w["HIST"] = macd(w["Close"])
        w["RSI"]                       = rsi(w["Close"])

    mo = get("1mo")
    if mo is not None and len(mo) >= 3:
        mo["MA5"]        = ma(mo["Close"], 5)
        mo["K"], mo["D"] = kd(mo)

    return d, w, mo


# ══════════════════════════════════════════════════════════════════
#  輔助分析
# ══════════════════════════════════════════════════════════════════

def win_rate(df, n=10):
    if len(df) < n: return 50
    r = df.tail(n)
    return int((r["Close"] >= r["Open"]).sum() / n * 100)

def board_ratio(df, n=5):
    r   = df.tail(n)
    ext = ((r["Close"] - r["Low"]) / (r["High"] - r["Low"] + 1e-9) * r["Volume"]).sum()
    tot = r["Volume"].sum()
    ep  = int(ext / tot * 100) if tot > 0 else 50
    return 100 - ep, ep

def main_force(df, n=5):
    r  = df.tail(n)
    up = r[r["Close"] >= r["Open"]]["Volume"].mean() or 0
    dn = r[r["Close"] <  r["Open"]]["Volume"].mean() or 0
    if up > dn * 1.3: return "進出見買", "籌碼集中", "高", "穩定",   RED,  "多方偏強"
    if dn > up * 1.3: return "進出見賣", "籌碼分散", "低", "不穩定", GRN,  "空方偏強"
    vt  = r["Volume"].diff().mean()
    txt = "中性偏多" if vt > 0 else "中性偏空"
    return txt, "籌碼適中", "中", "尚穩定", GOLD, "中性整理"

def kline_patterns(df):
    if len(df) < 2: return [("資料不足", "—", MUTED)]
    r     = df.iloc[-1]
    body  = abs(safe_float(r["Close"]) - safe_float(r["Open"]))
    total = safe_float(r["High"]) - safe_float(r["Low"]) + 1e-9
    upper = safe_float(r["High"]) - max(safe_float(r["Close"]), safe_float(r["Open"]))
    lower = min(safe_float(r["Close"]), safe_float(r["Open"])) - safe_float(r["Low"])
    bull  = safe_float(r["Close"]) >= safe_float(r["Open"])
    br    = body / total
    out   = []
    if br > 0.7: out.append(("長紅K" if bull else "長黑K",
                              "多方型態" if bull else "空方型態",
                              RED if bull else GRN))
    if br < 0.1: out.append(("十字線", "觀望訊號", GOLD))
    if lower > 2*body and upper < body:
        out.append(("錘子線", "底部訊號", RED if bull else GOLD))
    if upper > 2*body and lower < body:
        out.append(("流星線", "頂部訊號", GRN))
    if len(df) >= 2:
        prev = df.iloc[-2]
        pb   = abs(safe_float(prev["Close"]) - safe_float(prev["Open"]))
        if (bull and safe_float(r["Open"]) < safe_float(prev["Close"]) and
                safe_float(r["Close"]) > safe_float(prev["Open"]) and body > pb):
            out.append(("多頭吞噬", "強力買進訊號", RED))
        elif (not bull and safe_float(r["Open"]) > safe_float(prev["Close"]) and
              safe_float(r["Close"]) < safe_float(prev["Open"]) and body > pb):
            out.append(("空頭吞噬", "強力賣出訊號", GRN))
    if not out: out.append(("一般K線", "持續觀察", TEXT))
    return out

def chart_patterns(df):
    c = df["Close"].values
    if len(c) < 20: return [("持續整理", GOLD, "資料不足")]
    r = c[-20:]; mid = 10
    l1, l2 = r[:mid].min(), r[mid:].min()
    h1, h2 = r[:mid].max(), r[mid:].max()
    out = []
    if abs(l1-l2)/(l1+1e-9) < 0.03 and r[4:8].max() > l1*1.02:
        out.append(("W底型態", RED, "底部反轉訊號，留意突破"))
    if abs(h1-h2)/(h1+1e-9) < 0.03 and r[4:8].min() < h1*0.98:
        out.append(("M頭型態", GRN, "頭部反轉訊號，留意跌破"))
    highs = [c[-20+i] for i in range(20) if c[-20+i] == max(c[max(0,-20+i-3):-20+i+4])]
    lows  = [c[-20+i] for i in range(20) if c[-20+i] == min(c[max(0,-20+i-3):-20+i+4])]
    if len(highs) >= 2 and len(lows) >= 2:
        if highs[-1] < highs[0] and lows[-1] > lows[0]:
            out.append(("三角收斂", CYAN, "整理末段，留意方向突破"))
    if not out: out.append(("持續整理", GOLD, "未形成明確型態，等待方向"))
    return out

def trend_label(df_tf, tf_name):
    if df_tf is None or len(df_tf) < 3:
        return tf_name, "資料不足", MUTED, "—"
    r, p  = df_tf.iloc[-1], df_tf.iloc[-2]
    chg   = (safe_float(r["Close"]) - safe_float(p["Close"])) / (safe_float(p["Close"]) + 1e-9) * 100
    kv    = safe_float(r.get("K", 50), 50) if "K" in df_tf.columns else 50
    mv    = safe_float(r.get("MACD", 0)) if "MACD" in df_tf.columns else 0
    sv    = safe_float(r.get("SIG",  0)) if "SIG"  in df_tf.columns else 0
    ma5v  = safe_float(r.get("MA5", safe_float(r["Close"]))) if "MA5" in df_tf.columns else safe_float(r["Close"])
    trend = "多頭" if safe_float(r["Close"]) > ma5v else "空頭"
    col   = RED if trend == "多頭" else GRN
    kd_s  = "高鈍" if kv > 80 else ("低超" if kv < 20 else "中")
    ma_s  = "黃金" if mv > sv else "死亡"
    return tf_name, f"{trend} ({chg:+.1f}%)", col, f"KD:{kd_s} MACD:{ma_s}交叉"

def check_alerts(df, close, kv, dv, rv, mv, sv):
    alerts = []
    if len(df) < 3: return alerts
    prev_k = safe_float(df.iloc[-2].get("K",    50), 50)
    prev_d = safe_float(df.iloc[-2].get("D",    50), 50)
    prev_m = safe_float(df.iloc[-2].get("MACD",  0))
    prev_s = safe_float(df.iloc[-2].get("SIG",   0))
    if prev_k < prev_d and kv > dv: alerts.append(("🟢 KD黃金交叉", "買進訊號", GRN))
    if prev_k > prev_d and kv < dv: alerts.append(("🔴 KD死亡交叉", "賣出訊號", RED))
    if rv > 80:   alerts.append(("⚠️ RSI極度超買", f"RSI={rv:.1f}", RED))
    elif rv > 70: alerts.append(("⚠️ RSI超買",     f"RSI={rv:.1f}", ORNG))
    elif rv < 20: alerts.append(("✅ RSI極度超賣", f"RSI={rv:.1f}", GRN))
    elif rv < 30: alerts.append(("✅ RSI超賣",     f"RSI={rv:.1f}", CYAN))
    if "BBU" in df.columns:
        bbu = safe_float(df.iloc[-1]["BBU"])
        bbl = safe_float(df.iloc[-1]["BBL"])
        if close > bbu: alerts.append(("📈 突破布林上軌", f"{close:.0f}>{bbu:.0f}", RED))
        if close < bbl: alerts.append(("📉 跌破布林下軌", f"{close:.0f}<{bbl:.0f}", GRN))
        if "BBW" in df.columns:
            bbw = safe_float(df.iloc[-1]["BBW"])
            if bbw < df["BBW"].quantile(0.15):
                alerts.append(("🔔 布林帶極度收縮", "即將出現大波動", GOLD))
    if prev_m < prev_s and mv > sv: alerts.append(("📈 MACD黃金交叉", "DIF上穿DEA", RED))
    if prev_m > prev_s and mv < sv: alerts.append(("📉 MACD死亡交叉", "DIF下穿DEA", GRN))
    vol  = safe_float(df.iloc[-1]["Volume"])
    avg5 = safe_float(df["Volume"].tail(5).mean())
    if avg5 > 0:
        ratio = vol / avg5
        if ratio > 3.0: alerts.append(("🔥 超級爆量", f"{ratio:.1f}x均量", ORNG))
        elif ratio > 2.0: alerts.append(("🔥 爆量訊號", f"{ratio:.1f}x均量", ORNG))
    if "WR" in df.columns:
        wr_val = safe_float(df.iloc[-1]["WR"])
        if wr_val > -10: alerts.append(("⚡ 威廉%R超買", f"WR={wr_val:.1f}", RED))
        elif wr_val < -90: alerts.append(("⚡ 威廉%R超賣", f"WR={wr_val:.1f}", GRN))
    if "OBV" in df.columns and len(df) >= 5:
        obv_chg   = safe_float(df["OBV"].iloc[-1]) - safe_float(df["OBV"].iloc[-5])
        price_chg = safe_float(df["Close"].iloc[-1]) - safe_float(df["Close"].iloc[-5])
        if price_chg > 0 and obv_chg < 0:
            alerts.append(("⚠️ OBV價量背離", "價漲量縮，動能不足", ORNG))
        elif price_chg < 0 and obv_chg > 0:
            alerts.append(("✅ OBV正向背離", "價跌量增，底部蓄積", CYAN))
    if "VWAP" in df.columns:
        vwap_val = safe_float(df.iloc[-1]["VWAP"])
        prev_c   = safe_float(df.iloc[-2]["Close"])
        if prev_c < vwap_val and close > vwap_val:
            alerts.append(("📈 突破VWAP", f"{close:.0f}站上{vwap_val:.0f}", RED))
        elif prev_c > vwap_val and close < vwap_val:
            alerts.append(("📉 跌破VWAP", f"{close:.0f}跌破{vwap_val:.0f}", GRN))
    return alerts

def fibonacci_levels(df, n=60):
    r    = df.tail(n)
    hi   = float(r["High"].max())
    lo   = float(r["Low"].min())
    diff = hi - lo
    return {
        "0.0% (高點)": hi,  "23.6%": hi-diff*0.236,
        "38.2%": hi-diff*0.382, "50.0%": hi-diff*0.500,
        "61.8%": hi-diff*0.618, "78.6%": hi-diff*0.786,
        "100% (低點)": lo,
    }

def ai_summary(close, ma5, ma20, ma60, kv, dv, rv, mv, sv,
               hv, atr_v, wr_v=-50, cci_v=0, bbw_v=5):
    signals = []
    if ma5 > ma20 > ma60:   signals.append(("多", "均線多頭排列"))
    elif ma5 < ma20 < ma60: signals.append(("空", "均線空頭排列"))
    elif ma5 > ma20:        signals.append(("多", "短均線上穿中均線"))
    else:                   signals.append(("空", "短均線下穿中均線"))
    if kv > 80:             signals.append(("空", "KD高檔鈍化注意"))
    elif kv < 20:           signals.append(("多", "KD低檔超賣反彈"))
    elif kv > dv:           signals.append(("多", "KD黃金交叉"))
    else:                   signals.append(("空", "KD死亡交叉"))
    if rv < 30:             signals.append(("多", "RSI超賣反彈機會"))
    elif rv > 70:           signals.append(("空", "RSI超買注意拉回"))
    elif rv > 50:           signals.append(("多", "RSI強勢區間"))
    else:                   signals.append(("空", "RSI弱勢區間"))
    if mv > sv and hv > 0:  signals.append(("多", "MACD黃金交叉擴張"))
    elif mv > sv:           signals.append(("多", "MACD黃金交叉收斂"))
    elif mv < sv and hv < 0:signals.append(("空", "MACD死亡交叉擴張"))
    else:                   signals.append(("空", "MACD死亡交叉收斂"))
    if close > ma20:        signals.append(("多", "站上MA20"))
    else:                   signals.append(("空", "跌破MA20"))
    if wr_v < -80:          signals.append(("多", "威廉%R超賣區"))
    elif wr_v > -20:        signals.append(("空", "威廉%R超買區"))
    if cci_v < -100:        signals.append(("多", "CCI超賣反彈"))
    elif cci_v > 100:       signals.append(("空", "CCI超買回落"))
    bull  = sum(1 for s, _ in signals if s == "多")
    score = int(bull / max(len(signals), 1) * 100)
    at    = atr_v if atr_v and not math.isnan(atr_v) else close * 0.025
    vm    = 1.5 if bbw_v > 10 else 1.0
    if score >= 70:
        return "📈 偏多操作，可考慮布局", score, signals, close-at*1.5, close+at*2.5*vm, RED
    elif score <= 30:
        return "📉 偏空操作，謹慎持股", score, signals, close+at*1.0, close-at*2.0*vm, GRN
    return "⚖️ 中性觀望，等待訊號明朗", score, signals, close-at, close+at, GOLD


# ══════════════════════════════════════════════════════════════════
#  匯出工具
# ══════════════════════════════════════════════════════════════════

def export_csv(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue().encode("utf-8-sig")

def export_summary(ticker, name, close, pct, kv, rv, mv, sv,
                   verdict, score, signals, stop, target,
                   margin_data=None, inst_data=None):
    rr = abs(target-close) / (abs(close-stop) + 1e-9)
    lines = [
        "=" * 50,
        f"  {name}（{ticker}）技術分析報告 v3.0",
        "=" * 50,
        f"日期　　：{datetime.today().strftime('%Y-%m-%d %H:%M')}",
        f"收盤價　：{close:,.2f}　漲跌：{pct:+.2f}%",
        f"KD　　　：{kv:.1f}　RSI：{rv:.1f}",
        f"MACD　　：{'黃金交叉' if mv > sv else '死亡交叉'}",
        "",
        f"【AI 訊號評分】{score}/100",
        f"建議　　：{verdict}",
        f"目標價　：{target:,.2f}",
        f"停損價　：{stop:,.2f}",
        f"風報比　：{rr:.2f}",
        "",
    ]
    if inst_data:
        lines += [
            "【三大法人】",
            f"  外資：{inst_data.get('foreign', '—'):,}",
            f"  投信：{inst_data.get('invest',  '—'):,}",
            f"  自營：{inst_data.get('dealer',  '—'):,}",
            "",
        ]
    if margin_data:
        lines += [
            "【融資融券】",
            f"  融資餘額：{margin_data.get('margin_bal', '—'):,}",
            f"  融券餘額：{margin_data.get('short_bal',  '—'):,}",
            "",
        ]
    lines += ["【訊號明細】"]
    for s, desc in signals:
        lines.append(f"  {'▲' if s == '多' else '▼'} {desc}")
    lines += ["", "─"*50, "⚠ 本報告僅供學習研究，不構成投資建議。", "─"*50]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
#  Plotly 圖表
# ══════════════════════════════════════════════════════════════════

def _base_layout(height, margin=None):
    return dict(
        template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=BG,
        height=height, margin=margin or dict(l=52, r=6, t=8, b=6),
        font=dict(family="Arial", color=TEXT, size=10),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.005, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=9.5)),
    )

def build_main(df, show_bb, show_fib, show_vwap=True):
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.52, 0.16, 0.16, 0.16],
        vertical_spacing=0.008,
    )
    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="K線",
        increasing=dict(line=dict(color=RED, width=0.8), fillcolor=RED),
        decreasing=dict(line=dict(color=GRN, width=0.8), fillcolor=GRN),
        whiskerwidth=0.4,
    ), row=1, col=1)
    for n, c, w in [("MA5","MA5",GOLD),("MA20","MA20",ORNG),("MA60","MA60",CYAN)]:
        if c in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"], y=df[c], name=n,
                mode="lines", line=dict(color=w, width=1.2)), row=1, col=1)
    if show_vwap and "VWAP" in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["VWAP"], name="VWAP",
            mode="lines", line=dict(color=PURP, width=1.3, dash="dot")), row=1, col=1)
    if show_bb and "BBU" in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["BBU"], name="BB上",
            mode="lines", line=dict(color="rgba(160,80,255,.7)", width=1, dash="dot")),
            row=1, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=df["BBL"], name="BB下",
            mode="lines", line=dict(color="rgba(160,80,255,.7)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(160,80,255,.06)"), row=1, col=1)
    if show_fib:
        fib_c = {"0.0% (高點)":"rgba(255,48,48,0.6)","23.6%":"rgba(255,215,0,0.5)",
                 "38.2%":"rgba(0,200,100,0.6)","50.0%":"rgba(0,200,255,0.6)",
                 "61.8%":"rgba(0,200,100,0.6)","78.6%":"rgba(160,80,255,0.5)",
                 "100% (低點)":"rgba(0,200,100,0.6)"}
        for label, price in fibonacci_levels(df).items():
            fig.add_hline(y=price, line_dash="dash",
                line_color=fib_c.get(label,"rgba(255,215,0,0.4)"), line_width=1,
                annotation_text=f" Fib {label} {price:.0f}",
                annotation_font=dict(size=8, color=MUTED),
                annotation_position="right", row=1, col=1)
    if len(df) >= 2:
        pp = pivot_points(df)
        for lbl, prc, clr in [("P",pp["P"],GOLD),("R1",pp["R1"],RED),("S1",pp["S1"],GRN)]:
            fig.add_hline(y=prc, line_dash="longdash", line_color=clr,
                line_width=0.8, opacity=0.5,
                annotation_text=f" {lbl} {prc:.0f}",
                annotation_font=dict(size=7, color=clr),
                annotation_position="right", row=1, col=1)
    hi_i = int(df["High"].idxmax()); lo_i = int(df["Low"].idxmin())
    fig.add_annotation(x=df.loc[hi_i,"Date"], y=float(df.loc[hi_i,"High"]),
        text=f"近高 {float(df.loc[hi_i,'High']):.0f}",
        showarrow=True, arrowhead=2, arrowcolor=RED, ay=-25,
        font=dict(color=RED, size=9), bgcolor="rgba(255,48,48,.15)",
        bordercolor=RED, borderwidth=1, row=1, col=1)
    fig.add_annotation(x=df.loc[lo_i,"Date"], y=float(df.loc[lo_i,"Low"]),
        text=f"近低 {float(df.loc[lo_i,'Low']):.0f}",
        showarrow=True, arrowhead=2, arrowcolor=GRN, ay=25,
        font=dict(color=GRN, size=9), bgcolor="rgba(0,200,100,.15)",
        bordercolor=GRN, borderwidth=1, row=1, col=1)
    bar_c = [RED if c >= o else GRN for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="量",
        marker_color=bar_c, opacity=0.75), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Volume"].rolling(5).mean(),
        name="均量5", mode="lines", line=dict(color=GOLD, width=1.1)), row=2, col=1)
    if "K" in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["K"], name="K",
            mode="lines", line=dict(color=GOLD, width=1.3)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=df["D"], name="D",
            mode="lines", line=dict(color=ORNG, width=1.3)), row=3, col=1)
        for lvl, clr in [(80,RED),(50,"#334466"),(20,GRN)]:
            fig.add_hline(y=lvl, line_dash="dot", line_color=clr,
                          opacity=0.35, row=3, col=1)
    if "HIST" in df.columns:
        h = df["HIST"].fillna(0)
        fig.add_trace(go.Bar(x=df["Date"], y=h,
            marker_color=[RED if v >= 0 else GRN for v in h],
            opacity=0.7, name="柱"), row=4, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"],
            mode="lines", line=dict(color=CYAN, width=1.2), name="DIF"), row=4, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=df["SIG"],
            mode="lines", line=dict(color=GOLD, width=1.2), name="DEA"), row=4, col=1)
    layout = _base_layout(620)
    layout["xaxis_rangeslider_visible"] = False
    fig.update_layout(**layout)
    gc = "#142030"
    for i in range(1, 5):
        fig.update_xaxes(gridcolor=gc, zeroline=False, row=i, col=1,
                         showspikes=True, spikecolor="#33558A")
        fig.update_yaxes(gridcolor=gc, zeroline=False, row=i, col=1)
    return fig

def build_gauge(value):
    if   value >= 65: clr, lbl = RED,  "偏多"
    elif value >= 50: clr, lbl = GOLD, "中等偏多"
    elif value >= 35: clr, lbl = ORNG, "中等偏空"
    else:             clr, lbl = GRN,  "偏空"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number=dict(suffix="%", font=dict(size=30, color=clr, family="Arial")),
        title=dict(text=lbl, font=dict(size=12, color=MUTED)),
        gauge=dict(
            axis=dict(range=[0,100], tickfont=dict(size=9, color=MUTED)),
            bar=dict(color=clr, thickness=0.28),
            bgcolor=BG, borderwidth=0,
            steps=[dict(range=[0,35],color="#091A10"),
                   dict(range=[35,65],color="#141408"),
                   dict(range=[65,100],color="#1A0909")],
            threshold=dict(line=dict(color="#fff",width=2),thickness=0.7,value=value),
        ),
    ))
    fig.update_layout(paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                      height=200, margin=dict(l=15,r=15,t=30,b=10),
                      font=dict(family="Arial", color=TEXT))
    return fig

def build_obv_chart(df):
    if "OBV" not in df.columns: return None
    obv_ma = df["OBV"].rolling(10).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"].tail(60), y=df["OBV"].tail(60),
        mode="lines", line=dict(color=CYAN, width=1.3), name="OBV",
        fill="tozeroy", fillcolor="rgba(0,200,255,0.06)"))
    fig.add_trace(go.Scatter(x=df["Date"].tail(60), y=obv_ma.tail(60),
        mode="lines", line=dict(color=GOLD, width=1.1, dash="dot"), name="OBV MA10"))
    layout = _base_layout(150, dict(l=45,r=6,t=18,b=6))
    layout.update(xaxis=dict(gridcolor="#142030",zeroline=False),
                  yaxis=dict(gridcolor="#142030",zeroline=False))
    fig.update_layout(**layout)
    return fig


def build_wr_cci_chart(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], vertical_spacing=0.05)
    n = 40
    if "WR" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Date"].tail(n), y=df["WR"].tail(n),
            mode="lines", line=dict(color=PURP, width=1.3), name="威廉%R"
        ), row=1, col=1)
        for lvl, clr in [(-20, RED), (-50, MUTED), (-80, GRN)]:
            fig.add_hline(y=lvl, line_dash="dot", line_color=clr,
                          opacity=0.4, row=1, col=1)

    if "CCI" in df.columns:
        cv = df["CCI"].tail(n)
        fig.add_trace(go.Bar(
            x=df["Date"].tail(n), y=cv,
            marker_color=[RED if v > 0 else GRN for v in cv],
            opacity=0.7, name="CCI"
        ), row=2, col=1)
        for lvl, clr in [(100, RED), (0, MUTED), (-100, GRN)]:
            fig.add_hline(y=lvl, line_dash="dot", line_color=clr,
                          opacity=0.4, row=2, col=1)

    layout = _base_layout(180, dict(l=45, r=6, t=18, b=6))
    layout.update(legend=dict(orientation="h", y=1.08,
                               bgcolor="rgba(0,0,0,0)", font=dict(size=9)))
    fig.update_layout(**layout)
    gc = "#142030"
    for i in range(1, 3):
        fig.update_xaxes(gridcolor=gc, zeroline=False, row=i, col=1)
        fig.update_yaxes(gridcolor=gc, zeroline=False, row=i, col=1)
    return fig


def build_ai_radar(signals):
    categories  = ["均線", "KD", "RSI", "MACD", "量價", "威廉", "CCI"]
    keyword_map = {
        "均線": ["均線多頭", "均線空頭", "短均線"],
        "KD":   ["KD低檔",   "KD高檔",   "KD黃金", "KD死亡"],
        "RSI":  ["RSI超賣",  "RSI超買",  "RSI強勢", "RSI弱勢"],
        "MACD": ["MACD黃金", "MACD死亡"],
        "量價": ["站上MA20", "跌破MA20"],
        "威廉": ["威廉%R超賣", "威廉%R超買"],
        "CCI":  ["CCI超賣",  "CCI超買"],
    }
    vals = []
    for cat, keys in keyword_map.items():
        matched = [(s, d) for s, d in signals if any(k in d for k in keys)]
        vals.append(80 if (matched and matched[0][0] == "多") else
                    20 if matched else 50)
    vals_c = vals + vals[:1]
    cats_c = categories + categories[:1]
    fig = go.Figure(go.Scatterpolar(
        r=vals_c, theta=cats_c, fill="toself",
        fillcolor="rgba(0,200,255,0.12)",
        line=dict(color=CYAN, width=1.5),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=BG,
            radialaxis=dict(visible=True, range=[0, 100],
                            gridcolor=BORD, tickfont=dict(size=8, color=MUTED)),
            angularaxis=dict(gridcolor=BORD, tickfont=dict(size=9, color=TEXT)),
        ),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        height=240, margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False, font=dict(family="Arial", color=TEXT),
    )
    return fig


def build_bbw_chart(df):
    if "BBW" not in df.columns:
        return None
    n   = 60
    bbw = df["BBW"].tail(n)
    q15 = bbw.quantile(0.15)
    q85 = bbw.quantile(0.85)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"].tail(n), y=bbw,
        mode="lines", line=dict(color=PURP, width=1.3), name="BBW%",
        fill="tozeroy", fillcolor="rgba(160,80,255,0.06)"
    ))
    fig.add_hline(y=q15, line_dash="dot", line_color=GRN, opacity=0.5,
                  annotation_text=f"低波動 {q15:.1f}%",
                  annotation_font=dict(size=8, color=GRN))
    fig.add_hline(y=q85, line_dash="dot", line_color=RED, opacity=0.5,
                  annotation_text=f"高波動 {q85:.1f}%",
                  annotation_font=dict(size=8, color=RED))
    layout = _base_layout(130, dict(l=45, r=6, t=18, b=6))
    layout.update(
        title=dict(text="布林帶寬度 BBW%",
                   font=dict(size=10, color=MUTED), x=0.01),
        xaxis=dict(gridcolor="#142030", zeroline=False),
        yaxis=dict(gridcolor="#142030", zeroline=False),
        showlegend=False,
    )
    fig.update_layout(**layout)
    return fig


def build_margin_chart(margin_hist: list):
    """融資融券歷史趨勢圖（傳入 list of dict）"""
    if not margin_hist:
        return None
    df_m = pd.DataFrame(margin_hist)
    if df_m.empty or "date" not in df_m.columns:
        return None
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], vertical_spacing=0.05)
    if "margin_bal" in df_m.columns:
        fig.add_trace(go.Bar(
            x=df_m["date"], y=df_m["margin_bal"],
            marker_color=CYAN, opacity=0.75, name="融資餘額"
        ), row=1, col=1)
    if "short_bal" in df_m.columns:
        fig.add_trace(go.Bar(
            x=df_m["date"], y=df_m["short_bal"],
            marker_color=ORNG, opacity=0.75, name="融券餘額"
        ), row=2, col=1)
    layout = _base_layout(180, dict(l=55, r=6, t=18, b=6))
    layout.update(legend=dict(orientation="h", y=1.08,
                               bgcolor="rgba(0,0,0,0)", font=dict(size=9)))
    fig.update_layout(**layout)
    gc = "#142030"
    for i in range(1, 3):
        fig.update_xaxes(gridcolor=gc, zeroline=False, row=i, col=1)
        fig.update_yaxes(gridcolor=gc, zeroline=False, row=i, col=1)
    return fig


def build_chip_dist_chart(chip_data: list):
    """集保戶股權分散圓餅圖"""
    if not chip_data:
        return None
    labels = [r.get("level", "—") for r in chip_data]
    values = []
    for r in chip_data:
        try:
            values.append(float(str(r.get("pct", "0")).replace("%", "")))
        except Exception:
            values.append(0.0)
    colors = [RED, ORNG, GOLD, CYAN, GRN, PURP, TEXT, MUTED]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.55,
        marker=dict(colors=colors[:len(labels)],
                    line=dict(color=BG, width=1.5)),
        textfont=dict(size=9, color=TEXT),
        hovertemplate="%{label}<br>%{value:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        height=220, margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True,
        legend=dict(font=dict(size=8, color=MUTED),
                    bgcolor="rgba(0,0,0,0)"),
        font=dict(family="Arial", color=TEXT),
        annotations=[dict(text="持股\n分散", x=0.5, y=0.5,
                          font=dict(size=10, color=CYAN),
                          showarrow=False)],
    )
    return fig


# ══════════════════════════════════════════════════════════════════
#  HTML 小工具
# ══════════════════════════════════════════════════════════════════

def hdr(title, icon=""):
    return (f'<div style="color:{CYAN};font-size:.76rem;font-weight:700;'
            f'padding:3px 0 5px;border-bottom:1px solid {BORD};margin-bottom:7px;">'
            f'{icon} {title}</div>')

def row_item(lbl, val, vc=TEXT, bg_c=CARD):
    return (f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:3px 8px;border-radius:3px;background:{bg_c};margin:2px 0;">'
            f'<span style="color:{MUTED};font-size:.73rem;">{lbl}</span>'
            f'<span style="color:{vc};font-weight:700;font-size:.8rem;">{val}</span></div>')

def box(content, title="", icon=""):
    h = hdr(title, icon) if title else ""
    return (f'<div style="background:{CARD};border:1px solid {BORD};border-radius:6px;'
            f'padding:9px 11px;margin-bottom:7px;font-family:Arial,sans-serif;">'
            f'{h}{content}</div>')

def tag(txt, fg, bg_c):
    return (f'<span style="background:{bg_c};color:{fg};border-radius:3px;'
            f'padding:2px 7px;font-size:.72rem;font-weight:700;'
            f'margin:2px 2px 2px 0;display:inline-block;">{txt}</span>')


# ══════════════════════════════════════════════════════════════════
#  爬蟲資料 HTML 面板
# ══════════════════════════════════════════════════════════════════

def html_realtime(rt: dict) -> str:
    """即時報價面板"""
    if not rt:
        return box(f'<div style="color:{MUTED};font-size:.73rem;text-align:center;">'
                   f'⚠ 即時報價暫無資料（盤後/假日）</div>', "即時報價", "⚡")
    price  = rt.get("price", 0)
    chg    = rt.get("change", 0)
    pct    = rt.get("change_pct", 0)
    cc     = RED if chg >= 0 else GRN
    sym    = "▲" if chg >= 0 else "▼"
    source = rt.get("source", "—")
    t_str  = rt.get("time", "—")
    price_html = (
        f'<div style="text-align:center;padding:6px 0;">'
        f'<div style="color:{cc};font-size:2rem;font-weight:900;'
        f'letter-spacing:1px;">{price:,.2f}</div>'
        f'<div style="color:{cc};font-size:.85rem;font-weight:700;">'
        f'{sym} {abs(chg):.2f}（{abs(pct):.2f}%）</div>'
        f'<div style="color:{MUTED};font-size:.65rem;margin-top:3px;">'
        f'{source} · {t_str}</div></div>'
    )
    rows = "".join([
        row_item("開盤", f"{rt.get('open',  0):,.2f}"),
        row_item("最高", f"{rt.get('high',  0):,.2f}", RED),
        row_item("最低", f"{rt.get('low',   0):,.2f}", GRN),
        row_item("昨收", f"{rt.get('prev_close', 0):,.2f}"),
        row_item("成交量", f"{rt.get('volume', 0):,}"),
    ])
    return box(price_html + rows, "即時報價", "⚡")


def html_margin(margin: dict) -> str:
    """融資融券面板"""
    if not margin:
        return box(f'<div style="color:{MUTED};font-size:.73rem;text-align:center;">'
                   f'⚠ 融資融券資料暫無</div>', "融資融券", "💹")
    mb  = margin.get("margin_bal",  0)
    sb  = margin.get("short_bal",   0)
    mb_buy  = margin.get("margin_buy",  0)
    mb_sell = margin.get("margin_sell", 0)
    sc  = margin.get("short_sell",  0)
    sco = margin.get("short_cover", 0)
    dt  = margin.get("date", "—")

    # 融資使用率（簡易估算）
    ratio_color = RED if mb > sb * 3 else (GOLD if mb > sb else GRN)

    bar_ratio = min(int(mb / (mb + sb + 1) * 100), 100)
    bar_html = (
        f'<div style="background:#0A1428;border-radius:4px;overflow:hidden;'
        f'height:6px;margin:6px 0;">'
        f'<div style="background:{CYAN};height:100%;width:{bar_ratio}%;"></div>'
        f'</div>'
    )
    rows = "".join([
        row_item("融資買進",  f"{mb_buy:,}",  RED),
        row_item("融資賣出",  f"{mb_sell:,}", GRN),
        row_item("融資餘額",  f"{mb:,}",      ratio_color),
        row_item("融券賣出",  f"{sc:,}",      GRN),
        row_item("融券買進",  f"{sco:,}",     RED),
        row_item("融券餘額",  f"{sb:,}",      ORNG),
        row_item("資券比",
                 f"{mb/(sb+1):.1f}x" if sb > 0 else "—",
                 RED if mb > sb * 2 else GOLD),
        row_item("資料日期", dt, MUTED),
    ])
    return box(bar_html + rows, "融資融券", "💹")


def html_institutional(inst: dict) -> str:
    """三大法人面板"""
    if not inst:
        return box(f'<div style="color:{MUTED};font-size:.73rem;text-align:center;">'
                   f'⚠ 三大法人資料暫無</div>', "三大法人", "🏦")

    def fmt(v):
        if v is None: return "—"
        return f"+{v:,}" if v >= 0 else f"{v:,}"

    def clr(v):
        return RED if (v or 0) >= 0 else GRN

    foreign = inst.get("foreign", None)
    invest  = inst.get("invest",  None)
    dealer  = inst.get("dealer",  None)
    total   = inst.get("total",   None)
    dt      = inst.get("date",    "—")

    # 合力方向
    vals   = [v for v in [foreign, invest, dealer] if v is not None]
    net    = sum(vals)
    net_c  = RED if net >= 0 else GRN
    net_s  = f"合計 {fmt(int(net))}" if vals else "—"

    rows = "".join([
        row_item("外資買賣超",   fmt(foreign), clr(foreign)),
        row_item("投信買賣超",   fmt(invest),  clr(invest)),
        row_item("自營商買賣超", fmt(dealer),  clr(dealer)),
        row_item("三大合計",     net_s,        net_c),
        row_item("資料日期",     dt,           MUTED),
    ])
    return box(rows, "三大法人", "🏦")


def html_fundamental(fund: dict, pe_yf=None, mktcap=None) -> str:
    """基本面面板（整合 TWSE + yfinance）"""
    pe_twse  = fund.get("pe_ratio",  None)
    pb_twse  = fund.get("pb_ratio",  None)
    yld      = fund.get("yield_pct", None)
    dt       = fund.get("date",      "—")

    pe_s  = f"{pe_twse:.1f}x"  if pe_twse  else (f"{pe_yf:.1f}x" if pe_yf else "—")
    pb_s  = f"{pb_twse:.2f}x"  if pb_twse  else "—"
    yld_s = f"{yld:.2f}%"      if yld      else "—"
    mc_s  = f"{mktcap/1e8:,.0f} 億" if mktcap else "—"

    rows = "".join([
        row_item("本益比 (P/E)",  pe_s,  GOLD),
        row_item("股價淨值比",    pb_s,  CYAN),
        row_item("殖利率",        yld_s, GRN),
        row_item("市值",          mc_s,  TEXT),
        row_item("資料日期",      dt,    MUTED),
    ])
    return box(rows, "基本面數據", "📊")


def html_news(news_list: list) -> str:
    """新聞面板"""
    if not news_list:
        return box(f'<div style="color:{MUTED};font-size:.73rem;text-align:center;">'
                   f'⚠ 暫無相關新聞</div>', "最新消息", "📰")
    items = ""
    for n in news_list:
        title   = n.get("title",   "—")
        link    = n.get("link",    "#")
        pubdate = n.get("pubDate", "—")
        source  = n.get("source",  "—")
        items += (
            f'<div style="padding:5px 0;border-bottom:1px solid {BORD};">'
            f'<a href="{link}" target="_blank" style="color:{TEXT};'
            f'text-decoration:none;font-size:.74rem;line-height:1.4;">'
            f'{title}</a>'
            f'<div style="color:{MUTED};font-size:.64rem;margin-top:2px;">'
            f'{source} · {pubdate}</div></div>'
        )
    return box(items, "最新消息", "📰")


def html_ohlcv(df):
    r   = df.iloc[-1]; p = df.iloc[-2]
    cl  = safe_float(r["Close"]); pc = safe_float(p["Close"])
    chg = cl - pc; pct = chg / (pc + 1e-9) * 100
    cc  = RED if chg >= 0 else GRN; sym = "▲" if chg >= 0 else "▼"
    fields = [
        ("開", f"{safe_float(r['Open']):,.0f}"),
        ("高", f"{safe_float(r['High']):,.0f}"),
        ("低", f"{safe_float(r['Low']):,.0f}"),
        ("收", f"{cl:,.0f}"),
        ("量", f"{int(safe_float(r['Volume'])):,}"),
        ("漲跌", f"{sym}{abs(pct):.2f}%"),
    ]
    cells = "".join(
        f'<div style="text-align:center;flex:1;min-width:55px;">'
        f'<div style="color:{MUTED};font-size:.65rem;">{lb}</div>'
        f'<div style="color:{cc if lb=="漲跌" else TEXT};'
        f'font-size:.83rem;font-weight:700;">{v}</div></div>'
        for lb, v in fields
    )
    return box(f'<div style="display:flex;gap:4px;flex-wrap:wrap;">{cells}</div>')


def html_tech(df, ma5, ma20, ma60, kv, dv, rv, mv, sv, hv):
    if   ma5>ma20>ma60: tr,tc="多頭趨勢",RED
    elif ma5<ma20<ma60: tr,tc="空頭趨勢",GRN
    else:               tr,tc="盤整趨勢",GOLD
    if   ma5>ma20>ma60: ar,ac="多頭排列(5>20>60)",RED
    elif ma5<ma20<ma60: ar,ac="空頭排列(5<20<60)",GRN
    else:               ar,ac="交叉整理中",GOLD
    if   kv>80: ks,kc="KD高檔鈍化⚠",RED
    elif kv<20: ks,kc="KD低檔超賣✓",GRN
    else:       ks,kc="KD正常區間",GOLD
    if   rv>70: rs,rc=f"{rv:.0f} 超買",RED
    elif rv<30: rs,rc=f"{rv:.0f} 超賣",GRN
    else:       rs,rc=f"{rv:.0f}",TEXT
    macd_s = ("正值收斂" if mv>0 and hv>0 else "正值擴張" if mv>0
              else "負值收斂" if hv>0 else "負值擴張")
    macd_c  = RED if mv>0 else GRN
    cross   = "黃金交叉📈" if mv>sv else "死亡交叉📉"
    cross_c = RED if mv>sv else GRN
    cl  = safe_float(df.iloc[-1]["Close"]); pc = safe_float(df.iloc[-2]["Close"])
    vol = safe_float(df.iloc[-1]["Volume"]); av = safe_float(df["Volume"].tail(5).mean())
    vr  = vol/(av+1e-9)
    vs  = "量能擴張🔥" if vr>1.2 else ("量能萎縮" if vr<0.8 else "量能持平")
    vc  = RED if "擴" in vs else (GRN if "縮" in vs else GOLD)
    up  = cl>pc
    vp  = ("量增價漲✓" if up and vr>1 else "量縮價跌" if not up and vr<1 else "量價背離⚠")
    vpc = RED if "漲" in vp else (GRN if "跌" in vp else GOLD)
    vwap_s,vwap_c = "—",MUTED
    if "VWAP" in df.columns:
        vv = safe_float(df.iloc[-1]["VWAP"])
        if vv>0:
            vwap_s = f"站上{vv:,.0f}" if cl>vv else f"跌破{vv:,.0f}"
            vwap_c = RED if cl>vv else GRN
    return box("".join([
        row_item("趨勢方向",tr,tc), row_item("MA狀態",ar,ac),
        row_item("KD指標",ks,kc),   row_item("MACD",macd_s,macd_c),
        row_item("MACD交叉",cross,cross_c), row_item("RSI(14)",rs,rc),
        row_item("成交量",vs,vc),   row_item("量價關係",vp,vpc),
        row_item("VWAP",vwap_s,vwap_c),
    ]), "技術分析總覽", "◉")


def html_key_levels(df, close):
    r20=df.tail(20); r60=df.tail(60)
    res1=float(r20["High"].max()); res2=float(r60["High"].max())
    sup1=float(r20["Low"].min());  sup2=float(r60["Low"].min())
    pp = pivot_points(df) if len(df)>=2 else {}

    def lvl_box(label, val, col, bg_c, dist):
        return (f'<div style="background:{bg_c};border-radius:4px;'
                f'padding:6px 10px;margin:3px 0;border-left:3px solid {col};">'
                f'<div style="display:flex;justify-content:space-between;">'
                f'<span style="color:{MUTED};font-size:.71rem;">{label}</span>'
                f'<span style="color:{col};font-weight:900;font-size:.9rem;">'
                f'{val:,.0f}</span></div>'
                f'<div style="color:{MUTED};font-size:.67rem;">'
                f'距現價 {dist:+.1f}%</div></div>')

    cur = (f'<div style="text-align:center;padding:4px 0;">'
           f'<span style="color:{CYAN};font-size:.7rem;">'
           f'── 現價 {close:,.2f} ──</span></div>')
    pivot_html = ""
    if pp:
        pivot_html = (
            f'<div style="margin-top:5px;padding-top:5px;border-top:1px solid {BORD};">'
            + "".join(
                lvl_box(f"樞紐{k}", v,
                        RED if "R" in k else (GOLD if k=="P" else GRN),
                        "#1A0808" if "R" in k else ("#141408" if k=="P" else "#081A0E"),
                        (v-close)/close*100)
                for k, v in list(pp.items())[:5]
            ) + '</div>'
        )
    return box(
        lvl_box("壓力①（近20日）",res1,RED,"#1A0808",(res1-close)/close*100) +
        lvl_box("壓力②（近60日）",res2,RED,"#1A0808",(res2-close)/close*100) +
        cur +
        lvl_box("支撐①（近20日）",sup1,GRN,"#081A0E",-(close-sup1)/close*100) +
        lvl_box("支撐②（近60日）",sup2,GRN,"#081A0E",-(close-sup2)/close*100) +
        pivot_html +
        f'<div style="color:{MUTED};font-size:.7rem;text-align:center;margin-top:5px;">'
        f'守支撐 → 多方格局不變</div>',
        "關鍵價位", "🎯"
    )


def html_alerts(alerts):
    if not alerts:
        return box(f'<div style="color:{MUTED};font-size:.73rem;'
                   f'text-align:center;padding:6px 0;">✅ 目前無觸發警示</div>',
                   "即時警示", "🔔")
    rows = "".join(
        f'<div style="background:{BG};border-radius:4px;'
        f'padding:5px 9px;margin:3px 0;border-left:3px solid {c};">'
        f'<div style="color:{c};font-size:.76rem;font-weight:700;">{msg}</div>'
        f'<div style="color:{MUTED};font-size:.68rem;">{detail}</div></div>'
        for msg, detail, c in alerts
    )
    return box(rows, f"即時警示（{len(alerts)} 項）", "🔔")


def html_ai_summary(verdict, score, signals, stop, target, col, close):
    pct_w    = min(max(score, 0), 100)
    rr_ratio = abs(target-close) / (abs(close-stop) + 1e-9)
    score_bar = (f'<div style="background:{BG};border-radius:3px;'
                 f'overflow:hidden;height:8px;margin:5px 0;">'
                 f'<div style="background:{col};height:100%;width:{pct_w}%;"></div></div>')
    verdict_div = (
        f'<div style="background:{BG};border-radius:4px;'
        f'padding:7px 10px;margin-bottom:6px;border-left:3px solid {col};">'
        f'<div style="color:{col};font-size:.82rem;font-weight:700;">{verdict}</div>'
        f'<div style="display:flex;gap:12px;margin-top:4px;">'
        f'<span style="color:{MUTED};font-size:.7rem;">目標 '
        f'<span style="color:{col};font-weight:700;">{target:,.0f}</span></span>'
        f'<span style="color:{MUTED};font-size:.7rem;">停損 '
        f'<span style="color:{ORNG};font-weight:700;">{stop:,.0f}</span></span>'
        f'<span style="color:{MUTED};font-size:.7rem;">風報比 '
        f'<span style="color:{CYAN};font-weight:700;">{rr_ratio:.2f}</span>'
        f'</span></div></div>'
    )
    score_div = (f'<div style="display:flex;justify-content:space-between;'
                 f'align-items:center;margin-bottom:2px;">'
                 f'<span style="color:{MUTED};font-size:.72rem;">AI 訊號評分</span>'
                 f'<span style="color:{col};font-weight:900;font-size:.9rem;">'
                 f'{score}/100</span></div>')
    sig_rows = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;padding:2px 0;">'
        f'<span style="color:{"#FF3030" if s=="多" else "#00C864"};font-size:.72rem;">'
        f'{"▲" if s=="多" else "▼"}</span>'
        f'<span style="color:{MUTED};font-size:.71rem;">{desc}</span></div>'
        for s, desc in signals
    )
    return box(verdict_div + score_div + score_bar + sig_rows, "AI 訊號評分", "🤖")


def html_rsi_bar(rsi_val):
    if   rsi_val>70: clr,lbl=RED,"超買"
    elif rsi_val<30: clr,lbl=GRN,"超賣"
    else:            clr,lbl=GOLD,"中性"
    pct = min(max(rsi_val, 0), 100)
    bar = (f'<div style="background:{BG};border-radius:3px;overflow:hidden;'
           f'height:8px;margin:4px 0;">'
           f'<div style="background:{clr};height:100%;width:{pct:.0f}%;"></div></div>')
    return box(
        f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
        f'<span style="color:{MUTED};font-size:.73rem;">RSI(14)</span>'
        f'<span style="color:{clr};font-weight:700;font-size:.82rem;">'
        f'{rsi_val:.1f} {lbl}</span></div>' + bar
    )


def html_main_force(status, trend, cc, chip_c, chip_s, mf_col, df):
    recent = df.tail(5); rows_h = ""
    for _, r in recent.iterrows():
        is_b = safe_float(r["Close"]) >= safe_float(r["Open"])
        col  = RED if is_b else GRN
        lbl  = r["Date"].strftime("%m/%d") if hasattr(r["Date"],"strftime") else str(r["Date"])
        val  = f"{'買' if is_b else '賣'} {int(safe_float(r['Volume'])/1000):.0f}K"
        rows_h += row_item(lbl, val, col)
    top = (f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:7px;">'
           f'<div style="width:10px;height:10px;border-radius:50%;background:{mf_col};"></div>'
           f'<span style="color:{mf_col};font-weight:700;font-size:.84rem;">{status}</span>'
           f'<span style="color:{MUTED};font-size:.7rem;">{trend}</span></div>')
    foot = (f'<div style="margin-top:6px;">'
            f'{tag("集中度 "+chip_c,mf_col,CARD)}'
            f'{tag("穩定度 "+chip_s,mf_col,CARD)}'
            f'{tag(cc,mf_col,CARD)}</div>')
    return box(top + rows_h + foot, "主力出貨警示（近5日）", "◉")


def html_vol_price(df):
    cl=safe_float(df.iloc[-1]["Close"]); pc=safe_float(df.iloc[-2]["Close"])
    vol=safe_float(df.iloc[-1]["Volume"])
    a5=safe_float(df["Volume"].tail(5).mean()); a20=safe_float(df["Volume"].tail(20).mean())
    vr5=vol/(a5+1e-9); vr20=vol/(a20+1e-9)
    m20=safe_float(df["MA20"].iloc[-1]) if "MA20" in df.columns else 0
    m60=safe_float(df["MA60"].iloc[-1]) if "MA60" in df.columns else 0
    up=cl>pc
    vp=("量增價漲✓" if up and vr5>1 else "量縮價跌" if not up and vr5<1 else "量價背離⚠")
    vpc=RED if "漲" in vp else (GRN if "跌" in vp else GOLD)
    obv_s,obv_c="—",MUTED
    if "OBV" in df.columns and len(df)>=5:
        obv_chg=safe_float(df["OBV"].iloc[-1])-safe_float(df["OBV"].iloc[-5])
        obv_s="OBV上升✓" if obv_chg>0 else "OBV下降✗"
        obv_c=RED if obv_chg>0 else GRN
    return box("".join([
        row_item("量價關係",vp,vpc),
        row_item("均量比(5日)",f"{vr5:.2f}x",RED if vr5>1.2 else (GRN if vr5<0.8 else GOLD)),
        row_item("均量比(20日)",f"{vr20:.2f}x",RED if vr20>1.2 else (GRN if vr20<0.8 else GOLD)),
        row_item("OBV趨勢",obv_s,obv_c),
        row_item("距MA20",f"{(cl-m20)/cl*100:+.1f}%" if m20 else "—",RED if cl>m20 else GRN),
        row_item("距MA60",f"{(cl-m60)/cl*100:+.1f}%" if m60 else "—",RED if cl>m60 else GRN),
    ]), "量價結構分析", "◈")


def html_chip(df, mf_status, chip_c, chip_s, mf_col,
              foreign=None, invest=None, dealer=None):
    cl=safe_float(df.iloc[-1]["Close"])
    m20=safe_float(df["MA20"].iloc[-1]) if "MA20" in df.columns else 0
    m60=safe_float(df["MA60"].iloc[-1]) if "MA60" in df.columns else 0
    pos=("多方主導" if cl>m20>m60 else "空方主導" if cl<m20<m60 else "均線糾結")
    pc=RED if "多" in pos else (GRN if "空" in pos else GOLD)
    rows=[
        row_item("主力動向",mf_status,mf_col),
        row_item("籌碼集中度",chip_c,mf_col),
        row_item("籌碼穩定度",chip_s,mf_col),
        row_item("多空主導",pos,pc),
        row_item("MA20站穩","是✓" if cl>m20 else "否✗",RED if cl>m20 else GRN),
        row_item("主力進出",
                 "進出見買" if mf_status in ["進出見買","中性偏多"] else "進出見賣",
                 RED if mf_status in ["進出見買","中性偏多"] else GRN),
    ]
    if foreign is not None:
        def fmt(v): return "—" if v is None else (f"+{v:,}" if v>=0 else f"{v:,}")
        rows+=[
            row_item("外資買賣超",fmt(foreign),RED if (foreign or 0)>=0 else GRN),
            row_item("投信買賣超",fmt(invest), RED if (invest  or 0)>=0 else GRN),
            row_item("自營商買賣超",fmt(dealer),RED if (dealer or 0)>=0 else GRN),
        ]
    return box("".join(rows), "籌碼結構分析", "◈")


def html_day_script(df, close, atr_v):
    at = atr_v if atr_v and not math.isnan(atr_v) else close*0.025
    sc = [
        ("① 開高走高",RED,f"開盤>{close:.0f} 量能同步",
         f"目標 {close+at:.0f}/{close+2*at:.0f}",f"停損 {close-at*.5:.0f}"),
        ("② 盤整整理",GOLD,f"盤在{close-at*.5:.0f}~{close+at*.5:.0f}",
         "觀望 等突破方向",f"停損 {close-at:.0f}"),
        ("③ 開低回測",GRN,f"開盤<{close:.0f} 觀察量能",
         f"目標 {close-at:.0f}/{close-2*at:.0f}",f"停損 {close+at*.5:.0f}"),
    ]
    cards = "".join(
        f'<div style="background:{BG};border-radius:4px;padding:7px 9px;'
        f'margin:3px 0;border-left:3px solid {c};">'
        f'<div style="color:{c};font-size:.76rem;font-weight:700;margin-bottom:3px;">{t}</div>'
        f'<div style="color:{MUTED};font-size:.68rem;">{cond}</div>'
        f'<div style="color:{TEXT};font-size:.69rem;">{tgt}</div>'
        f'<div style="color:{ORNG};font-size:.69rem;">{stp}</div></div>'
        for t,c,cond,tgt,stp in sc
    )
    return box(cards, f"日操作劇本（{datetime.today().strftime('%m/%d')}）", "◐")


def html_kline_patterns(kp, cp):
    krows = "".join(
        f'<div style="display:flex;justify-content:space-between;'
        f'padding:3px 8px;border-radius:3px;background:{CARD};margin:2px 0;">'
        f'<span style="color:{TEXT};font-size:.75rem;">{n}</span>'
        f'<span style="color:{c};font-size:.73rem;">{t}</span></div>'
        for n,t,c in kp
    )
    crows = "".join(
        f'<div style="padding:4px 8px;border-radius:3px;background:{CARD};'
        f'margin:3px 0;border-left:2px solid {c};">'
        f'<div style="color:{c};font-size:.75rem;font-weight:700;">{n}</div>'
        f'<div style="color:{MUTED};font-size:.69rem;">{d}</div></div>'
        for n,c,d in cp
    )
    return box(krows +
               f'<div style="color:{MUTED};font-size:.7rem;margin:6px 0 3px;">型態分析</div>'
               + crows, "K線型態", "◆")


def html_multiperiod(daily, weekly, monthly):
    results = [trend_label(daily,"日K"), trend_label(weekly,"週K"),
               trend_label(monthly,"月K")]
    rows_h = ""
    for tf,tr,col,note in results:
        rows_h += (f'<div style="background:{CARD};border-radius:4px;'
                   f'padding:5px 9px;margin:3px 0;">'
                   f'<div style="display:flex;justify-content:space-between;">'
                   f'<span style="color:{TEXT};font-size:.78rem;font-weight:700;">{tf}</span>'
                   f'<span style="color:{col};font-size:.75rem;font-weight:700;">{tr}</span></div>'
                   f'<div style="color:{MUTED};font-size:.67rem;">{note}</div></div>')
    return box(rows_h, "多週期分析", "◍")


def build_header_html(ticker, name, close, chg, pct, vol, mktcap, pe,
                      ma5, ma20, ma60, kv, dv, rv, mv, sv,
                      ip, ep, iv, ev, date_s):
    cc=RED if chg>=0 else GRN; sym="▲" if chg>=0 else "▼"
    mktcap_s=f"{mktcap/1e8:,.0f} 億" if mktcap else "—"
    pe_s=f"{pe:.1f}" if pe else "—"
    badges="".join(
        f'<div style="text-align:center;background:{CARD};border-radius:4px;padding:4px 10px;">'
        f'<div style="color:{MUTED};font-size:.6rem;">{lb}</div>'
        f'<div style="color:{vc};font-size:.8rem;font-weight:700;">{vl}</div></div>'
        for lb,vl,vc in [
            ("成交量",f"{vol:,}",TEXT),("日期",date_s,TEXT),
            ("市值",mktcap_s,TEXT),("本益比",pe_s,TEXT),
            ("內盤",f"{iv:,}({ip}%)",GRN),("外盤",f"{ev:,}({ep}%)",RED),
            ("內外盤比",f"{ip}:{ep}",CYAN),
        ]
    )
    return (
        f'<div style="background:linear-gradient(90deg,#05080F,#0B1528,#05080F);'
        f'border:1px solid {BORD};border-radius:8px;padding:10px 16px;'
        f'font-family:Arial,sans-serif;margin-bottom:6px;">'
        f'<div style="display:flex;align-items:center;flex-wrap:wrap;gap:14px;">'
        f'<div><span style="color:#8090A0;font-size:.78rem;">{ticker.replace(".TW","")}</span>'
        f'<span style="color:#E0EAF4;font-size:.82rem;font-weight:700;margin-left:5px;">'
        f'{name}</span></div>'
        f'<div style="display:flex;align-items:baseline;gap:9px;">'
        f'<span style="color:{cc};font-size:2.2rem;font-weight:900;'
        f'letter-spacing:1px;line-height:1;">{close:,.2f}</span>'
        f'<span style="color:{cc};font-size:.94rem;font-weight:700;">'
        f'{sym} {abs(chg):.2f}（{abs(pct):.2f}%）</span></div>'
        f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-left:auto;">'
        f'{badges}</div></div>'
        f'<div style="display:flex;gap:14px;margin-top:6px;flex-wrap:wrap;">'
        f'<span style="color:{MUTED};font-size:.68rem;">日K線圖</span>'
        f'<span style="color:{GOLD};font-size:.7rem;">MA5 {ma5:,.2f}</span>'
        f'<span style="color:{ORNG};font-size:.7rem;">MA20 {ma20:,.2f}</span>'
        f'<span style="color:{CYAN};font-size:.7rem;">MA60 {ma60:,.2f}</span>'
        f'<span style="color:{MUTED};font-size:.7rem;">KD {kv:.0f}/{dv:.0f}</span>'
        f'<span style="color:{MUTED};font-size:.7rem;">RSI {rv:.0f}</span>'
        f'<span style="color:{"#FF3030" if mv>sv else "#00C864"};font-size:.7rem;">'
        f'MACD {"黃金↑" if mv>sv else "死亡↓"}</span>'
        f'</div></div>'
    )


# ══════════════════════════════════════════════════════════════════
#  側邊欄
# ══════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown(
            f'<div style="color:{CYAN};font-size:.9rem;font-weight:700;'
            f'padding:6px 0;border-bottom:1px solid {BORD};margin-bottom:10px;">'
            f'📋 自選股清單</div>', unsafe_allow_html=True)
        if "watchlist" not in st.session_state:
            st.session_state["watchlist"] = [
                "2330.TW","2317.TW","2454.TW","2412.TW","2308.TW"]
        new_stock = st.text_input("新增股票代號", placeholder="如 2303 或 2303.TW")
        if st.button("➕ 加入自選", use_container_width=True) and new_stock.strip():
            fmt = new_stock.strip() if "." in new_stock else new_stock.strip()+".TW"
            if fmt not in st.session_state["watchlist"]:
                st.session_state["watchlist"].append(fmt)
        st.markdown(f'<div style="color:{MUTED};font-size:.7rem;margin:6px 0;">'
                    f'點擊切換股票</div>', unsafe_allow_html=True)
        to_remove = None
        for stk in st.session_state["watchlist"]:
            c1, c2 = st.columns([4, 1])
            with c1:
                if st.button(stk.replace(".TW",""), key=f"ws_{stk}",
                             use_container_width=True):
                    st.session_state["ticker"] = stk
                    st.rerun()
            with c2:
                if st.button("✕", key=f"wd_{stk}"):
                    to_remove = stk
        if to_remove:
            st.session_state["watchlist"].remove(to_remove)
            st.rerun()
        st.markdown("---")
        st.markdown(f'<div style="color:{CYAN};font-size:.76rem;font-weight:700;'
                    f'margin-bottom:6px;">⚙️ 顯示設定</div>', unsafe_allow_html=True)
        st.session_state["show_vwap"]  = st.checkbox("顯示 VWAP",
            value=st.session_state.get("show_vwap", True))
        st.session_state["show_pivot"] = st.checkbox("顯示樞紐點",
            value=st.session_state.get("show_pivot", True))
        st.session_state["show_rt"]    = st.checkbox("顯示即時報價",
            value=st.session_state.get("show_rt", True))
        st.markdown("---")
        st.markdown(f'<div style="color:{MUTED};font-size:.68rem;text-align:center;">'
                    f'⚠ 僅供學習研究<br>不構成投資建議</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  主頁面
# ══════════════════════════════════════════════════════════════════

def main():
    render_sidebar()

    # ── Banner ────────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:linear-gradient(90deg,#05080F,#0C1828,#05080F);'
        f'border-bottom:2px solid {CYAN};padding:10px 16px;margin-bottom:8px;'
        f'border-radius:8px;display:flex;align-items:center;gap:12px;">'
        f'<span style="font-size:1.6rem;">📈</span><div>'
        f'<h1 style="color:#E0EAF4;margin:0;font-size:1.1rem;font-weight:900;'
        f'letter-spacing:1px;">台股技術分析儀表板 v3.0</h1>'
        f'<p style="color:{MUTED};margin:2px 0 0;font-size:.72rem;">'
        f'K線·均線·KD·MACD·RSI·布林·費波·VWAP·OBV·威廉%R·CCI·'
        f'即時報價·融資融券·三大法人·新聞·籌碼分散</p></div></div>',
        unsafe_allow_html=True)

    # ── 控制列 ────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns([3, 3, 1.5, 1.5, 1.5, 2])
    with c1:
        ticker_input = st.text_input("股票代號",
            value=st.session_state.get("ticker","2330.TW"),
            placeholder="2330 / 2330.TW / TSLA")
    with c2:
        period_days = st.slider("查詢天數", 30, 365,
            st.session_state.get("period", 90), 15)
    with c3:
        show_bb  = st.checkbox("布林通道",
            value=st.session_state.get("show_bb", False))
    with c4:
        show_fib = st.checkbox("費波那契",
            value=st.session_state.get("show_fib", False))
    with c5:
        show_vwap = st.checkbox("VWAP",
            value=st.session_state.get("show_vwap", True))
    with c6:
        st.markdown("<br>", unsafe_allow_html=True)
        query_btn = st.button("🔍 查詢分析", use_container_width=True)

    st.session_state.update({
        "ticker": ticker_input, "period": period_days,
        "show_bb": show_bb, "show_fib": show_fib, "show_vwap": show_vwap,
    })
    if "loaded" not in st.session_state or query_btn:
        st.session_state["loaded"] = True
    if not st.session_state.get("loaded"):
        return

    ticker = ticker_input.strip()
    if not ticker:
        st.error("請輸入股票代號"); return
    if "." not in ticker:
        ticker += ".TW"
    stock_id = ticker.replace(".TW","").replace(".TWO","")

    # ── 平行抓取所有資料 ─────────────────────────────────────────
    with st.spinner(f"正在抓取 {ticker} 全部資料…"):
        daily, weekly, monthly = fetch_all(ticker, period_days)
        rt_data   = crawl_realtime(stock_id)
        inst_data = crawl_institutional(stock_id)
        margin    = crawl_margin(stock_id)
        fund_data = crawl_fundamental(stock_id)
        chip_dist = crawl_chip_distribution(stock_id)

    if daily is None or daily.empty:
        st.error(f"❌ 無法取得 {ticker}，請確認代號或網路。"); return
    daily = daily.tail(int(period_days)).reset_index(drop=True)
    if len(daily) < 3:
        st.error("資料筆數不足，請增加查詢天數。"); return

    # 公司資訊
    try:
        info     = yf.Ticker(ticker).info
        name     = info.get("longName", info.get("shortName", ticker))
        sector   = info.get("sector",   "科技")
        industry = info.get("industry", "半導體")
        pe       = info.get("trailingPE", None)
        mktcap   = info.get("marketCap",  None)
    except Exception:
        name=ticker; sector=industry="—"; pe=mktcap=None

    # 新聞（非同步抓，不阻塞主流程）
    try:
        news_list = crawl_news(stock_id, name)
    except Exception:
        news_list = []

    # 最新值
    r, p   = daily.iloc[-1], daily.iloc[-2]
    close  = safe_float(r["Close"])
    prev_c = safe_float(p["Close"])
    chg    = close - prev_c
    pct    = chg / (prev_c + 1e-9) * 100
    vol    = int(safe_float(r["Volume"]))
    ma5    = safe_float(r.get("MA5",  0))
    ma20   = safe_float(r.get("MA20", 0))
    ma60   = safe_float(r.get("MA60", 0))
    kv     = safe_float(r.get("K",   50), 50)
    dv     = safe_float(r.get("D",   50), 50)
    rv     = safe_float(r.get("RSI", 50), 50)
    mv     = safe_float(r.get("MACD", 0))
    sv     = safe_float(r.get("SIG",  0))
    hv     = safe_float(r.get("HIST", 0))
    atr_v  = safe_float(r.get("ATR",  close*0.02))
    wr_v   = safe_float(r.get("WR",  -50), -50)
    cci_v  = safe_float(r.get("CCI",   0))
    bbw_v  = safe_float(r.get("BBW",   5))

    ip, ep = board_ratio(daily)
    iv, ev = int(vol*ip/100), int(vol*ep/100)
    wr     = win_rate(daily)

    mf_stat, mf_trend, chip_c, chip_s, mf_col, mf_cc = main_force(daily)
    kp      = kline_patterns(daily)
    cp      = chart_patterns(daily)
    alerts  = check_alerts(daily, close, kv, dv, rv, mv, sv)
    verdict, score, signals, stop, target, ai_col = ai_summary(
        close, ma5, ma20, ma60, kv, dv, rv, mv, sv, hv,
        atr_v, wr_v, cci_v, bbw_v)
    date_s  = (r["Date"].strftime("%m/%d")
               if hasattr(r["Date"],"strftime") else str(r["Date"]))

    # 三大法人（整合爬蟲結果）
    foreign = inst_data.get("foreign") if inst_data else None
    invest  = inst_data.get("invest")  if inst_data else None
    dealer  = inst_data.get("dealer")  if inst_data else None

    # ── 警示 toast ────────────────────────────────────────────────
    for msg, detail, _ in alerts[:5]:
        st.toast(f"{msg}：{detail}", icon="🔔")

    # ── 標頭 ─────────────────────────────────────────────────────
    st.markdown(build_header_html(
        ticker, name, close, chg, pct, vol, mktcap, pe,
        ma5, ma20, ma60, kv, dv, rv, mv, sv,
        ip, ep, iv, ev, date_s), unsafe_allow_html=True)

    # ── 主體四欄 ─────────────────────────────────────────────────
    col_left, col_mid, col_right, col_far = st.columns([42, 20, 20, 20])

    with col_left:
        st.plotly_chart(build_main(daily, show_bb, show_fib, show_vwap),
                        use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown(html_ohlcv(daily) + html_rsi_bar(rv),
                    unsafe_allow_html=True)

        # 指標 Tab
        tab1, tab2, tab3 = st.tabs(["📊 OBV", "⚡ 威廉%R + CCI", "📐 布林帶寬"])
        with tab1:
            obv_fig = build_obv_chart(daily)
            if obv_fig:
                st.plotly_chart(obv_fig, use_container_width=True,
                                config={"displayModeBar": False})
            else:
                st.caption("OBV 資料不足")
        with tab2:
            wr_fig = build_wr_cci_chart(daily)
            if wr_fig:
                st.plotly_chart(wr_fig, use_container_width=True,
                                config={"displayModeBar": False})
        with tab3:
            bbw_fig = build_bbw_chart(daily)
            if bbw_fig:
                st.plotly_chart(bbw_fig, use_container_width=True,
                                config={"displayModeBar": False})

    # ── 中欄 ──────────────────────────────────────────────────────
    with col_mid:
        # 即時報價（爬蟲）
        if st.session_state.get("show_rt", True):
            st.markdown(html_realtime(rt_data), unsafe_allow_html=True)

        st.markdown(html_alerts(alerts), unsafe_allow_html=True)
        st.markdown(html_main_force(mf_stat, mf_trend, mf_cc,
                                    chip_c, chip_s, mf_col, daily),
                    unsafe_allow_html=True)
        st.markdown(
            f'<div style="color:{CYAN};font-size:.76rem;font-weight:700;'
            f'padding:3px 0 5px;border-bottom:1px solid {BORD};'
            f'margin-bottom:4px;font-family:Arial,sans-serif;">'
            f'⚡ 短線勝率（近10日）</div>',
            unsafe_allow_html=True)
        st.plotly_chart(build_gauge(wr), use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown(html_key_levels(daily, close), unsafe_allow_html=True)

        # 融資融券（爬蟲）
        st.markdown(html_margin(margin), unsafe_allow_html=True)

    # ── 右欄 ──────────────────────────────────────────────────────
    with col_right:
        st.markdown(
            html_tech(daily, ma5, ma20, ma60, kv, dv, rv, mv, sv, hv) +
            html_institutional(inst_data) +
            html_fundamental(fund_data, pe, mktcap),
            unsafe_allow_html=True)

        # 內外盤結構
        board_html = (
            f'<div style="background:{CARD};border:1px solid {BORD};'
            f'border-radius:6px;padding:9px 11px;margin-bottom:7px;'
            f'font-family:Arial,sans-serif;">'
            + f'<div style="color:{CYAN};font-size:.76rem;font-weight:700;'
            f'padding:3px 0 5px;border-bottom:1px solid {BORD};margin-bottom:7px;">'
            f'◎ 內外盤結構</div>'
            + "".join([
                row_item("內盤（賣）", f"{iv:,} ({ip}%)", GRN),
                row_item("外盤（買）", f"{ev:,} ({ep}%)", RED),
                row_item("內外盤比",   f"{ip}:{ep}",      CYAN),
                row_item("買賣判斷",
                         "外盤積極" if ep > ip else "內盤偏重",
                         RED if ep > ip else GRN),
            ])
            + '</div>'
        )
        st.markdown(board_html, unsafe_allow_html=True)

        # 集保戶股權分散圖
        if chip_dist:
            st.markdown(
                f'<div style="color:{CYAN};font-size:.76rem;font-weight:700;'
                f'padding:3px 0 5px;border-bottom:1px solid {BORD};'
                f'margin-bottom:4px;font-family:Arial,sans-serif;">'
                f'🥧 集保戶持股分散</div>',
                unsafe_allow_html=True)
            dist_fig = build_chip_dist_chart(chip_dist)
            if dist_fig:
                st.plotly_chart(dist_fig, use_container_width=True,
                                config={"displayModeBar": False})

    # ── 最右欄 ────────────────────────────────────────────────────
    with col_far:
        st.markdown(
            html_ai_summary(verdict, score, signals,
                            stop, target, ai_col, close),
            unsafe_allow_html=True)
        st.plotly_chart(build_ai_radar(signals), use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown(
            html_vol_price(daily) +
            html_chip(daily, mf_stat, chip_c, chip_s, mf_col,
                      foreign, invest, dealer) +
            html_day_script(daily, close, atr_v),
            unsafe_allow_html=True)

    # ── 底部列：K線型態 + 多週期 + 新聞 ─────────────────────────
    bot1, bot2, bot3 = st.columns([1, 1, 2])
    with bot1:
        st.markdown(html_kline_patterns(kp, cp), unsafe_allow_html=True)
    with bot2:
        st.markdown(html_multiperiod(daily, weekly, monthly),
                    unsafe_allow_html=True)
    with bot3:
        # 新聞面板（爬蟲）
        st.markdown(html_news(news_list), unsafe_allow_html=True)

    # ── 下載列 ───────────────────────────────────────────────────
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "⬇️ 下載 K線資料 CSV",
            data=export_csv(daily),
            file_name=f"{ticker}_{datetime.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "📄 下載分析摘要 TXT",
            data=export_summary(
                ticker, name, close, pct, kv, rv, mv, sv,
                verdict, score, signals, stop, target,
                margin_data=margin,
                inst_data=inst_data,
            ),
            file_name=f"{ticker}_report_{datetime.today().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # ── 免責聲明 ──────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:{CARD};border:1px solid {BORD};'
        f'border-radius:5px;padding:6px 12px;margin-top:4px;text-align:center;">'
        f'<span style="color:{MUTED};font-size:.7rem;">'
        f'⚠ 資料來源：Yahoo Finance · TWSE · TPEX · Yahoo新聞｜'
        f'本儀表板僅供學習與研究，不構成投資建議。投資有風險，操作請審慎。'
        f'</span></div>',
        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
