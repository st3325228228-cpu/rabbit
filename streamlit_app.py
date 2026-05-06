"""
台股技術分析儀表板 ── Streamlit 完整進階版
依賴：pip install streamlit yfinance plotly pandas numpy requests
執行：streamlit run stock_dashboard_streamlit.py
"""

import warnings, math, io, requests
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ══════════════════════════════════════════════════════════════════
#  頁面設定（必須在最前面）
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="台股技術分析儀表板",
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

# ══════════════════════════════════════════════════════════════════
#  全域 CSS
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
  .stApp, .main, [data-testid="stAppViewContainer"] {{
      background-color: {BG} !important;
      color: {TEXT} !important;
  }}
  [data-testid="stHeader"], [data-testid="stSidebar"] {{
      background-color: {PANEL} !important;
  }}
  .stTextInput input, .stNumberInput input {{
      background-color: {CARD} !important;
      border: 1px solid {BORD} !important;
      color: #E0EAF4 !important;
      border-radius: 5px !important;
  }}
  .stSlider > div > div {{ background-color: {CYAN} !important; }}
  .stButton button {{
      background: linear-gradient(90deg, #0A1E38, #122840) !important;
      color: {CYAN} !important;
      border: 1px solid {CYAN} !important;
      font-weight: 700 !important;
      border-radius: 5px !important;
  }}
  .stButton button:hover {{
      background: linear-gradient(90deg, #122840, #1A3050) !important;
      color: #fff !important;
  }}
  .stCheckbox label {{ color: {TEXT} !important; }}
  .js-plotly-plot, .plotly-graph-div {{ background: transparent !important; }}
  #MainMenu, footer, [data-testid="stToolbar"] {{ visibility: hidden; }}
  [data-testid="column"] {{ padding: 0 4px !important; }}
  ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
  ::-webkit-scrollbar-track {{ background: {BG}; }}
  ::-webkit-scrollbar-thumb {{ background: {BORD}; border-radius: 3px; }}
  label[data-testid="stWidgetLabel"] > p {{ color: {MUTED} !important; font-size: .78rem !important; }}
  hr {{ border-color: {BORD} !important; }}
  /* sidebar */
  [data-testid="stSidebar"] .stButton button {{
      font-size: .75rem !important;
      padding: 3px 8px !important;
  }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  技術指標
# ══════════════════════════════════════════════════════════════════

def ma(s, n):
    return s.rolling(n).mean()

def kd(df, n=9):
    lo  = df["Low"].rolling(n).min()
    hi  = df["High"].rolling(n).max()
    rsv = (df["Close"] - lo) / (hi - lo + 1e-9) * 100
    k   = rsv.ewm(com=2, adjust=False).mean()
    d   = k.ewm(com=2, adjust=False).mean()
    return k, d

def macd(s, f=12, sl=26, sig=9):
    m  = s.ewm(span=f,  adjust=False).mean() - s.ewm(span=sl, adjust=False).mean()
    sg = m.ewm(span=sig, adjust=False).mean()
    return m, sg, m - sg

def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / (l + 1e-9))

def bb(s, n=20, k=2):
    m   = s.rolling(n).mean()
    std = s.rolling(n).std()
    return m + k*std, m, m - k*std

def atr(df, n=14):
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"]  - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

# ══════════════════════════════════════════════════════════════════
#  資料抓取
# ══════════════════════════════════════════════════════════════════

def _clean(df):
    df = df.reset_index()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    dc = "Datetime" if "Datetime" in df.columns else "Date"
    df["Date"] = pd.to_datetime(df[dc])
    for c in ["Open","High","Low","Close","Volume"]:
        if c in df.columns:
            df[c] = df[c].astype(float)
    return df.sort_values("Date").reset_index(drop=True)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_all(ticker, days):
    end   = datetime.today()
    start = end - timedelta(days=max(int(days)+30, 220))

    def get(iv):
        try:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=True, progress=False, interval=iv)
            return _clean(df) if not df.empty else None
        except Exception:
            return None

    d = get("1d")
    if d is None:
        return None, None, None

    d["MA5"]  = ma(d["Close"], 5)
    d["MA20"] = ma(d["Close"], 20)
    d["MA60"] = ma(d["Close"], 60)
    d["K"], d["D"]                 = kd(d)
    d["MACD"], d["SIG"], d["HIST"] = macd(d["Close"])
    d["RSI"]                       = rsi(d["Close"])
    d["BBU"], d["BBM"], d["BBL"]   = bb(d["Close"])
    d["ATR"]                       = atr(d)

    w = get("1wk")
    if w is not None and len(w) >= 5:
        w["MA5"], w["MA20"]            = ma(w["Close"], 5), ma(w["Close"], 20)
        w["K"], w["D"]                 = kd(w)
        w["MACD"], w["SIG"], w["HIST"] = macd(w["Close"])

    mo = get("1mo")
    if mo is not None and len(mo) >= 3:
        mo["MA5"]        = ma(mo["Close"], 5)
        mo["K"], mo["D"] = kd(mo)

    return d, w, mo

# ══════════════════════════════════════════════════════════════════
#  三大法人資料
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_institutional(stock_id):
    """抓取近日三大法人買賣超（TWSE OpenAPI）"""
    try:
        date_str = datetime.today().strftime("%Y%m%d")
        url = (f"https://www.twse.com.tw/fund/T86"
               f"?response=json&date={date_str}&selectType=ALLBUT0999")
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        data = resp.json()
        for row in data.get("data", []):
            if row[0] == stock_id:
                def p(x):
                    try: return int(x.replace(",","").replace(" ",""))
                    except: return 0
                foreign = p(row[4])
                invest  = p(row[10])
                dealer  = p(row[14])
                return foreign, invest, dealer
    except Exception:
        pass
    return None, None, None

# ══════════════════════════════════════════════════════════════════
#  輔助分析
# ══════════════════════════════════════════════════════════════════

def win_rate(df, n=10):
    if len(df) < n:
        return 50
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
    vt = r["Volume"].diff().mean()
    if up > dn * 1.3:
        return "進出見買", "籌碼集中", "高", "穩定",   RED,  "多方偏強"
    if dn > up * 1.3:
        return "進出見賣", "籌碼分散", "低", "不穩定", GRN,  "空方偏強"
    txt = "中性偏多" if vt > 0 else "中性偏空"
    return txt, "籌碼適中", "中", "尚穩定", GOLD, "中性整理"

def kline_patterns(df):
    r      = df.iloc[-1]
    body   = abs(float(r["Close"]) - float(r["Open"]))
    total  = float(r["High"]) - float(r["Low"]) + 1e-9
    upper  = float(r["High"]) - max(float(r["Close"]), float(r["Open"]))
    lower  = min(float(r["Close"]), float(r["Open"])) - float(r["Low"])
    bull   = float(r["Close"]) >= float(r["Open"])
    br     = body / total
    out = []
    if br > 0.7:
        out.append(("長紅K" if bull else "長黑K", "多方型態" if bull else "空方型態", RED if bull else GRN))
    if br < 0.1:
        out.append(("十字線", "觀望訊號", GOLD))
    if lower > 2*body and upper < body:
        out.append(("錘子線", "底部訊號", RED if bull else GOLD))
    if upper > 2*body and lower < body:
        out.append(("流星線", "頂部訊號", GRN))
    if not out:
        out.append(("一般K線", "持續觀察", TEXT))
    return out

def chart_patterns(df):
    c = df["Close"].values
    if len(c) < 20:
        return [("持續整理", GOLD, "資料不足")]
    r   = c[-20:]
    mid = 10
    l1, l2 = r[:mid].min(), r[mid:].min()
    mh     = r[4:8].max()
    h1, h2 = r[:mid].max(), r[mid:].max()
    ml     = r[4:8].min()
    out = []
    if abs(l1-l2)/l1 < 0.03 and mh > l1*1.02:
        out.append(("W底型態", RED, "底部反轉訊號，留意突破"))
    if abs(h1-h2)/h1 < 0.03 and ml < h1*0.98:
        out.append(("M頭型態", GRN, "頭部反轉訊號，留意跌破"))
    if not out:
        out.append(("持續整理", GOLD, "未形成明確型態，等待方向"))
    return out

def trend_label(df_tf, tf_name):
    if df_tf is None or len(df_tf) < 3:
        return tf_name, "資料不足", MUTED, "—"
    r, p = df_tf.iloc[-1], df_tf.iloc[-2]
    chg  = (float(r["Close"]) - float(p["Close"])) / float(p["Close"]) * 100
    kv   = float(r.get("K", 50))   if "K"    in df_tf.columns else 50
    mv   = float(r.get("MACD", 0)) if "MACD" in df_tf.columns else 0
    sv   = float(r.get("SIG",  0)) if "SIG"  in df_tf.columns else 0
    ma5v = float(r.get("MA5", float(r["Close"]))) if "MA5" in df_tf.columns else float(r["Close"])
    trend = "多頭" if float(r["Close"]) > ma5v else "空頭"
    col   = RED if trend == "多頭" else GRN
    kd_s  = "高鈍" if kv > 80 else ("低超" if kv < 20 else "中")
    ma_s  = "黃金" if mv > sv else "死亡"
    return tf_name, f"{trend} ({chg:+.1f}%)", col, f"KD:{kd_s} MACD:{ma_s}交叉"

# ══════════════════════════════════════════════════════════════════
#  警示系統
# ══════════════════════════════════════════════════════════════════

def check_alerts(df, close, kv, dv, rv, mv, sv):
    alerts = []
    if len(df) < 3:
        return alerts

    prev_k = float(df.iloc[-2]["K"]) if "K" in df.columns else 50
    prev_d = float(df.iloc[-2]["D"]) if "D" in df.columns else 50

    # KD 交叉
    if prev_k < prev_d and kv > dv:
        alerts.append(("🟢 KD黃金交叉", "買進訊號", GRN))
    if prev_k > prev_d and kv < dv:
        alerts.append(("🔴 KD死亡交叉", "賣出訊號", RED))

    # RSI 極值
    if rv > 80:
        alerts.append(("⚠️ RSI超買", f"RSI={rv:.1f}，注意拉回", RED))
    elif rv < 20:
        alerts.append(("✅ RSI超賣", f"RSI={rv:.1f}，留意反彈", GRN))

    # 布林通道突破
    if "BBU" in df.columns and "BBL" in df.columns:
        bbu = float(df.iloc[-1]["BBU"])
        bbl = float(df.iloc[-1]["BBL"])
        if close > bbu:
            alerts.append(("📈 突破布林上軌", f"{close:.0f} > {bbu:.0f}", RED))
        if close < bbl:
            alerts.append(("📉 跌破布林下軌", f"{close:.0f} < {bbl:.0f}", GRN))

    # MACD 交叉
    prev_m = float(df.iloc[-2]["MACD"]) if "MACD" in df.columns else 0
    prev_s = float(df.iloc[-2]["SIG"])  if "SIG"  in df.columns else 0
    if prev_m < prev_s and mv > sv:
        alerts.append(("📈 MACD黃金交叉", "DIF上穿DEA", RED))
    if prev_m > prev_s and mv < sv:
        alerts.append(("📉 MACD死亡交叉", "DIF下穿DEA", GRN))

    # 均量比放量
    vol   = float(df.iloc[-1]["Volume"])
    avg5  = float(df["Volume"].tail(5).mean())
    if avg5 > 0 and vol / avg5 > 2.0:
        alerts.append(("🔥 爆量訊號", f"量能 {vol/avg5:.1f}x 均量", ORNG))

    return alerts

# ══════════════════════════════════════════════════════════════════
#  費波那契回撤
# ══════════════════════════════════════════════════════════════════

def fibonacci_levels(df, n=60):
    r    = df.tail(n)
    hi   = float(r["High"].max())
    lo   = float(r["Low"].min())
    diff = hi - lo
    return {
        "0.0% (高點)":  hi,
        "23.6%":        hi - diff * 0.236,
        "38.2%":        hi - diff * 0.382,
        "50.0%":        hi - diff * 0.500,
        "61.8%":        hi - diff * 0.618,
        "100% (低點)":  lo,
    }

# ══════════════════════════════════════════════════════════════════
#  AI 訊號評分
# ══════════════════════════════════════════════════════════════════

def ai_summary(close, ma5, ma20, ma60, kv, dv, rv, mv, sv, hv, atr_v):
    signals = []

    if ma5 > ma20 > ma60:
        signals.append(("多", "均線多頭排列"))
    elif ma5 < ma20 < ma60:
        signals.append(("空", "均線空頭排列"))

    if kv > 80:
        signals.append(("空", "KD高檔鈍化注意"))
    elif kv < 20:
        signals.append(("多", "KD低檔超賣反彈"))

    if rv < 30:
        signals.append(("多", "RSI超賣反彈機會"))
    elif rv > 70:
        signals.append(("空", "RSI超買注意拉回"))

    if mv > sv and hv > 0:
        signals.append(("多", "MACD黃金交叉擴張"))
    elif mv < sv and hv < 0:
        signals.append(("空", "MACD死亡交叉擴張"))

    if close > ma20:
        signals.append(("多", "站上MA20"))
    else:
        signals.append(("空", "跌破MA20"))

    bull  = sum(1 for s, _ in signals if s == "多")
    total = len(signals) if signals else 1
    score = int(bull / total * 100)

    at = atr_v if atr_v and not math.isnan(atr_v) else close * 0.025

    if score >= 70:
        verdict = "📈 偏多操作，可考慮布局"
        stop    = close - at * 1.5
        target  = close + at * 2.5
        col     = RED
    elif score <= 30:
        verdict = "📉 偏空操作，謹慎持股"
        stop    = close + at * 1.0
        target  = close - at * 2.0
        col     = GRN
    else:
        verdict = "⚖️ 中性觀望，等待訊號明朗"
        stop    = close - at
        target  = close + at
        col     = GOLD

    return verdict, score, signals, stop, target, col

# ══════════════════════════════════════════════════════════════════
#  匯出工具
# ══════════════════════════════════════════════════════════════════

def export_csv(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue().encode("utf-8-sig")

def export_summary(ticker, name, close, pct, kv, rv, mv, sv,
                   verdict, score, signals, stop, target):
    lines = [
        f"{'='*45}",
        f"  {name}（{ticker}）技術分析報告",
        f"{'='*45}",
        f"日期　　：{datetime.today().strftime('%Y-%m-%d %H:%M')}",
        f"收盤價　：{close:,.2f}　漲跌：{pct:+.2f}%",
        f"KD　　　：{kv:.1f}　RSI：{rv:.1f}",
        f"MACD　　：{'黃金交叉' if mv > sv else '死亡交叉'}",
        f"",
        f"【AI 訊號評分】{score}/100",
        f"建議　　：{verdict}",
        f"目標價　：{target:,.2f}",
        f"停損價　：{stop:,.2f}",
        f"",
        f"【訊號明細】",
    ]
    for s, desc in signals:
        lines.append(f"  {'▲' if s=='多' else '▼'} {desc}")
    lines += [
        f"",
        f"{'─'*45}",
        f"⚠ 本報告僅供學習與研究，不構成投資建議。",
        f"  投資有風險，操作請審慎。",
        f"{'─'*45}",
    ]
    return "\n".join(lines)

# ══════════════════════════════════════════════════════════════════
#  Plotly 圖表
# ══════════════════════════════════════════════════════════════════

def build_main(df, show_bb, show_fib):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.60, 0.20, 0.20],
        vertical_spacing=0.01,
    )

    # K 線
    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="K線",
        increasing=dict(line=dict(color=RED, width=0.8), fillcolor=RED),
        decreasing=dict(line=dict(color=GRN, width=0.8), fillcolor=GRN),
        whiskerwidth=0.4,
    ), row=1, col=1)

    for n, c, w in [("MA5","MA5",GOLD),("MA20","MA20",ORNG),("MA60","MA60",CYAN)]:
        fig.add_trace(go.Scatter(x=df["Date"], y=df[c], name=n,
            mode="lines", line=dict(color=w, width=1.2)), row=1, col=1)

    if show_bb:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["BBU"], name="BB上",
            mode="lines", line=dict(color="rgba(160,80,255,.7)", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=df["BBL"], name="BB下",
            mode="lines", line=dict(color="rgba(160,80,255,.7)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(160,80,255,.06)"), row=1, col=1)

    # 費波那契水平線
    if show_fib:
        fib_colors = {
            "0.0% (高點)":  "rgba(255,48,48,0.6)",
            "23.6%":        "rgba(255,215,0,0.5)",
            "38.2%":        "rgba(0,200,100,0.6)",
            "50.0%":        "rgba(0,200,255,0.6)",
            "61.8%":        "rgba(0,200,100,0.6)",
            "100% (低點)":  "rgba(0,200,100,0.6)",
        }
        for label, price in fibonacci_levels(df).items():
            fig.add_hline(
                y=price, line_dash="dash",
                line_color=fib_colors.get(label, "rgba(255,215,0,0.4)"),
                line_width=1,
                annotation_text=f" Fib {label} {price:.0f}",
                annotation_font=dict(size=8, color=MUTED),
                annotation_position="right",
                row=1, col=1,
            )

    # 近高近低標註
    hi_i = int(df["High"].idxmax())
    lo_i = int(df["Low"].idxmin())
    fig.add_annotation(
        x=df.loc[hi_i,"Date"], y=float(df.loc[hi_i,"High"]),
        text=f"近高 {float(df.loc[hi_i,'High']):.0f}",
        showarrow=True, arrowhead=2, arrowcolor=RED, ay=-25,
        font=dict(color=RED, size=9), bgcolor="rgba(255,48,48,.15)",
        bordercolor=RED, borderwidth=1, row=1, col=1)
    fig.add_annotation(
        x=df.loc[lo_i,"Date"], y=float(df.loc[lo_i,"Low"]),
        text=f"近低 {float(df.loc[lo_i,'Low']):.0f}",
        showarrow=True, arrowhead=2, arrowcolor=GRN, ay=25,
        font=dict(color=GRN, size=9), bgcolor="rgba(0,200,100,.15)",
        bordercolor=GRN, borderwidth=1, row=1, col=1)

    # 成交量
    bar_c = [RED if c >= o else GRN for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="量",
        marker_color=bar_c, opacity=0.75), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Volume"].rolling(5).mean(),
        name="均量", mode="lines", line=dict(color=GOLD, width=1.1)), row=2, col=1)

    # KD
    fig.add_trace(go.Scatter(x=df["Date"], y=df["K"], name="K",
        mode="lines", line=dict(color=GOLD, width=1.3)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["D"], name="D",
        mode="lines", line=dict(color=ORNG, width=1.3)), row=3, col=1)
    for lvl, clr in [(80, RED), (50, "#334466"), (20, GRN)]:
        fig.add_hline(y=lvl, line_dash="dot", line_color=clr, opacity=0.35, row=3, col=1)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=BG,
        height=560, margin=dict(l=52, r=6, t=8, b=6),
        legend=dict(orientation="h", y=1.005, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=9.5)),
        xaxis_rangeslider_visible=False,
        font=dict(family="Arial", color=TEXT, size=10),
        hovermode="x unified",
    )
    gc = "#142030"
    for i in range(1, 4):
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
        mode="gauge+number",
        value=value,
        number=dict(suffix="%", font=dict(size=30, color=clr, family="Arial")),
        title=dict(text=lbl, font=dict(size=12, color=MUTED)),
        gauge=dict(
            axis=dict(range=[0,100], tickfont=dict(size=9, color=MUTED),
                      tickwidth=1, tickcolor="#224"),
            bar=dict(color=clr, thickness=0.28),
            bgcolor=BG,
            borderwidth=0,
            steps=[
                dict(range=[0,35],   color="#091A10"),
                dict(range=[35,65],  color="#141408"),
                dict(range=[65,100], color="#1A0909"),
            ],
            threshold=dict(line=dict(color="#fff",width=2), thickness=0.7, value=value),
        ),
    ))
    fig.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        height=200, margin=dict(l=15, r=15, t=30, b=10),
        font=dict(family="Arial", color=TEXT),
    )
    return fig


def build_macd(df):
    h = df["HIST"].fillna(0)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Date"].tail(40), y=h.tail(40),
        marker_color=[RED if v >= 0 else GRN for v in h.tail(40)],
        opacity=0.75, name="柱"))
    fig.add_trace(go.Scatter(x=df["Date"].tail(40), y=df["MACD"].tail(40),
        mode="lines", line=dict(color=CYAN, width=1.3), name="DIF"))
    fig.add_trace(go.Scatter(x=df["Date"].tail(40), y=df["SIG"].tail(40),
        mode="lines", line=dict(color=GOLD, width=1.3), name="DEA"))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=BG,
        height=150, margin=dict(l=45, r=6, t=18, b=6),
        legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=9)),
        font=dict(family="Arial", color=TEXT, size=9),
        xaxis=dict(gridcolor="#142030", zeroline=False),
        yaxis=dict(gridcolor="#142030", zeroline=False),
        hovermode="x unified",
    )
    return fig


def build_ai_score_chart(score, signals):
    """AI 評分雷達圖"""
    categories = ["均線", "KD", "RSI", "MACD", "量價"]
    # 從 signals 對應分數
    keyword_map = {
        "均線": ["均線多頭", "均線空頭"],
        "KD":   ["KD低檔", "KD高檔"],
        "RSI":  ["RSI超賣", "RSI超買"],
        "MACD": ["MACD黃金", "MACD死亡"],
        "量價": ["站上MA20", "跌破MA20"],
    }
    vals = []
    for cat, keys in keyword_map.items():
        matched = [s for s, d in signals if any(k in d for k in keys)]
        if matched:
            vals.append(80 if matched[0] == "多" else 20)
        else:
            vals.append(50)
    vals += vals[:1]
    cats  = categories + categories[:1]

    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats, fill="toself",
        fillcolor=f"rgba(0,200,255,0.15)",
        line=dict(color=CYAN, width=1.5),
        name="技術評分",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=BG,
            radialaxis=dict(visible=True, range=[0,100],
                            gridcolor=BORD, tickfont=dict(size=8, color=MUTED)),
            angularaxis=dict(gridcolor=BORD, tickfont=dict(size=9, color=TEXT)),
        ),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        height=220, margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
        font=dict(family="Arial", color=TEXT),
    )
    return fig

# ══════════════════════════════════════════════════════════════════
#  HTML 小工具
# ══════════════════════════════════════════════════════════════════

def hdr(title, icon=""):
    return (f'<div style="color:{CYAN};font-size:.76rem;font-weight:700;'
            f'padding:3px 0 5px;border-bottom:1px solid {BORD};margin-bottom:7px;">'
            f'{icon} {title}</div>')

def row_item(lbl, val, vc=TEXT, bg=CARD):
    return (f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:3px 8px;border-radius:3px;background:{bg};margin:2px 0;">'
            f'<span style="color:{MUTED};font-size:.73rem;">{lbl}</span>'
            f'<span style="color:{vc};font-weight:700;font-size:.8rem;">{val}</span></div>')

def box(content, title="", icon=""):
    h = hdr(title, icon) if title else ""
    return (f'<div style="background:{CARD};border:1px solid {BORD};border-radius:6px;'
            f'padding:9px 11px;margin-bottom:7px;font-family:Arial,sans-serif;">{h}{content}</div>')

def tag(txt, fg, bg):
    return (f'<span style="background:{bg};color:{fg};border-radius:3px;'
            f'padding:2px 7px;font-size:.72rem;font-weight:700;margin:2px 2px 2px 0;'
            f'display:inline-block;">{txt}</span>')

# ══════════════════════════════════════════════════════════════════
#  各面板 HTML 建構函式
# ══════════════════════════════════════════════════════════════════

def html_ohlcv(df):
    r   = df.iloc[-1]
    p   = df.iloc[-2]
    cl  = float(r["Close"]); pc = float(p["Close"])
    chg = cl - pc; pct = chg / pc * 100
    cc  = RED if chg >= 0 else GRN
    sym = "▲" if chg >= 0 else "▼"
    fields = [
        ("開", f"{float(r['Open']):,.0f}"),
        ("高", f"{float(r['High']):,.0f}"),
        ("低", f"{float(r['Low']):,.0f}"),
        ("收", f"{float(r['Close']):,.0f}"),
        ("量", f"{int(r['Volume']):,}"),
        ("漲跌", f"{sym}{abs(pct):.2f}%"),
    ]
    cells = "".join(
        f'<div style="text-align:center;flex:1;min-width:55px;">'
        f'<div style="color:{MUTED};font-size:.65rem;">{lb}</div>'
        f'<div style="color:{cc if lb=="漲跌" else TEXT};font-size:.83rem;font-weight:700;">{v}</div></div>'
        for lb, v in fields
    )
    return box(f'<div style="display:flex;gap:4px;flex-wrap:wrap;">{cells}</div>')


def html_tech(df, ma5, ma20, ma60, kv, dv, rv, mv, sv, hv):
    if   ma5 > ma20 > ma60: tr, tc = "多頭趨勢", RED
    elif ma5 < ma20 < ma60: tr, tc = "空頭趨勢", GRN
    else:                   tr, tc = "盤整趨勢", GOLD

    if   ma5 > ma20 > ma60: ar, ac = "多頭排列 (5>20>60)", RED
    elif ma5 < ma20 < ma60: ar, ac = "空頭排列 (5<20<60)", GRN
    else:                   ar, ac = "交叉整理中",           GOLD

    if   kv > 80: ks, kc = "KD高檔鈍化 ⚠", RED
    elif kv < 20: ks, kc = "KD低檔超賣 ✓", GRN
    else:         ks, kc = "KD正常區間",    GOLD

    if   rv > 70: rs, rc = f"{rv:.0f} 超買", RED
    elif rv < 30: rs, rc = f"{rv:.0f} 超賣", GRN
    else:         rs, rc = f"{rv:.0f}",       TEXT

    macd_s = ("正值收斂" if mv>0 and hv>0 else
              "正值擴張" if mv>0 else
              "負值收斂" if hv>0 else "負值擴張")
    macd_c = RED if mv > 0 else GRN
    cross  = "黃金交叉 📈" if mv > sv else "死亡交叉 📉"
    cross_c = RED if mv > sv else GRN

    cl   = float(df.iloc[-1]["Close"]); pc = float(df.iloc[-2]["Close"])
    vol  = float(df.iloc[-1]["Volume"]); av = float(df["Volume"].tail(5).mean())
    vr   = vol / av if av > 0 else 1
    vs   = "量能擴張 🔥" if vr > 1.2 else ("量能萎縮" if vr < 0.8 else "量能持平")
    vc   = RED if "擴" in vs else (GRN if "縮" in vs else GOLD)
    up   = cl > pc
    vp   = ("量增價漲 ✓" if up and vr>1 else
            "量縮價跌"   if not up and vr<1 else
            "量價背離 ⚠")
    vpc  = RED if "漲" in vp else (GRN if "跌" in vp else GOLD)

    return box("".join([
        row_item("趨勢方向", tr,     tc),
        row_item("MA狀態",   ar,     ac),
        row_item("KD指標",   ks,     kc),
        row_item("MACD",     macd_s, macd_c),
        row_item("MACD交叉", cross,  cross_c),
        row_item("RSI(14)",  rs,     rc),
        row_item("成交量",   vs,     vc),
        row_item("量價關係", vp,     vpc),
    ]), "技術分析總覽", "◉")


def html_industry(sector, industry):
    items = ["半導體設備備貨積極", "受惠先進製程擴產需求",
             "產業景氣維持高檔",   "設備廠表現相對強勢"]
    rows = "".join(
        f'<div style="color:{MUTED};font-size:.71rem;padding:2px 0;">• {i}</div>'
        for i in items
    )
    return box(
        f'<div style="color:{CYAN};font-size:.74rem;margin-bottom:5px;">{sector} / {industry}</div>{rows}',
        "產業概況", "◈"
    )


def html_company(name, pe, mktcap):
    pe_s = f"{pe:.1f}x" if pe else "—"
    mc_s = f"{mktcap/1e8:,.0f} 億" if mktcap else "—"
    items = [f"本益比：{pe_s}", f"市值：{mc_s}",
             "技術領先，毛利率佳", "訂單能見度高", "營運成長動能強勁"]
    rows = "".join(
        f'<div style="color:{MUTED};font-size:.71rem;padding:2px 0;">• {i}</div>'
        for i in items
    )
    return box(rows, "公司概況", "◇")


def html_board(ip, ep, iv, ev):
    bar = (f'<div style="background:#0A1428;border-radius:4px;overflow:hidden;'
           f'height:7px;margin:7px 0;">'
           f'<div style="background:{GRN};height:100%;width:{ip}%;float:left;"></div>'
           f'<div style="background:{RED};height:100%;width:{ep}%;float:left;"></div></div>')
    sig = "外盤大於內盤，買盤較積極" if ep > ip else "內盤大於外盤，賣壓較重"
    nums = (
        f'<div style="display:flex;justify-content:space-between;margin-bottom:6px;">'
        f'<div style="text-align:center;flex:1;">'
        f'  <div style="color:{MUTED};font-size:.65rem;">內盤（賣）</div>'
        f'  <div style="color:{GRN};font-size:1.1rem;font-weight:900;">{iv:,}</div>'
        f'  <div style="color:{GRN};font-size:.71rem;">({ip:.1f}%)</div></div>'
        f'<div style="text-align:center;flex:1;">'
        f'  <div style="color:{MUTED};font-size:.65rem;">外盤（買）</div>'
        f'  <div style="color:{RED};font-size:1.1rem;font-weight:900;">{ev:,}</div>'
        f'  <div style="color:{RED};font-size:.71rem;">({ep:.1f}%)</div></div>'
        f'<div style="text-align:center;flex:1;">'
        f'  <div style="color:{MUTED};font-size:.65rem;">內外盤比</div>'
        f'  <div style="color:{CYAN};font-size:1rem;font-weight:900;">{ip}:{ep}</div></div>'
        f'</div>'
    )
    foot = f'<div style="color:{MUTED};font-size:.7rem;text-align:center;">{sig}</div>'
    return box(nums + bar + foot, "內外盤結構", "◎")


def html_main_force(status, trend, cc, chip_c, chip_s, mf_col, df):
    recent = df.tail(5)
    rows_h = ""
    for _, r in recent.iterrows():
        is_b = float(r["Close"]) >= float(r["Open"])
        col  = RED if is_b else GRN
        lbl  = r["Date"].strftime("%m/%d") if hasattr(r["Date"], "strftime") else str(r["Date"])
        val  = f"{'買' if is_b else '賣'} {int(r['Volume']/1000):.0f}K"
        rows_h += row_item(lbl, val, col)

    top = (f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:7px;">'
           f'<div style="width:10px;height:10px;border-radius:50%;background:{mf_col};"></div>'
           f'<span style="color:{mf_col};font-weight:700;font-size:.84rem;">{status}</span>'
           f'<span style="color:{MUTED};font-size:.7rem;">{trend}</span></div>')
    foot = (f'<div style="margin-top:6px;">'
            f'{tag(f"集中度 {chip_c}", mf_col, CARD)}'
            f'{tag(f"穩定度 {chip_s}", mf_col, CARD)}'
            f'{tag(cc, mf_col, CARD)}</div>')
    return box(top + rows_h + foot, "主力出貨警示（近5日）", "◉")


def html_key_levels(df, close):
    r20  = df.tail(20)
    r60  = df.tail(60)
    res1 = float(r20["High"].max())
    res2 = float(r60["High"].max())
    sup1 = float(r20["Low"].min())
    sup2 = float(r60["Low"].min())

    def lvl_box(label, val, col, bg, dist):
        return (f'<div style="background:{bg};border-radius:4px;padding:6px 10px;margin:3px 0;'
                f'border-left:3px solid {col};">'
                f'<div style="display:flex;justify-content:space-between;">'
                f'<span style="color:{MUTED};font-size:.71rem;">{label}</span>'
                f'<span style="color:{col};font-weight:900;font-size:.9rem;">{val:,.0f}</span></div>'
                f'<div style="color:{MUTED};font-size:.67rem;">距現價 {dist:+.1f}%</div></div>')

    cur = (f'<div style="text-align:center;padding:4px 0;">'
           f'<span style="color:{CYAN};font-size:.7rem;">── 現價 {close:,.2f} ──</span></div>')
    foot = f'<div style="color:{MUTED};font-size:.7rem;text-align:center;margin-top:5px;">守支撐 → 多方格局不變</div>'

    return box(
        lvl_box("壓力①（近20日）", res1, RED, "#1A0808", (res1-close)/close*100) +
        lvl_box("壓力②（近60日）", res2, RED, "#1A0808", (res2-close)/close*100) +
        cur +
        lvl_box("支撐①（近20日）", sup1, GRN, "#081A0E", -(close-sup1)/close*100) +
        lvl_box("支撐②（近60日）", sup2, GRN, "#081A0E", -(close-sup2)/close*100) +
        foot,
        "關鍵價位", "🎯"
    )


def html_kline_patterns(kp, cp):
    krows = "".join(
        f'<div style="display:flex;justify-content:space-between;padding:3px 8px;'
        f'border-radius:3px;background:{CARD};margin:2px 0;">'
        f'<span style="color:{TEXT};font-size:.75rem;">{n}</span>'
        f'<span style="color:{c};font-size:.73rem;">{t}</span></div>'
        for n, t, c in kp
    )
    crows = "".join(
        f'<div style="padding:4px 8px;border-radius:3px;background:{CARD};margin:3px 0;'
        f'border-left:2px solid {c};">'
        f'<div style="color:{c};font-size:.75rem;font-weight:700;">{n}</div>'
        f'<div style="color:{MUTED};font-size:.69rem;">{d}</div></div>'
        for n, c, d in cp
    )
    sep = f'<div style="color:{MUTED};font-size:.7rem;margin:6px 0 3px;">型態分析（不破底）</div>'
    return box(krows + sep + crows, "K線型態", "◆")


def html_multiperiod(daily, weekly, monthly):
    results = [
        trend_label(daily,   "日K"),
        trend_label(weekly,  "週K"),
        trend_label(monthly, "月K"),
    ]
    rows_h = ""
    for tf, tr, col, note in results:
        rows_h += (
            f'<div style="background:{CARD};border-radius:4px;padding:5px 9px;margin:3px 0;">'
            f'<div style="display:flex;justify-content:space-between;">'
            f'<span style="color:{TEXT};font-size:.78rem;font-weight:700;">{tf}</span>'
            f'<span style="color:{col};font-size:.75rem;font-weight:700;">{tr}</span></div>'
            f'<div style="color:{MUTED};font-size:.67rem;">{note}</div></div>'
        )
    return box(rows_h, "多週期分析", "◍")


def html_vol_price(df):
    cl   = float(df.iloc[-1]["Close"])
    pc   = float(df.iloc[-2]["Close"])
    vol  = float(df.iloc[-1]["Volume"])
    a5   = float(df["Volume"].tail(5).mean())
    a20  = float(df["Volume"].tail(20).mean())
    vr5  = vol / a5  if a5  > 0 else 1
    vr20 = vol / a20 if a20 > 0 else 1
    m20  = float(df["MA20"].iloc[-1]) if "MA20" in df.columns else 0
    m60  = float(df["MA60"].iloc[-1]) if "MA60" in df.columns else 0
    up   = cl > pc
    vp   = ("量增價漲 ✓" if up and vr5>1 else "量縮價跌" if not up and vr5<1 else "量價背離 ⚠")
    vpc  = RED if "漲" in vp else (GRN if "跌" in vp else GOLD)
    return box("".join([
        row_item("量價關係",     vp,               vpc),
        row_item("均量比(5日)",  f"{vr5:.2f}x",    RED if vr5>1.2 else (GRN if vr5<0.8 else GOLD)),
        row_item("均量比(20日)", f"{vr20:.2f}x",   RED if vr20>1.2 else (GRN if vr20<0.8 else GOLD)),
        row_item("距MA20",       f"{(cl-m20)/cl*100:+.1f}%" if m20 else "—",
                 RED if cl > m20 else GRN),
        row_item("距MA60",       f"{(cl-m60)/cl*100:+.1f}%" if m60 else "—",
                 RED if cl > m60 else GRN),
    ]), "量價結構分析", "◈")


def html_chip(df, mf_status, chip_c, chip_s, mf_col,
              foreign=None, invest=None, dealer=None):
    cl  = float(df.iloc[-1]["Close"])
    m20 = float(df["MA20"].iloc[-1]) if "MA20" in df.columns else 0
    m60 = float(df["MA60"].iloc[-1]) if "MA60" in df.columns else 0
    pos = "多方主導" if cl > m20 > m60 else ("空方主導" if cl < m20 < m60 else "均線糾結")
    pc  = RED if "多" in pos else (GRN if "空" in pos else GOLD)

    rows = [
        row_item("主力動向",   mf_status,   mf_col),
        row_item("籌碼集中度", f"{chip_c}", mf_col),
        row_item("籌碼穩定度", f"{chip_s}", mf_col),
        row_item("多空主導",   pos,         pc),
        row_item("MA20站穩",   "是 ✓" if cl > m20 else "否 ✗",
                 RED if cl > m20 else GRN),
        row_item("主力進出",   "進出見買" if mf_status in ["進出見買","中性偏多"] else "進出見賣",
                 RED if mf_status in ["進出見買","中性偏多"] else GRN),
    ]

    # 三大法人（若有資料）
    if foreign is not None:
        def fmt(v):
            if v is None: return "—"
            return f"+{v:,}" if v >= 0 else f"{v:,}"
        fc = RED if (foreign or 0) >= 0 else GRN
        ic = RED if (invest  or 0) >= 0 else GRN
        dc = RED if (dealer  or 0) >= 0 else GRN
        rows += [
            row_item("外資買賣超", fmt(foreign), fc),
            row_item("投信買賣超", fmt(invest),  ic),
            row_item("自營商買賣超", fmt(dealer), dc),
        ]

    return box("".join(rows), "籌碼結構分析", "◈")


def html_day_script(df, close, atr_v):
    at = atr_v if atr_v and not math.isnan(atr_v) else close * 0.025
    sc = [
        ("① 開高走高", RED,  f"開盤 > {close:.0f} 量能同步",
         f"目標 {close+at:.0f} / {close+2*at:.0f}", f"停損 {close-at*.5:.0f}"),
        ("② 盤整整理", GOLD, f"盤在 {close-at*.5:.0f}~{close+at*.5:.0f}",
         "觀望 等突破方向", f"停損 {close-at:.0f}"),
        ("③ 開低回測", GRN,  f"開盤 < {close:.0f} 觀察量能",
         f"目標 {close-at:.0f} / {close-2*at:.0f}", f"停損 {close+at*.5:.0f}"),
    ]
    cards = "".join(
        f'<div style="background:{BG};border-radius:4px;padding:7px 9px;margin:3px 0;'
        f'border-left:3px solid {c};">'
        f'<div style="color:{c};font-size:.76rem;font-weight:700;margin-bottom:3px;">{t}</div>'
        f'<div style="color:{MUTED};font-size:.68rem;">{cond}</div>'
        f'<div style="color:{TEXT};font-size:.69rem;">{tgt}</div>'
        f'<div style="color:{ORNG};font-size:.69rem;">{stp}</div></div>'
        for t, c, cond, tgt, stp in sc
    )
    return box(cards, f"日操作劇本（{datetime.today().strftime('%m/%d')}）", "◐")


def html_rsi_bar(rsi_val):
    if   rsi_val > 70: clr, lbl = RED,  "超買"
    elif rsi_val < 30: clr, lbl = GRN,  "超賣"
    else:              clr, lbl = GOLD, "中性"
    pct = min(max(rsi_val, 0), 100)
    bar = (f'<div style="background:{BG};border-radius:3px;overflow:hidden;height:8px;margin:4px 0;">'
           f'<div style="background:{clr};height:100%;width:{pct:.0f}%;"></div></div>')
    return box(
        f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
        f'<span style="color:{MUTED};font-size:.73rem;">RSI(14)</span>'
        f'<span style="color:{clr};font-weight:700;font-size:.82rem;">{rsi_val:.1f} {lbl}</span></div>'
        + bar
    )


def html_alerts(alerts):
    if not alerts:
        return box(
            f'<div style="color:{MUTED};font-size:.73rem;text-align:center;padding:6px 0;">'
            f'✅ 目前無觸發警示</div>',
            "即時警示", "🔔"
        )
    rows = "".join(
        f'<div style="background:{BG};border-radius:4px;padding:5px 9px;margin:3px 0;'
        f'border-left:3px solid {c};">'
        f'<div style="color:{c};font-size:.76rem;font-weight:700;">{msg}</div>'
        f'<div style="color:{MUTED};font-size:.68rem;">{detail}</div></div>'
        for msg, detail, c in alerts
    )
    return box(rows, f"即時警示（{len(alerts)} 項）", "🔔")


def html_ai_summary(verdict, score, signals, stop, target, col):
    score_bar = (
        f'<div style="background:{BG};border-radius:3px;overflow:hidden;height:8px;margin:5px 0;">'
        f'<div style="background:{col};height:100
