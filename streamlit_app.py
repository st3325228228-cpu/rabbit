"""
台股技術分析儀表板 ── Streamlit 版本
依賴：pip install streamlit yfinance plotly pandas numpy
執行：streamlit run stock_dashboard_streamlit.py
"""

import warnings, math
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
    initial_sidebar_state="collapsed",
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
  /* 全域背景 */
  .stApp, .main, [data-testid="stAppViewContainer"] {{
      background-color: {BG} !important;
      color: {TEXT} !important;
  }}
  /* 頂部欄 & 側欄 */
  [data-testid="stHeader"], [data-testid="stSidebar"] {{
      background-color: {PANEL} !important;
  }}
  /* 輸入元件 */
  .stTextInput input, .stNumberInput input {{
      background-color: {CARD} !important;
      border: 1px solid {BORD} !important;
      color: #E0EAF4 !important;
      border-radius: 5px !important;
  }}
  .stSlider > div > div {{ background-color: {CYAN} !important; }}
  /* 按鈕 */
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
  /* Checkbox */
  .stCheckbox label {{ color: {TEXT} !important; }}
  /* plotly 圖 */
  .js-plotly-plot, .plotly-graph-div {{ background: transparent !important; }}
  /* 隱藏 Streamlit 標誌 */
  #MainMenu, footer, [data-testid="stToolbar"] {{ visibility: hidden; }}
  /* 欄位間距 */
  [data-testid="column"] {{ padding: 0 4px !important; }}
  /* 卷軸 */
  ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
  ::-webkit-scrollbar-track {{ background: {BG}; }}
  ::-webkit-scrollbar-thumb {{ background: {BORD}; border-radius: 3px; }}
  /* label */
  label[data-testid="stWidgetLabel"] > p {{ color: {MUTED} !important; font-size: .78rem !important; }}
  /* 分隔線 */
  hr {{ border-color: {BORD} !important; }}
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
    d["K"], d["D"]             = kd(d)
    d["MACD"], d["SIG"], d["HIST"] = macd(d["Close"])
    d["RSI"]                   = rsi(d["Close"])
    d["BBU"], d["BBM"], d["BBL"] = bb(d["Close"])
    d["ATR"]                   = atr(d)

    w = get("1wk")
    if w is not None and len(w) >= 5:
        w["MA5"], w["MA20"] = ma(w["Close"], 5), ma(w["Close"], 20)
        w["K"], w["D"]      = kd(w)
        w["MACD"], w["SIG"], w["HIST"] = macd(w["Close"])

    mo = get("1mo")
    if mo is not None and len(mo) >= 3:
        mo["MA5"]        = ma(mo["Close"], 5)
        mo["K"], mo["D"] = kd(mo)

    return d, w, mo

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
        return "進出見買", "籌碼集中", "高", "穩定",    RED,  "多方偏強"
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
    kv   = float(r.get("K", 50)) if "K" in df_tf.columns else 50
    mv   = float(r.get("MACD", 0)) if "MACD" in df_tf.columns else 0
    sv   = float(r.get("SIG",  0)) if "SIG"  in df_tf.columns else 0
    ma5v = float(r.get("MA5", float(r["Close"]))) if "MA5" in df_tf.columns else float(r["Close"])
    trend = "多頭" if float(r["Close"]) > ma5v else "空頭"
    col   = RED if trend == "多頭" else GRN
    kd_s  = "高鈍" if kv > 80 else ("低超" if kv < 20 else "中")
    ma_s  = "黃金" if mv > sv else "死亡"
    return tf_name, f"{trend} ({chg:+.1f}%)", col, f"KD:{kd_s} MACD:{ma_s}交叉"

# ══════════════════════════════════════════════════════════════════
#  Plotly 圖表
# ══════════════════════════════════════════════════════════════════

def build_main(df, show_bb):
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

    bar_c = [RED if c >= o else GRN for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="量",
        marker_color=bar_c, opacity=0.75), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Volume"].rolling(5).mean(),
        name="均量", mode="lines", line=dict(color=GOLD, width=1.1)), row=2, col=1)

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
    rows = "".join(f'<div style="color:{MUTED};font-size:.71rem;padding:2px 0;">• {i}</div>' for i in items)
    return box(
        f'<div style="color:{CYAN};font-size:.74rem;margin-bottom:5px;">{sector} / {industry}</div>{rows}',
        "產業概況", "◈"
    )


def html_company(name, pe, mktcap):
    pe_s = f"{pe:.1f}x" if pe else "—"
    mc_s = f"{mktcap/1e8:,.0f} 億" if mktcap else "—"
    items = [f"本益比：{pe_s}", f"市值：{mc_s}",
             "技術領先，毛利率佳", "訂單能見度高", "營運成長動能強勁"]
    rows = "".join(f'<div style="color:{MUTED};font-size:.71rem;padding:2px 0;">• {i}</div>' for i in items)
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
        row_item("量價關係",    vp,                  vpc),
        row_item("均量比(5日)", f"{vr5:.2f}x",   RED if vr5>1.2 else (GRN if vr5<0.8 else GOLD)),
        row_item("均量比(20日)",f"{vr20:.2f}x",  RED if vr20>1.2 else (GRN if vr20<0.8 else GOLD)),
        row_item("距MA20",      f"{(cl-m20)/cl*100:+.1f}%" if m20 else "—",
                 RED if cl > m20 else GRN),
        row_item("距MA60",      f"{(cl-m60)/cl*100:+.1f}%" if m60 else "—",
                 RED if cl > m60 else GRN),
    ]), "量價結構分析", "◈")


def html_chip(df, mf_status, chip_c, chip_s, mf_col):
    cl  = float(df.iloc[-1]["Close"])
    m20 = float(df["MA20"].iloc[-1]) if "MA20" in df.columns else 0
    m60 = float(df["MA60"].iloc[-1]) if "MA60" in df.columns else 0
    pos = "多方主導" if cl > m20 > m60 else ("空方主導" if cl < m20 < m60 else "均線糾結")
    pc  = RED if "多" in pos else (GRN if "空" in pos else GOLD)
    return box("".join([
        row_item("主力動向",   mf_status,   mf_col),
        row_item("籌碼集中度", f"{chip_c}", mf_col),
        row_item("籌碼穩定度", f"{chip_s}", mf_col),
        row_item("多空主導",   pos,         pc),
        row_item("MA20站穩", "是 ✓" if cl > m20 else "否 ✗",
                 RED if cl > m20 else GRN),
        row_item("主力進出", "進出見買" if mf_status in ["進出見買","中性偏多"] else "進出見賣",
                 RED if mf_status in ["進出見買","中性偏多"] else GRN),
    ]), "籌碼結構分析", "◈")


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

# ══════════════════════════════════════════════════════════════════
#  主頁面
# ══════════════════════════════════════════════════════════════════

def build_header_html(ticker, name, close, chg, pct, vol, mktcap, pe,
                      ma5, ma20, ma60, kv, dv, rv, mv, sv,
                      ip, ep, iv, ev, date_s):
    cc  = RED  if chg >= 0 else GRN
    sym = "▲"  if chg >= 0 else "▼"
    mktcap_s = f"{mktcap/1e8:,.0f} 億" if mktcap else "—"
    pe_s     = f"{pe:.1f}" if pe else "—"

    return f"""
    <div style="background:linear-gradient(90deg,#05080F,#0B1528,#05080F);
                border:1px solid {BORD};border-radius:8px;
                padding:10px 16px;font-family:Arial,sans-serif;margin-bottom:6px;">
      <div style="display:flex;align-items:center;flex-wrap:wrap;gap:14px;">
        <div>
          <span style="color:#8090A0;font-size:.78rem;">{ticker.replace('.TW','')}</span>
          <span style="color:#E0EAF4;font-size:.82rem;font-weight:700;margin-left:5px;">{name}</span>
        </div>
        <div style="display:flex;align-items:baseline;gap:9px;">
          <span style="color:{cc};font-size:2.2rem;font-weight:900;letter-spacing:1px;line-height:1;">
            {close:,.2f}
          </span>
          <span style="color:{cc};font-size:.94rem;font-weight:700;">
            {sym} {abs(chg):.2f}（{abs(pct):.2f}%）
          </span>
        </div>
        <div style="display:flex;gap:10px;flex-wrap:wrap;margin-left:auto;">
          {''.join(
            f'<div style="text-align:center;background:{CARD};border-radius:4px;padding:4px 10px;">'
            f'<div style="color:{MUTED};font-size:.6rem;">{lb}</div>'
            f'<div style="color:{vc};font-size:.8rem;font-weight:700;">{vl}</div></div>'
            for lb, vl, vc in [
                ("成交量",   f"{vol:,}",       TEXT),
                ("日期",     date_s,           TEXT),
                ("市值",     mktcap_s,         TEXT),
                ("本益比",   pe_s,             TEXT),
                ("內盤",     f"{iv:,} ({ip}%)", GRN),
                ("外盤",     f"{ev:,} ({ep}%)", RED),
                ("內外盤比", f"{ip}:{ep}",     CYAN),
            ])}
        </div>
      </div>
      <div style="display:flex;gap:14px;margin-top:6px;flex-wrap:wrap;">
        <span style="color:{MUTED};font-size:.68rem;">日K線圖</span>
        <span style="color:{GOLD};font-size:.7rem;">MA5 {ma5:,.2f}</span>
        <span style="color:{ORNG};font-size:.7rem;">MA20 {ma20:,.2f}</span>
        <span style="color:{CYAN};font-size:.7rem;">MA60 {ma60:,.2f}</span>
        <span style="color:{MUTED};font-size:.7rem;">KD {kv:.0f}/{dv:.0f}</span>
        <span style="color:{MUTED};font-size:.7rem;">RSI {rv:.0f}</span>
        <span style="color:{RED if mv>sv else GRN};font-size:.7rem;">
          MACD {'黃金↑' if mv>sv else '死亡↓'}
        </span>
      </div>
    </div>"""


def main():
    # ── Banner ────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:linear-gradient(90deg,#05080F,#0C1828,#05080F);
                border-bottom:2px solid {CYAN};padding:10px 16px;margin-bottom:8px;
                border-radius:8px;display:flex;align-items:center;gap:12px;">
      <span style="font-size:1.6rem;">📈</span>
      <div>
        <h1 style="color:#E0EAF4;margin:0;font-size:1.1rem;font-weight:900;letter-spacing:1px;">
          台股技術分析儀表板
        </h1>
        <p style="color:{MUTED};margin:2px 0 0;font-size:.72rem;">
          K線 · 均線 · KD · MACD · RSI · 布林通道 · 關鍵價位 · 籌碼 · 日操作劇本
        </p>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── 控制列 ────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([3, 3, 2, 2])
    with c1:
        ticker_input = st.text_input(
            "股票代號",
            value=st.session_state.get("ticker", "2330.TW"),
            placeholder="2330 / 2330.TW / TSLA",
            help="台股加 .TW，如 2330.TW",
        )
    with c2:
        period_days = st.slider("查詢天數", min_value=30, max_value=365,
                                value=st.session_state.get("period", 90), step=15)
    with c3:
        show_bb = st.checkbox("顯示布林通道",
                              value=st.session_state.get("show_bb", False))
    with c4:
        st.markdown("<br>", unsafe_allow_html=True)
        query_btn = st.button("🔍 查詢分析", use_container_width=True)

    # 儲存狀態
    st.session_state["ticker"]  = ticker_input
    st.session_state["period"]  = period_days
    st.session_state["show_bb"] = show_bb

    # 自動首次載入 或 按鈕觸發
    if "loaded" not in st.session_state or query_btn:
        st.session_state["loaded"] = True

    if not st.session_state.get("loaded"):
        return

    ticker = ticker_input.strip()
    if not ticker:
        st.error("請輸入股票代號")
        return
    if "." not in ticker:
        ticker += ".TW"

    with st.spinner(f"正在抓取 {ticker} 資料…"):
        daily, weekly, monthly = fetch_all(ticker, period_days)

    if daily is None or daily.empty:
        st.error(f"❌ 無法取得 {ticker}，請確認代號或網路。")
        return

    daily = daily.tail(int(period_days)).reset_index(drop=True)
    if len(daily) < 3:
        st.error("資料筆數不足，請增加查詢天數。")
        return

    # 公司資訊
    try:
        info     = yf.Ticker(ticker).info
        name     = info.get("longName", info.get("shortName", ticker))
        sector   = info.get("sector",   "科技")
        industry = info.get("industry", "半導體")
        pe       = info.get("trailingPE", None)
        mktcap   = info.get("marketCap",  None)
    except Exception:
        name = ticker; sector = industry = "—"; pe = mktcap = None

    # 最新值
    r, p  = daily.iloc[-1], daily.iloc[-2]
    s     = lambda x: float(x) if not (isinstance(x, float) and math.isnan(x)) else 0.0
    close = s(r["Close"]);  prev_c = s(p["Close"])
    chg   = close - prev_c; pct    = chg / prev_c * 100
    vol   = int(r["Volume"])
    ma5   = s(r.get("MA5",  0))
    ma20  = s(r.get("MA20", 0))
    ma60  = s(r.get("MA60", 0))
    kv    = s(r.get("K",   50))
    dv    = s(r.get("D",   50))
    rv    = s(r.get("RSI", 50))
    mv    = s(r.get("MACD", 0))
    sv    = s(r.get("SIG",  0))
    hv    = s(r.get("HIST", 0))
    atr_v = s(r.get("ATR",  close * 0.02))

    ip, ep     = board_ratio(daily)
    iv, ev     = int(vol * ip / 100), int(vol * ep / 100)
    wr         = win_rate(daily)
    mf_stat, mf_trend, chip_c, chip_s, mf_col, mf_cc = main_force(daily)
    kp         = kline_patterns(daily)
    cp         = chart_patterns(daily)
    date_s     = (r["Date"].strftime("%m/%d") if hasattr(r["Date"], "strftime") else str(r["Date"]))

    # ── 標頭 ─────────────────────────────────────────────────────
    st.markdown(build_header_html(
        ticker, name, close, chg, pct, vol, mktcap, pe,
        ma5, ma20, ma60, kv, dv, rv, mv, sv,
        ip, ep, iv, ev, date_s
    ), unsafe_allow_html=True)

    # ── 主體四欄 ─────────────────────────────────────────────────
    col_left, col_mid, col_right, col_far = st.columns([42, 20, 20, 20])

    with col_left:
        st.plotly_chart(build_main(daily, show_bb),
                        use_container_width=True, config={"displayModeBar": False})
        st.markdown(html_ohlcv(daily) + html_rsi_bar(rv), unsafe_allow_html=True)
        st.plotly_chart(build_macd(daily),
                        use_container_width=True, config={"displayModeBar": False})

    with col_mid:
        st.markdown(html_main_force(mf_stat, mf_trend, mf_cc,
                                    chip_c, chip_s, mf_col, daily),
                    unsafe_allow_html=True)
        # 勝率 gauge
        st.markdown(f"""
        <div style="color:{CYAN};font-size:.76rem;font-weight:700;
             padding:3px 0 5px;border-bottom:1px solid {BORD};margin-bottom:4px;
             font-family:Arial,sans-serif;">
          ⚡ 短線勝率（近10日）
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(build_gauge(wr),
                        use_container_width=True, config={"displayModeBar": False})
        st.markdown(html_key_levels(daily, close), unsafe_allow_html=True)

    with col_right:
        st.markdown(
            html_tech(daily, ma5, ma20, ma60, kv, dv, rv, mv, sv, hv) +
            html_industry(sector, industry) +
            html_company(name, pe, mktcap) +
            html_board(ip, ep, iv, ev),
            unsafe_allow_html=True
        )

    with col_far:
        st.markdown(
            html_vol_price(daily) +
            html_chip(daily, mf_stat, chip_c, chip_s, mf_col) +
            html_day_script(daily, close, atr_v),
            unsafe_allow_html=True
        )

    # ── 底部列：K線型態 ＋ 多週期分析 ────────────────────────────
    bot1, bot2 = st.columns([1, 1])
    with bot1:
        st.markdown(html_kline_patterns(kp, cp), unsafe_allow_html=True)
    with bot2:
        st.markdown(html_multiperiod(daily, weekly, monthly), unsafe_allow_html=True)

    # ── 免責聲明 ──────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:{CARD};border:1px solid {BORD};border-radius:5px;
                padding:6px 12px;margin-top:4px;text-align:center;">
      <span style="color:{MUTED};font-size:.7rem;">
        ⚠ 資料來源：Yahoo Finance｜本儀表板僅供學習與研究，不構成投資建議。投資有風險，操作請審慎。
      </span>
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
