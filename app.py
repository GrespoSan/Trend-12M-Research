from __future__ import annotations

import io
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ============================================================
# CONFIG — REGOLA CONGELATA
# ============================================================

st.set_page_config(page_title="Multi-Horizon Trend Weekly V1.0", layout="wide")

LOOKBACKS = {"1M": 21, "3M": 63, "12M": 252}
EWMA_CENTER_DAYS = 60
EWMA_DELTA = EWMA_CENTER_DAYS / (EWMA_CENTER_DAYS + 1.0)  # delta/(1-delta)=60
EWMA_ALPHA = 1.0 - EWMA_DELTA
ANNUAL_DAYS = 261
ASSET_VOL_TARGET = 0.40  # metodologia Hurst/Ooi/Pedersen per singolo mercato
PORTFOLIO_VOL_TARGET = 0.10
PORTFOLIO_VOL_LOOKBACK_WEEKS = 26  # implementazione trasparente per target di portafoglio
PORTFOLIO_SCALE_CAP = 3.0          # solo controllo numerico/rischio, non segnale

DEFAULT_UNIVERSE = {
    "S&P 500 Future": "ES=F",
    "Nasdaq 100 Future": "NQ=F",
    "Dow Future": "YM=F",
    "Russell 2000 Future": "RTY=F",
    "DAX (proxy cash)": "^GDAXI",
    "Euro Stoxx 50 (proxy cash)": "^STOXX50E",
    "Gold Future": "GC=F",
    "Silver Future": "SI=F",
    "WTI Future": "CL=F",
    "Natural Gas Future": "NG=F",
    "Copper Future": "HG=F",
    "US 30Y Treasury Future": "ZB=F",
    "US 10Y Treasury Future": "ZN=F",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
}


# ============================================================
# DATA
# ============================================================

def parse_universe_text(text: str) -> tuple[dict[str, str], list[str]]:
    out: dict[str, str] = {}
    bad: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "," in line:
            name, ticker = [x.strip() for x in line.split(",", 1)]
        else:
            name = ticker = line
        if not ticker:
            bad.append(raw)
            continue
        out[name or ticker] = ticker
    return out, bad


def normalize_yf_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    x = df.copy()
    if isinstance(x.columns, pd.MultiIndex):
        try:
            if ticker in x.columns.get_level_values(-1):
                x = x.xs(ticker, axis=1, level=-1)
        except Exception:
            pass
        if isinstance(x.columns, pd.MultiIndex):
            x.columns = [c[0] if isinstance(c, tuple) else c for c in x.columns]
    wanted = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in x.columns]
    x = x[wanted].copy()
    if "Close" not in x.columns:
        return pd.DataFrame()
    x.index = pd.to_datetime(x.index).tz_localize(None)
    x = x[~x.index.duplicated(keep="last")].sort_index()
    for c in x.columns:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["Close"])
    x = x[x["Close"] > 0]
    return x


@st.cache_data(ttl=12 * 60 * 60, show_spinner=False)
def download_history(ticker: str, start_date: date, end_date: date) -> tuple[pd.DataFrame, str]:
    errors: list[str] = []
    try:
        raw = yf.download(
            ticker,
            start=start_date.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
        )
        df = normalize_yf_frame(raw, ticker)
        if not df.empty:
            return df, "OK · yf.download"
        errors.append("yf.download vuoto")
    except Exception as exc:
        errors.append(f"yf.download: {type(exc).__name__}: {str(exc)[:180]}")

    try:
        raw = yf.Ticker(ticker).history(
            period="max",
            interval="1d",
            auto_adjust=False,
            actions=False,
            repair=False,
            raise_errors=True,
        )
        df = normalize_yf_frame(raw, ticker)
        if not df.empty:
            s, e = pd.Timestamp(start_date), pd.Timestamp(end_date)
            df = df[(df.index >= s) & (df.index <= e)].copy()
        if not df.empty:
            return df, "OK · Ticker.history fallback"
        errors.append("Ticker.history vuoto")
    except Exception as exc:
        errors.append(f"Ticker.history: {type(exc).__name__}: {str(exc)[:180]}")

    return pd.DataFrame(), " | ".join(errors)


# ============================================================
# CORE STRATEGY
# ============================================================

def ewma_annual_vol(close: pd.Series) -> pd.Series:
    r = close.pct_change()
    mean = r.ewm(alpha=EWMA_ALPHA, adjust=False, min_periods=30).mean()
    var = ((r - mean) ** 2).ewm(alpha=EWMA_ALPHA, adjust=False, min_periods=30).mean()
    return np.sqrt(var * ANNUAL_DAYS)


def prepare_asset_daily(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["Daily Return"] = x["Close"].pct_change()
    x["EWMA Vol"] = ewma_annual_vol(x["Close"])
    for label, lag in LOOKBACKS.items():
        x[f"Ret {label}"] = x["Close"] / x["Close"].shift(lag) - 1.0
        x[f"Sign {label}"] = np.sign(x[f"Ret {label}"])
    return x


def build_weekly_asset_stream(
    name: str,
    ticker: str,
    df: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Ogni settimana:
    - rebalance alla chiusura dell'ultima seduta disponibile della settimana;
    - segnale calcolato SOLO con dati precedenti alla seduta di rebalance;
    - 1M/3M/12M = segno del rendimento passato;
    - posizione di ciascun orizzonte = sign * 40% / vol_EWMA;
    - ritorno misurato dalla chiusura di rebalance a quella della settimana successiva.

    Nei normali calendari corrisponde al venerdì close basato su dati fino a giovedì.
    In una settimana festiva usa l'ultima seduta effettivamente disponibile e mantiene
    comunque il vincolo no-lookahead usando dati strettamente precedenti.
    """
    if df is None or df.empty or len(df) < 280:
        return pd.DataFrame()

    x = prepare_asset_daily(df)
    x = x.loc[x.index <= pd.Timestamp(end_date)].copy()
    if x.empty:
        return pd.DataFrame()

    # Chiave settimana con fine venerdì. Se venerdì non esiste, prendiamo l'ultima seduta disponibile.
    week_period = x.index.to_period("W-FRI")
    rows: list[dict] = []

    for _, idxs in pd.Series(x.index, index=x.index).groupby(week_period):
        dates = list(idxs.values)
        if not dates:
            continue
        rebalance_date = pd.Timestamp(max(dates))
        if rebalance_date.date() < start_date or rebalance_date.date() > end_date:
            continue

        pos = x.index.get_loc(rebalance_date)
        if not isinstance(pos, (int, np.integer)) or pos < 1:
            continue
        info_date = x.index[pos - 1]  # informazione disponibile PRIMA del close di ingresso
        info = x.loc[info_date]
        entry_close = float(x.loc[rebalance_date, "Close"])

        sigs = []
        valid = True
        for label in LOOKBACKS:
            val = info.get(f"Sign {label}", np.nan)
            if pd.isna(val) or float(val) == 0:
                valid = False
                break
            sigs.append(float(val))
        vol = info.get("EWMA Vol", np.nan)
        if not valid or pd.isna(vol) or float(vol) <= 0 or entry_close <= 0:
            continue

        rows.append({
            "Week": rebalance_date.to_period("W-FRI").end_time.normalize(),
            "Entry Date": rebalance_date,
            "Info Date": pd.Timestamp(info_date),
            "Asset": name,
            "Ticker": ticker,
            "Entry Close": entry_close,
            "Vol EWMA": float(vol),
            "Signal 1M": sigs[0],
            "Signal 3M": sigs[1],
            "Signal 12M": sigs[2],
            "Trend Score": float(np.mean(sigs)),
            "Asset Vol Scalar": float(ASSET_VOL_TARGET / float(vol)),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values("Entry Date").reset_index(drop=True)
    out["Exit Date"] = out["Entry Date"].shift(-1)
    out["Exit Close"] = out["Entry Close"].shift(-1)
    out = out.dropna(subset=["Exit Date", "Exit Close"]).copy()
    out["Underlying Return"] = out["Exit Close"] / out["Entry Close"] - 1.0

    # Tre orizzonti pesati ugualmente. Equivalentemente: mean(signals) * 40%/vol.
    out["Core Return"] = out["Trend Score"] * out["Asset Vol Scalar"] * out["Underlying Return"]
    out["Direction"] = np.where(out["Trend Score"] > 0, "LONG", np.where(out["Trend Score"] < 0, "SHORT", "MIXED"))
    return out


def build_core_portfolio(streams: pd.DataFrame, requested_assets: int) -> pd.DataFrame:
    if streams is None or streams.empty:
        return pd.DataFrame()
    x = streams.copy()
    x["Week"] = pd.to_datetime(x["Week"])
    p = x.groupby("Week").agg(
        Core_Return=("Core Return", "mean"),
        Asset_Count=("Asset", "nunique"),
        Long_Count=("Trend Score", lambda s: int((s > 0).sum())),
        Short_Count=("Trend Score", lambda s: int((s < 0).sum())),
        Mixed_Count=("Trend Score", lambda s: int((s == 0).sum())),
        Gross_Exposure=("Asset Vol Scalar", "mean"),
    ).reset_index().sort_values("Week")
    p["Coverage %"] = p["Asset_Count"] / max(int(requested_assets), 1)
    p["Year"] = p["Week"].dt.year
    return p


def add_portfolio_vol_target(core: pd.DataFrame) -> pd.DataFrame:
    """
    Secondo livello di risk scaling, separato dal segnale.
    Usa SOLO i ritorni core precedenti per stimare la vol settimanale a 26 settimane.
    È una implementazione pratica/trasparente, non una replica esatta della matrice
    var-cov giornaliera del paper.
    """
    if core is None or core.empty:
        return pd.DataFrame()
    p = core.copy()
    prior = p["Core_Return"].shift(1)
    vol = prior.ewm(span=PORTFOLIO_VOL_LOOKBACK_WEEKS, adjust=False, min_periods=13).std(bias=False) * math.sqrt(52)
    scale = PORTFOLIO_VOL_TARGET / vol
    scale = scale.clip(lower=0.0, upper=PORTFOLIO_SCALE_CAP)
    p["Portfolio Vol Est"] = vol
    p["Portfolio Scale"] = scale
    p["Return 10% Target"] = p["Core_Return"] * p["Portfolio Scale"]
    return p


# ============================================================
# METRICS
# ============================================================

def profit_factor(r: pd.Series) -> float:
    x = pd.to_numeric(r, errors="coerce").dropna()
    if x.empty:
        return np.nan
    gains = float(x[x > 0].sum())
    losses = float(x[x < 0].sum())
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return gains / abs(losses)


def metrics_weekly(r: pd.Series) -> dict:
    x = pd.to_numeric(r, errors="coerce").dropna()
    if x.empty:
        return {k: np.nan for k in ["Weeks", "PF", "Mean Weekly", "Win %", "CAGR", "Vol", "Sharpe", "Max DD", "Total"]}
    equity = (1.0 + x).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    years = len(x) / 52.0
    total_growth = float(equity.iloc[-1])
    cagr = total_growth ** (1 / years) - 1.0 if years > 0 and total_growth > 0 else np.nan
    vol = float(x.std(ddof=1) * math.sqrt(52)) if len(x) > 1 else np.nan
    ann_mean = float(x.mean() * 52)
    sharpe = ann_mean / vol if vol and not pd.isna(vol) and vol > 0 else np.nan
    return {
        "Weeks": int(len(x)),
        "PF": profit_factor(x),
        "Mean Weekly": float(x.mean()),
        "Win %": float((x > 0).mean()),
        "CAGR": float(cagr),
        "Vol": vol,
        "Sharpe": sharpe,
        "Max DD": float(dd.min()),
        "Total": float(total_growth - 1.0),
    }


def equity_frame(p: pd.DataFrame, col: str) -> pd.DataFrame:
    x = p[["Week", col]].dropna().copy()
    x["Equity"] = (1.0 + x[col]).cumprod()
    x["Peak"] = x["Equity"].cummax()
    x["Drawdown"] = x["Equity"] / x["Peak"] - 1.0
    return x


def yearly_table(p: pd.DataFrame, col: str) -> pd.DataFrame:
    x = p[["Week", col]].dropna().copy()
    if x.empty:
        return pd.DataFrame()
    x["Year"] = x["Week"].dt.year
    rows = []
    for year, part in x.groupby("Year"):
        m = metrics_weekly(part[col])
        rows.append({
            "Year": int(year), "Weeks": m["Weeks"], "PF": m["PF"], "CAGR": m["CAGR"],
            "Vol": m["Vol"], "Sharpe": m["Sharpe"], "Max DD": m["Max DD"], "Return": m["Total"]
        })
    return pd.DataFrame(rows)


def decade_table(p: pd.DataFrame, col: str) -> pd.DataFrame:
    x = p[["Week", col]].dropna().copy()
    if x.empty:
        return pd.DataFrame()
    x["Decade"] = (x["Week"].dt.year // 10) * 10
    rows = []
    for dec, part in x.groupby("Decade"):
        m = metrics_weekly(part[col])
        years = sorted(part["Week"].dt.year.unique())
        ytbl = yearly_table(part.rename(columns={col: "R"}), "R")
        rows.append({
            "Period": f"{int(dec)}-{int(dec)+9}", "Weeks": m["Weeks"], "PF": m["PF"],
            "CAGR": m["CAGR"], "Vol": m["Vol"], "Sharpe": m["Sharpe"], "Max DD": m["Max DD"],
            "Positive Years %": float((ytbl["Return"] > 0).mean()) if not ytbl.empty else np.nan,
            "Total": m["Total"], "Years": len(years)
        })
    return pd.DataFrame(rows)


def per_asset_table(streams: pd.DataFrame) -> pd.DataFrame:
    if streams is None or streams.empty:
        return pd.DataFrame()
    rows = []
    for (asset, ticker), part in streams.groupby(["Asset", "Ticker"]):
        m = metrics_weekly(part["Core Return"])
        rows.append({
            "Asset": asset, "Ticker": ticker, "Weeks": m["Weeks"], "PF": m["PF"],
            "Mean Weekly": m["Mean Weekly"], "Sharpe": m["Sharpe"], "Max DD": m["Max DD"],
            "Total": m["Total"], "Avg Trend Score": float(part["Trend Score"].mean()),
            "Avg Vol Scalar": float(part["Asset Vol Scalar"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["Sharpe", "PF"], ascending=[False, False], na_position="last")


def horizon_contribution(streams: pd.DataFrame) -> pd.DataFrame:
    if streams is None or streams.empty:
        return pd.DataFrame()
    rows = []
    for label in LOOKBACKS:
        tmp = streams.copy()
        tmp["HReturn"] = tmp[f"Signal {label}"] * tmp["Asset Vol Scalar"] * tmp["Underlying Return"]
        weekly = tmp.groupby("Week")["HReturn"].mean().sort_index()
        m = metrics_weekly(weekly)
        rows.append({"Horizon": label, **m})
    return pd.DataFrame(rows)


def bootstrap_mean_ci(r: pd.Series, n_boot: int = 5000, seed: int = 42) -> tuple[float, float]:
    x = pd.to_numeric(r, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) < 52:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    q = np.quantile(means, [0.025, 0.975])
    return float(q[0]), float(q[1])


def cost_stress(p: pd.DataFrame, cost_bps_per_week=(0, 1, 2, 5, 10)) -> pd.DataFrame:
    if p is None or p.empty:
        return pd.DataFrame()
    rows = []
    for bps in cost_bps_per_week:
        r = p["Core_Return"] - float(bps) / 10000.0
        m = metrics_weekly(r)
        rows.append({"Cost bps/week portfolio": bps, **m})
    return pd.DataFrame(rows)


def fmt_pct(v) -> str:
    return "n/d" if pd.isna(v) else f"{v:.2%}"


def fmt_num(v) -> str:
    if pd.isna(v):
        return "n/d"
    if np.isinf(v):
        return "∞"
    return f"{v:.2f}"


# ============================================================
# EXPORT
# ============================================================

def excel_report(settings: dict, core: pd.DataFrame, targeted: pd.DataFrame, yearly_core: pd.DataFrame,
                 yearly_target: pd.DataFrame, decades_core: pd.DataFrame, per_asset: pd.DataFrame,
                 horizons: pd.DataFrame, streams: pd.DataFrame, costs: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        pd.DataFrame([{"Setting": k, "Value": v} for k, v in settings.items()]).to_excel(writer, sheet_name="Settings", index=False)
        core.to_excel(writer, sheet_name="Core_Weekly", index=False)
        targeted.to_excel(writer, sheet_name="Portfolio_10pct", index=False)
        yearly_core.to_excel(writer, sheet_name="Yearly_Core", index=False)
        yearly_target.to_excel(writer, sheet_name="Yearly_10pct", index=False)
        decades_core.to_excel(writer, sheet_name="Decades_Core", index=False)
        per_asset.to_excel(writer, sheet_name="Per_Asset", index=False)
        horizons.to_excel(writer, sheet_name="Horizons", index=False)
        streams.to_excel(writer, sheet_name="Asset_Week", index=False)
        costs.to_excel(writer, sheet_name="Cost_Stress", index=False)
        wb = writer.book
        header_fmt = wb.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
        for sheet_name, df in {
            "Core_Weekly": core, "Portfolio_10pct": targeted, "Yearly_Core": yearly_core,
            "Yearly_10pct": yearly_target, "Decades_Core": decades_core, "Per_Asset": per_asset,
            "Horizons": horizons, "Asset_Week": streams, "Cost_Stress": costs,
        }.items():
            ws = writer.sheets[sheet_name]
            ws.freeze_panes(1, 0)
            for j, c in enumerate(df.columns):
                ws.write(0, j, c, header_fmt)
                ws.set_column(j, j, min(max(len(str(c)) + 2, 12), 24))
    output.seek(0)
    return output.getvalue()


# ============================================================
# UI
# ============================================================

st.title("Multi-Horizon Trend Weekly Research V1.0.1")
st.caption(
    "Time-Series Momentum multi-asset: segnali 1M + 3M + 12M, rebalance settimanale, "
    "volatility scaling. Nessuna ottimizzazione dei lookback."
)

with st.sidebar:
    st.header("Test")
    start_date = st.date_input("Data inizio", value=date(2000, 1, 1), min_value=date(1980, 1, 1))
    end_date = st.date_input("Data fine", value=date.today(), max_value=date.today())

    st.divider()
    st.subheader("Universo")
    uploaded = st.file_uploader("Lista asset .txt — opzionale", type=["txt"])
    if uploaded is None:
        universe = DEFAULT_UNIVERSE.copy()
        universe_source = "predefinito"
        st.caption(f"Universo predefinito: **{len(universe)} asset**")
        st.text_area("Asset", "\n".join(f"{k},{v}" for k, v in universe.items()), height=250, disabled=True)
    else:
        txt = uploaded.getvalue().decode("utf-8-sig", errors="replace")
        universe, bad = parse_universe_text(txt)
        universe_source = uploaded.name
        if bad:
            st.warning("Righe non valide: " + " | ".join(bad))

    run = st.button("Esegui Multi-Horizon Trend", type="primary", width="stretch")

    st.divider()
    st.caption(
        "Regola congelata: 21 / 63 / 252 sedute; segnale = segno del rendimento passato; "
        "EWMA vol con centro di massa 60 giorni; singolo mercato target 40% vol; "
        "orizzonti pesati ugualmente."
    )

if not run:
    st.info(
        "Questa V1 non cerca il lookback migliore. Serve a verificare la struttura 1M/3M/12M "
        "su più mercati e periodi."
    )
    st.stop()

if start_date >= end_date:
    st.error("La data iniziale deve precedere la data finale.")
    st.stop()
if not universe:
    st.error("Nessun asset disponibile.")
    st.stop()

# Buffer sufficiente per 252 sedute + EWMA.
download_start = start_date - timedelta(days=650)
streams_list = []
errors = []
progress = st.progress(0)
status = st.empty()

for i, (name, ticker) in enumerate(universe.items(), start=1):
    status.write(f"Scarico e analizzo {name} ({ticker})…")
    df, diag = download_history(ticker, download_start, end_date)
    if df.empty:
        errors.append(f"{name} ({ticker}): {diag}")
        progress.progress(i / len(universe))
        continue
    s = build_weekly_asset_stream(name, ticker, df, start_date, end_date)
    if s.empty:
        errors.append(f"{name} ({ticker}): storico insufficiente / nessuna settimana valida")
    else:
        streams_list.append(s)
    progress.progress(i / len(universe))

status.empty()
progress.empty()

if not streams_list:
    st.error("Nessun risultato calcolabile.")
    if errors:
        st.code("\n".join(errors))
    st.stop()

streams = pd.concat(streams_list, ignore_index=True)
core = build_core_portfolio(streams, len(universe))
targeted = add_portfolio_vol_target(core)

m_core = metrics_weekly(core["Core_Return"])
m_target = metrics_weekly(targeted["Return 10% Target"])
yearly_core = yearly_table(core, "Core_Return")
yearly_target = yearly_table(targeted, "Return 10% Target")
decades_core = decade_table(core, "Core_Return")
per_asset = per_asset_table(streams)
horizons = horizon_contribution(streams)
costs = cost_stress(core)
ci_lo, ci_hi = bootstrap_mean_ci(core["Core_Return"])

st.subheader("Core multi-horizon — asset-level volatility scaling")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Settimane", m_core["Weeks"])
c2.metric("Asset con dati", int(streams["Asset"].nunique()))
c3.metric("Profit Factor", fmt_num(m_core["PF"]))
c4.metric("CAGR", fmt_pct(m_core["CAGR"]))
c5.metric("Sharpe", fmt_num(m_core["Sharpe"]))
c6.metric("Max DD", fmt_pct(m_core["Max DD"]))

c7, c8, c9 = st.columns(3)
c7.metric("Vol realizzata", fmt_pct(m_core["Vol"]))
c8.metric("Settimane positive", fmt_pct(m_core["Win %"]))
c9.metric("Bootstrap 95% media weekly", f"{fmt_pct(ci_lo)} → {fmt_pct(ci_hi)}")

st.caption(
    "Core = media dei ritorni dei mercati dopo scaling 40%/vol EWMA e combinazione uguale dei tre orizzonti. "
    "Non è ancora il 10% portfolio target."
)

st.subheader("Overlay portfolio 10% vol target")
t1, t2, t3, t4 = st.columns(4)
t1.metric("PF", fmt_num(m_target["PF"]))
t2.metric("CAGR", fmt_pct(m_target["CAGR"]))
t3.metric("Vol realizzata", fmt_pct(m_target["Vol"]))
t4.metric("Max DD", fmt_pct(m_target["Max DD"]))
st.caption(
    "Questo overlay usa solo la volatilità settimanale CORE precedente (EWMA 26 settimane) per puntare al 10% annuo. "
    "È separato dal segnale e non pretende di replicare esattamente la matrice var-cov del paper."
)

st.subheader("Equity Core")
ef = equity_frame(core, "Core_Return")
st.line_chart(ef.set_index("Week")[["Equity"]], height=320)

st.subheader("Drawdown Core")
st.line_chart(ef.set_index("Week")[["Drawdown"]], height=240)

st.subheader("Robustezza per decennio — Core")
show = decades_core.copy()
if not show.empty:
    for c in ["CAGR", "Vol", "Max DD", "Positive Years %", "Total"]:
        show[c] = show[c].map(fmt_pct)
    for c in ["PF", "Sharpe"]:
        show[c] = show[c].map(fmt_num)
    st.dataframe(show, width="stretch", hide_index=True)

st.subheader("Contributo dei tre orizzonti")
show = horizons.copy()
if not show.empty:
    for c in ["Mean Weekly", "Win %", "CAGR", "Vol", "Max DD", "Total"]:
        show[c] = show[c].map(fmt_pct)
    for c in ["PF", "Sharpe"]:
        show[c] = show[c].map(fmt_num)
    st.dataframe(show, width="stretch", hide_index=True)

st.subheader("Risultati per asset")
show = per_asset.copy()
if not show.empty:
    for c in ["Mean Weekly", "Max DD", "Total"]:
        show[c] = show[c].map(fmt_pct)
    for c in ["PF", "Sharpe", "Avg Trend Score", "Avg Vol Scalar"]:
        show[c] = show[c].map(fmt_num)
    st.dataframe(show, width="stretch", hide_index=True)

st.subheader("Risultati per anno — Core")
show = yearly_core.copy()
if not show.empty:
    for c in ["CAGR", "Vol", "Max DD", "Return"]:
        show[c] = show[c].map(fmt_pct)
    for c in ["PF", "Sharpe"]:
        show[c] = show[c].map(fmt_num)
    st.dataframe(show, width="stretch", hide_index=True)

st.subheader("Cost Stress — Core")
show = costs.copy()
if not show.empty:
    for c in ["Mean Weekly", "Win %", "CAGR", "Vol", "Max DD", "Total"]:
        show[c] = show[c].map(fmt_pct)
    for c in ["PF", "Sharpe"]:
        show[c] = show[c].map(fmt_num)
    st.dataframe(show, width="stretch", hide_index=True)

st.subheader("Ultime settimane / posizioni")
last = streams.sort_values(["Week", "Asset"], ascending=[False, True]).head(120).copy()
for c in ["Underlying Return", "Core Return", "Vol EWMA"]:
    last[c] = last[c].map(fmt_pct)
st.dataframe(last, width="stretch", hide_index=True)

if errors:
    with st.expander("Diagnostica asset"):
        st.code("\n".join(errors))

settings = {
    "Project": "Multi-Horizon Trend Weekly Research V1.0.1",
    "Start": start_date,
    "End": end_date,
    "Universe": universe_source,
    "Requested assets": len(universe),
    "Assets with data": int(streams["Asset"].nunique()),
    "Horizons": "1M=21, 3M=63, 12M=252 sessions",
    "Signal": "sign of past price return",
    "Rebalance": "weekly, last available close; info strictly prior",
    "EWMA volatility": "center of mass 60 days",
    "Asset volatility target": "40% annualized",
    "Horizon weights": "equal 1/3 each",
    "Core portfolio": "equal weight across available markets",
    "Portfolio overlay": "10% target using prior 26-week EWMA core vol; cap 3x",
    "Costs in main result": "0",
}
excel_bytes = excel_report(settings, core, targeted, yearly_core, yearly_target, decades_core, per_asset, horizons, streams, costs)
st.download_button(
    "📥 Scarica report Excel",
    data=excel_bytes,
    file_name=f"multi_horizon_trend_weekly_{start_date}_{end_date}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    width="stretch",
)

st.caption(
    "Ricerca statistica. Yahoo continuous futures / proxy cash non sono una replica perfetta di futures roll-adjusted, "
    "excess returns, costi e execution reali."
)
