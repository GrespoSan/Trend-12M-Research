import io
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Trend 12M Research V1.2", layout="wide")
LOOKBACK_SESSIONS = 252

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


def parse_universe_text(text: str) -> tuple[dict, list]:
    out, bad = {}, []
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
        if ticker in x.columns.get_level_values(-1):
            try:
                x = x.xs(ticker, axis=1, level=-1)
            except Exception:
                pass
        if isinstance(x.columns, pd.MultiIndex):
            x.columns = [c[0] if isinstance(c, tuple) else c for c in x.columns]
    wanted = ["Open", "High", "Low", "Close", "Volume"]
    x = x[[c for c in wanted if c in x.columns]].copy()
    if "Open" not in x.columns or "Close" not in x.columns:
        return pd.DataFrame()
    x.index = pd.to_datetime(x.index).tz_localize(None)
    x = x[~x.index.duplicated(keep="last")].sort_index()
    for c in x.columns:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["Open", "Close"])
    return x[(x["Open"] > 0) & (x["Close"] > 0)]


@st.cache_data(ttl=12 * 60 * 60, show_spinner=False)
def download_history(
    ticker: str,
    start_date: date,
    end_date: date,
) -> tuple[pd.DataFrame, str]:
    """
    Loader robusto V1.1.

    Primo tentativo:
    - yf.download con la stessa struttura già usata nell'app stagionale.

    Fallback:
    - yf.Ticker(...).history(period="max") e taglio locale del periodo.

    Restituisce anche un messaggio diagnostico: non nasconde più le eccezioni.
    """
    errors = []

    # --- Tentativo 1: yf.download standard ---
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

        errors.append("yf.download ha restituito un DataFrame vuoto")

    except Exception as exc:
        errors.append(
            f"yf.download: {type(exc).__name__}: {str(exc)[:220]}"
        )

    # --- Tentativo 2: Ticker.history MAX + filtro locale ---
    try:
        hist = yf.Ticker(ticker).history(
            period="max",
            interval="1d",
            auto_adjust=False,
            actions=False,
            repair=False,
            raise_errors=True,
        )

        df = normalize_yf_frame(hist, ticker)

        if not df.empty:
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
            df = df.loc[
                (df.index >= start_ts)
                & (df.index <= end_ts)
            ].copy()

        if not df.empty:
            return df, "OK · Ticker.history fallback"

        errors.append("Ticker.history ha restituito dati vuoti")

    except Exception as exc:
        errors.append(
            f"Ticker.history: {type(exc).__name__}: {str(exc)[:220]}"
        )

    return pd.DataFrame(), " | ".join(errors)


def build_monthly_trades(name, ticker, df, start_date, end_date, monthly_cost_bps):
    if df is None or df.empty or len(df) <= LOOKBACK_SESSIONS + 5:
        return pd.DataFrame()
    x = df.copy()
    x["Prev Close"] = x["Close"].shift(1)
    x["Prev Close 252"] = x["Close"].shift(LOOKBACK_SESSIONS + 1)
    x["Signal"] = np.where(
        x["Prev Close"] > x["Prev Close 252"], 1,
        np.where(x["Prev Close"] < x["Prev Close 252"], -1, 0),
    )
    valid = x["Prev Close"].notna() & x["Prev Close 252"].notna()
    x.loc[~valid, "Signal"] = np.nan
    month_first = x.groupby(x.index.to_period("M"), sort=True).head(1).copy()
    month_first["Exit Date"] = month_first.index.to_series().shift(-1)
    month_first["Exit Open"] = month_first["Open"].shift(-1)
    mask = (month_first.index.date >= start_date) & (month_first.index.date <= end_date)
    month_first = month_first.loc[mask].dropna(subset=["Signal", "Open", "Exit Open", "Exit Date"])
    month_first = month_first[month_first["Signal"] != 0].copy()
    if month_first.empty:
        return pd.DataFrame()
    month_first["Underlying Return"] = month_first["Exit Open"] / month_first["Open"] - 1.0
    month_first["Gross Return"] = month_first["Signal"] * month_first["Underlying Return"]
    month_first["Cost"] = float(monthly_cost_bps) / 10000.0
    month_first["Net Return"] = month_first["Gross Return"] - month_first["Cost"]
    month_first["Direction"] = np.where(month_first["Signal"] > 0, "LONG", "SHORT")
    month_first["Asset"] = name
    month_first["Ticker"] = ticker
    month_first["Entry Date"] = month_first.index
    month_first["Entry Open"] = month_first["Open"]
    month_first["Momentum 12M %"] = month_first["Prev Close"] / month_first["Prev Close 252"] - 1.0
    cols = ["Entry Date", "Exit Date", "Asset", "Ticker", "Direction", "Signal", "Entry Open", "Exit Open",
            "Prev Close", "Prev Close 252", "Momentum 12M %", "Underlying Return", "Gross Return", "Cost", "Net Return"]
    return month_first[cols].reset_index(drop=True)


def max_drawdown_from_returns(returns):
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return np.nan
    equity = (1.0 + r).cumprod()
    dd = equity / equity.cummax() - 1.0
    return float(dd.min())


def profit_factor(returns):
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return np.nan
    gains = float(r[r > 0].sum())
    losses = float(r[r < 0].sum())
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return gains / abs(losses)


def cagr_from_monthly(returns):
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return np.nan
    total = float((1.0 + r).prod())
    years = len(r) / 12.0
    if years <= 0 or total <= 0:
        return np.nan
    return total ** (1.0 / years) - 1.0


def metrics_from_returns(returns):
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return {k: np.nan for k in ["Mesi positivi %", "Media mensile", "Mediana mensile", "Profit Factor", "CAGR", "Vol ann.", "Max DD", "Totale"]} | {"Mesi": 0}
    return {
        "Mesi": int(len(r)),
        "Mesi positivi %": float((r > 0).mean()),
        "Media mensile": float(r.mean()),
        "Mediana mensile": float(r.median()),
        "Profit Factor": profit_factor(r),
        "CAGR": cagr_from_monthly(r),
        "Vol ann.": float(r.std(ddof=1) * np.sqrt(12)) if len(r) > 1 else np.nan,
        "Max DD": max_drawdown_from_returns(r),
        "Totale": float((1.0 + r).prod() - 1.0),
    }


def build_portfolio(trades):
    """
    Portfolio equal-weight per vero mese di calendario.
    I mercati possono avere una prima seduta diversa nello stesso mese:
    si aggrega quindi per YYYY-MM e non per Entry Date esatta.
    """
    if trades is None or trades.empty:
        return pd.DataFrame()
    x = trades.copy()
    x["Entry Date"] = pd.to_datetime(x["Entry Date"])
    x["Month"] = x["Entry Date"].dt.to_period("M").dt.to_timestamp()
    p = x.groupby("Month").agg(
        Portfolio_Return=("Net Return", "mean"),
        Gross_Return=("Gross Return", "mean"),
        Asset_Count=("Asset", "nunique"),
        Long_Count=("Direction", lambda s: int((s == "LONG").sum())),
        Short_Count=("Direction", lambda s: int((s == "SHORT").sum())),
    ).reset_index().sort_values("Month")
    p = p.rename(columns={"Month": "Entry Date"})
    p["Equity"] = (1.0 + p["Portfolio_Return"]).cumprod()
    p["Peak"] = p["Equity"].cummax()
    p["Drawdown"] = p["Equity"] / p["Peak"] - 1.0
    p["Year"] = p["Entry Date"].dt.year
    return p


def build_yearly(portfolio):
    rows = []
    if portfolio is None or portfolio.empty:
        return pd.DataFrame()
    for year, part in portfolio.groupby("Year"):
        m = metrics_from_returns(part["Portfolio_Return"])
        rows.append({"Anno": int(year), "Mesi": m["Mesi"], "Rendimento": m["Totale"], "Mesi + %": m["Mesi positivi %"], "PF": m["Profit Factor"], "Max DD": m["Max DD"]})
    return pd.DataFrame(rows).sort_values("Anno")


def build_per_asset(trades):
    rows = []
    if trades is None or trades.empty:
        return pd.DataFrame()
    for (asset, ticker), part in trades.groupby(["Asset", "Ticker"]):
        m = metrics_from_returns(part["Net Return"])
        long_r = part.loc[part["Direction"] == "LONG", "Net Return"]
        short_r = part.loc[part["Direction"] == "SHORT", "Net Return"]
        rows.append({
            "Asset": asset, "Ticker": ticker, "Trade/Mesi": m["Mesi"], "PF": m["Profit Factor"],
            "Media mese": m["Media mensile"], "Totale composto": m["Totale"], "Max DD": m["Max DD"],
            "LONG mesi": int((part["Direction"] == "LONG").sum()), "LONG media": float(long_r.mean()) if not long_r.empty else np.nan,
            "SHORT mesi": int((part["Direction"] == "SHORT").sum()), "SHORT media": float(short_r.mean()) if not short_r.empty else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["PF", "Media mese"], ascending=[False, False], na_position="last")


def build_subperiods(portfolio):
    if portfolio is None or portfolio.empty:
        return pd.DataFrame()
    first_year, last_year = int(portfolio["Year"].min()), int(portfolio["Year"].max())
    rows = []
    for start in range((first_year // 10) * 10, last_year + 1, 10):
        end = min(start + 9, last_year)
        part = portfolio[(portfolio["Year"] >= start) & (portfolio["Year"] <= end)]
        if part.empty:
            continue
        m = metrics_from_returns(part["Portfolio_Return"])
        y = build_yearly(part)
        rows.append({"Periodo": f"{start}-{end}", "Mesi": m["Mesi"], "PF": m["Profit Factor"], "Media mensile": m["Media mensile"],
                     "CAGR": m["CAGR"], "Max DD": m["Max DD"], "Anni positivi %": float((y["Rendimento"] > 0).mean()) if not y.empty else np.nan, "Totale": m["Totale"]})
    return pd.DataFrame(rows)


def bootstrap_mean_ci(returns, n_boot=5000, seed=42):
    r = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    if len(r) < 24:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(r, size=(n_boot, len(r)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def stress_costs(trades, levels_bps=(0, 2, 5, 10, 20)):
    rows = []
    for bps in levels_bps:
        t = trades.copy()
        t["Net Return"] = t["Gross Return"] - bps / 10000.0
        p = build_portfolio(t)
        m = metrics_from_returns(p["Portfolio_Return"])
        y = build_yearly(p)
        rows.append({"Costo bps/mese/asset": bps, "PF": m["Profit Factor"], "Media mensile": m["Media mensile"], "CAGR": m["CAGR"],
                     "Max DD": m["Max DD"], "Anni positivi %": float((y["Rendimento"] > 0).mean()) if not y.empty else np.nan, "Totale": m["Totale"]})
    return pd.DataFrame(rows)


def excel_report(settings, summary, portfolio, yearly, subperiods, per_asset, trades, costs):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        pd.DataFrame([{"Impostazione": k, "Valore": v} for k, v in settings.items()]).to_excel(writer, sheet_name="Impostazioni", index=False)
        summary.to_excel(writer, sheet_name="Riepilogo", index=False)
        portfolio.to_excel(writer, sheet_name="Portfolio_Mensile", index=False)
        yearly.to_excel(writer, sheet_name="Per_Anno", index=False)
        subperiods.to_excel(writer, sheet_name="Per_Decennio", index=False)
        per_asset.to_excel(writer, sheet_name="Per_Asset", index=False)
        trades.to_excel(writer, sheet_name="Trade_Asset_Mese", index=False)
        costs.to_excel(writer, sheet_name="Cost_Stress", index=False)
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            ws.freeze_panes(1, 0)
            ws.set_column(0, 20, 18)
    output.seek(0)
    return output.getvalue()


def fmt_pct(v):
    return "n/d" if pd.isna(v) else f"{v:.2%}"


def fmt_num(v):
    if pd.isna(v):
        return "n/d"
    if np.isinf(v):
        return "∞"
    return f"{v:.2f}"


st.title("Trend 12M Research V1.2")
st.caption("Time-Series Momentum minimale: una decisione al mese, lookback fisso 252 sedute, LONG/SHORT, nessun target e nessuno stop.")

with st.sidebar:
    st.header("Impostazioni test")
    start_date = st.date_input("Data inizio", value=date(2000, 1, 1), min_value=date(1980, 1, 1))
    end_date = st.date_input("Data fine", value=date.today(), max_value=date.today())
    monthly_cost_bps = st.number_input("Costo round-trip per mese/asset (bps)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
    st.divider()
    st.subheader("Universo")
    uploaded = st.file_uploader("Lista asset .txt — opzionale", type=["txt"])
    if uploaded is None:
        universe = DEFAULT_UNIVERSE.copy()
        universe_source = "predefinito"
        st.caption(f"Universo predefinito: **{len(universe)} asset**")
        st.text_area("Asset", "\n".join(f"{k},{v}" for k, v in universe.items()), height=260, disabled=True)
    else:
        text = uploaded.getvalue().decode("utf-8-sig", errors="replace")
        universe, bad_rows = parse_universe_text(text)
        universe_source = uploaded.name
        st.caption(f"Asset validi: **{len(universe)}**")
        if bad_rows:
            st.warning("Righe non valide: " + " | ".join(bad_rows))
    run = st.button("Esegui Trend 12M", type="primary", width="stretch")
    st.caption("Regola congelata: Close(T−1) > Close(T−253) = LONG; inferiore = SHORT. Entry alla prima apertura reale del mese.")

if not run:
    st.info("La V1 non ottimizza nulla: prima verifichiamo se l'effetto è robusto su più mercati e più decenni.")
    st.stop()

if start_date >= end_date:
    st.error("La data iniziale deve precedere la data finale.")
    st.stop()
if not universe:
    st.error("Nessun asset disponibile.")
    st.stop()

download_start = start_date - timedelta(days=550)
all_trades, errors = [], []
progress, status = st.progress(0), st.empty()
for i, (name, ticker) in enumerate(universe.items(), start=1):
    status.write(f"Scarico e analizzo {name} ({ticker})…")
    df, download_info = download_history(ticker, download_start, end_date)
    if df.empty:
        errors.append(
            f"{name} ({ticker}): dati non disponibili · {download_info}"
        )
    else:
        t = build_monthly_trades(name, ticker, df, start_date, end_date, monthly_cost_bps)
        if t.empty:
            errors.append(f"{name} ({ticker}): storico insufficiente o nessun trade")
        else:
            all_trades.append(t)
    progress.progress(i / len(universe))
status.empty(); progress.empty()

if not all_trades:
    st.error(
        "Nessun trade calcolabile. Ora sotto trovi l'errore reale restituito "
        "dal provider dati, invece del generico 'dati non disponibili'."
    )
    if errors:
        st.code("\n".join(errors))
    st.stop()

trades = pd.concat(all_trades, ignore_index=True)
trades["Entry Date"] = pd.to_datetime(trades["Entry Date"])
trades["Exit Date"] = pd.to_datetime(trades["Exit Date"])
portfolio = build_portfolio(trades)
if not portfolio.empty:
    portfolio["Coverage %"] = portfolio["Asset_Count"] / max(len(universe), 1)
yearly = build_yearly(portfolio)
per_asset = build_per_asset(trades)
subperiods = build_subperiods(portfolio)
costs = stress_costs(trades)
m = metrics_from_returns(portfolio["Portfolio_Return"])
positive_years = float((yearly["Rendimento"] > 0).mean()) if not yearly.empty else np.nan
ci_lo, ci_hi = bootstrap_mean_ci(portfolio["Portfolio_Return"])
summary_dict = {
    "Mesi portfolio": m["Mesi"],
    "Asset con dati": int(trades["Asset"].nunique()),
    "Copertura media asset %": float(portfolio["Coverage %"].mean()) if not portfolio.empty else np.nan,
    "Copertura minima asset %": float(portfolio["Coverage %"].min()) if not portfolio.empty else np.nan,
    "Profit Factor": m["Profit Factor"],
    "Media mensile": m["Media mensile"],
    "Mediana mensile": m["Mediana mensile"],
    "Mesi positivi %": m["Mesi positivi %"],
    "CAGR": m["CAGR"],
    "Vol ann.": m["Vol ann."],
    "Max DD": m["Max DD"],
    "Anni positivi %": positive_years,
    "Totale composto": m["Totale"],
    "Bootstrap media 95% low": ci_lo,
    "Bootstrap media 95% high": ci_hi,
}
summary = pd.DataFrame([{"Metrica": k, "Valore": v} for k, v in summary_dict.items()])

st.subheader("Risultato portfolio equal-weight")
c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Mesi", m["Mesi"]); c2.metric("Asset", int(trades["Asset"].nunique())); c3.metric("Profit Factor", fmt_num(m["Profit Factor"]))
c4.metric("Media mensile", fmt_pct(m["Media mensile"])); c5.metric("CAGR", fmt_pct(m["CAGR"])); c6.metric("Max DD", fmt_pct(m["Max DD"]))
c7,c8,c9 = st.columns(3)
c7.metric("Mesi positivi", fmt_pct(m["Mesi positivi %"])); c8.metric("Anni positivi", fmt_pct(positive_years)); c9.metric("Bootstrap 95% media mese", f"{fmt_pct(ci_lo)} → {fmt_pct(ci_hi)}")
st.caption(
    "Portfolio = media semplice dei rendimenti mensili degli asset disponibili. "
    "V1.2 aggrega tutti i mercati per vero mese di calendario (YYYY-MM). "
    "Nessuna leva e nessun volatility targeting."
)
if not portfolio.empty and "Coverage %" in portfolio.columns:
    st.caption(
        f"Copertura media universo: {portfolio['Coverage %'].mean():.1%} · "
        f"copertura minima: {portfolio['Coverage %'].min():.1%}. "
        "Nei primi anni alcuni ticker Yahoo possono avere storico più corto."
    )

st.subheader("Equity"); st.line_chart(portfolio.set_index("Entry Date")[["Equity"]], height=320)
st.subheader("Drawdown"); st.line_chart(portfolio.set_index("Entry Date")[["Drawdown"]], height=250)

st.subheader("Robustezza per decennio")
show = subperiods.copy()
if not show.empty:
    for c in ["Media mensile","CAGR","Max DD","Anni positivi %","Totale"]: show[c]=show[c].map(fmt_pct)
    show["PF"] = show["PF"].map(fmt_num)
    st.dataframe(show, width="stretch", hide_index=True)

st.subheader("Risultati per anno")
show = yearly.copy()
if not show.empty:
    for c in ["Rendimento","Mesi + %","Max DD"]: show[c]=show[c].map(fmt_pct)
    show["PF"] = show["PF"].map(fmt_num)
    st.dataframe(show, width="stretch", hide_index=True)

st.subheader("Risultati per asset")
show = per_asset.copy()
if not show.empty:
    for c in ["Media mese","Totale composto","Max DD","LONG media","SHORT media"]: show[c]=show[c].map(fmt_pct)
    show["PF"] = show["PF"].map(fmt_num)
    st.dataframe(show, width="stretch", hide_index=True)

st.subheader("Cost Stress")
show = costs.copy()
for c in ["Media mensile","CAGR","Max DD","Anni positivi %","Totale"]: show[c]=show[c].map(fmt_pct)
show["PF"] = show["PF"].map(fmt_num)
st.dataframe(show, width="stretch", hide_index=True)

st.subheader("Ultimi trade mensili")
last_trades = trades.sort_values(["Entry Date","Asset"], ascending=[False, True]).head(100).copy()
for c in ["Momentum 12M %","Underlying Return","Gross Return","Cost","Net Return"]: last_trades[c]=last_trades[c].map(fmt_pct)
st.dataframe(last_trades, width="stretch", hide_index=True)

if errors:
    with st.expander("Diagnostica download / asset con problemi"):
        st.code("\n".join(errors))

settings = {
    "Progetto": "Trend 12M Research V1.2", "Data inizio": start_date, "Data fine": end_date,
    "Lookback": "252 sedute", "Decisione": "Prima apertura del mese", "Segnale LONG": "Close T-1 > Close T-253",
    "Segnale SHORT": "Close T-1 < Close T-253", "Exit": "Prima apertura del mese successivo", "Target": "Nessuno",
    "Stop": "Nessuno", "Volatility targeting": "OFF", "Portfolio": "Equal-weight per mese di calendario", "Costo bps/mese/asset": float(monthly_cost_bps),
    "Universo": universe_source, "Numero asset richiesti": len(universe), "Numero asset con dati": int(trades["Asset"].nunique()),
}
excel_bytes = excel_report(settings, summary, portfolio, yearly, subperiods, per_asset, trades, costs)
st.download_button("📥 Scarica report Excel", data=excel_bytes, file_name=f"trend_12m_research_{start_date}_{end_date}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
st.caption("Ricerca statistica: continuous futures Yahoo e proxy cash non equivalgono necessariamente a una serie tradabile reale comprensiva di roll, slippage e costi del broker.")
