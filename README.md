# Trend 12M Research V1.0

Progetto Streamlit indipendente per testare una versione minimale di Time-Series Momentum.

## Regola congelata
- una decisione al mese
- LONG se Close(T-1) > Close(T-253)
- SHORT se Close(T-1) < Close(T-253)
- entry alla prima apertura reale del mese
- exit/continuazione alla prima apertura del mese successivo
- nessun target, stop, EMA, RSI, stagionalità o altro filtro
- nessuna ottimizzazione del lookback

## Perché T-253
All'open di T l'ultimo Close noto è T-1. Il confronto con T-253 dà 252 sedute di distanza e non usa dati futuri.

## Portfolio
Equal-weight mensile fra tutti gli asset con dati disponibili. In V1 non c'è volatility targeting né leva: vogliamo prima verificare il puro edge direzionale.

## Output
- PF, media mensile, CAGR, Max DD
- mesi e anni positivi
- bootstrap 95% della media mensile
- equity e drawdown
- risultati per decennio, anno e asset
- contributi LONG/SHORT per asset
- Cost Stress 0/2/5/10/20 bps per mese e asset
- export Excel

## Avvio
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Primo test consigliato
Periodo più lungo possibile (es. dal 2000), costo iniziale 0 bps, universo predefinito. Prima guardare soprattutto robustezza per decennio e per asset; solo dopo il Cost Stress.

Un risultato forte soltanto nell'ultimo decennio non basta.


## V1.1 — Download fix

Corretto il loader dati.

La V1.0 nascondeva ogni eccezione di yfinance e trasformava qualsiasi problema
di download nel generico messaggio "dati non disponibili".

La V1.1 usa:
1. `yf.download` standard, con la stessa impostazione già usata nel progetto
   stagionalità;
2. fallback `Ticker.history(period="max")`;
3. diagnostica reale dell'errore se entrambi i metodi falliscono.

La logica Trend 12M NON è stata modificata.


## V1.2 — Fix portfolio mensile

Corretto un errore importante della V1.0/V1.1.

Prima il portfolio veniva aggregato sulla data esatta della prima seduta del
mese. Mercati con calendari/festività differenti potevano quindi creare due
osservazioni separate nello stesso mese.

Ora:
- aggregazione per YYYY-MM;
- una sola osservazione per mese di calendario;
- CAGR, equity, drawdown, statistiche annuali/decennali e Cost Stress corretti;
- aggiunta copertura percentuale dell'universo disponibile.

La regola Trend 12M è invariata.
