# Multi-Horizon Trend Weekly Research V1.0

Progetto separato dal Trend 12M minimale.

## Ipotesi congelata

- tre orizzonti: 1M / 3M / 12M = 21 / 63 / 252 sedute;
- LONG se il rendimento passato dell'orizzonte è positivo, SHORT se negativo;
- i tre orizzonti sono pesati ugualmente;
- rebalance settimanale;
- segnale calcolato con dati strettamente precedenti al prezzo di rebalance;
- stima volatilità EWMA con centro di massa 60 giorni;
- posizione del singolo mercato scalata come 40% / volatilità annualizzata;
- core portfolio = media dei mercati disponibili;
- nessuna ottimizzazione di lookback, soglie o filtri.

Questa struttura riprende la metodologia descritta da Hurst, Ooi e Pedersen
("Demystifying Managed Futures") per i segnali 1M/3M/12M, il rebalance weekly e
il sizing volatility-scaled del singolo mercato.

## Overlay 10% portfolio

L'app mostra anche un secondo livello separato dal segnale:

- stima la volatilità del CORE usando esclusivamente le 26 settimane precedenti;
- scala il core verso un target del 10% annuo;
- massimo moltiplicatore 3x.

Questa è un'implementazione pratica e trasparente per il nostro dataset e NON
pretende di replicare esattamente la matrice var-cov utilizzata nello studio.
Il giudizio sull'edge va fatto prima di tutto sul CORE.

## Output

- PF, CAGR, Sharpe, Max DD, volatilità;
- bootstrap 95% della media settimanale;
- equity e drawdown;
- risultati per decennio;
- risultati per anno;
- contributo 1M / 3M / 12M;
- risultati per asset;
- Cost Stress;
- dettaglio asset/settimana;
- export Excel.

## Primo test

- 2000-01-01 → oggi
- universo predefinito
- nessuna modifica alla regola

La qualità dello storico Yahoo non è uniforme tra tutti i futures; il report
mostra la coverage settimanale e i risultati vanno letti anche per periodi con
copertura comparabile.
