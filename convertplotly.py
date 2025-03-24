import json
import datetime
import yfinance as yf
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
from fpdf import FPDF

# === CHARGER LES DONNÉES DU BACKTEST ===
with open("2eme million.json") as f:
    data = json.load(f)

# --- Préparer les données du capital
dates = []
equities = []

for key in sorted(data["rollingWindow"].keys()):
    try:
        date_str = key.split("_")[1]
        date_obj = datetime.datetime.strptime(date_str, "%Y%m%d")
        equity = float(data["rollingWindow"][key]["portfolioStatistics"]["endEquity"])
        if equity <= 0 or equity > 10_000_000:
            continue
        dates.append(date_obj)
        equities.append(equity)
    except:
        continue

# === RÉCUPÉRER LES INDICES AVEC YFINANCE ===
tickers = {
    "S&P 500": "^GSPC",
    "DAX": "^GDAXI",
    "CAC 40": "^FCHI",
    "Nikkei 225": "^N225"
}

start_date = min(dates).strftime("%Y-%m-%d")
end_date = max(dates).strftime("%Y-%m-%d")

# Télécharger les données
raw_data = yf.download(list(tickers.values()), start=start_date, end=end_date)
adj_close = raw_data["Close"] if isinstance(raw_data.columns, pd.MultiIndex) else raw_data
adj_close = adj_close[~adj_close.index.duplicated(keep="first")]
normalized_indices = adj_close / adj_close.iloc[0] * equities[0]

# === ALIGNEMENT DES DONNÉES ===
strategy_series = pd.Series(data=equities, index=pd.to_datetime(dates), name="Stratégie")
strategy_series = strategy_series[~strategy_series.index.duplicated(keep="first")]
combined_df = pd.concat([strategy_series, normalized_indices], axis=1)
combined_df.dropna(inplace=True)

# === GRAPHIQUE COMPARATIF PLOTLY ===
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=combined_df.index,
    y=combined_df["Stratégie"],
    mode='lines',
    name='Stratégie',
    line=dict(color='blue', width=2)
))

for label, ticker in tickers.items():
    if ticker in combined_df.columns:
        fig.add_trace(go.Scatter(
            x=combined_df.index,
            y=combined_df[ticker],
            mode='lines',
            name=label,
            line=dict(width=1)
        ))

fig.update_layout(
    title="Stratégie vs Indices (Capital normalisé)",
    xaxis=dict(
        title="Date",
        dtick="M24",  # afficher un tick tous les 24 mois = 2 ans
        tickformat="%Y"
    ),
    yaxis_title="Capital ($)",
    template="plotly_white",
    width=1000,
    height=600
)

pio.write_image(fig, "comparison_plot.png", format="png")

# === CALCUL DES STATS ===
start_date_obj = min(dates)
end_date_obj = max(dates)
nb_years = (end_date_obj - start_date_obj).days // 365

start_equity = equities[0]
end_equity = equities[-1]
net_profit = end_equity - start_equity
net_return_pct = ((end_equity / start_equity) - 1) * 100 if start_equity > 0 else 0

sharpe_list, drawdown_list, ret_list, alpha_list, beta_list = [], [], [], [], []

for entry in data["rollingWindow"].values():
    stats = entry.get("portfolioStatistics", {})
    try:
        sharpe_list.append(float(stats["sharpeRatio"]))
        drawdown_list.append(float(stats["drawdown"]))
        ret_list.append(float(stats["totalNetProfit"]))
        alpha_list.append(float(stats["alpha"]))
        beta_list.append(float(stats["beta"]))
    except:
        continue

def avg(lst):
    return sum(lst) / len(lst) if lst else 0

# === PDF FINAL ===
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.set_title("Rapport de Backtest")

pdf.set_font("Arial", 'B', 14)
pdf.cell(200, 10, txt="Rapport de Backtest - Stratégie vs Indices", ln=True, align="C")

pdf.image("comparison_plot.png", x=10, y=25, w=190)

pdf.set_xy(10, 165)
pdf.set_font("Arial", size=11)
pdf.multi_cell(0, 8, txt=f"""
Période analysée : {nb_years} ans
Capital initial : ${start_equity:,.2f}
Capital final : ${end_equity:,.2f}
Profit net : ${net_profit:,.2f} ({net_return_pct:.2f}%)

Sharpe ratio moyen : {avg(sharpe_list):.2f}
Drawdown moyen : {avg(drawdown_list):.2%}
Rendement mensuel moyen : {avg(ret_list):.2%}
Alpha moyen : {avg(alpha_list):.4f}
Bêta moyen : {avg(beta_list):.4f}
""")

pdf.output("rapport_backtest.pdf")
print("✅ Rapport généré avec succès : rapport_backtest.pdf")
