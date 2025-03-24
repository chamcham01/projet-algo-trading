import streamlit as st
import json
import datetime
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
from fpdf import FPDF

st.title("Backtest - Stratégie vs Actifs de référence")

asset_type = st.selectbox("Choisir l'actif de comparaison", ["Indices", "ETF", "Cryptos"])

if asset_type == "Indices":
    tickers = {
        "S&P 500": "^GSPC",
        "DAX": "^GDAXI",
        "CAC 40": "^FCHI",
        "Nikkei 225": "^N225"
    }
elif asset_type == "ETF":
    tickers = {
        "SPY": "SPY",
        "QQQ": "QQQ",
        "EFA": "EFA",
        "EEM": "EEM"
    }
elif asset_type == "Cryptos":
    tickers = {
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD",
        "Binance Coin": "BNB-USD",
        "XRP": "XRP-USD",
        "Cardano": "ADA-USD",
        "Solana": "SOL-USD",
        "Polkadot": "DOT-USD",
        "Dogecoin": "DOGE-USD",
        "Avalanche": "AVAX-USD",
        "Litecoin": "LTC-USD"
    }

uploaded_file = st.file_uploader("Uploader le fichier JSON du backtest", type="json")

if uploaded_file:
    data = json.load(uploaded_file)

    # Récupération des dates et de l'equity de la stratégie
    dates, equities = [], []
    for key in sorted(data["rollingWindow"].keys()):
        try:
            date_str = key.split("_")[1]
            date_obj = datetime.datetime.strptime(date_str, "%Y%m%d")
            equity = float(data["rollingWindow"][key]["portfolioStatistics"]["endEquity"])
            # Ajustez éventuellement le filtre
            if equity <= 0 or equity > 10_000_000:
                continue
            dates.append(date_obj)
            equities.append(equity)
        except:
            continue

    if not dates:
        st.error("Aucune donnée valide dans le fichier JSON.")
        st.stop()

    start_date = min(dates).strftime("%Y-%m-%d")
    end_date = max(dates).strftime("%Y-%m-%d")

    # Télécharger les données via yfinance
    raw_data = yf.download(list(tickers.values()), start=start_date, end=end_date)

    if raw_data.empty:
        st.warning("Aucune donnée Yahoo Finance pour cette période et ces actifs.")
        st.stop()

    # Extraire le 'Close' si MultiIndex
    if isinstance(raw_data.columns, pd.MultiIndex):
        adj_close = raw_data["Close"]
    else:
        adj_close = raw_data

    # Supprimer d'éventuelles lignes ou colonnes totalement vides
    adj_close.dropna(axis=0, how="all", inplace=True)
    adj_close.dropna(axis=1, how="all", inplace=True)

    # Si tout a été supprimé
    if adj_close.empty:
        st.warning("Toutes les colonnes ou lignes sont vides pour ces actifs.")
        st.stop()

    # Forward-fill + back-fill pour éviter les NaN internes
    adj_close.fillna(method="ffill", inplace=True)
    adj_close.fillna(method="bfill", inplace=True)

    # Créer la série de la stratégie
    strategy_series = pd.Series(equities, index=pd.to_datetime(dates), name="Stratégie")

    # Restreindre la série et adj_close aux dates communes
    common_index = strategy_series.index.intersection(adj_close.index)
    strategy_series = strategy_series.loc[common_index]
    adj_close = adj_close.loc[common_index]

    # S'il n'y a plus de dates communes
    if strategy_series.empty or adj_close.empty:
        st.warning("Aucune date commune entre la stratégie et les données de marché.")
        st.stop()

    # Vérifier que la première ligne n'est pas NaN
    # (après ffill/bfill, normalement ça ne devrait plus être NaN,
    #  sauf si l'actif n'existait vraiment pas encore)
    if adj_close.iloc[0].isna().any():
        st.warning("La première ligne contient encore des NaN : actif trop récent ?")
        st.stop()

    # Normaliser
    normalized_indices = adj_close.div(adj_close.iloc[0]) * strategy_series.iloc[0]

    # Combiner
    combined_df = pd.concat([strategy_series, normalized_indices], axis=1)

    # Tracer
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
        title=f"Stratégie vs {asset_type} (Capital normalisé)",
        xaxis_title="Date",
        yaxis_title="Capital ($)",
        template="plotly_white",
        width=800,
        height=500
    )

    st.plotly_chart(fig)

    # Statistiques simples
    start_equity, end_equity = strategy_series.iloc[0], strategy_series.iloc[-1]
    net_profit = end_equity - start_equity
    net_return_pct = (end_equity / start_equity - 1) * 100

    def avg(lst):
        return sum(lst)/len(lst) if lst else 0

    sharpe, drawdown, ret, alpha, beta = [], [], [], [], []
    for entry in data["rollingWindow"].values():
        stats = entry.get("portfolioStatistics", {})
        try:
            sharpe.append(float(stats["sharpeRatio"]))
            drawdown.append(float(stats["drawdown"]))
            ret.append(float(stats["totalNetProfit"]))
            alpha.append(float(stats["alpha"]))
            beta.append(float(stats["beta"]))
        except:
            continue

    st.subheader("📊 Statistiques")
    st.write(f"""
    - **Période analysée :** {(max(dates)-min(dates)).days // 365} ans
    - **Capital initial :** ${start_equity:,.2f}
    - **Capital final :** ${end_equity:,.2f}
    - **Profit net :** ${net_profit:,.2f} ({net_return_pct:.2f}%)
    - **Sharpe ratio moyen :** {avg(sharpe):.2f}
    - **Drawdown moyen :** {avg(drawdown):.2%}
    - **Rendement mensuel moyen :** {avg(ret):.2%}
    - **Alpha moyen :** {avg(alpha):.4f}
    - **Bêta moyen :** {avg(beta):.4f}
    """)

    if st.button("📄 Générer le rapport PDF"):
        fig.write_image("comparison_plot.png")

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, f"Rapport de Backtest - Stratégie vs {asset_type}", ln=True, align="C")
        pdf.image("comparison_plot.png", x=10, y=25, w=190)
        pdf.set_xy(10, 165)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 8, txt=f"""
        Période analysée : {(max(dates)-min(dates)).days // 365} ans
        Capital initial : ${start_equity:,.2f}
        Capital final : ${end_equity:,.2f}
        Profit net : ${net_profit:,.2f} ({net_return_pct:.2f}%)

        Sharpe ratio moyen : {avg(sharpe):.2f}
        Drawdown moyen : {avg(drawdown):.2%}
        Rendement mensuel moyen : {avg(ret):.2%}
        Alpha moyen : {avg(alpha):.4f}
        Bêta moyen : {avg(beta):.4f}
        """)
        pdf.output("rapport_backtest.pdf")

        with open("rapport_backtest.pdf", "rb") as f:
            st.download_button("Télécharger le PDF", f, "rapport_backtest.pdf")
