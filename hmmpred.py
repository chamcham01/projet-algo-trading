import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn import hmm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------------------------------------------------------
# 1. Téléchargement des données journalières via yfinance
# --------------------------------------------------------------------
ticker = "^GSPC"  # S&P 500
start_date = "2010-01-01"
end_date = "2025-01-01"  # pour inclure toute l'année 2024

df = yf.download(ticker, start=start_date, end=end_date)
df["LogClose"] = np.log(df["Close"])

# --------------------------------------------------------------------
# 2. Agrégation en données mensuelles
# --------------------------------------------------------------------
# On utilise la dernière valeur de chaque mois pour le 'Close'
df_monthly = df.resample('M').last()
df_monthly["Monthly_Return"] = df_monthly["LogClose"].diff()
df_monthly.dropna(inplace=True)

# --------------------------------------------------------------------
# 3. Séparation Train/Test
# --------------------------------------------------------------------
# Train : jusqu'à fin 2023 ; Test : de Janvier 2024 à Décembre 2024
train_end_date = "2023-12-31"
df_train = df_monthly.loc[:train_end_date].copy()
df_test = df_monthly.loc["2024-01-01":].copy()

# --------------------------------------------------------------------
# 4. Entraînement du HMM sur les rendements mensuels du train
# --------------------------------------------------------------------
returns_train = df_train["Monthly_Return"].values.reshape(-1, 1)
n_states = 3  # par exemple, on choisit 3 états
model = hmm.GaussianHMM(n_components=n_states, n_iter=200, covariance_type="full", random_state=42)
model.fit(returns_train)
print("HMM entraîné sur les rendements mensuels jusqu'à fin 2023.")

# --------------------------------------------------------------------
# 5. Prévision one-step-ahead sur la période test (mensuelle)
# --------------------------------------------------------------------
predicted_returns = []
predicted_states = []
# On initialise avec la distribution du dernier point de train
last_state_distribution = model.predict_proba(returns_train)[-1, :]

returns_test = df_test["Monthly_Return"].values

for r_obs in returns_test:
    # a) Prévision a priori de la distribution des états
    new_state_distribution = last_state_distribution @ model.transmat_
    # Calcul de l'espérance du rendement mensuel prédit
    expected_return = np.sum(new_state_distribution * model.means_.flatten())
    predicted_returns.append(expected_return)
    predicted_state = np.argmax(new_state_distribution)
    predicted_states.append(predicted_state)

    # b) Mise à jour par la vraisemblance : on calcule la densité pour chaque état
    densities = []
    for s in range(n_states):
        mean_s = model.means_[s, 0]
        # Extraction de la variance selon le type de covariance
        var_s = model.covars_[s, 0, 0] if model.covariance_type == "full" else model.covars_[s]
        pdf_val = (1 / np.sqrt(2 * np.pi * var_s)) * np.exp(-0.5 * ((r_obs - mean_s) ** 2 / var_s))
        densities.append(pdf_val)
    densities = np.array(densities)
    # Mise à jour de la distribution (posteriori)
    posterior = new_state_distribution * densities
    posterior /= np.sum(posterior)
    last_state_distribution = posterior

df_test["Predicted_Monthly_Return"] = predicted_returns

# --------------------------------------------------------------------
# 6. Définition des tendances mensuelles
# --------------------------------------------------------------------
# Tendance réelle : bullish (1) si Monthly_Return > 0, bearish (-1) sinon
df_test["Actual_Trend"] = np.where(df_test["Monthly_Return"] > 0, 1, -1)
# Tendance prédite : bullish (1) si le rendement prédit > 0, bearish (-1) sinon
df_test["Predicted_Trend"] = np.where(df_test["Predicted_Monthly_Return"] > 0, 1, -1)

# Calcul des rendements cumulés mensuels
df_test["Cumulative_Actual"] = df_test["Monthly_Return"].cumsum()
df_test["Cumulative_Predicted"] = df_test["Predicted_Monthly_Return"].cumsum()

# --------------------------------------------------------------------
# 7. Visualisation avec Plotly
# --------------------------------------------------------------------
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    subplot_titles=["Tendance mensuelle (2024) : Réelle vs. Prédite",
                                    "Rendement cumulé mensuel (2024) : Réel vs. Prédit"],
                    vertical_spacing=0.15)

# Sous-figure 1 : Tendance mensuelle
fig.add_trace(
    go.Scatter(
        x=df_test.index,
        y=df_test["Actual_Trend"],
        mode="lines+markers",
        name="Tendance Réelle",
        line=dict(color="black"),
        marker=dict(symbol="circle")
    ),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(
        x=df_test.index,
        y=df_test["Predicted_Trend"],
        mode="lines+markers",
        name="Tendance Prédite",
        line=dict(color="red"),
        marker=dict(symbol="diamond")
    ),
    row=1, col=1
)
fig.update_yaxes(title_text="Tendance", tickvals=[-1, 1], ticktext=["Bearish", "Bullish"], row=1, col=1)

# Sous-figure 2 : Rendement cumulé mensuel
fig.add_trace(
    go.Scatter(
        x=df_test.index,
        y=df_test["Cumulative_Actual"],
        mode="lines",
        name="Cumul Réel",
        line=dict(color="black")
    ),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(
        x=df_test.index,
        y=df_test["Cumulative_Predicted"],
        mode="lines",
        name="Cumul Prédit",
        line=dict(color="red")
    ),
    row=2, col=1
)
fig.update_yaxes(title_text="Cumulative Return", row=2, col=1)
fig.update_xaxes(title_text="Date", row=2, col=1)

fig.update_layout(title="Prédiction de Tendances Mensuelles via HMM : Réel vs. Prédit (2024)",
                  hovermode="x unified")
fig.show()
