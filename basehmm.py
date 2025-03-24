import yfinance as yf
import numpy as np
import pandas as pd
from hmmlearn import hmm
import plotly.graph_objects as go

# Télécharger les données
data = yf.download("^GSPC", start="1998-01-01", end="2023-01-01")

# Aplatir colonnes si nécessaire
if isinstance(data.columns, pd.MultiIndex):
    data.columns = ['_'.join(col).strip() for col in data.columns.values]

# Récupérer la colonne "Close"
close_col = [col for col in data.columns if 'Close' in col][0]
data = data[[close_col]].copy()
data.rename(columns={close_col: "Close"}, inplace=True)
data.dropna(inplace=True)

# Calcul des log-rendements
rets = np.log(data['Close']).diff().dropna()

# Modèle HMM
model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=200, random_state=42)
model.fit(rets.values.reshape(-1, 1))
hidden_states = model.predict(rets.values.reshape(-1, 1))

# Intégrer les états
data = data.iloc[1:].copy()
data['État'] = hidden_states
data['Date'] = data.index

# Nommer les états
state_means = model.means_.flatten()
state_order = np.argsort(state_means)
state_names = ["Baissier", "Neutre", "Haussier"]
state_map = {state_order[i]: state_names[i] for i in range(3)}
data['Régime'] = data['État'].map(state_map)

# Détecter les changements
data['Changement'] = data['Régime'].ne(data['Régime'].shift())

# Points de changement
changement_df = data[data['Changement']].copy()

# Initialiser le graphique
fig = go.Figure()

# Courbe S&P 500
fig.add_trace(go.Scatter(
    x=data['Date'],
    y=data['Close'],
    mode='lines',
    name='S&P 500',
    line=dict(color='black')
))

# Ajouter les points de changement
colors = {"Baissier": "red", "Neutre": "orange", "Haussier": "green"}
for regime in data['Régime'].unique():
    points = changement_df[changement_df['Régime'] == regime]
    fig.add_trace(go.Scatter(
        x=points['Date'],
        y=points['Close'],
        mode='markers',
        name=f"Changement vers {regime}",
        marker=dict(size=8, color=colors[regime], symbol="x")
    ))

# Ajouter les zones colorées pour chaque régime
shapes = []
start_idx = 0
for i in range(1, len(data)):
    if data['Régime'].iloc[i] != data['Régime'].iloc[i - 1] or i == len(data) - 1:
        end_idx = i
        start_date = data['Date'].iloc[start_idx]
        end_date = data['Date'].iloc[end_idx]
        regime = data['Régime'].iloc[start_idx]
        shapes.append(dict(
            type="rect",
            xref="x", yref="paper",
            x0=start_date, x1=end_date,
            y0=0, y1=1,
            fillcolor=colors[regime],
            opacity=0.1,
            layer="below",
            line_width=0,
        ))
        start_idx = i

# Ajouter les formes au layout
fig.update_layout(
    shapes=shapes,
    title="Détection de régimes de marché (HMM) sur le S&P 500",
    xaxis_title="Date",
    yaxis_title="Cours de clôture",
    template="plotly_white",
    legend_title="Légende"
)

fig.show()
