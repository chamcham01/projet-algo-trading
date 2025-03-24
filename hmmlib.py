import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from hmmlearn import hmm
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# --------------------------------------------------------------------
# 1. Récupération des données via yfinance
# --------------------------------------------------------------------

ticker = "^GSPC"  # Code Yahoo Finance du S&P 500
start_date = "1928-01-01"
end_date = "2024-03-19"

# Téléchargement des données
df = yf.download(ticker, start=start_date, end=end_date)

# --------------------------------------------------------------------
# 2. Prétraitement : Calcul des rendements log
# --------------------------------------------------------------------

df["LogClose"] = np.log(df["Close"])
df["Return"] = df["LogClose"].diff()
df.dropna(inplace=True)  # Suppression des NaN

# Conversion en numpy array (HMM attend un array de shape (N, 1))
returns = df["Return"].values.reshape(-1, 1)

# --------------------------------------------------------------------
# 3. Ajustement des modèles HMM avec différents nombres d'états
# --------------------------------------------------------------------

def fit_hmm_models(returns, n_states_list, covariance_type="full", n_iter=200, random_state=42):
    """
    Essaie différents modèles HMM et compare leurs log-vraisemblance, AIC et BIC.
    """
    results = []
    N = len(returns)

    for n_states in n_states_list:
        model = hmm.GaussianHMM(n_components=n_states,
                                covariance_type=covariance_type,
                                n_iter=n_iter,
                                random_state=random_state)
        model.fit(returns)  # Entraînement du modèle

        # Log-vraisemblance
        log_likelihood = model.score(returns)

        # Nombre de paramètres estimés (approximation pour HMM gaussien dimension=1)
        n_params = (n_states - 1) + n_states * (n_states - 1) + n_states * 1 + n_states * 1

        # Calcul des critères AIC et BIC
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(N) - 2 * log_likelihood

        results.append({
            "n_states": n_states,
            "logL": log_likelihood,
            "AIC": aic,
            "BIC": bic,
            "model": model
        })

    return pd.DataFrame(results)

# Essai avec 2, 3, 4 états
n_states_candidates = [2, 3, 4]
results_df = fit_hmm_models(returns, n_states_candidates)

# Sélection du meilleur modèle basé sur BIC
best_index = results_df["BIC"].idxmin()
best_model = results_df.loc[best_index, "model"]
best_n_states = results_df.loc[best_index, "n_states"]

print(results_df[["n_states", "logL", "AIC", "BIC"]])
print(f"\nMeilleur modèle selon BIC : {best_n_states} états.")

# --------------------------------------------------------------------
# 4. Décodage des régimes de marché
# --------------------------------------------------------------------

# États les plus probables (Viterbi)
hidden_states = best_model.predict(returns)

# Probabilités filtrées (probabilité d'être dans chaque état à chaque instant)
state_probs = best_model.predict_proba(returns)

# Ajout des résultats au DataFrame
df["State"] = hidden_states
for i in range(best_n_states):
    df[f"Prob_State_{i}"] = state_probs[:, i]

# --------------------------------------------------------------------
# 5. Visualisation des régimes de marché avec Plotly
# --------------------------------------------------------------------

# Définition d'une palette de couleurs (Seaborn)
palette = sns.color_palette("Set1", n_colors=best_n_states)

def to_rgba(color, alpha=0.2):
    """Convertit un tuple (r, g, b) de Seaborn en string 'rgba(r, g, b, alpha)'."""
    return f"rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, {alpha})"

# Calcul du rendement cumulé
df["Cumulative_Return"] = df["Return"].cumsum()

# Création d'une figure Plotly avec 2 sous-graphiques
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.6, 0.4],
    vertical_spacing=0.08,
    subplot_titles=["S&P 500 : Rendement cumulé et régimes HMM", "Probabilités filtrées des régimes"]
)

# Sous-graphe 1 : Rendement cumulé
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Cumulative_Return"],
        mode='lines',
        line=dict(color='blue'),
        name='Rendement cumulé'
    ),
    row=1, col=1
)

# Pour la coloration, on va créer des rectangles (shapes)
# Pour connaître le domaine vertical du premier subplot, on utilise yaxis de la figure
# (Les shapes en mode "paper" utilisent des coordonnées entre 0 et 1)
domain1 = fig.layout["yaxis"]["domain"]

shapes = []
prev_state = hidden_states[0]
start_idx = 0

for i in range(1, len(hidden_states)):
    if hidden_states[i] != prev_state:
        shapes.append(dict(
            type="rect",
            xref="x1",
            yref="paper",  # On utilise les coordonnées "paper" et le domaine du premier subplot
            x0=df.index[start_idx],
            x1=df.index[i-1],
            y0=domain1[0],
            y1=domain1[1],
            fillcolor=to_rgba(palette[prev_state], 0.2),
            line=dict(width=0)
        ))
        prev_state = hidden_states[i]
        start_idx = i

# Dernier segment
shapes.append(dict(
    type="rect",
    xref="x1",
    yref="paper",
    x0=df.index[start_idx],
    x1=df.index[-1],
    y0=domain1[0],
    y1=domain1[1],
    fillcolor=to_rgba(palette[prev_state], 0.2),
    line=dict(width=0)
))

# Sous-graphe 2 : Probabilités filtrées
for i in range(best_n_states):
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[f"Prob_State_{i}"],
            mode='lines',
            name=f"Prob état {i}",
            line=dict(color=to_rgba(palette[i], 1))
        ),
        row=2, col=1
    )

# Mise à jour finale de la mise en forme
fig.update_layout(
    shapes=shapes,
    title="Détection de régimes cachés (HMM) sur le S&P 500",
    hovermode="x unified"
)

fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
fig.update_yaxes(title_text="Probabilité", row=2, col=1)

fig.show()
