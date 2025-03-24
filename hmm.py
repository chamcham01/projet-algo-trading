import matplotlib
matplotlib.use('TkAgg')  # Important : doit être placé avant d'importer pyplot

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# (le reste de ton code reste identique)

# Téléchargement des données historiques (S&P500 et CAC40)
tickers = ['^GSPC', '^FCHI']
data = yf.download(tickers, start='2020-01-01', end='2024-01-01', auto_adjust=True)

# Sélectionner explicitement les prix de clôture ajustés
data_close = data['Close'].dropna()
data_close.columns = ['SP500', 'CAC40']

# Calcul des rendements quotidiens
returns = data_close.pct_change().dropna()
print(returns.head())

# Définir les observations (hausse = 1, baisse = 0)
observations_sp500 = (returns['SP500'] > 0).astype(int).values
observations_cac40 = (returns['CAC40'] > 0).astype(int).values

# Implémentation de l'algorithme Viterbi
def viterbi(obs, pi, A, B):
    N = A.shape[0]
    T = len(obs)

    delta = np.zeros((N, T))
    psi = np.zeros((N, T), dtype=int)

    delta[:, 0] = pi * B[:, obs[0]]
    for t in range(1, T):
        for j in range(N):
            delta[j, t] = np.max(delta[:, t-1] * A[:, j]) * B[j, obs[t]]
            psi[j, t] = np.argmax(delta[:, t-1] * A[:, j])

    states = np.zeros(T, dtype=int)
    states[T-1] = np.argmax(delta[:, T-1])

    for t in reversed(range(1, T)):
        states[t-1] = psi[states[t], t]

    return states

# Paramètres du modèle (0 = Bull, 1 = Bear)
pi = np.array([0.5, 0.5])
A = np.array([[0.95, 0.05],
              [0.10, 0.90]])
B = np.array([[0.7, 0.3],
              [0.4, 0.6]])

# Application du modèle aux données
returns['Regime_SP500'] = viterbi(observations_sp500, pi, A, B)
returns['Regime_CAC40'] = viterbi(observations_cac40, pi, A, B)

# Visualisation des régimes cachés pour le S&P500
fig, ax = plt.subplots(figsize=(14,6))
returns['SP500'].cumsum().plot(ax=ax, title='S&P500 avec Régimes Cachés')
for i in range(len(returns)):
    color = 'green' if returns['Regime_SP500'].iloc[i] == 0 else 'red'
    ax.axvspan(returns.index[i], returns.index[min(i+1, len(returns)-1)], color=color, alpha=0.2)
plt.xlabel('Date')
plt.ylabel('Rendement cumulé')
plt.show()

# Visualisation des régimes cachés pour le CAC40
fig, ax = plt.subplots(figsize=(14,6))
returns['CAC40'].cumsum().plot(ax=ax, title='CAC40 avec Régimes Cachés')
for i in range(len(returns)):
    color = 'green' if returns['Regime_CAC40'].iloc[i] == 0 else 'red'
    ax.axvspan(returns.index[i], returns.index[min(i+1, len(returns)-1)], color=color, alpha=0.2)
plt.xlabel('Date')
plt.ylabel('Rendement cumulé')
plt.show()
