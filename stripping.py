import streamlit as st
import datetime
import re
from math import log, exp
from bisect import bisect_left
import matplotlib

matplotlib.use('Agg')  # Backend non interactif pour Streamlit
import matplotlib.pyplot as plt

# --- Paramètres d'onglet pour Streamlit ---
st.set_page_config(
    page_title="Stripping de Courbe de Taux",
    layout="wide",
    initial_sidebar_state="auto",
    # Vous pouvez aussi configurer la couleur primaire ici :
    # theme={"primaryColor": "#007acc"}
)

# === Injection de CSS personnalisé ===
st.markdown("""
    <style>
    /* Contexte général de l'application */
    .reportview-container {
        background: #f5f7fa;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Ajustement général des marges */
    .css-18e3th9 {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Titres centraux */
    h1, h2, h3 {
        color: #333333;
        text-align: center;
        font-weight: 600;
    }

    /* Conteneur principal des onglets (baseweb) */
    [data-baseweb="tab-list"] {
        justify-content: center;
        background: #ffffff;
        border-bottom: 2px solid #e6e6e6;
        flex-wrap: wrap; /* Permet le retour à la ligne si besoin */
        margin-bottom: 1rem;
    }

    /* Style de base pour les onglets */
    [data-baseweb="tab-list"] button {
        border: none !important;
        border-bottom: none !important;
        color: #555 !important;
        font-size: 16px;
        padding: 10px 20px;
        margin: 0 0.5rem;
        transition: all 0.2s ease-in-out;
        background: none; /* retire tout fond bizarre */
        white-space: nowrap; /* évite la troncature des mots */
    }

    /* Onglet au survol : bordure bleue fine */
    [data-baseweb="tab-list"] button:hover {
        border-bottom: 2px solid #007acc !important;
        color: #007acc !important;
        cursor: pointer;
    }

    /* Onglet actif : bordure bleue épaisse et texte bleu */
    [data-baseweb="tab-list"] button[aria-selected="true"] {
        border-bottom: 3px solid #007acc !important;
        color: #007acc !important;
    }

    /* Retire les focus outlines si nécessaire */
    [data-baseweb="tab-list"] button:focus {
        outline: none !important;
        box-shadow: none !important;
    }
    </style>
""", unsafe_allow_html=True)


# === Fonctions utilitaires et calculs (identiques) ===
def adjust_date(start_date, offset):
    match = re.match(r'(\d+)(\D+)', offset)
    value = int(match.group(1))
    unit = match.group(2).upper()
    if unit == 'D':
        new_date = start_date + datetime.timedelta(days=value)
    elif unit == 'M':
        month = start_date.month + value
        year = start_date.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        day = start_date.day
        try:
            new_date = datetime.date(year, month, day)
        except ValueError:
            new_date = datetime.date(year, month, 28)
            if day > 28:
                new_date += datetime.timedelta(days=min(day - 28, 3))
    elif unit == 'Y':
        year = start_date.year + value
        try:
            new_date = datetime.date(year, start_date.month, start_date.day)
        except ValueError:
            new_date = datetime.date(year, start_date.month, 28) + datetime.timedelta(days=1)
    else:
        raise ValueError(f"Unknown unit: {unit}")
    while new_date.weekday() >= 5:
        new_date += datetime.timedelta(days=1)
    return new_date


def fraction_annee(start_date, end_date, convention):
    delta_days = (end_date - start_date).days
    if convention == 'ACT/360':
        return delta_days / 360.0
    elif convention == '30/360':
        d1 = min(start_date.day, 30)
        d2 = min(end_date.day, 30) if d1 < 30 else 30
        total_days = 360 * (end_date.year - start_date.year) + 30 * (end_date.month - start_date.month) + (d2 - d1)
        return total_days / 360.0
    else:
        raise ValueError(f"Unknown convention: {convention}")


class ZeroCurve:
    def __init__(self, curve_date):
        self.curve_date = curve_date
        self.times = []
        self.zero_rates = []

    def add_point(self, t, z):
        index = bisect_left(self.times, t)
        if index < len(self.times) and self.times[index] == t:
            self.zero_rates[index] = z
        else:
            self.times.insert(index, t)
            self.zero_rates.insert(index, z)

    def get_zero_rate(self, t):
        if not self.times:
            return 0.0
        if t <= self.times[0]:
            return self.zero_rates[0]
        if t >= self.times[-1]:
            return self.zero_rates[-1]
        index = bisect_left(self.times, t)
        x0, x1 = self.times[index - 1], self.times[index]
        y0, y1 = self.zero_rates[index - 1], self.zero_rates[index]
        return y0 + (y1 - y0) * (t - x0) / (x1 - x0)


def depositRate(curve_date, start_date, end_date, rate, day_count_conv):
    tau = fraction_annee(start_date, end_date, day_count_conv)
    t = (end_date - curve_date).days / 365.0
    return log(1 + rate / 100 * tau) / t


def fraRate(curve_date, start_date, end_date, rate, day_count_conv, curve):
    t1 = (start_date - curve_date).days / 365.0
    t2 = (end_date - curve_date).days / 365.0
    tau = fraction_annee(start_date, end_date, day_count_conv)
    z1 = curve.get_zero_rate(t1)
    forward_rate = log(1 + rate / 100 * tau) / (t2 - t1)
    return (z1 * t1 + forward_rate * (t2 - t1)) / t2


def swapRate(curve_date, start_date, end_date, swap_rate, day_count_conv, curve, payment_freq='1Y'):
    payment_dates = []
    current_date = start_date
    while current_date < end_date:
        next_date = adjust_date(current_date, payment_freq)
        if next_date > end_date:
            next_date = end_date
        payment_dates.append(next_date)
        current_date = next_date
    payment_times = [(d - curve_date).days / 365.0 for d in payment_dates]
    tau_values = [fraction_annee(payment_dates[i - 1] if i > 0 else start_date, d, day_count_conv)
                  for i, d in enumerate(payment_dates)]
    sum_tau_df = 0.0
    rate_conv = swap_rate / 100
    for t, tau in zip(payment_times[:-1], tau_values[:-1]):
        df = exp(-curve.get_zero_rate(t) * t)
        sum_tau_df += tau * df
    t_maturity = payment_times[-1]
    df_maturity = (1 - rate_conv * sum_tau_df) / (1 + rate_conv * tau_values[-1])
    return -log(df_maturity) / t_maturity


def swapRate_multi(curve_date, start_date, end_date, swap_rate, day_count_conv, forward_curve, discount_curve,
                   payment_freq='1Y'):
    payment_dates = []
    current_date = start_date
    while current_date < end_date:
        next_date = adjust_date(current_date, payment_freq)
        if next_date > end_date:
            next_date = end_date
        payment_dates.append(next_date)
        current_date = next_date
    payment_times = [(d - curve_date).days / 365.0 for d in payment_dates]
    tau_values = [fraction_annee(payment_dates[i - 1] if i > 0 else start_date, d, day_count_conv)
                  for i, d in enumerate(payment_dates)]
    sum_tau_df = 0.0
    rate_conv = swap_rate / 100
    for t, tau in zip(payment_times[:-1], tau_values[:-1]):
        df = exp(-discount_curve.get_zero_rate(t) * t)
        sum_tau_df += tau * df
    t_maturity = payment_times[-1]
    df_maturity = (1 - rate_conv * sum_tau_df) / (1 + rate_conv * tau_values[-1])
    return -log(df_maturity) / t_maturity


def process_curve(curve_date, instruments, params):
    curve = ZeroCurve(curve_date)
    instruments_sorted = sorted(
        [{
            'type': inst['instrument'],
            'start': adjust_date(curve_date, inst['start_offset']),
            'end': adjust_date(adjust_date(curve_date, inst['start_offset']), inst['maturity_offset']),
            'rate': inst['rate']
        } for inst in instruments],
        key=lambda x: (x['end'] - curve_date).days
    )
    for inst in instruments_sorted:
        t = (inst['end'] - curve_date).days / 365.0
        if inst['type'] == 'DEPOSIT':
            z = depositRate(curve_date, inst['start'], inst['end'], inst['rate'], params['dec_deposit'])
            curve.add_point(t, z)
        elif inst['type'] == 'FRA':
            z = fraRate(curve_date, inst['start'], inst['end'], inst['rate'], params['dec_fra'], curve)
            curve.add_point(t, z)
        elif inst['type'] in ['SWAP', 'SWAPOIS']:
            z = swapRate(curve_date, inst['start'], inst['end'], inst['rate'], params['dec_fixe'], curve,
                         params['freq_payment_fixe'])
            curve.add_point(t, z)
    return curve


def process_curve_multi(curve_date, instruments, params, discount_curve):
    curve = ZeroCurve(curve_date)
    instruments_sorted = sorted(
        [{
            'type': inst['instrument'],
            'start': adjust_date(curve_date, inst['start_offset']),
            'end': adjust_date(adjust_date(curve_date, inst['start_offset']), inst['maturity_offset']),
            'rate': inst['rate']
        } for inst in instruments],
        key=lambda x: (x['end'] - curve_date).days
    )
    for inst in instruments_sorted:
        t = (inst['end'] - curve_date).days / 365.0
        if inst['type'] == 'DEPOSIT':
            z = depositRate(curve_date, inst['start'], inst['end'], inst['rate'], params['dec_deposit'])
            curve.add_point(t, z)
        elif inst['type'] == 'FRA':
            z = fraRate(curve_date, inst['start'], inst['end'], inst['rate'], params['dec_fra'], curve)
            curve.add_point(t, z)
        elif inst['type'] in ['SWAP', 'SWAPOIS']:
            z = swapRate_multi(curve_date, inst['start'], inst['end'], inst['rate'], params['dec_fixe'],
                               curve, discount_curve, params['freq_payment_fixe'])
            curve.add_point(t, z)
    return curve


def parse_instruments(raw_data):
    return [
        {
            'instrument': row[0],
            'start_offset': row[1],
            'maturity_offset': row[2],
            'rate': row[3]
        }
        for row in raw_data
    ]


# === Données d'entrée (simplifiées pour l'exemple) ===
eonia_data = [
    ['SWAPOIS', '0D', '1D', 0.962],
    ['SWAPOIS', '0D', '7D', 0.9792],
    ['SWAPOIS', '0D', '14D', 0.9935],
    ['SWAPOIS', '0D', '1M', 0.9878],
    ['SWAPOIS', '0D', '2M', 0.975],
    ['SWAPOIS', '0D', '3M', 0.97],
    ['SWAPOIS', '0D', '4M', 0.975],
    ['SWAPOIS', '0D', '5M', 0.97],
    ['SWAPOIS', '0D', '6M', 0.9705],
    ['SWAPOIS', '0D', '7M', 0.965],
    ['SWAPOIS', '0D', '8M', 0.96],
    ['SWAPOIS', '0D', '9M', 0.955],
    ['SWAPOIS', '0D', '10M', 0.955],
    ['SWAPOIS', '0D', '11M', 0.95],
    ['SWAPOIS', '0D', '1Y', 0.945],
    ['SWAPOIS', '0D', '18M', 0.945],
    ['SWAPOIS', '0D', '2Y', 0.95],
    ['SWAPOIS', '0D', '30M', 0.96],
    ['SWAPOIS', '0D', '3Y', 0.98],
    ['SWAPOIS', '0D', '4Y', 1.03],
    ['SWAPOIS', '0D', '5Y', 1.109],
    ['SWAPOIS', '0D', '6Y', 1.213],
    ['SWAPOIS', '0D', '7Y', 1.337],
    ['SWAPOIS', '0D', '8Y', 1.468],
    ['SWAPOIS', '0D', '9Y', 1.596],
    ['SWAPOIS', '0D', '10Y', 1.716],
    ['SWAPOIS', '0D', '11Y', 1.825],
    ['SWAPOIS', '0D', '12Y', 1.923],
    ['SWAPOIS', '0D', '15Y', 2.159],
    ['SWAPOIS', '0D', '20Y', 2.402],
    ['SWAPOIS', '0D', '25Y', 2.5145],
    ['SWAPOIS', '0D', '30Y', 2.573]
]
euribor3m_data = [
    ['DEPOSIT', '2D', '3M', 1.08],
    ['FRA', '1M', '4M', 1.085],
    ['FRA', '2M', '5M', 1.081],
    ['FRA', '3M', '6M', 1.078],
    ['FRA', '4M', '7M', 1.08],
    ['FRA', '5M', '8M', 1.076],
    ['FRA', '6M', '9M', 1.077],
    ['FRA', '7M', '10M', 1.075],
    ['FRA', '8M', '11M', 1.076],
    ['FRA', '9M', '12M', 1.077],
    ['FRA', '12M', '15M', 1.086],
    ['FRA', '15M', '18M', 1.1015],
    ['FRA', '18M', '21M', 1.1265],
    ['FRA', '21M', '24M', 1.149],
    ['SWAP', '2D', '3Y', 1.137],
    ['SWAP', '2D', '4Y', 1.2045],
    ['SWAP', '2D', '5Y', 1.293],
    ['SWAP', '2D', '6Y', 1.403],
    ['SWAP', '2D', '7Y', 1.528],
    ['SWAP', '2D', '8Y', 1.661],
    ['SWAP', '2D', '9Y', 1.791],
    ['SWAP', '2D', '10Y', 1.913],
    ['SWAP', '2D', '11Y', 2.023],
    ['SWAP', '2D', '12Y', 2.121],
    ['SWAP', '2D', '15Y', 2.353],
    ['SWAP', '2D', '20Y', 2.578],
    ['SWAP', '2D', '25Y', 2.679],
    ['SWAP', '2D', '30Y', 2.72]
]
euribor6m_data = [
    ['DEPOSIT', '2D', '6M', 1.181],
    ['SWAP', '2D', '1Y', 1.185],
    ['SWAP', '2D', '18M', 1.193],
    ['SWAP', '2D', '2Y', 1.208],
    ['SWAP', '2D', '3Y', 1.2593],
    ['SWAP', '2D', '4Y', 1.3336],
    ['SWAP', '2D', '5Y', 1.419],
    ['SWAP', '2D', '6Y', 1.5285],
    ['SWAP', '2D', '7Y', 1.656],
    ['SWAP', '2D', '8Y', 1.786],
    ['SWAP', '2D', '9Y', 1.911],
    ['SWAP', '2D', '10Y', 2.028],
    ['SWAP', '2D', '11Y', 2.1405],
    ['SWAP', '2D', '12Y', 2.2275],
    ['SWAP', '2D', '13Y', 2.3156],
    ['SWAP', '2D', '14Y', 2.388],
    ['SWAP', '2D', '15Y', 2.4503],
    ['SWAP', '2D', '16Y', 2.5],
    ['SWAP', '2D', '17Y', 2.548],
    ['SWAP', '2D', '18Y', 2.589],
    ['SWAP', '2D', '19Y', 2.624],
    ['SWAP', '2D', '20Y', 2.6583],
    ['SWAP', '2D', '21Y', 2.678],
    ['SWAP', '2D', '22Y', 2.7],
    ['SWAP', '2D', '23Y', 2.7215],
    ['SWAP', '2D', '24Y', 2.74],
    ['SWAP', '2D', '25Y', 2.7483],
    ['SWAP', '2D', '26Y', 2.755],
    ['SWAP', '2D', '27Y', 2.764],
    ['SWAP', '2D', '28Y', 2.772],
    ['SWAP', '2D', '29Y', 2.78],
    ['SWAP', '2D', '30Y', 2.786]
]

# === Paramètres des courbes ===
curve_date = datetime.date(2025, 2, 3)
eonia_params = {
    'freq_fixing_var': '1D',
    'dec_var': 'ACT/360',
    'freq_payment_fixe': '1Y',
    'dec_fixe': 'ACT/360',
    'dec_deposit': 'ACT/360',
    'dec_fra': 'ACT/360'
}
euribor3m_params = {
    'freq_fixing_var': '3M',
    'dec_var': '30/360',
    'freq_payment_fixe': '1Y',
    'dec_fixe': 'ACT/360',
    'dec_deposit': 'ACT/360',
    'dec_fra': 'ACT/360'
}
euribor6m_params = {
    'freq_fixing_var': '6M',
    'dec_var': '30/360',
    'freq_payment_fixe': '1Y',
    'dec_fixe': 'ACT/360',
    'dec_deposit': 'ACT/360',
    'dec_fra': 'ACT/360'
}

# === Construction des courbes ===
eonia_instruments = parse_instruments(eonia_data)
euribor3m_instruments = parse_instruments(euribor3m_data)
euribor6m_instruments = parse_instruments(euribor6m_data)

eonia_curve = process_curve(curve_date, eonia_instruments, eonia_params)
euribor3m_curve = process_curve(curve_date, euribor3m_instruments, euribor3m_params)
euribor6m_curve = process_curve(curve_date, euribor6m_instruments, euribor6m_params)

euribor3m_multi = process_curve_multi(curve_date, euribor3m_instruments, euribor3m_params, eonia_curve)
euribor6m_multi = process_curve_multi(curve_date, euribor6m_instruments, euribor6m_params, eonia_curve)


# === Calcul des taux forward (exemple) ===
def compute_forward_rates(zero_curve, forward_period):
    times = []
    forward_rates = []
    dt = 7 / 365.0
    t = 0.0
    max_t = 30
    while t + forward_period <= max_t:
        rate_start = zero_curve.get_zero_rate(t)
        rate_end = zero_curve.get_zero_rate(t + forward_period)
        fwd = (rate_end * (t + forward_period) - rate_start * t) / forward_period
        times.append(t)
        forward_rates.append(fwd * 100)
        t += dt
    return times, forward_rates


fwd3m_mono_times, fwd3m_mono = compute_forward_rates(euribor3m_curve, 3 / 12)
fwd3m_multi_times, fwd3m_multi = compute_forward_rates(euribor3m_multi, 3 / 12)
fwd6m_mono_times, fwd6m_mono = compute_forward_rates(euribor6m_curve, 6 / 12)
fwd6m_multi_times, fwd6m_multi = compute_forward_rates(euribor6m_multi, 6 / 12)


# === Création des figures (exemples) ===
def fig_comparison():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(euribor3m_curve.times, [z * 100 for z in euribor3m_curve.zero_rates],
            color='blue', linestyle='--', label='EURIBOR3M Mono')
    ax.plot(euribor3m_multi.times, [z * 100 for z in euribor3m_multi.zero_rates],
            color='blue', linestyle='-', label='EURIBOR3M Multi')
    ax.plot(euribor6m_curve.times, [z * 100 for z in euribor6m_curve.zero_rates],
            color='red', linestyle='--', label='EURIBOR6M Mono')
    ax.plot(euribor6m_multi.times, [z * 100 for z in euribor6m_multi.zero_rates],
            color='red', linestyle='-', label='EURIBOR6M Multi')
    ax.set_title('Impact du Multi-Courbe sur les Taux Zéro-Coupon', fontsize=14)
    ax.set_xlabel('Maturité (années)', fontsize=12)
    ax.set_ylabel('Taux ZC Continu (%)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 3.5)
    fig.tight_layout()
    return fig


def fig_stripped_vs_raw():
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    line_styles = ['-', '--']
    curves = {
        "EONIA": eonia_curve,
        "EURIBOR3M": euribor3m_curve,
        "EURIBOR6M": euribor6m_curve
    }
    for idx, (name, curve) in enumerate(curves.items()):
        ax.plot(curve.times, [z * 100 for z in curve.zero_rates],
                color=colors[idx], linestyle=line_styles[0],
                linewidth=2, label=f'{name} Strippée')
        raw_data = (eonia_data if name == "EONIA"
                    else euribor3m_data if name == "EURIBOR3M"
        else euribor6m_data)
        raw_points = sorted([
            ((adjust_date(adjust_date(curve_date, inst[1]), inst[2]) - curve_date).days / 365.0,
             float(inst[3]))
            for inst in raw_data
        ], key=lambda x: x[0])
        raw_t, raw_z = zip(*raw_points)
        ax.plot(raw_t, raw_z, color=colors[idx], linestyle=line_styles[1],
                linewidth=1.5, alpha=0.7, label=f'{name} Données Brutes')
    ax.set_title('Comparaison Courbes Strippées vs Données Brutes (03/02/2025)', fontsize=14)
    ax.set_xlabel('Maturité (années)', fontsize=12)
    ax.set_ylabel('Taux (%)', fontsize=12)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(0, 35)
    ax.set_ylim(0, 3.5)
    fig.tight_layout()
    return fig


def fig_forward_3M():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(fwd3m_mono_times, fwd3m_mono, linestyle='--', color='blue', label='EURIBOR3M Mono')
    ax.plot(fwd3m_multi_times, fwd3m_multi, linestyle='-', color='blue', label='EURIBOR3M Multi')
    ax.set_xlabel("Fixing (années)", fontsize=12)
    ax.set_ylabel("Taux Forward 3M (%)", fontsize=12)
    ax.set_title("Taux Forward 3M : Mono-courbe vs Multi-courbe", fontsize=14)
    ax.legend()
    ax.grid(True)
    ax.set_xlim(0, 30)
    fig.tight_layout()
    return fig


def fig_forward_6M():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(fwd6m_mono_times, fwd6m_mono, linestyle='--', color='red', label='EURIBOR6M Mono')
    ax.plot(fwd6m_multi_times, fwd6m_multi, linestyle='-', color='red', label='EURIBOR6M Multi')
    ax.set_xlabel("Fixing (années)", fontsize=12)
    ax.set_ylabel("Taux Forward 6M (%)", fontsize=12)
    ax.set_title("Taux Forward 6M : Mono-courbe vs Multi-courbe", fontsize=14)
    ax.legend()
    ax.grid(True)
    ax.set_xlim(0, 30)
    fig.tight_layout()
    return fig


# === Navigation par onglets (en haut) ===
tabs = st.tabs([
    "Introduction",
    "Mono-courbe stripping",
    "Multi-courbe stripping",
    "Taux Forward 3M",
    "Taux Forward 6M"
])

with tabs[0]:
    st.title("Stripping de Courbe de Taux – Introduction")
    st.markdown(
        """
        **Contexte du Projet :**  
        Ce projet traite des données de trois courbes de taux :  
        - **EONIA**
        - **EURIBOR3M**
        - **EURIBOR6M**

        **Objectifs :**
        1. **Mono-courbe stripping :** Construire la courbe zéro-coupon pour chaque indice et vérifier la valorisation des instruments.
        2. **Multi-courbe stripping :** Utiliser la courbe EONIA comme courbe de discount pour EURIBOR3M et EURIBOR6M.
        3. **Taux Forward :** Calculer et comparer les taux forward (3M et 6M) sur un horizon de 30 ans avec des fixings hebdomadaires.

        Utilisez les onglets ci-dessus pour explorer chaque étape du projet.
        """
    )

with tabs[1]:
    st.title("Mono-courbe Stripping")
    st.markdown(
        """
        **Méthodologie :**  
        La courbe zéro-coupon est construite pour chaque indice (EONIA, EURIBOR3M, EURIBOR6M) à partir des instruments de marché correspondants.

        **Ce que montre la figure :**
        - La courbe strippée (calculée) est affichée en continu.
        - Les points bruts des taux observés sont également tracés.
        - La comparaison permet de vérifier la cohérence entre la valorisation des instruments et la courbe obtenue.
        """
    )
    st.pyplot(fig_stripped_vs_raw())

with tabs[2]:
    st.title("Multi-courbe Stripping (Discount EONIA)")
    st.markdown(
        """
        **Méthodologie :**  
        Pour EURIBOR3M et EURIBOR6M, le stripping est réalisé en utilisant la courbe EONIA comme courbe de discount.

        **Ce que montre la figure :**
        - Les courbes zéro-coupon obtenues par l'approche mono-courbe (en pointillés) sont comparées aux courbes multi-courbe (en continu).
        - Cette comparaison met en lumière l'effet du discounting externe sur la structure des taux.
        """
    )
    st.pyplot(fig_comparison())

with tabs[3]:
    st.title("Taux Forward 3M")
    st.markdown(
        """
        **Calcul des taux Forward 3M :**  
        Les taux forward sur une période de 3 mois sont calculés à partir des courbes EURIBOR3M (mono‑courbe et multi‑courbe)
        avec des fixings hebdomadaires de 0 à 30 ans.

        **Ce que montre la figure :**
        - La ligne en pointillés représente l'approche mono-courbe.
        - La ligne continue représente l'approche multi-courbe.
        - La convergence sur le court terme et les divergences sur le long terme sont clairement visibles.
        """
    )
    st.pyplot(fig_forward_3M())

with tabs[4]:
    st.title("Taux Forward 6M")
    st.markdown(
        """
        **Calcul des taux Forward 6M :**  
        Les taux forward sur une période de 6 mois sont calculés à partir des courbes EURIBOR6M (mono‑courbe et multi‑courbe)
        avec des fixings hebdomadaires de 0 à 30 ans.

        **Ce que montre la figure :**
        - La ligne en pointillés correspond aux taux forward issus de l'approche mono‑courbe.
        - La ligne continue correspond à l'approche multi‑courbe.
        - Des divergences plus marquées apparaissent sur le long terme, soulignant l'impact du discounting externe.
        """
    )
    st.pyplot(fig_forward_6M())
