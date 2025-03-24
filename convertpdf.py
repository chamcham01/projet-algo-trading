import json
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from fpdf import FPDF

# Charger les données JSON
with open("2eme million.json") as f:
    data = json.load(f)

# Extraire et convertir les dates et capital
dates = []
equities = []

for key in sorted(data["rollingWindow"].keys()):
    try:
        date_str = key.split("_")[1]  # ex: '20000131'
        date_obj = datetime.datetime.strptime(date_str, "%Y%m%d")
        equity = float(data["rollingWindow"][key]["portfolioStatistics"]["endEquity"])
        dates.append(date_obj)
        equities.append(equity)
    except Exception:
        continue  # Ignore les erreurs s'il manque des données

# Tracer l'évolution du capital
plt.figure(figsize=(12, 6))
plt.plot(dates, equities, marker='o', linestyle='-')
plt.title("Évolution du capital (End Equity)")
plt.xlabel("Date")
plt.ylabel("Capital (€)")
plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # un tick par an
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("performance.png")
plt.close()

# Créer un PDF avec le graphique
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="Rapport de Backtest QuantConnect", ln=True, align="C")
pdf.image("performance.png", x=10, y=30, w=190)
pdf.output("rapport_backtest.pdf")

print("✅ Rapport PDF généré avec succès : rapport_backtest.pdf")
