"""Exemple : afficher les signaux du jour de tous les modèles.  python exemple.py"""
from ia_signaux import signal_v4, signal_v5, signaux_crypto_v2, signal_matiere

print("=== Crypto v4 (top 3 + filtre Bitcoin) ===")
s = signal_v4()
print("Données du", s["date_des_donnees"], "| filtre Bitcoin positif :", s["filtre_bitcoin_positif"])
print("Portefeuille conseillé :", s["portefeuille"])

print("\n=== Crypto v5 (répartition IA) ===")
print(signal_v5()["portefeuille"])

for version in ["achat_seul", "achat_vente"]:
    print(f"\n=== Crypto v2 — {version} ===")
    for x in signaux_crypto_v2(version):
        print(f"{x['symbole']:9} {x['signal']:8} votes {x['votes']}  (décision {x['bougie_de_decision']})")

try:                                      # nécessite DATABENTO_API_KEY
    print("\n=== Matières premières ===")
    for m in ["SI", "LE", "HG"]:
        x = signal_matiere(m)
        print(f"{x['nom']:7} {x['signal']:8} votes {x['votes']}")
except RuntimeError as e:
    print("Matières premières ignorées :", e)
