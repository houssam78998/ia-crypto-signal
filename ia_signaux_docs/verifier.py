"""Vérifie que ce paquet donne les mêmes signaux que l'application officielle (test en argent fictif).
    python verifier.py
Compare avec https://houssam78998.github.io/ia-crypto-signal/ . Un écart d'un jour peut venir de l'heure de mise à jour."""
import requests
from ia_signaux import signal_v4, signal_v5, signal_crypto_v2

SITE = "https://houssam78998.github.io/ia-crypto-signal/data/"
ok = True
app = requests.get(SITE + "app.json", timeout=30).json()
s = signal_v4()
cl_app = [c[0] for c in app["classement"]["scores"][:3]]
cl_moi = [c["symbole"].replace("USDT", "") for c in s["classement"][:3]]
print(f"v4  : classement application {cl_app} ({app['classement']['date']}) | paquet {cl_moi} ({s['date_des_donnees']})")
ok &= cl_app == cl_moi

v2 = requests.get(SITE + "crypto_v2.json", timeout=30).json()
for p in v2["portefeuilles"]:
    for m in p["marches"]:
        x = signal_crypto_v2(m["code"], p["code"])
        same = x["signal"] == m["signal"]
        ok &= same
        print(f"v2 {p['code']:11} {m['nom']:5}: application {m['signal']:8} | paquet {x['signal']:8} {'✓' if same else '✗'}")
print("\nTOUT EST IDENTIQUE ✓" if ok else "\nDifférence trouvée : vérifie que les deux utilisent les mêmes données (même jour).")
