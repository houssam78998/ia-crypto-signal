"""Crypto v2 : robots v11 (renforcement PPO + autoencodeur), un vote par jour sur la bougie 00:00–04:00 UTC.

Deux versions :
  - "achat_seul"  : XRP, XLM, SOL, DOT, AVAX — ACHETER ou DEHORS ;
  - "achat_vente" : ETH, XRP, ADA, DOGE, SOL, LTC, DOT, AVAX — ACHETER, DEHORS ou VENDRE (pari à la baisse).
"""
import numpy as np
import pandas as pd
from .commun import MODELES, bougies_binance, bougies_4h, observations, charger_robots, voter, NOMS_SIGNAL

VERSIONS = {
    "achat_seul": ["XRPUSDT", "XLMUSDT", "SOLUSDT", "DOTUSDT", "AVAXUSDT"],
    "achat_vente": ["ETHUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "SOLUSDT", "LTCUSDT", "DOTUSDT", "AVAXUSDT"],
}
DEBUT_TEST = pd.Timestamp("2026-10-21 00:00")     # début du test officiel (sert à rejouer la position)
HEURE_DECISION = 0                                # bougie de 4 h qui commence à 00:00 UTC
_CACHE = {}


def _robots(symbole, version):
    k = (symbole, version)
    if k not in _CACHE:
        _CACHE[k] = charger_robots(MODELES / "crypto_v2" / symbole, version + "_{}.zip")
    return _CACHE[k]


def signal_crypto_v2(symbole, version="achat_seul", position=None, bougies_horaires=None, rejouer_depuis=DEBUT_TEST):
    """Signal du jour pour une crypto.

    symbole  : ex. "XRPUSDT" (doit faire partie de la version choisie).
    version  : "achat_seul" ou "achat_vente".
    position : ta position actuelle (1 = acheté, 0 = dehors, -1 = vendu à découvert). Les robots en tiennent compte.
               Si None, la position est retrouvée en rejouant les votes depuis `rejouer_depuis` (comme l'application).
    bougies_horaires : DataFrame (ts, open, high, low, close, volume) si tu as déjà les données ; sinon téléchargées sur Binance.
    """
    if version not in VERSIONS:
        raise ValueError("version doit être 'achat_seul' ou 'achat_vente'")
    if symbole not in VERSIONS[version]:
        raise ValueError(f"{symbole} n'a pas de robot validé dans la version {version} : {VERSIONS[version]}")
    debut = pd.Timestamp(rejouer_depuis) if position is None else pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=3)
    h = bougies_horaires if bougies_horaires is not None else bougies_binance(symbole, "1h", debut - pd.Timedelta(days=90))
    info, OBS = observations(bougies_4h(h), MODELES / "crypto_v2" / symbole)
    robots, achat_seul = _robots(symbole, version), version == "achat_seul"
    t = info["t"]
    decisions = [i for i in range(len(t)) if t.iloc[i].hour == HEURE_DECISION and t.iloc[i] >= debut]
    if not decisions:
        raise RuntimeError("Pas encore de bougie de décision (00:00–04:00 UTC) dans les données.")
    pos, votes, avant = (0 if position is None else int(position)), None, 0
    if position is None:
        for i in decisions:                       # rejoue les décisions jour après jour, comme le test officiel
            avant = pos
            pos, votes = voter(robots, OBS[i], pos, achat_seul)
    else:
        avant = pos
    i = decisions[-1]
    if position is not None:
        pos, votes = voter(robots, OBS[i], pos, achat_seul)
    ts = t.iloc[i]
    return {"symbole": symbole, "version": version, "signal": NOMS_SIGNAL[pos], "position": pos,
            "votes": votes, "position_avant": avant, "change": pos != avant,
            "bougie_de_decision": str(ts), "decision_valable_a_partir_de": str(ts + pd.Timedelta(hours=4)),
            "prix": float(info["prix"].iloc[i]),
            "prochaine_decision": str((ts + pd.Timedelta(days=1, hours=4)))}


def signaux_crypto_v2(version="achat_seul"):
    """Signal du jour pour toutes les cryptos d'une version, avec la part conseillée (parts égales)."""
    out = [signal_crypto_v2(s, version) for s in VERSIONS[version]]
    part = 100.0 / len(out)
    for o in out:
        o["part_pct"] = part if o["position"] != 0 else 0.0
    return out
