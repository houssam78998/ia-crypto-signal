"""Matières premières (CME) : robots v11 sur l'argent (SI), les bovins (LE) et le cuivre (HG). ACHETER ou DEHORS.

Les données CME ne sont pas gratuites : il faut une clé Databento (https://databento.com), à mettre dans la
variable d'environnement DATABENTO_API_KEY. Environ 100 jours de bougies horaires coûtent quelques centimes.
"""
import os
import pandas as pd
from .commun import MODELES, bougies_4h, observations, charger_robots, voter, NOMS_SIGNAL, STRIDE

MARCHES = {"SI": "Argent", "LE": "Bovins", "HG": "Cuivre"}
DEBUT_TEST = pd.Timestamp("2026-10-20 20:00")     # début du test officiel (sert à rejouer la position)
_CACHE = {}


def bougies_databento(marche, debut, fin=None, cle=None, cout_max=2.0):
    """Bougies horaires du contrat le plus échangé (colonnes ts, instrument_id, open, high, low, close, volume)."""
    import databento as db
    cle = cle or os.environ.get("DATABENTO_API_KEY")
    if not cle:
        raise RuntimeError("Clé Databento manquante : variable d'environnement DATABENTO_API_KEY.")
    client = db.Historical(cle)
    dispo = pd.Timestamp(client.metadata.get_dataset_range(dataset="GLBX.MDP3")["end"]).tz_convert(None).floor("h")
    fin = min(pd.Timestamp(fin), dispo) if fin else dispo
    kw = dict(dataset="GLBX.MDP3", schema="ohlcv-1h", stype_in="continuous", symbols=[f"{marche}.v.0"],
              start=pd.Timestamp(debut).isoformat(), end=fin.isoformat())
    cout = float(client.metadata.get_cost(**kw))
    if cout > cout_max:
        raise RuntimeError(f"Coût Databento estimé {cout:.2f} $ > limite {cout_max} $ (paramètre cout_max).")
    df = client.timeseries.get_range(**kw).to_df().reset_index().rename(columns={"ts_event": "ts"})
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df[["ts", "instrument_id", "open", "high", "low", "close", "volume"]].dropna()


def signal_matiere(marche, position=None, bougies_horaires=None, rejouer_depuis=DEBUT_TEST, cle_databento=None):
    """Signal actuel des 3 robots d'un marché ("SI", "LE" ou "HG").
    Les robots votent toutes les 6 bougies de 4 h (environ une fois par jour de bourse).
    position : 1 = acheté, 0 = dehors. Si None, elle est retrouvée en rejouant les votes depuis `rejouer_depuis`."""
    if marche not in MARCHES:
        raise ValueError("marche doit être SI, LE ou HG")
    debut = pd.Timestamp(rejouer_depuis) if position is None else pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=5)
    h = bougies_horaires if bougies_horaires is not None else bougies_databento(marche, debut - pd.Timedelta(days=120), cle=cle_databento)
    info, OBS = observations(bougies_4h(h, instrument=True), MODELES / "matieres" / marche)
    if marche not in _CACHE:
        _CACHE[marche] = charger_robots(MODELES / "matieres" / marche, "robot_{}.zip")
    robots, t = _CACHE[marche], info["t"]
    i0 = int(t.searchsorted(debut))
    if i0 >= len(t):
        raise RuntimeError("Pas encore de bougie de 4 h après la date de départ.")
    if position is None:
        pos, last, votes, avant = 0, None, None, 0
        for i in range(i0, len(t)):
            if last is None or i - last >= STRIDE:
                avant = pos
                pos, votes = voter(robots, OBS[i], pos, True); last = i
        i = last
    else:
        i = len(t) - 1; avant = int(position)
        pos, votes = voter(robots, OBS[i], avant, True)
    ts = t.iloc[i]
    return {"marche": marche, "nom": MARCHES[marche], "signal": NOMS_SIGNAL[pos], "position": pos, "votes": votes,
            "position_avant": avant, "bougie_de_decision": str(ts), "prix": float(info["prix"].iloc[i]),
            "derniere_bougie": str(t.iloc[-1])}
