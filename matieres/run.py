"""Moteur du test en argent fictif « matières premières » (robots v11, renforcement + autoencodeur).

Chaque soir de semaine (GitHub Actions) :
  1. télécharge les nouvelles bougies horaires CME (Databento) de l'argent, des bovins et du cuivre ;
  2. reconstruit exactement les mêmes indicateurs que la v11 (bougies de 4 h, autoencodeur, contexte) ;
  3. fait voter les 3 robots de chaque marché toutes les 6 bougies de 4 h (le rythme de leur entraînement) ;
  4. met à jour un portefeuille fictif de 1 000 $ par marché, comparé à « acheter et garder » ;
  5. écrit docs/data/matieres.json pour l'application.
Aucun ordre réel n'est passé. Le moteur crypto (engine/run.py) n'est pas concerné.
"""
import os, json, math
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch, torch.nn as nn

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DATA_DIR, MODEL_DIR = ROOT / "donnees", ROOT / "modeles"
STATE_PATH = ROOT / "etat.json"
OUT_PATH = REPO / "docs" / "data" / "matieres.json"
BASE_PATH, RECENT_PATH = DATA_DIR / "horaire_base.csv.gz", DATA_DIR / "horaire_recent.csv"

MARKETS = {
    "SI": {"nom": "Argent", "unite": "$ / once",
           "appris": "Il achète surtout quand le marché est agité (60 à 70 % du temps) et moins quand il est calme (34 à 53 %).",
           "backtest": {"robot": 25.5, "garder": 9.6}},
    "LE": {"nom": "Bovins", "unite": "cents / livre",
           "appris": "Il achète les baisses : 86 à 100 % du temps quand la tendance du mois baisse, seulement 17 à 32 % quand elle monte.",
           "backtest": {"robot": 7.7, "garder": 8.2}},
    "HG": {"nom": "Cuivre", "unite": "$ / livre",
           "appris": "Il achète les baisses : 65 à 71 % du temps quand la tendance du mois baisse, presque jamais (6 à 12 %) quand elle monte.",
           "backtest": {"robot": 6.7, "garder": 7.8}},
}
PAPER_START = pd.Timestamp(os.environ.get("MATIERES_PAPER_START", "2026-10-20 20:00"))   # UTC
PAPER_END = pd.Timestamp(os.environ.get("MATIERES_PAPER_END", "2027-04-20 23:59"))
CAPITAL = 1000.0
COST = 0.00025                      # par unité de position changée (0,05 % aller-retour)
TRAIN_END = "2017-01-01"            # période d'apprentissage de la v11 (sert à normaliser l'autoencodeur)
WINDOW, LATENT, STRIDE = 60, 8, 6
DATA_START = "2010-06-06"
MAX_COST_USD = float(os.environ.get("MATIERES_MAX_COST", "8"))


# ---------------------------------------------------------------- données
def fetch(start, end):
    import databento as db
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise RuntimeError("Secret DATABENTO_API_KEY absent : ajoute-le dans Settings → Secrets → Actions.")
    client = db.Historical(key)
    avail = pd.Timestamp(client.metadata.get_dataset_range(dataset="GLBX.MDP3")["end"]).tz_convert(None)
    end = min(pd.Timestamp(end), avail.floor("h"))
    if pd.Timestamp(start) >= end:
        return pd.DataFrame(columns=["ts", "root", "instrument_id", "open", "high", "low", "close", "volume"])
    kw = dict(dataset="GLBX.MDP3", schema="ohlcv-1h", stype_in="continuous",
              symbols=[f"{r}.v.0" for r in MARKETS], start=pd.Timestamp(start).isoformat(), end=end.isoformat())
    cost = float(client.metadata.get_cost(**kw))
    print(f"Databento : {kw['start']} → {kw['end']}, coût estimé {cost:.4f} $")
    if cost > MAX_COST_USD:
        raise RuntimeError(f"Coût {cost:.2f} $ supérieur à la limite {MAX_COST_USD} $ : arrêt par sécurité.")
    df = client.timeseries.get_range(**kw).to_df().reset_index().rename(columns={"ts_event": "ts"})
    if df.empty:
        return pd.DataFrame(columns=["ts", "root", "instrument_id", "open", "high", "low", "close", "volume"])
    df["root"] = df["symbol"].astype(str).str.split(".").str[0]
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
    return df[["ts", "root", "instrument_id", "open", "high", "low", "close", "volume"]]


def load_hourly(now):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not BASE_PATH.exists():
        base = fetch(DATA_START, PAPER_START.floor("D"))
        base.to_csv(BASE_PATH, index=False, compression="gzip")
    base = pd.read_csv(BASE_PATH, parse_dates=["ts"])
    recent = pd.read_csv(RECENT_PATH, parse_dates=["ts"]) if RECENT_PATH.exists() else base.iloc[0:0]
    allh = pd.concat([base, recent], ignore_index=True)
    last = allh["ts"].max()
    new = fetch(last - pd.Timedelta(days=2), now)             # 2 jours de recouvrement en cas de correction
    if len(new):
        recent = pd.concat([recent, new], ignore_index=True)
        recent = recent[recent["ts"] >= base["ts"].max() + pd.Timedelta(hours=1)]
        recent = recent.drop_duplicates(["ts", "root"], keep="last").sort_values(["root", "ts"])
        recent.to_csv(RECENT_PATH, index=False)
        allh = pd.concat([base, recent], ignore_index=True)
    allh["ts"] = pd.to_datetime(allh["ts"]).astype("datetime64[ns]")
    for c in ["open", "high", "low", "close", "volume"]:
        allh[c] = pd.to_numeric(allh[c], errors="coerce").astype("float64")     # évite les colonnes « object » après fusion
    allh = allh.dropna(subset=["open", "high", "low", "close"])
    return allh.drop_duplicates(["ts", "root"], keep="last")


# ---------------------------------------------------------------- mêmes calculs que la v11
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv1d(3, 32, 5, padding=2), nn.GELU(), nn.Conv1d(32, 32, 5, stride=2, padding=2), nn.GELU(),
                                 nn.Flatten(), nn.Linear(32 * (WINDOW // 2), 64), nn.GELU(), nn.Linear(64, LATENT))
        self.dec = nn.Sequential(nn.Linear(LATENT, 64), nn.GELU(), nn.Linear(64, WINDOW * 3))

    def forward(self, x):
        z = self.enc(x.transpose(1, 2)); return z, self.dec(z).view(-1, WINDOW, 3)


def make_bars(hourly, market):
    h = hourly[hourly["root"] == market].copy()
    h = h[(h["close"] > 0) & (h["open"] > 0)].sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    same = h["instrument_id"].eq(h["instrument_id"].shift())
    h["r"] = np.where(same, np.log(h["close"] / h["close"].shift()), np.log(h["close"] / h["open"]))
    h.loc[0, "r"] = 0.0; h["r"] = h["r"].clip(-0.3, 0.3); h["rng"] = np.log(h["high"] / h["low"])
    g = h.set_index("ts").groupby(pd.Grouper(freq="4h"))
    b = pd.DataFrame({"r": g["r"].sum(), "rng": g["rng"].max(), "vol": g["volume"].sum(), "n": g["r"].count(),
                      "prix": g["close"].last()})
    b = b[b["n"] > 0].copy()
    b = b[b.index + pd.Timedelta(hours=4) <= h["ts"].max() + pd.Timedelta(hours=1)]     # seulement les bougies terminées
    b["vol_30"] = b["r"].rolling(180, min_periods=60).std()
    return b.dropna()


def features(bars, market):
    v = bars["vol_30"].values; lv = np.log1p(bars["vol"].values)
    lvz = (lv - pd.Series(lv).rolling(180, min_periods=60).mean().values) / pd.Series(lv).rolling(180, min_periods=60).std().values
    F = np.stack([np.clip(bars["r"].values / v, -8, 8), np.clip(bars["rng"].values / v, 0, 20) / 4,
                  np.nan_to_num(np.clip(lvz, -5, 5))], axis=1).astype(np.float32)
    idx = np.arange(WINDOW, len(bars))
    WIN = np.stack([F[i - WINDOW + 1:i + 1] for i in idx])
    t_idx = bars.index[idx]
    ae = AE(); ae.load_state_dict(torch.load(MODEL_DIR / market / "autoencodeur.pt", map_location="cpu")); ae.eval()
    with torch.no_grad():
        Z = np.concatenate([ae(torch.tensor(WIN[i:i + 4096]))[0].numpy() for i in range(0, len(WIN), 4096)])
    tm = t_idx < TRAIN_END
    tm_v11 = tm.copy(); tm_v11[-1] = False                   # la v11 excluait la toute dernière bougie
    Z = ((Z - Z[tm_v11].mean(0)) / (Z[tm_v11].std(0) + 1e-6)).clip(-5, 5)
    r = bars["r"].values; c = np.concatenate([[0.0], np.cumsum(r)])

    def cum(n):
        out = np.full(len(r), np.nan); out[n - 1:] = c[n:] - c[:-n]; return out
    vol_ratio = np.log(pd.Series(r).rolling(30, min_periods=10).std().values / v)
    extra = np.stack([np.clip(r / v, -8, 8), np.clip(cum(6) / (v * np.sqrt(6)), -6, 6), np.clip(cum(30) / (v * np.sqrt(30)), -6, 6),
                      np.clip(cum(130) / (v * np.sqrt(130)), -6, 6), np.clip(vol_ratio, -3, 3),
                      bars.index.dayofweek / 4.0], axis=1)
    OBS = np.concatenate([Z, np.nan_to_num(extra[idx])], axis=1).astype(np.float32)
    return pd.DataFrame({"t": t_idx, "r": r[idx], "prix": bars["prix"].values[idx]}), OBS


# ---------------------------------------------------------------- portefeuille fictif
def fresh_state():
    return {m: {"dernier_traite": None, "derniere_decision": None, "position": 0, "robot": 1.0, "garder": 1.0,
                "frais": 0.0, "changements": 0, "decisions": [], "courbe": [], "votes": None,
                "prix_reference": None, "signal_depuis": None} for m in MARKETS}


def update_market(m, S, info, OBS, agents):
    t = info["t"]
    start_i = int(np.searchsorted(t.values, np.datetime64(PAPER_START), side="left"))
    if start_i >= len(t):
        return False
    i0 = start_i if S["dernier_traite"] is None else int(np.searchsorted(t.values, np.datetime64(pd.Timestamp(S["dernier_traite"])), side="right"))
    last_dec = None if S["derniere_decision"] is None else int(np.searchsorted(t.values, np.datetime64(pd.Timestamp(S["derniere_decision"]))))
    for i in range(i0, len(t)):
        if t[i] > PAPER_END:
            break
        if S["dernier_traite"] is not None or i > start_i:
            # gain ou perte de la bougie i avec la position décidée avant
            S["robot"] *= math.exp(S["position"] * info["r"].iloc[i]); S["garder"] *= math.exp(info["r"].iloc[i])
        if last_dec is None or i - last_dec >= STRIDE:
            obs = np.append(OBS[i], float(S["position"])).astype(np.float32)
            votes = [int(a.predict(obs, deterministic=True)[0]) for a in agents]
            new = int(np.mean(votes) > 0.5)
            if last_dec is None:
                S["garder"] *= (1 - COST)                       # « garder » achète une fois au départ
            if new != S["position"] or last_dec is None:
                if new != S["position"]:
                    S["robot"] *= (1 - COST * abs(new - S["position"])); S["frais"] += CAPITAL * S["robot"] * COST
                    S["changements"] += 1
                S["signal_depuis"] = str(t[i])
                S["prix_reference"] = float(info["prix"].iloc[i])
                S["decisions"].append({"date": str(t[i]), "fin": str(t[i] + pd.Timedelta(hours=4)), "initial": last_dec is None,
                                       "signal": "ACHETER" if new else "DEHORS",
                                       "prix": float(info["prix"].iloc[i]), "votes": votes, "change": new != S["position"] or last_dec is None})
            S["position"] = new; S["votes"] = votes; S["derniere_decision"] = str(t[i]); last_dec = i
        S["dernier_traite"] = str(t[i])
    return True


def sharpe(vals):
    v = np.array(vals, dtype=float)
    if len(v) < 10:
        return None
    r = np.diff(np.log(v)); return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0


def max_dd(vals):
    v = np.array(vals, dtype=float)
    return float((v / np.maximum.accumulate(v) - 1).min() * 100) if len(v) else 0.0


def main():
    from stable_baselines3 import PPO
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    hourly = load_hourly(now)
    state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else fresh_state()
    today = now.strftime("%Y-%m-%d")
    out_markets = []
    for m, meta in MARKETS.items():
        bars = make_bars(hourly, m)
        info, OBS = features(bars, m)
        agents = [PPO.load(MODEL_DIR / m / f"robot_{s}.zip", device="cpu") for s in range(3)]
        S = state[m]
        started = update_market(m, S, info, OBS, agents)
        # aperçu (non enregistré) si le test n'a pas commencé
        obs_now = np.append(OBS[-1], float(S["position"])).astype(np.float32)
        votes_now = [int(a.predict(obs_now, deterministic=True)[0]) for a in agents]
        if started:
            if not S["courbe"] or S["courbe"][-1]["d"] != today:
                S["courbe"].append({"d": today, "robot": round(CAPITAL * S["robot"], 2), "garder": round(CAPITAL * S["garder"], 2)})
            else:
                S["courbe"][-1].update({"robot": round(CAPITAL * S["robot"], 2), "garder": round(CAPITAL * S["garder"], 2)})
        bars_per_day = len(info[info["t"] >= info["t"].iloc[-1] - pd.Timedelta(days=28)]) / 20.0
        last_dec_t = pd.Timestamp(S["derniere_decision"]) if S["derniere_decision"] else None
        since = 0 if last_dec_t is None else int((info["t"] > last_dec_t).sum())
        days_next = max(0.0, (STRIDE - since) / max(bars_per_day, 0.5))
        rv = [p["robot"] for p in S["courbe"]]; gv = [p["garder"] for p in S["courbe"]]
        last = S["decisions"][-1] if S["decisions"] else None
        out_markets.append({
            "code": m, "nom": meta["nom"], "unite": meta["unite"], "appris": meta["appris"], "backtest": meta["backtest"],
            "commence": started,
            "signal": ("ACHETER" if S["position"] else "DEHORS") if started else None,
            "apercu": None if started else {"signal": "ACHETER" if np.mean(votes_now) > 0.5 else "DEHORS", "votes": votes_now},
            "votes": S["votes"], "signal_depuis": S["signal_depuis"],
            "signal_depuis_fin": str(pd.Timestamp(S["signal_depuis"]) + pd.Timedelta(hours=4)) if S["signal_depuis"] else None, "prix_reference": S["prix_reference"],
            "dernier_changement": last, "prix_actuel": float(info["prix"].iloc[-1]), "bougie": str(info["t"].iloc[-1] + pd.Timedelta(hours=4)),
            "prochaine_decision_jours": round(days_next, 1),
            "valeur": round(CAPITAL * S["robot"], 2), "rendement_pct": round((S["robot"] - 1) * 100, 2),
            "garder_valeur": round(CAPITAL * S["garder"], 2), "garder_rendement_pct": round((S["garder"] - 1) * 100, 2),
            "frais": round(S["frais"], 2), "changements": S["changements"],
            "pire_baisse_pct": round(max_dd(rv), 2), "garder_pire_baisse_pct": round(max_dd(gv), 2),
            "sharpe": sharpe(rv), "garder_sharpe": sharpe(gv),
            "courbe": S["courbe"], "decisions": S["decisions"][-60:],
        })
    STATE_PATH.write_text(json.dumps(state, indent=1))

    started = [x for x in out_markets if x["commence"]]
    total = {"valeur": round(sum(x["valeur"] for x in started), 2), "garder_valeur": round(sum(x["garder_valeur"] for x in started), 2),
             "capital": CAPITAL * len(MARKETS)}
    finished = now >= PAPER_END
    crit, verdict = [], {"statut": "en cours", "criteres": [], "texte": ""}
    if started:
        beat = sum(1 for x in started if x["valeur"] > x["garder_valeur"])
        tr_r = [np.sum([x["courbe"][i]["robot"] for x in started]) for i in range(min(len(x["courbe"]) for x in started))]
        tr_g = [np.sum([x["courbe"][i]["garder"] for x in started]) for i in range(min(len(x["courbe"]) for x in started))]
        s_r, s_g = sharpe(tr_r), sharpe(tr_g)
        crit = [{"nom": "Le portefeuille des robots rapporte plus que « garder »", "ok": total["valeur"] > total["garder_valeur"]},
                {"nom": "Meilleur rapport gain / risque (Sharpe) que « garder »", "ok": (s_r or 0) > (s_g or 0)},
                {"nom": f"Au moins 2 marchés sur 3 battent leur « garder » ({beat}/3)", "ok": beat >= 2}]
        n_ok = sum(c["ok"] for c in crit)
        if finished:
            st = "validé" if n_ok == 3 else ("à prolonger" if n_ok == 2 else "non validé")
            verdict = {"statut": st, "criteres": crit, "texte": {"validé": "Les robots ont battu « garder » sur les 3 critères pendant 6 mois.",
                       "à prolonger": "Résultat serré : 2 critères sur 3. Il faut prolonger le test avant de conclure.",
                       "non validé": "Les robots n'ont pas battu « garder ». Ils ne doivent pas être utilisés avec de l'argent réel."}[st]}
        else:
            verdict = {"statut": "en cours", "criteres": crit, "texte": "Verdict définitif le " + PAPER_END.strftime("%d/%m/%Y") + ". Les critères ci-dessus changent chaque jour."}
    days_total = (PAPER_END - PAPER_START).days
    out = {"genere_le": now.strftime("%Y-%m-%d %H:%M UTC"), "donnees_jusqu_au": max(x["bougie"] for x in out_markets),
           "test": {"debut": PAPER_START.strftime("%Y-%m-%d"), "fin": PAPER_END.strftime("%Y-%m-%d"), "capital": CAPITAL,
                    "jours_ecoules": max(0, min(days_total, (now - PAPER_START).days)), "jours_total": days_total,
                    "frais_aller_retour_pct": COST * 200, "termine": bool(finished)},
           "marches": out_markets, "total": total, "verdict": verdict}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    for x in out_markets:
        print(f"{x['nom']}: {x['signal'] or ('aperçu ' + x['apercu']['signal'])} | robot {x['valeur']} $ | garder {x['garder_valeur']} $")


if __name__ == "__main__":
    main()
