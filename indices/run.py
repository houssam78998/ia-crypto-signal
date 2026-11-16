"""Moteur du test en argent fictif « Indices » : robots des notebooks C2 (équipes de 3) et C3 (équipes de 10).

Trois cas : Nasdaq achat seulement, Nasdaq achat et vente, S&P 500 achat seulement.
Pour chaque cas, deux équipes de robots votent chaque soir de bourse après la clôture américaine :
  - « c2 » : les 3 robots du notebook C2 ; il faut au moins 2 robots d'accord, sinon DEHORS ;
  - « c3 » : les 10 robots du notebook C3 ; il faut plus de la moitié d'accord, sinon DEHORS.
Chaque portefeuille fictif part de 1 000 $ et est comparé à « acheter et garder » le même indice.
Aucun entraînement ici : les robots ont été entraînés dans Colab. Aucun ordre réel.

⚠️ Le notebook C3 a classé ces cas « chance » (achat seulement) ou « fragile » (Nasdaq achat et vente) :
ce test est une observation sur de nouvelles données, pas un modèle validé.
"""
import os, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import torch, torch.nn as nn

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DATA_DIR, MODEL_DIR = ROOT / "donnees", ROOT / "modeles"
STATE_PATH = ROOT / "etat.json"
OUT_PATH = REPO / "docs" / "data" / "indices.json"

CAS = [("QQQ_achat_seul", "QQQ", "Nasdaq", "achat_seul", False),
       ("QQQ_achat_vente", "QQQ", "Nasdaq", "achat_vente", True),
       ("SPY_achat_seul", "SPY", "S&P 500", "achat_seul", False)]
NOMS_CAS = {"QQQ_achat_seul": "Nasdaq — achat seulement", "QQQ_achat_vente": "Nasdaq — achat et vente", "SPY_achat_seul": "S&P 500 — achat seulement"}
EQUIPES = {"c2": {"nom": "Équipe de 3 robots (notebook C2)", "graines": [0, 1, 2]},
           "c3": {"nom": "Équipe de 10 robots (notebook C3)", "graines": list(range(10))}}
# Résultats des notebooks sur 2019 → 13 nov. 2026 : Sharpe du vote, Sharpe de « garder », verdict
NOTEBOOK = {("QQQ_achat_seul", "c2"): (0.863, 0.810, "à approfondir"), ("QQQ_achat_vente", "c2"): (1.196, 0.810, "à approfondir"),
            ("SPY_achat_seul", "c2"): (0.971, 0.805, "validé"),
            ("QQQ_achat_seul", "c3"): (0.727, 0.810, "chance"), ("QQQ_achat_vente", "c3"): (1.099, 0.810, "fragile"),
            ("SPY_achat_seul", "c3"): (0.596, 0.805, "chance")}
PAPER_START = pd.Timestamp(os.environ.get("IND_PAPER_START", "2026-11-17"))      # première décision : clôture de ce jour
PAPER_END = pd.Timestamp(os.environ.get("IND_PAPER_END", "2027-05-17"))
CAPITAL, FRAIS = 1000.0, 0.0005
FIN_APPRENTISSAGE, FIN_VALIDATION = "2016-01-01", "2019-01-01"
FENETRE, LATENT = 60, 8
COUT_MAX = float(os.environ.get("IND_COUT_MAX", "0.50"))                         # plafond Databento par calcul, en dollars
OFFLINE = os.environ.get("IND_OFFLINE") == "1"
SIGNAL = {1: "ACHETER", 0: "DEHORS", -1: "VENDRE"}


# ---------------------------------------------------------------- données
def base(t):
    b = pd.read_csv(DATA_DIR / f"{t.lower()}_base.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    return b[["Open", "High", "Low", "Close", "Volume"]]


def _databento(t, debut):
    """Bougies journalières Databento. Renvoie (DataFrame, description de la source) ou lève une erreur."""
    import databento as db
    cle = os.environ.get("DATABENTO_API_KEY")
    if not cle:
        raise RuntimeError("secret DATABENTO_API_KEY absent")
    client = db.Historical(cle); erreurs = []
    for dataset in ["EQUS.MINI", "DBEQ.BASIC", "XNAS.ITCH" if t == "QQQ" else "ARCX.PILLAR"]:
        try:
            fin = pd.Timestamp(client.metadata.get_dataset_range(dataset=dataset)["end"]).tz_convert(None)
            kw = dict(dataset=dataset, schema="ohlcv-1d", symbols=[t], stype_in="raw_symbol", start=debut.isoformat(), end=fin.isoformat())
            cout = float(client.metadata.get_cost(**kw))
            print(f"  Databento {dataset} {t} : coût estimé {cout:.4f} $")
            if cout > COUT_MAX:
                erreurs.append(f"{dataset} : coût {cout:.2f} $ > plafond {COUT_MAX} $"); continue
            df = client.timeseries.get_range(**kw).to_df().reset_index()
            if not len(df):
                erreurs.append(f"{dataset} : aucune donnée"); continue
            df["Date"] = pd.to_datetime(df["ts_event"], utc=True).dt.tz_localize(None).dt.normalize()
            df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
            df = df.groupby("Date").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
            return df.astype(float), f"Databento {dataset} (coût {cout:.4f} $)"
        except Exception as e:
            erreurs.append(f"{dataset} : {str(e)[:120]}")
    raise RuntimeError(" | ".join(erreurs))


def _yahoo(t):
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{t}", params={"range": "1y", "interval": "1d"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]; q = res["indicators"]["quote"][0]
    df = pd.DataFrame({"Open": q["open"], "High": q["high"], "Low": q["low"], "Close": q["close"], "Volume": q["volume"]},
                      index=pd.to_datetime(res["timestamp"], unit="s").normalize()).dropna()
    df.index.name = "Date"
    return df.astype(float), "Yahoo Finance"


def charger(t):
    """Historique complet = fichier de base (prix du notebook) + jours récents téléchargés. Renvoie (prix, source)."""
    b = base(t); rp = DATA_DIR / f"recent_{t}.csv"
    rec = pd.read_csv(rp, parse_dates=["Date"]).set_index("Date") if rp.exists() else pd.DataFrame(columns=b.columns, index=pd.DatetimeIndex([], name="Date"))
    source = "fichiers enregistrés"
    if not OFFLINE:
        debut = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=200)
        nouveau, erreurs = None, []
        for nom, f in (("Databento", lambda: _databento(t, debut)), ("Yahoo", lambda: _yahoo(t))):
            for essai in range(3):
                try:
                    nouveau, source = f(); break
                except Exception as e:
                    err = f"{nom} : {str(e)[:300]}"
                    if essai == 2 or "absent" in str(e) or "coût" in str(e): erreurs.append(err); break
                    time.sleep(20 * (essai + 1))
            if nouveau is not None: break
        if nouveau is None:
            raise RuntimeError(f"Aucune source de prix pour {t} : " + " || ".join(erreurs))
        if erreurs: print("  Sources en échec :", " || ".join(erreurs))
        maintenant = pd.Timestamp.now(tz="UTC").tz_localize(None)
        nouveau = nouveau[(nouveau.index + pd.Timedelta(hours=22) <= maintenant) & (nouveau["Close"] > 0) & (nouveau["Low"] > 0)]
        commun = nouveau.index.intersection(b.index)
        if len(commun) < 20:
            raise RuntimeError(f"{t} : pas assez de jours en commun avec le fichier de base ({len(commun)}) pour vérifier la source {source}")
        ecart = float((nouveau.loc[commun, "Close"] / b.loc[commun, "Close"] - 1).abs().median())
        if ecart > 0.02:
            raise RuntimeError(f"{t} : les prix de {source} diffèrent de {ecart:.1%} du fichier de base : source refusée")
        ratio = float((b.loc[commun, "Volume"] / nouveau.loc[commun, "Volume"].replace(0, np.nan)).median())
        nouveau = nouveau.assign(Volume=nouveau["Volume"] * ratio)                # même échelle de volume que le fichier de base
        nouveau = nouveau[nouveau.index > b.index[-1]]
        rec = pd.concat([rec[~rec.index.isin(nouveau.index)], nouveau]).sort_index()
        rec.index.name = "Date"; rec.to_csv(rp, float_format="%.6f")
        print(f"  {t} : source {source}, écart de prix {ecart:.2%}, {len(rec)} jour(s) après le {b.index[-1].date()}")
    df = pd.concat([b, rec[rec.index > b.index[-1]]]).sort_index()
    return df[~df.index.duplicated(keep="last")].astype(float), source


# ---------------------------------------------------------------- mêmes calculs que les notebooks C2 / C3
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv1d(3, 32, 5, padding=2), nn.GELU(), nn.Conv1d(32, 32, 5, stride=2, padding=2), nn.GELU(),
                                 nn.Flatten(), nn.Linear(32 * (FENETRE // 2), 64), nn.GELU(), nn.Linear(64, LATENT))
        self.dec = nn.Sequential(nn.Linear(LATENT, 64), nn.GELU(), nn.Linear(64, FENETRE * 3))

    def forward(self, x):
        z = self.enc(x.transpose(1, 2)); return z, self.dec(z).view(-1, FENETRE, 3)


def preparer(t, df):
    """Renvoie dates, rendement de chaque jour, observations des robots, prix de clôture."""
    c = df["Close"]; r = np.log(c / c.shift()).fillna(0.0).values
    v = pd.Series(r).rolling(63, min_periods=30).std().values
    rng_ = np.log(df["High"] / df["Low"]).values; lv = np.log1p(df["Volume"].values)
    lvz = (lv - pd.Series(lv).rolling(63, min_periods=30).mean().values) / pd.Series(lv).rolling(63, min_periods=30).std().values
    F = np.stack([np.clip(r / v, -8, 8), np.clip(rng_ / v, 0, 20) / 4, np.nan_to_num(np.clip(lvz, -5, 5))], 1).astype(np.float32)
    debut = int(np.argmax(~np.isnan(v))) + FENETRE; idx = np.arange(debut, len(df)); dates = df.index[idx]
    W = np.nan_to_num(np.stack([F[i - FENETRE + 1:i + 1] for i in idx]))
    ae = AE(); ae.load_state_dict(torch.load(MODEL_DIR / t / "autoencodeur.pt", map_location="cpu")); ae.eval()
    with torch.no_grad(): Z = ae(torch.tensor(W))[0].numpy()
    npath = MODEL_DIR / t / "normalisation.json"
    if npath.exists():
        nm = json.loads(npath.read_text()); mu, sd = np.array(nm["moyenne"]), np.array(nm["ecart"])
    else:                                                         # calculée une fois, sur la période d'apprentissage seulement
        app = np.asarray(dates < FIN_APPRENTISSAGE); mu, sd = Z[app].mean(0), Z[app].std(0)
        npath.write_text(json.dumps({"moyenne": mu.tolist(), "ecart": sd.tolist(), "fenetres": int(app.sum())}))
    Z = ((Z - mu) / (sd + 1e-6)).clip(-5, 5)
    cs = np.concatenate([[0.0], np.cumsum(r)])

    def cum(n):
        o = np.full(len(r), np.nan); o[n - 1:] = cs[n:] - cs[:-n]; return o
    sma = (c / c.rolling(200).mean() - 1).values
    extra = np.stack([np.clip(r / v, -8, 8), np.clip(cum(5) / (v * np.sqrt(5)), -6, 6), np.clip(cum(21) / (v * np.sqrt(21)), -6, 6),
                      np.clip(cum(63) / (v * np.sqrt(63)), -6, 6), np.clip(np.log(pd.Series(r).rolling(10).std().values / v), -3, 3),
                      np.clip(sma * 5, -3, 3)], 1)
    OBS = np.concatenate([Z, np.nan_to_num(extra[idx])], 1).astype(np.float32)
    return dates, r[idx], OBS, c.values[idx]


def gain(pos, r):
    """Gain logarithmique d'un jour pour une position (+1 acheté, 0 dehors, −1 vendu) : une baisse de 10 % rapporte 10 % au vendeur."""
    return math.log(max(1e-9, 1 + pos * (math.exp(r) - 1)))


def majorite(votes):
    for c in (1, -1):
        if votes.count(c) * 2 > len(votes): return c
    return 0


def voter(robots, obs, pos, court):
    o = np.append(obs, float(pos)).astype(np.float32)
    votes = [int(m.predict(o, deterministic=True)[0]) - (1 if court else 0) for m in robots]
    return majorite(votes), votes


def verifier(robots, dates, R, OBS, court):
    """Rejoue le test des notebooks (2019 → dernier jour du fichier de base) : doit redonner leur Sharpe."""
    a = int(np.searchsorted(dates, np.datetime64(FIN_VALIDATION))); b = int(np.searchsorted(dates, np.datetime64(base_fin), side="right"))
    positions = [-1.0, 0.0, 1.0] if court else [0.0, 1.0]
    tabs = [{p: m.predict(np.column_stack([OBS[a:b - 1], np.full(b - 1 - a, p, dtype=np.float32)]).astype(np.float32), deterministic=True)[0] - (1 if court else 0)
             for p in positions} for m in robots]
    pos, rets = 0.0, []
    for k in range(b - 1 - a):
        n = float(majorite([int(T[pos][k]) for T in tabs]))
        rets.append(gain(n, R[a + k + 1]) - FRAIS * abs(n - pos)); pos = n
    rets = np.array(rets)
    return round(float(rets.mean() / rets.std() * np.sqrt(252)), 3) if rets.std() > 0 else 0.0


# ---------------------------------------------------------------- portefeuilles fictifs
def neuf():
    return {"dernier_traite": None, "commence": False, "position": 0, "robot": 1.0, "garder": 1.0, "frais": 0.0, "changements": 0,
            "votes": None, "signal_depuis": None, "prix_reference": None, "decisions": [], "courbe": []}


def avancer(S, robots, dates, R, OBS, prix, court):
    """Traite chaque jour de bourse terminé depuis le dernier calcul : gain du jour, puis vote à la clôture."""
    i0 = int(np.searchsorted(dates, np.datetime64(PAPER_START)))
    if S["dernier_traite"]:
        i0 = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(S["dernier_traite"])), side="right"))
    for i in range(i0, len(dates)):
        d = pd.Timestamp(dates[i])
        if d > PAPER_END: break
        if S["commence"]:
            S["robot"] *= math.exp(gain(S["position"], R[i])); S["garder"] *= math.exp(R[i])
        n, votes = voter(robots, OBS[i], S["position"], court)
        premier = not S["commence"]
        if premier: S["garder"] *= (1 - FRAIS)
        change = n != S["position"]
        if change:
            S["robot"] *= (1 - FRAIS * abs(n - S["position"])); S["frais"] += CAPITAL * S["robot"] * FRAIS * abs(n - S["position"]); S["changements"] += 1
        if change or premier:
            S["signal_depuis"], S["prix_reference"] = str(d.date()), float(prix[i])
        S["decisions"].append({"date": str(d.date()), "signal": SIGNAL[n], "ancien": SIGNAL[S["position"]], "prix": float(prix[i]),
                               "votes": votes, "change": bool(change or premier), "initial": premier})
        S["decisions"] = S["decisions"][-150:]
        S["position"], S["votes"], S["commence"], S["dernier_traite"] = n, votes, True, str(d.date())
        S["courbe"].append({"d": str(d.date()), "robot": round(CAPITAL * S["robot"], 2), "garder": round(CAPITAL * S["garder"], 2)})


def sharpe(v):
    v = np.array(v, dtype=float)
    if len(v) < 10: return None
    r = np.diff(np.log(v)); return round(float(r.mean() / r.std() * np.sqrt(252)), 2) if r.std() > 0 else 0.0


def pire_baisse(v):
    v = np.array(v, dtype=float)
    return round(float((v / np.maximum.accumulate(v) - 1).min() * 100), 2) if len(v) else 0.0


def main():
    global base_fin
    from stable_baselines3 import PPO
    maintenant = pd.Timestamp.now(tz="UTC").tz_localize(None)
    etat = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
    etat.setdefault("verification", {}); etat.setdefault("portefeuilles", {})
    prep, sources = {}, {}
    for t in sorted({c[1] for c in CAS}):
        df, sources[t] = charger(t); base_fin = base(t).index[-1]
        prep[t] = preparer(t, df) + (base_fin,)
        print(f"{t} : {len(df)} jours, dernière clôture {df.index[-1].date()} à {df['Close'].iloc[-1]:.2f}")
    sortie, classement = [], []
    for code, t, marche, version, court in CAS:
        dates, R, OBS, prix, base_fin = prep[t]
        tous = {g: PPO.load(MODEL_DIR / t / f"{version}_{g}.zip", device="cpu") for g in range(10)}
        equipes = []
        for eq, E in EQUIPES.items():
            robots = [tous[g] for g in E["graines"]]; cle = f"{code}:{eq}"
            if cle not in etat["verification"]:
                etat["verification"][cle] = verifier(robots, dates, R, OBS, court)
                print(f"  Vérification {cle} : Sharpe recalculé {etat['verification'][cle]:.3f} (notebook : {NOTEBOOK[(code, eq)][0]:.3f})")
            S = etat["portefeuilles"].setdefault(cle, neuf())
            avancer(S, robots, dates, R, OBS, prix, court)
            apercu = None
            if not S["commence"]:
                n, votes = voter(robots, OBS[-1], 0, court); apercu = {"signal": SIGNAL[n], "votes": votes}
            rv, gv = [p["robot"] for p in S["courbe"]], [p["garder"] for p in S["courbe"]]
            val, gval = (rv[-1], gv[-1]) if rv else (CAPITAL, CAPITAL)
            sr, sg = sharpe(rv), sharpe(gv)
            crit = [{"nom": "Les robots finissent au-dessus de « acheter et garder »", "ok": val > gval},
                    {"nom": "Meilleur rapport gain / risque (Sharpe) que « garder »", "ok": (sr or 0) > (sg or 0)},
                    {"nom": "Pire baisse plus petite que « garder »", "ok": pire_baisse(rv) > pire_baisse(gv)}]
            k = sum(c["ok"] for c in crit)
            if not S["commence"]:
                verdict = {"statut": "pas commencé", "criteres": [], "texte": "Première décision à la clôture du " + PAPER_START.strftime("%d/%m/%Y") + "."}
            elif maintenant >= PAPER_END + pd.Timedelta(days=1):
                st = "validé" if k == 3 else ("à prolonger" if k == 2 else "non validé")
                verdict = {"statut": st, "criteres": crit, "texte": {"validé": "Les robots ont battu « garder » sur les 3 critères pendant 6 mois.",
                           "à prolonger": "Résultat serré : 2 critères sur 3. Il faut prolonger le test avant de conclure.",
                           "non validé": "Les robots n'ont pas battu « garder »."}[st]}
            else:
                verdict = {"statut": "en cours", "criteres": crit, "texte": "Verdict définitif le " + PAPER_END.strftime("%d/%m/%Y") + ". Les critères changent chaque jour."}
            nb = NOTEBOOK[(code, eq)]
            equipes.append({"code": eq, "nom": E["nom"], "robots": len(robots), "commence": S["commence"], "apercu": apercu,
                            "signal": SIGNAL[S["position"]] if S["commence"] else None, "votes": S["votes"],
                            "signal_depuis": S["signal_depuis"], "prix_reference": S["prix_reference"], "derniere_decision": S["dernier_traite"],
                            "dernier_changement": next((d for d in reversed(S["decisions"]) if d["change"]), None),
                            "valeur": val, "rendement_pct": round((val / CAPITAL - 1) * 100, 2),
                            "garder_valeur": gval, "garder_rendement_pct": round((gval / CAPITAL - 1) * 100, 2),
                            "sharpe": sr, "garder_sharpe": sg, "pire_baisse_pct": pire_baisse(rv), "garder_pire_baisse_pct": pire_baisse(gv),
                            "frais": round(S["frais"], 2), "changements": S["changements"], "courbe": S["courbe"], "decisions": S["decisions"][-60:],
                            "notebook": {"sharpe_vote": nb[0], "sharpe_garder": nb[1], "verdict": nb[2], "sharpe_recalcule": etat["verification"][cle]},
                            "verdict": verdict})
            classement.append({"cas": code, "nom": NOMS_CAS[code], "equipe": eq, "robots": len(robots), "valeur": val, "rendement_pct": round((val / CAPITAL - 1) * 100, 2),
                               "garder_rendement_pct": round((gval / CAPITAL - 1) * 100, 2), "avance_pct": round((val / gval - 1) * 100, 2),
                               "signal": SIGNAL[S["position"]] if S["commence"] else None})
        sortie.append({"code": code, "ticker": t, "marche": marche, "version": version, "nom": NOMS_CAS[code], "achat_vente": court,
                       "prix": float(prix[-1]), "derniere_cloture": str(pd.Timestamp(dates[-1]).date()), "equipes": equipes})
    classement.sort(key=lambda x: -x["avance_pct"])
    STATE_PATH.write_text(json.dumps(etat, indent=1, default=float))
    jours_total = (PAPER_END - PAPER_START).days
    out = {"genere_le": maintenant.strftime("%Y-%m-%d %H:%M UTC"), "donnees_jusqu_au": max(c["derniere_cloture"] for c in sortie),
           "sources": sources,
           "test": {"debut": PAPER_START.strftime("%Y-%m-%d"), "fin": PAPER_END.strftime("%Y-%m-%d"), "capital": CAPITAL,
                    "jours_ecoules": max(0, min(jours_total, (maintenant - PAPER_START).days)), "jours_total": jours_total,
                    "frais_pct": FRAIS * 100, "termine": bool(maintenant >= PAPER_END + pd.Timedelta(days=1))},
           "cas": sortie, "classement": classement}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    for c in sortie:
        print(c["nom"], "|", " ; ".join(f"{e['code']} {e['signal'] or ('aperçu ' + e['apercu']['signal'])} {e['valeur']:.2f} $ (garder {e['garder_valeur']:.2f} $)" for e in c["equipes"]))


base_fin = None
if __name__ == "__main__":
    main()
