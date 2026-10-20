"""Moteur du test en argent fictif « Crypto v2 » (robots v11 crypto : renforcement + autoencodeur).

Deux portefeuilles fictifs de 1 000 $ chacun, répartis à parts égales :
  - « Achat seulement »  : XRP, XLM, SOL, DOT, AVAX — chaque robot ACHÈTE ou reste DEHORS ;
  - « Achat et vente »   : ETH, XRP, ADA, DOGE, SOL, LTC, DOT, AVAX — chaque robot ACHÈTE, reste DEHORS
                           ou VEND À DÉCOUVERT (pari à la baisse, fictif).
Chaque jour (GitHub Actions, vers 04:30 UTC) :
  1. télécharge les bougies horaires Binance (data-api.binance.vision) ;
  2. reconstruit exactement les indicateurs de la v11 (bougies de 4 h, autoencodeur, contexte) ;
  3. fait voter les 3 robots de chaque crypto sur la bougie 00:00–04:00 UTC (même rythme que leur test) ;
  4. met à jour les portefeuilles fictifs, comparés à « acheter et garder » les mêmes cryptos ;
  5. écrit docs/data/crypto_v2.json pour l'application.
Aucun entraînement ici : les robots ont été entraînés dans Colab (notebook crypto v11). Aucun ordre réel.
"""
import os, io, json, math, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import torch, torch.nn as nn

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DATA_DIR, MODEL_DIR = ROOT / "donnees", ROOT / "modeles"
STATE_PATH = ROOT / "etat.json"
OUT_PATH = REPO / "docs" / "data" / "crypto_v2.json"

NAMES = {"XRPUSDT": "XRP", "XLMUSDT": "XLM", "SOLUSDT": "SOL", "DOTUSDT": "DOT", "AVAXUSDT": "AVAX",
         "ETHUSDT": "ETH", "ADAUSDT": "ADA", "DOGEUSDT": "DOGE", "LTCUSDT": "LTC"}
PORTFOLIOS = {
    "achat_seul": {"nom": "Achat seulement", "long_only": True,
                   "coins": ["XRPUSDT", "XLMUSDT", "SOLUSDT", "DOTUSDT", "AVAXUSDT"]},
    "achat_vente": {"nom": "Achat et vente", "long_only": False,
                    "coins": ["ETHUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "SOLUSDT", "LTCUSDT", "DOTUSDT", "AVAXUSDT"]},
}
# Résultats du test historique v11 (2024 → oct. 2026), rendement annuel % et Sharpe : robots (vote) contre « garder »
BACKTEST = {
    ("achat_seul", "XRPUSDT"): (60.3, 31.2, 0.82, 0.34), ("achat_seul", "XLMUSDT"): (66.2, 12.6, 0.71, 0.13),
    ("achat_seul", "SOLUSDT"): (9.5, 0.7, 0.15, 0.01), ("achat_seul", "DOTUSDT"): (-4.5, -50.9, -0.11, -0.83),
    ("achat_seul", "AVAXUSDT"): (6.6, -38.8, 0.12, -0.54),
    ("achat_vente", "ETHUSDT"): (15.0, 2.3, 0.21, 0.03), ("achat_vente", "XRPUSDT"): (86.5, 31.2, 0.80, 0.34),
    ("achat_vente", "ADAUSDT"): (46.1, -29.9, 0.41, -0.38), ("achat_vente", "DOGEUSDT"): (83.4, -4.0, 0.66, -0.04),
    ("achat_vente", "SOLUSDT"): (42.8, 0.7, 0.44, 0.01), ("achat_vente", "LTCUSDT"): (135.4, -7.0, 1.19, -0.10),
    ("achat_vente", "DOTUSDT"): (215.4, -50.9, 1.35, -0.83), ("achat_vente", "AVAXUSDT"): (125.3, -38.8, 0.90, -0.54),
}
# Ce que les robots ont appris (part du temps acheté / vendu selon la tendance du dernier mois, test 2024–2026)
APPRIS = {
    ("achat_seul", "XRPUSDT"): "Achète 78 % du temps quand la tendance monte, seulement 25 à 27 % sinon.",
    ("achat_seul", "XLMUSDT"): "Achète 71 % du temps quand la tendance monte, 20 % quand elle baisse.",
    ("achat_seul", "SOLUSDT"): "Achète presque toujours (98 %) quand la tendance monte, 14 % quand elle baisse.",
    ("achat_seul", "DOTUSDT"): "Très prudent : 45 % du temps acheté en tendance haussière, presque jamais en baisse (8 %).",
    ("achat_seul", "AVAXUSDT"): "Achète 66 % du temps en tendance haussière, presque jamais en baisse (5 %).",
    ("achat_vente", "ETHUSDT"): "Surtout acheteur : 100 % en hausse, 87 % sans tendance ; vend à moitié en baisse.",
    ("achat_vente", "XRPUSDT"): "Achète en hausse (76 %), vend 77 % du temps sinon.",
    ("achat_vente", "ADAUSDT"): "Suit la tendance : achète 94 % en hausse, vend 87 % en baisse.",
    ("achat_vente", "DOGEUSDT"): "Suit la tendance : achète 94 % en hausse, vend 89 % en baisse.",
    ("achat_vente", "SOLUSDT"): "Suit la tendance : achète 92 % en hausse, vend 91 % en baisse.",
    ("achat_vente", "LTCUSDT"): "Surtout vendeur : vend 97 % en baisse, 89 % sans tendance, et encore 43 % en hausse.",
    ("achat_vente", "DOTUSDT"): "Surtout vendeur : vend 93 % en baisse, 78 % sans tendance ; achète 62 % en hausse.",
    ("achat_vente", "AVAXUSDT"): "Surtout vendeur : vend 99 % en baisse, 89 % sans tendance ; achète 68 % en hausse.",
}
PAPER_START = pd.Timestamp(os.environ.get("CV2_PAPER_START", "2026-10-21 00:00"))   # bougie 00:00–04:00 UTC
PAPER_END = pd.Timestamp(os.environ.get("CV2_PAPER_END", "2027-04-21 04:00"))
CAPITAL = 1000.0
COST = 0.0015                       # 0,10 % de frais Binance + 0,05 % de glissement par unité de position changée
TRAIN_END, VAL_END = "2023-01-01", "2024-01-01"
WINDOW, LATENT, STRIDE = 60, 8, 6
DATA_START = "2018-01-01"
KEEP_DAYS = 400                     # historique horaire gardé dans le dépôt (largement assez pour les indicateurs)
DECISION_HOUR = 0                   # bougie de 4 h qui commence à 00:00 UTC, terminée à 04:00 UTC
OFFLINE = os.environ.get("CV2_OFFLINE") == "1"   # tests locaux sans réseau


# ---------------------------------------------------------------- données Binance
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "tb_base", "tb_quote", "ignore"]


def _to_ms(x):
    x = pd.to_numeric(x, errors="coerce")
    return np.where(x > 1e14, x // 1000, x)          # les archives Binance sont en microsecondes depuis 2025


def _clean(raw):
    raw = raw.copy()
    raw["ts"] = pd.to_datetime(pd.Series(_to_ms(raw["open_time"])).astype("int64").values, unit="ms")
    ct = pd.to_datetime(pd.Series(_to_ms(raw["close_time"])).astype("int64").values, unit="ms")
    raw = raw[ct < pd.Timestamp.now(tz="UTC").tz_localize(None)]          # seulement les heures terminées
    for c in ["open", "high", "low", "close", "volume"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    return raw[["ts", "open", "high", "low", "close", "volume"]].dropna()


def download_history(symbol):
    sess, frames = requests.Session(), []
    end = pd.Timestamp.now(tz="UTC").tz_localize(None)
    cur = end.to_period("M").to_timestamp()
    parts = [("monthly", m.strftime("%Y-%m")) for m in pd.date_range(DATA_START, cur - pd.Timedelta(days=1), freq="MS")]
    parts += [("daily", d.strftime("%Y-%m-%d")) for d in pd.date_range(cur, end.normalize() - pd.Timedelta(days=1), freq="D")]
    for kind, part in parts:
        url = f"https://data.binance.vision/data/spot/{kind}/klines/{symbol}/1h/{symbol}-1h-{part}.zip"
        for attempt in range(3):
            try:
                r = sess.get(url, timeout=60)
                if r.status_code in (403, 404):
                    break
                r.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    df = pd.read_csv(z.open(z.namelist()[0]), header=None)
                if not str(df.iloc[0, 0]).strip().isdigit():
                    df = df.iloc[1:]
                df = df.iloc[:, :12]; df.columns = COLS
                frames.append(df); break
            except Exception:
                time.sleep(2 * (attempt + 1))
    return _clean(pd.concat(frames, ignore_index=True))


def recent_api(symbol, limit=1000):
    for attempt in range(3):
        try:
            r = requests.get("https://data-api.binance.vision/api/v3/klines",
                             params={"symbol": symbol, "interval": "1h", "limit": limit}, timeout=30)
            r.raise_for_status()
            return _clean(pd.DataFrame(r.json(), columns=COLS))
        except Exception as e:
            print(f"  {symbol} : nouvel essai ({e})"); time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Impossible de joindre Binance pour {symbol}")


def load_hourly(symbol):
    """Renvoie (bougies horaires, premier_lancement). Au premier lancement, tout l'historique depuis 2018 est
    téléchargé (il sert à normaliser l'autoencodeur exactement comme dans Colab), puis seuls 400 jours sont gardés."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{symbol}_1h.csv.gz"
    first = not path.exists() or not (MODEL_DIR / symbol / "normalisation.json").exists()
    if OFFLINE:
        h = pd.read_csv(path, parse_dates=["ts"]); return h, first
    if first:
        print(f"{NAMES[symbol]} : premier lancement, téléchargement de l'historique depuis {DATA_START}…")
        h = download_history(symbol)
    else:
        h = pd.read_csv(path, parse_dates=["ts"])
    h = pd.concat([h, recent_api(symbol)], ignore_index=True)
    h["ts"] = pd.to_datetime(h["ts"], format="mixed").dt.floor("h").astype("datetime64[ns]")
    h = h.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)
    keep = h[h["ts"] >= h["ts"].max() - pd.Timedelta(days=KEEP_DAYS)]
    keep.to_csv(path, index=False, compression="gzip", float_format="%.8g")
    return h, first


# ---------------------------------------------------------------- mêmes calculs que la v11
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv1d(3, 32, 5, padding=2), nn.GELU(), nn.Conv1d(32, 32, 5, stride=2, padding=2), nn.GELU(),
                                 nn.Flatten(), nn.Linear(32 * (WINDOW // 2), 64), nn.GELU(), nn.Linear(64, LATENT))
        self.dec = nn.Sequential(nn.Linear(LATENT, 64), nn.GELU(), nn.Linear(64, WINDOW * 3))

    def forward(self, x):
        z = self.enc(x.transpose(1, 2)); return z, self.dec(z).view(-1, WINDOW, 3)


def make_bars(h):
    h = h[(h["close"] > 0) & (h["open"] > 0)].sort_values("ts").drop_duplicates("ts").reset_index(drop=True).copy()
    h["r"] = np.log(h["close"] / h["close"].shift())
    h.loc[0, "r"] = 0.0; h["r"] = h["r"].clip(-0.3, 0.3); h["rng"] = np.log(h["high"] / h["low"])
    g = h.set_index("ts").groupby(pd.Grouper(freq="4h"))
    b = pd.DataFrame({"r": g["r"].sum(), "rng": g["rng"].max(), "vol": g["volume"].sum(), "n": g["r"].count(),
                      "prix": g["close"].last()})
    b = b[b["n"] > 0].copy()
    b = b[b.index + pd.Timedelta(hours=4) <= h["ts"].max() + pd.Timedelta(hours=1)]     # seulement les bougies terminées
    b["vol_30"] = b["r"].rolling(180, min_periods=60).std()
    return b.dropna()


def features(bars, symbol):
    v = bars["vol_30"].values; lv = np.log1p(bars["vol"].values)
    lvz = (lv - pd.Series(lv).rolling(180, min_periods=60).mean().values) / pd.Series(lv).rolling(180, min_periods=60).std().values
    F = np.stack([np.clip(bars["r"].values / v, -8, 8), np.clip(bars["rng"].values / v, 0, 20) / 4,
                  np.nan_to_num(np.clip(lvz, -5, 5))], axis=1).astype(np.float32)
    idx = np.arange(WINDOW, len(bars))
    WIN = np.stack([F[i - WINDOW + 1:i + 1] for i in idx])
    t_idx = bars.index[idx]
    ae = AE(); ae.load_state_dict(torch.load(MODEL_DIR / symbol / "autoencodeur.pt", map_location="cpu")); ae.eval()
    with torch.no_grad():
        Z = np.concatenate([ae(torch.tensor(WIN[i:i + 4096]))[0].numpy() for i in range(0, len(WIN), 4096)])
    npath = MODEL_DIR / symbol / "normalisation.json"
    if npath.exists():
        nm = json.loads(npath.read_text()); mu, sd = np.array(nm["moyenne"]), np.array(nm["ecart"])
    else:
        tm = np.asarray(t_idx < TRAIN_END)
        if tm.sum() < 500:
            raise RuntimeError(f"{symbol} : pas assez d'historique avant {TRAIN_END} pour normaliser l'autoencodeur")
        mu, sd = Z[tm].mean(0), Z[tm].std(0)
        npath.write_text(json.dumps({"moyenne": mu.tolist(), "ecart": sd.tolist(), "fenetres": int(tm.sum())}))
    Z = ((Z - mu) / (sd + 1e-6)).clip(-5, 5)
    r = bars["r"].values; c = np.concatenate([[0.0], np.cumsum(r)])

    def cum(n):
        out = np.full(len(r), np.nan); out[n - 1:] = c[n:] - c[:-n]; return out
    vol_ratio = np.log(pd.Series(r).rolling(30, min_periods=10).std().values / v)
    extra = np.stack([np.clip(r / v, -8, 8), np.clip(cum(6) / (v * np.sqrt(6)), -6, 6), np.clip(cum(30) / (v * np.sqrt(30)), -6, 6),
                      np.clip(cum(130) / (v * np.sqrt(130)), -6, 6), np.clip(vol_ratio, -3, 3),
                      bars.index.dayofweek / 4.0], axis=1)
    OBS = np.concatenate([Z, np.nan_to_num(extra[idx])], axis=1).astype(np.float32)
    return pd.DataFrame({"t": t_idx, "r": r[idx], "prix": bars["prix"].values[idx]}), OBS


def vote(agents, obs, long_only):
    acts = [int(a.predict(obs, deterministic=True)[0]) for a in agents]
    pos = acts if long_only else [x - 1 for x in acts]
    new = int(np.mean(pos) > 0.5) if long_only else int(np.sign(np.sum(pos)))
    return new, pos


def verify(info, OBS, agents, long_only, key):
    """Rejoue le test v11 (2024 → aujourd'hui) : doit redonner à peu près le Sharpe trouvé dans Colab."""
    t = info["t"].values; a = int(np.searchsorted(t, np.datetime64(VAL_END)))
    CS = np.concatenate([[0.0], np.cumsum(info["r"].values[1:])])
    pos, rets = 0.0, []
    for i in range(a, len(t) - 1 - STRIDE, STRIDE):
        new, _ = vote(agents, np.append(OBS[i], pos).astype(np.float32), long_only)
        rets.append(new * (CS[i + STRIDE] - CS[i]) - COST * abs(new - pos)); pos = float(new)
    rets = np.array(rets); s = rets.mean() / rets.std() * np.sqrt(365.25) if rets.std() > 0 else 0.0
    print(f"  Vérification {key[0]} {NAMES[key[1]]} : Sharpe recalculé {s:.2f} (Colab : {BACKTEST[key][2]:.2f})")
    return round(float(s), 2)


# ---------------------------------------------------------------- portefeuille fictif (une « part » par crypto)
def fresh_sleeve():
    return {"dernier_traite": None, "derniere_decision": None, "position": 0, "robot": 1.0, "garder": 1.0,
            "frais": 0.0, "changements": 0, "decisions": [], "votes": None, "prix_reference": None, "signal_depuis": None}


def is_decision(ts, last_dec):
    if ts.hour == DECISION_HOUR:
        return True
    return last_dec is not None and ts - pd.Timestamp(last_dec) >= pd.Timedelta(hours=28)   # rattrapage si une bougie manque


def update_sleeve(S, info, OBS, agents, long_only, part):
    t = info["t"]
    start_i = int(np.searchsorted(t.values, np.datetime64(PAPER_START), side="left"))
    if start_i >= len(t):
        return False
    i0 = start_i if S["dernier_traite"] is None else int(np.searchsorted(t.values, np.datetime64(pd.Timestamp(S["dernier_traite"])), side="right"))
    for i in range(i0, len(t)):
        ts = pd.Timestamp(t.iloc[i])
        if ts > PAPER_END:
            break
        if S["dernier_traite"] is not None or i > start_i:
            S["robot"] *= math.exp(S["position"] * info["r"].iloc[i]); S["garder"] *= math.exp(info["r"].iloc[i])
        first = S["derniere_decision"] is None
        if (first and ts.hour == DECISION_HOUR) or (not first and is_decision(ts, S["derniere_decision"])):
            new, votes = vote(agents, np.append(OBS[i], float(S["position"])).astype(np.float32), long_only)
            if first:
                S["garder"] *= (1 - COST)
            changed = new != S["position"]
            if changed:
                S["robot"] *= (1 - COST * abs(new - S["position"])); S["frais"] += part * S["robot"] * COST * abs(new - S["position"])
                S["changements"] += 1
            if changed or first:
                S["signal_depuis"] = str(ts); S["prix_reference"] = float(info["prix"].iloc[i])
            S["decisions"].append({"date": str(ts), "fin": str(ts + pd.Timedelta(hours=4)), "initial": first,
                                   "signal": {1: "ACHETER", 0: "DEHORS", -1: "VENDRE"}[new], "ancien": {1: "ACHETER", 0: "DEHORS", -1: "VENDRE"}[S["position"]],
                                   "prix": float(info["prix"].iloc[i]), "votes": votes, "change": bool(changed or first)})
            S["decisions"] = S["decisions"][-120:]
            S["position"] = new; S["votes"] = votes; S["derniere_decision"] = str(ts)
        elif first:
            pass                                         # on attend la première bougie de décision
        S["dernier_traite"] = str(ts)
    return S["derniere_decision"] is not None


def sharpe(vals):
    v = np.array(vals, dtype=float)
    if len(v) < 10:
        return None
    r = np.diff(np.log(v)); return round(float(r.mean() / r.std() * np.sqrt(365)), 2) if r.std() > 0 else 0.0


def max_dd(vals):
    v = np.array(vals, dtype=float)
    return round(float((v / np.maximum.accumulate(v) - 1).min() * 100), 2) if len(v) else 0.0


def main():
    from stable_baselines3 import PPO
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    today = now.strftime("%Y-%m-%d")
    state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
    state.setdefault("courbes", {k: [] for k in PORTFOLIOS}); state.setdefault("verification", {})
    coins = sorted({c for p in PORTFOLIOS.values() for c in p["coins"]})
    infos, last_bar = {}, None
    for sym in coins:
        h, first = load_hourly(sym)
        info, OBS = features(make_bars(h), sym)
        infos[sym] = (info, OBS, first)
        last_bar = info["t"].iloc[-1] if last_bar is None else max(last_bar, info["t"].iloc[-1])
        print(f"{NAMES[sym]} : dernière bougie {info['t'].iloc[-1]} (terminée à +4 h), prix {info['prix'].iloc[-1]:.6g}")

    out_pf = []
    for key, P in PORTFOLIOS.items():
        part = CAPITAL / len(P["coins"]); markets = []
        for sym in P["coins"]:
            info, OBS, first = infos[sym]
            agents = [PPO.load(MODEL_DIR / sym / f"{key}_{s}.zip", device="cpu") for s in range(3)]
            vk = f"{key}:{sym}"
            if first or vk not in state["verification"]:
                state["verification"][vk] = verify(info, OBS, agents, P["long_only"], (key, sym))
            S = state.setdefault(vk, fresh_sleeve())
            started = update_sleeve(S, info, OBS, agents, P["long_only"], part)
            new_now, votes_now = vote(agents, np.append(OBS[-1], float(S["position"])).astype(np.float32), P["long_only"])
            last = next((d for d in reversed(S["decisions"]) if d["change"]), None)
            bt = BACKTEST[(key, sym)]
            markets.append({
                "code": sym, "nom": NAMES[sym], "commence": started,
                "signal": {1: "ACHETER", 0: "DEHORS", -1: "VENDRE"}[S["position"]] if started else None,
                "apercu": None if started else {"signal": {1: "ACHETER", 0: "DEHORS", -1: "VENDRE"}[new_now], "votes": votes_now},
                "votes": S["votes"], "signal_depuis": S["signal_depuis"],
                "signal_depuis_fin": str(pd.Timestamp(S["signal_depuis"]) + pd.Timedelta(hours=4)) if S["signal_depuis"] else None,
                "prix_reference": S["prix_reference"], "dernier_changement": last,
                "derniere_decision": S["derniere_decision"],
                "prix_actuel": float(info["prix"].iloc[-1]),
                "part": round(part, 2), "valeur": round(part * S["robot"], 2), "rendement_pct": round((S["robot"] - 1) * 100, 2),
                "garder_valeur": round(part * S["garder"], 2), "garder_rendement_pct": round((S["garder"] - 1) * 100, 2),
                "frais": round(S["frais"], 2), "changements": S["changements"],
                "backtest": {"robot": bt[0], "garder": bt[1], "sharpe_robot": bt[2], "sharpe_garder": bt[3],
                             "sharpe_recalcule": state["verification"].get(vk)},
                "appris": APPRIS[(key, sym)], "decisions": S["decisions"][-60:],
            })
        started = [m for m in markets if m["commence"]]
        val = round(sum(m["valeur"] for m in started) + part * (len(markets) - len(started)), 2)
        gval = round(sum(m["garder_valeur"] for m in started) + part * (len(markets) - len(started)), 2)
        C = state["courbes"][key]
        if started:
            pt = {"d": today, "robot": val, "garder": gval}
            if C and C[-1]["d"] == today:
                C[-1] = pt
            else:
                C.append(pt)
        rv, gv = [p["robot"] for p in C], [p["garder"] for p in C]
        s_r, s_g = sharpe(rv), sharpe(gv)
        beat = sum(1 for m in started if m["valeur"] > m["garder_valeur"])
        need = len(markets) // 2 + 1
        crit = [{"nom": "Le portefeuille des robots finit au-dessus de « acheter et garder » les mêmes cryptos", "ok": val > gval},
                {"nom": "Meilleur rapport gain / risque (Sharpe) que « garder »", "ok": (s_r or 0) > (s_g or 0)},
                {"nom": f"Au moins {need} cryptos sur {len(markets)} battent leur « garder » ({beat}/{len(markets)})", "ok": beat >= need}]
        n_ok = sum(c["ok"] for c in crit)
        if not started:
            verdict = {"statut": "pas commencé", "criteres": [], "texte": "Le test commence le " + PAPER_START.strftime("%d/%m/%Y") + " à 04:00 UTC."}
        elif now >= PAPER_END:
            st = "validé" if n_ok == 3 else ("à prolonger" if n_ok == 2 else "non validé")
            verdict = {"statut": st, "criteres": crit, "texte": {
                "validé": "Les robots ont battu « garder » sur les 3 critères pendant 6 mois.",
                "à prolonger": "Résultat serré : 2 critères sur 3. Il faut prolonger le test avant de conclure.",
                "non validé": "Les robots n'ont pas battu « garder ». Ils ne doivent pas être utilisés avec de l'argent réel."}[st]}
        else:
            verdict = {"statut": "en cours", "criteres": crit, "texte": "Verdict définitif le " + PAPER_END.strftime("%d/%m/%Y") + ". Les critères changent chaque jour."}
        out_pf.append({"code": key, "nom": P["nom"], "long_only": P["long_only"], "commence": bool(started),
                       "capital": CAPITAL, "valeur": val, "rendement_pct": round((val / CAPITAL - 1) * 100, 2),
                       "garder_valeur": gval, "garder_rendement_pct": round((gval / CAPITAL - 1) * 100, 2),
                       "sharpe": s_r, "garder_sharpe": s_g, "pire_baisse_pct": max_dd(rv), "garder_pire_baisse_pct": max_dd(gv),
                       "frais": round(sum(m["frais"] for m in markets), 2), "changements": sum(m["changements"] for m in markets),
                       "courbe": C, "marches": markets, "verdict": verdict})
    STATE_PATH.write_text(json.dumps(state, indent=1, default=float))
    nxt = (now.normalize() + pd.Timedelta(hours=4, minutes=30))
    if nxt <= now:
        nxt += pd.Timedelta(days=1)
    days_total = (PAPER_END - PAPER_START).days
    out = {"genere_le": now.strftime("%Y-%m-%d %H:%M UTC"), "donnees_jusqu_au": str(last_bar + pd.Timedelta(hours=4)),
           "prochaine_mise_a_jour": str(nxt),
           "test": {"debut": PAPER_START.strftime("%Y-%m-%d"), "fin": PAPER_END.strftime("%Y-%m-%d"), "capital": CAPITAL,
                    "jours_ecoules": max(0, min(days_total, (now - PAPER_START).days)), "jours_total": days_total,
                    "frais_pct": COST * 100, "termine": bool(now >= PAPER_END)},
           "portefeuilles": out_pf}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    for p in out_pf:
        print(f"{p['nom']} : {p['valeur']} $ (garder {p['garder_valeur']} $) | " +
              ", ".join(f"{m['nom']} {m['signal'] or ('aperçu ' + m['apercu']['signal'])}" for m in p["marches"]))


if __name__ == "__main__":
    main()
