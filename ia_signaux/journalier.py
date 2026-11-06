"""Crypto v4 (classement IA, top 3) et crypto v5 (répartition choisie par l'IA), bougies journalières Binance."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from .commun import MODELES, bougies_binance

SYMBOLES = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "SOLUSDT", "TRXUSDT",
            "LTCUSDT", "LINKUSDT", "DOTUSDT", "AVAXUSDT", "BCHUSDT", "XLMUSDT", "ATOMUSDT", "MATICUSDT"]
TOP_K, TREND_SMA, MIN_COINS = 3, 100, 5


def panel_journalier(jours=1000):
    """Bougies journalières terminées des 16 cryptos (colonnes date, symbol, open, high, low, close, volume, quote_volume).
    MATIC n'est plus cotée sur Binance : elle est simplement absente, comme dans le moteur officiel depuis 2024."""
    debut = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=jours)

    def un(s):
        try:
            d = bougies_binance(s, "1d", debut)
        except Exception:
            return pd.DataFrame()
        d["date"] = d["ts"].dt.normalize(); d["symbol"] = s
        return d[["date", "symbol", "open", "high", "low", "close", "volume", "quote_volume"]]
    with ThreadPoolExecutor(max_workers=8) as ex:
        parts = [p for p in ex.map(un, SYMBOLES) if len(p)]
    return pd.concat(parts, ignore_index=True)


def _pivots(panel):
    O = panel.pivot(index="date", columns="symbol", values="open").sort_index()
    C = panel.pivot(index="date", columns="symbol", values="close").sort_index()
    QV = panel.pivot(index="date", columns="symbol", values="quote_volume").sort_index()
    idx = pd.date_range(C.index.min(), C.index.max(), freq="D")
    O, C, QV = O.reindex(idx), C.reindex(idx), QV.reindex(idx)
    O.columns.name = C.columns.name = QV.columns.name = None
    logC = np.log(C); r1 = logC.diff(); raw = {}
    for k in (1, 3, 7, 14, 30, 60, 90): raw[f"ret_{k}"] = logC - logC.shift(k)
    for k in (7, 30, 90): raw[f"vol_{k}"] = r1.rolling(k, min_periods=k).std()
    for k in (20, 50, 100, 200): raw[f"sma_{k}"] = C / C.rolling(k, min_periods=k).mean() - 1
    raw["dist_haut_90"] = C / C.rolling(90, min_periods=90).max() - 1
    raw["volume_7_30"] = np.log(QV.rolling(7, min_periods=7).mean() / QV.rolling(30, min_periods=30).mean())
    raw["sharpe_30"] = raw["ret_30"] / (raw["vol_30"] * np.sqrt(30))
    valid = C.notna()
    for f in raw.values(): valid &= f.notna()
    n_valid = valid.sum(axis=1)
    ranked = {n: f.where(valid).rank(axis=1, pct=True) - 0.5 for n, f in raw.items()}
    btc = C["BTCUSDT"]
    market = pd.DataFrame({"mkt_btc_ret_30": np.log(btc / btc.shift(30)), "mkt_btc_sma_100": btc / btc.rolling(100).mean() - 1,
                           "mkt_breadth_50": (C > C.rolling(50, min_periods=50).mean()).where(valid).sum(axis=1) / n_valid.replace(0, np.nan),
                           "mkt_vol_30": r1.where(valid).mean(axis=1).rolling(30).std()}, index=idx)
    return dict(idx=idx, O=O, C=C, raw=raw, ranked=ranked, valid=valid, n_valid=n_valid, market=market, btc=btc)


def _filtre(P, d):
    b = P["btc"]; sma = b.rolling(TREND_SMA).mean()
    return bool(b.loc[d] > sma.loc[d]), float(b.loc[d] / sma.loc[d] - 1) * 100


def _dernier_modele(dossier):
    fichiers = sorted((MODELES / dossier).glob("modele_cutoff_*.pt"))
    if not fichiers:
        raise RuntimeError(f"Aucun modèle dans {MODELES / dossier}")
    return torch.load(fichiers[-1], weights_only=False), fichiers[-1].stem.split("_")[-1]


# ------------------------------------------------------------------------------------------ v4
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def signal_v4(panel=None):
    """Classement IA du jour (16 cryptos) et portefeuille conseillé : top 3 à parts égales si le filtre Bitcoin est positif,
    sinon tout en USDT. Décision prise à la clôture du dernier jour complet, à exécuter à l'ouverture suivante."""
    P = _pivots(panel if panel is not None else panel_journalier())
    saved, cutoff = _dernier_modele("v4")
    feats = saved["features"]
    d = P["C"].dropna(how="all").index[-1]
    ok = P["valid"].loc[d]
    if P["n_valid"].loc[d] < MIN_COINS:
        raise RuntimeError("Pas assez de cryptos valides ce jour-là.")
    X = pd.DataFrame({n: P["ranked"][n].loc[d] for n in P["ranked"]})[ok]
    for c in P["market"].columns:
        X[c] = P["market"].loc[d, c]
    X = X[feats].values
    Xs = (X - np.asarray(saved["mean"])) / np.asarray(saved["scale"])
    preds = []
    for st in saved["states"]:
        m = MLP(len(feats)); m.load_state_dict(st); m.eval()
        with torch.no_grad():
            preds.append(m(torch.tensor(Xs, dtype=torch.float32)).numpy())
    score = pd.Series(np.mean(preds, axis=0), index=ok[ok].index).sort_values(ascending=False)
    filtre, ecart = _filtre(P, d)
    top = list(score.index[:TOP_K]) if filtre else []
    return {"date_des_donnees": str(d.date()), "a_executer_le": str((d + pd.Timedelta(days=1)).date()),
            "filtre_bitcoin_positif": filtre, "btc_vs_moyenne_100j_pct": round(ecart, 2),
            "classement": [{"symbole": s, "score": round(float(v), 4)} for s, v in score.items()],
            "portefeuille": {s: round(100 / TOP_K, 2) for s in top} or {"USDT": 100.0},
            "modele": f"entraîné jusqu'au {cutoff}"}


# ------------------------------------------------------------------------------------------ v5
class Alloc(nn.Module):
    def __init__(self, d, dm):
        super().__init__()
        self.coin = nn.Sequential(nn.Linear(d + dm, 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1))
        self.cash = nn.Sequential(nn.Linear(dm, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, x, m, v):
        mm = m[:, None, :].expand(-1, x.shape[1], -1)
        s = self.coin(torch.cat([x, mm], -1)).squeeze(-1).masked_fill(~v, -1e9)
        return torch.softmax(s, dim=1)


def signal_v5(panel=None):
    """Répartition du jour choisie par l'IA v5 (pourcentages par crypto), ou 100 % USDT si le filtre Bitcoin est négatif."""
    P = _pivots(panel if panel is not None else panel_journalier())
    pack, cutoff = _dernier_modele("v5")
    absf = {"abs_vol_30": np.log(P["raw"]["vol_30"]), "abs_ret_30": P["raw"]["ret_30"], "abs_sma_100": P["raw"]["sma_100"]}
    X = np.nan_to_num(np.stack([P["ranked"][n].values for n in P["ranked"]] + [absf[n].values for n in absf], axis=2).astype(np.float64))
    M = P["market"].values.astype(np.float64)
    V = P["valid"].values & (P["n_valid"].values[:, None] >= 8)
    i = int(np.where(~np.isnan(P["C"].values).all(axis=1))[0][-1]); d = P["idx"][i]
    Xs = torch.tensor(((X[i:i + 1] - pack["mu"]) / pack["sd"]).clip(-8, 8), dtype=torch.float32)
    Ms = torch.tensor(np.nan_to_num((M[i:i + 1] - pack["mmu"]) / pack["msd"]).clip(-8, 8), dtype=torch.float32)
    v = torch.tensor(V[i:i + 1]); acc = np.zeros(X.shape[1])
    for st in pack["states"]:
        m = Alloc(X.shape[2], M.shape[1]); m.load_state_dict(st); m.eval()
        with torch.no_grad():
            acc += m(Xs, Ms, v).numpy()[0]
    w = pd.Series(acc / len(pack["states"]), index=list(P["C"].columns))
    filtre, ecart = _filtre(P, d)
    rep = {s: round(float(x) * 100, 2) for s, x in w.sort_values(ascending=False).items() if x > 0.005} if filtre else {"USDT": 100.0}
    return {"date_des_donnees": str(d.date()), "a_executer_le": str((d + pd.Timedelta(days=1)).date()),
            "filtre_bitcoin_positif": filtre, "btc_vs_moyenne_100j_pct": round(ecart, 2),
            "portefeuille": rep, "modele": f"entraîné jusqu'au {cutoff}"}
