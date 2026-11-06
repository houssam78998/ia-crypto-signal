"""Outils communs : données Binance publiques, autoencodeur et indicateurs des robots v11."""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import torch, torch.nn as nn

MODELES = Path(__file__).resolve().parent / "modeles"
API = "https://data-api.binance.vision/api/v3/klines"      # données publiques, sans clé
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "tb_base", "tb_quote", "ignore"]
WINDOW, LATENT, STRIDE = 60, 8, 6


def _get(params):
    for essai in range(4):
        try:
            r = requests.get(API, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:                                  # réseau instable : on réessaie
            if essai == 3:
                raise RuntimeError(f"Binance injoignable ({e})")
            time.sleep(2 * (essai + 1))


def bougies_binance(symbole, intervalle, debut):
    """Bougies terminées depuis `debut` (Timestamp UTC sans fuseau). Colonnes : ts, open, high, low, close, volume, quote_volume."""
    pas = {"1h": 3_600_000, "1d": 86_400_000}[intervalle]
    start = int(pd.Timestamp(debut).timestamp() * 1000)
    lignes = []
    while True:
        bloc = _get({"symbol": symbole, "interval": intervalle, "startTime": start, "limit": 1000})
        if not bloc:
            break
        lignes += bloc
        start = bloc[-1][0] + pas
        if len(bloc) < 1000:
            break
    df = pd.DataFrame(lignes, columns=COLS)
    if df.empty:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume", "quote_volume"])
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms")
    fin = pd.to_datetime(df["close_time"], unit="ms")
    df = df[fin < pd.Timestamp.now(tz="UTC").tz_localize(None)]             # seulement les bougies terminées
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["ts", "open", "high", "low", "close", "volume", "quote_volume"]].dropna().reset_index(drop=True)


class AE(nn.Module):
    """Autoencodeur des robots v11 : résume 60 bougies de 4 h en 8 nombres."""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv1d(3, 32, 5, padding=2), nn.GELU(), nn.Conv1d(32, 32, 5, stride=2, padding=2), nn.GELU(),
                                 nn.Flatten(), nn.Linear(32 * (WINDOW // 2), 64), nn.GELU(), nn.Linear(64, LATENT))
        self.dec = nn.Sequential(nn.Linear(LATENT, 64), nn.GELU(), nn.Linear(64, WINDOW * 3))

    def forward(self, x):
        z = self.enc(x.transpose(1, 2)); return z, self.dec(z).view(-1, WINDOW, 3)


def bougies_4h(h, instrument=False):
    """Bougies horaires -> bougies de 4 h (mêmes calculs que la v11). `instrument` : gère les changements de contrat (CME)."""
    h = h[(h["close"] > 0) & (h["open"] > 0)].sort_values("ts").drop_duplicates("ts").reset_index(drop=True).copy()
    if instrument:
        meme = h["instrument_id"].eq(h["instrument_id"].shift())
        h["r"] = np.where(meme, np.log(h["close"] / h["close"].shift()), np.log(h["close"] / h["open"]))
    else:
        h["r"] = np.log(h["close"] / h["close"].shift())
    h.loc[0, "r"] = 0.0; h["r"] = h["r"].clip(-0.3, 0.3); h["rng"] = np.log(h["high"] / h["low"])
    g = h.set_index("ts").groupby(pd.Grouper(freq="4h"))
    b = pd.DataFrame({"r": g["r"].sum(), "rng": g["rng"].max(), "vol": g["volume"].sum(), "n": g["r"].count(), "prix": g["close"].last()})
    b = b[b["n"] > 0].copy()
    b = b[b.index + pd.Timedelta(hours=4) <= h["ts"].max() + pd.Timedelta(hours=1)]     # seulement les bougies terminées
    b["vol_30"] = b["r"].rolling(180, min_periods=60).std()
    return b.dropna()


def observations(bars, dossier):
    """Indicateurs vus par les robots pour chaque bougie de 4 h. `dossier` contient autoencodeur.pt et normalisation.json."""
    v = bars["vol_30"].values; lv = np.log1p(bars["vol"].values)
    lvz = (lv - pd.Series(lv).rolling(180, min_periods=60).mean().values) / pd.Series(lv).rolling(180, min_periods=60).std().values
    F = np.stack([np.clip(bars["r"].values / v, -8, 8), np.clip(bars["rng"].values / v, 0, 20) / 4,
                  np.nan_to_num(np.clip(lvz, -5, 5))], axis=1).astype(np.float32)
    idx = np.arange(WINDOW, len(bars))
    WIN = np.stack([F[i - WINDOW + 1:i + 1] for i in idx])
    ae = AE(); ae.load_state_dict(torch.load(Path(dossier) / "autoencodeur.pt", map_location="cpu")); ae.eval()
    with torch.no_grad():
        Z = np.concatenate([ae(torch.tensor(WIN[i:i + 4096]))[0].numpy() for i in range(0, len(WIN), 4096)])
    nm = json.loads((Path(dossier) / "normalisation.json").read_text())
    Z = ((Z - np.array(nm["moyenne"])) / (np.array(nm["ecart"]) + 1e-6)).clip(-5, 5)
    r = bars["r"].values; c = np.concatenate([[0.0], np.cumsum(r)])

    def cum(n):
        out = np.full(len(r), np.nan); out[n - 1:] = c[n:] - c[:-n]; return out
    vol_ratio = np.log(pd.Series(r).rolling(30, min_periods=10).std().values / v)
    extra = np.stack([np.clip(r / v, -8, 8), np.clip(cum(6) / (v * np.sqrt(6)), -6, 6), np.clip(cum(30) / (v * np.sqrt(30)), -6, 6),
                      np.clip(cum(130) / (v * np.sqrt(130)), -6, 6), np.clip(vol_ratio, -3, 3),
                      bars.index.dayofweek / 4.0], axis=1)
    OBS = np.concatenate([Z, np.nan_to_num(extra[idx])], axis=1).astype(np.float32)
    return pd.DataFrame({"t": bars.index[idx], "r": r[idx], "prix": bars["prix"].values[idx]}), OBS


def charger_robots(dossier, nom_fichier):
    """Charge les 3 robots PPO (stable-baselines3) d'un dossier. nom_fichier ex. 'achat_seul_{}.zip' ou 'robot_{}.zip'."""
    from stable_baselines3 import PPO
    return [PPO.load(Path(dossier) / nom_fichier.format(s), device="cpu") for s in range(3)]


def voter(robots, obs, position, achat_seul):
    """Vote des 3 robots. Renvoie (nouvelle position 1/0/-1, votes de chaque robot)."""
    o = np.append(obs, float(position)).astype(np.float32)
    acts = [int(a.predict(o, deterministic=True)[0]) for a in robots]
    votes = acts if achat_seul else [x - 1 for x in acts]
    nouvelle = int(np.mean(votes) > 0.5) if achat_seul else int(np.sign(np.sum(votes)))
    return nouvelle, votes


NOMS_SIGNAL = {1: "ACHETER", 0: "DEHORS", -1: "VENDRE"}
