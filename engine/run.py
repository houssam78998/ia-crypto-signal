# -*- coding: utf-8 -*-
"""
Moteur du signal IA crypto (v4) — exécuté chaque nuit par GitHub Actions.
1. Met à jour les bougies journalières Binance (16 cryptos).
2. Calcule les indicateurs et le classement IA (mêmes réglages que la v4 testée en walk-forward).
3. Tient deux portefeuilles FICTIFS de 1 000 USDT : signal quotidien et signal hebdomadaire (dimanche).
4. Écrit docs/data/app.json, lu par l'application mobile.
Aucun ordre réel n'est passé.
"""
import os, io, time, json, zipfile, warnings, random, math
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

ROOT = Path(os.environ.get("SIGNAL_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = ROOT / "data"
MODEL_DIR = DATA_DIR / "modeles"
ARCHIVE_DIR = Path(os.environ.get("ARCHIVE_DIR", "/tmp/archives_binance"))
APP_JSON = ROOT / "docs" / "data" / "app.json"
STATE_PATH = DATA_DIR / "etat.json"
PANEL_PATH = DATA_DIR / "panel.csv.gz"
for _d in (DATA_DIR, MODEL_DIR, ARCHIVE_DIR, APP_JSON.parent):
    _d.mkdir(parents=True, exist_ok=True)

# ---- Réglages identiques à la v4 (ne pas modifier pendant le test) ----
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "SOLUSDT", "TRXUSDT",
           "LTCUSDT", "LINKUSDT", "DOTUSDT", "AVAXUSDT", "BCHUSDT", "XLMUSDT", "ATOMUSDT", "MATICUSDT"]
START_DATE = "2018-01-01"
HORIZON = 7
EMBARGO_DAYS = HORIZON + 2
TOP_K = 3
TREND_SMA = 100
FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005
MIN_COINS = 5
N_SEEDS = int(os.environ.get("N_SEEDS", 5))
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", 60))
PATIENCE = int(os.environ.get("PATIENCE", 8))
SEED = 42
# ---- Test en argent fictif ----
PAPER_START = pd.Timestamp(os.environ.get("PAPER_START", "2026-10-18"))
TEST_WEEKS = 26
PAPER_END = PAPER_START + pd.Timedelta(weeks=TEST_WEEKS)   # dernière valorisation
INITIAL_CAPITAL = 1_000.0
RETRAIN_DAYS = 28
PORTFOLIOS = {"quotidien": "Signal quotidien", "hebdomadaire": "Signal hebdomadaire (dimanche)"}
MINOR_USDT = 0.02    # ajustement < 2 % du portefeuille = mineur (facultatif à exécuter en vrai)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(max(1, os.cpu_count() or 1))
def log(*a):
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}]", *a, flush=True)

def _to_ms(series):
    # Les archives spot Binance sont en microsecondes depuis 2025.
    s = pd.to_numeric(series, errors="coerce")
    return np.where(s > 1e14, s // 1000, s)

def _periods(start, end):
    first = start.to_period("M").to_timestamp().tz_localize("UTC")
    current = end.to_period("M").to_timestamp().tz_localize("UTC")
    out, m = [], first
    while m < current:
        out.append(("monthly", m)); m = m + pd.offsets.MonthBegin(1)
    d = current
    while d <= end.normalize():
        out.append(("daily", d)); d += pd.Timedelta(days=1)
    return out

COLS = ["open_time_ms", "open", "high", "low", "close", "volume", "close_time_ms",
        "quote_volume", "trade_count", "taker_buy_base", "taker_buy_quote", "ignore"]

def download_symbol(symbol):
    start, end = pd.Timestamp(START_DATE, tz="UTC"), pd.Timestamp.now(tz="UTC")
    folder = ARCHIVE_DIR / symbol; folder.mkdir(parents=True, exist_ok=True)
    sess, frames = requests.Session(), []
    for kind, dt in _periods(start, end):
        part = dt.strftime("%Y-%m" if kind == "monthly" else "%Y-%m-%d")
        fn = f"{symbol}-1d-{part}.zip"
        url = f"https://data.binance.vision/data/spot/{kind}/klines/{symbol}/1d/{fn}"
        local = folder / fn
        try:
            if local.exists():
                content = local.read_bytes()
            else:
                r = None
                for attempt in range(3):
                    try:
                        r = sess.get(url, timeout=45); break
                    except requests.RequestException:
                        time.sleep(2 * (attempt + 1))
                if r is None or r.status_code in (403, 404):
                    continue                       # crypto pas encore cotée (ou jour pas encore publié)
                r.raise_for_status()
                content = r.content
                if kind == "monthly":
                    local.write_bytes(content)
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                names = [n for n in z.namelist() if n.lower().endswith(".csv")]
                if not names:
                    continue
                with z.open(names[0]) as f:
                    part_df = pd.read_csv(f, header=None)
            if len(part_df) and not str(part_df.iloc[0, 0]).strip().replace(".", "", 1).isdigit():
                part_df = part_df.iloc[1:]
            if part_df.shape[1] < 12 or part_df.empty:
                continue
            part_df = part_df.iloc[:, :12]; part_df.columns = COLS
            frames.append(part_df)
        except Exception as e:
            print(f"  {fn} ignoré ({e})")
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    raw["open_time_ms"] = _to_ms(raw["open_time_ms"]); raw["close_time_ms"] = _to_ms(raw["close_time_ms"])
    raw = raw.dropna(subset=["open_time_ms"])
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "close_time_ms"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw["date"] = pd.to_datetime(raw["open_time_ms"].astype("int64"), unit="ms").dt.normalize()
    closed = pd.to_datetime(raw["close_time_ms"], unit="ms") < pd.Timestamp.utcnow().tz_localize(None)
    raw = raw[closed].drop_duplicates("date").sort_values("date")
    raw["symbol"] = symbol
    return raw[["date", "symbol", "open", "high", "low", "close", "volume", "quote_volume"]].dropna()



def recent_from_api(symbol, days=60):
    try:
        r = requests.get("https://data-api.binance.vision/api/v3/klines",
                         params={"symbol": symbol, "interval": "1d", "limit": days}, timeout=20)
        if r.status_code != 200:
            return pd.DataFrame()
        df = pd.DataFrame(r.json(), columns=COLS)
        for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["date"] = pd.to_datetime(df["open_time_ms"].astype("int64"), unit="ms").dt.normalize()
        df = df[pd.to_datetime(df["close_time_ms"].astype("int64"), unit="ms") < pd.Timestamp.utcnow().tz_localize(None)]
        df["symbol"] = symbol
        return df[["date", "symbol", "open", "high", "low", "close", "volume", "quote_volume"]].dropna()
    except Exception:
        return pd.DataFrame()

def update_panel():
    if PANEL_PATH.exists():
        panel = pd.read_csv(PANEL_PATH, parse_dates=["date"])
        log(f"Historique chargé : {len(panel):,} lignes jusqu'au {panel.date.max().date()}")
    else:
        log("Premier lancement : téléchargement de l'historique Binance (quelques minutes)…")
        with ThreadPoolExecutor(max_workers=8) as ex:
            parts = list(ex.map(download_symbol, SYMBOLS))
        panel = pd.concat([p for p in parts if len(p)], ignore_index=True)
    recent = [recent_from_api(s) for s in SYMBOLS]
    recent = [p for p in recent if len(p)]
    if recent:
        log(f"API Binance : {len(recent)} cryptos mises à jour")
    else:
        log("API Binance indisponible : complément par les archives journalières")
        global START_DATE
        keep = START_DATE
        START_DATE = (pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
        with ThreadPoolExecutor(max_workers=8) as ex:
            recent = [p for p in ex.map(download_symbol, SYMBOLS) if len(p)]
        START_DATE = keep
    panel = pd.concat([panel] + recent, ignore_index=True)
    panel = panel.drop_duplicates(["date", "symbol"], keep="last").sort_values(["date", "symbol"]).reset_index(drop=True)
    panel.to_csv(PANEL_PATH, index=False, compression="gzip")
    log(f"Données jusqu'au {panel.date.max().date()} inclus")
    return panel

def display(x):
    pass

if os.environ.get("SIGNAL_PANEL_OVERRIDE"):
    panel = pd.read_csv(os.environ["SIGNAL_PANEL_OVERRIDE"], parse_dates=["date"])   # tests
else:
    panel = update_panel()

# ======================= INDICATEURS (v4) =======================
O = panel.pivot(index="date", columns="symbol", values="open").sort_index()
C = panel.pivot(index="date", columns="symbol", values="close").sort_index()
QV = panel.pivot(index="date", columns="symbol", values="quote_volume").sort_index()
full_idx = pd.date_range(C.index.min(), C.index.max(), freq="D", name="date")
O, C, QV = O.reindex(full_idx), C.reindex(full_idx), QV.reindex(full_idx)
O.columns.name = C.columns.name = QV.columns.name = None
COINS = list(C.columns)

logC = np.log(C)
r1 = logC.diff()
raw_feats = {}
for k in (1, 3, 7, 14, 30, 60, 90):
    raw_feats[f"ret_{k}"] = logC - logC.shift(k)
for k in (7, 30, 90):
    raw_feats[f"vol_{k}"] = r1.rolling(k, min_periods=k).std()
for k in (20, 50, 100, 200):
    raw_feats[f"sma_{k}"] = C / C.rolling(k, min_periods=k).mean() - 1
raw_feats["dist_haut_90"] = C / C.rolling(90, min_periods=90).max() - 1
raw_feats["volume_7_30"] = np.log(QV.rolling(7, min_periods=7).mean() / QV.rolling(30, min_periods=30).mean())
raw_feats["sharpe_30"] = raw_feats["ret_30"] / (raw_feats["vol_30"] * np.sqrt(30))

valid = C.notna()
for f in raw_feats.values():
    valid &= f.notna()
n_valid = valid.sum(axis=1)

ranked = {name: f.where(valid).rank(axis=1, pct=True) - 0.5 for name, f in raw_feats.items()}

btc = C["BTCUSDT"]
market = pd.DataFrame({
    "mkt_btc_ret_30": np.log(btc / btc.shift(30)),
    "mkt_btc_sma_100": btc / btc.rolling(100).mean() - 1,
    "mkt_breadth_50": (C > C.rolling(50, min_periods=50).mean()).where(valid).sum(axis=1) / n_valid.replace(0, np.nan),
    "mkt_vol_30": r1.where(valid).mean(axis=1).rolling(30).std(),
}, index=full_idx)

fwd = np.log(O.shift(-(HORIZON + 1)) / O.shift(-1)).where(valid)
label = fwd.rank(axis=1, pct=True) - 0.5

def to_long(df, name):
    return df.reset_index().melt(id_vars="date", var_name="symbol", value_name=name)

long = to_long(valid, "valid")
for name, df in ranked.items():
    long[name] = to_long(df, name)[name].values
long["fwd"] = to_long(fwd, "fwd")["fwd"].values
long["label"] = to_long(label, "label")["label"].values
long = long[long["valid"].astype(bool)].drop(columns="valid")
long = long.merge(market, left_on="date", right_index=True, how="left")
long["n_coins"] = long["date"].map(n_valid)
long = long[long["n_coins"] >= MIN_COINS].dropna(subset=list(market.columns)).reset_index(drop=True)

FEATURES = list(ranked.keys()) + list(market.columns)
print(f"Échantillons : {len(long):,} (crypto × jour) | {len(FEATURES)} indicateurs")
print(f"Période utilisable : {long.date.min().date()} → {long.date.max().date()}")
# ======================= MODÈLE (v4) =======================
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_ensemble(Xtr, ytr, Xva, yva):
    models = []
    Xtr_t, ytr_t = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)
    Xva_t, yva_t = torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva, dtype=torch.float32)
    for s in range(N_SEEDS):
        torch.manual_seed(SEED + s)
        rng = np.random.default_rng(SEED + s)
        m = MLP(Xtr.shape[1])
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
        best, best_state, bad = np.inf, None, 0
        for ep in range(MAX_EPOCHS):
            m.train()
            perm = rng.permutation(len(Xtr_t))
            for i in range(0, len(perm), 512):
                b = perm[i:i + 512]
                opt.zero_grad()
                loss = ((m(Xtr_t[b]) - ytr_t[b]) ** 2).mean()
                loss.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                vl = float(((m(Xva_t) - yva_t) ** 2).mean())
            if vl < best - 1e-6:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        m.load_state_dict(best_state); m.eval()
        models.append(m)
    return models

def predict(models, X):
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        return np.mean([m(X_t).numpy() for m in models], axis=0)



def get_model(signal_date):
    cutoff_max = signal_date - pd.Timedelta(days=EMBARGO_DAYS)
    for p in sorted(MODEL_DIR.glob("modele_cutoff_*.pt"), reverse=True):
        cutoff = pd.Timestamp(p.stem.replace("modele_cutoff_", ""))
        if cutoff_max - pd.Timedelta(days=RETRAIN_DAYS) < cutoff <= cutoff_max:
            saved = torch.load(p, weights_only=False)
            if saved["features"] != FEATURES:
                continue
            models = []
            for st in saved["states"]:
                m = MLP(len(FEATURES)); m.load_state_dict(st); m.eval(); models.append(m)
            sc = StandardScaler(); sc.mean_ = np.asarray(saved["mean"]); sc.scale_ = np.asarray(saved["scale"])
            sc.var_ = sc.scale_ ** 2; sc.n_features_in_ = len(FEATURES)
            return models, sc, cutoff
    train = long[(long["date"] <= cutoff_max) & long["label"].notna()]
    dates = np.sort(train["date"].unique())
    cut = pd.Timestamp(dates[int(len(dates) * 0.85)])
    tr = train[train["date"] < cut - pd.Timedelta(days=EMBARGO_DAYS)]
    va = train[train["date"] >= cut]
    sc = StandardScaler().fit(tr[FEATURES])
    log(f"Entraînement d'un nouveau modèle (données jusqu'au {cutoff_max.date()})")
    models = train_ensemble(sc.transform(tr[FEATURES]), tr["label"].values,
                            sc.transform(va[FEATURES]), va["label"].values)
    torch.save({"features": FEATURES, "states": [m.state_dict() for m in models],
                "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist()},
               MODEL_DIR / f"modele_cutoff_{cutoff_max:%Y-%m-%d}.pt")
    return models, sc, cutoff_max

def ranking(d):
    models, sc, _ = get_model(d)
    g = long[long["date"] == d].copy()
    g["pred"] = predict(models, sc.transform(g[FEATURES]))
    return g.sort_values("pred", ascending=False)[["symbol", "pred"]]

# ======================= PORTEFEUILLES FICTIFS =======================
btc_sma = btc.rolling(TREND_SMA).mean()
state = json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else {}
state.setdefault("classements", {})
for p in PORTFOLIOS:
    state.setdefault(p, {"cash": INITIAL_CAPITAL, "qty": {}, "history": []})
state.setdefault("btc_qty", None)

available = set(pd.to_datetime(long["date"].unique()))
last_day = C.index.max()
decision_days = [d for d in C.index if PAPER_START <= d < PAPER_END and d in available]
cost_rate = FEE_RATE + SLIPPAGE_RATE

def processed(p):
    return {h["date"] for h in state[p]["history"]}

def rebalance(p, d, top, filt, btc_pct):
    pf = state[p]
    px = C.loc[d]
    qty = pd.Series(pf["qty"], dtype=float).reindex(C.columns).fillna(0.0)
    cur = (qty * px).fillna(0.0)
    value = pf["cash"] + float(cur.sum())
    w = pd.Series(0.0, index=C.columns)
    if top:
        w[top] = 1.0 / len(top)
    cost = float((w * value - cur).abs().sum() * cost_rate)
    target = w * (value - cost)
    new_qty = (target / px).fillna(0.0)
    cash = value - cost - float(target.sum())
    actions = []
    for s in C.columns:
        delta = float(target[s] - cur[s])
        if abs(delta) > 0.01:
            kind = "ACHETER" if delta > 0 else "VENDRE"
            entry = (cur[s] < 0.01 and delta > 0)
            exit_ = (target[s] < 0.01 and delta < 0)
            major = entry or exit_ or abs(delta) >= MINOR_USDT * value
            actions.append({"crypto": s.replace("USDT", ""), "action": kind, "usdt": round(abs(delta), 2),
                            "prix": float(px[s]), "frais": round(abs(delta) * cost_rate, 4),
                            "type": "entrée" if entry else ("sortie" if exit_ else "ajustement"), "majeur": bool(major)})
    pf["cash"] = cash
    pf["qty"] = {s: float(q) for s, q in new_qty.items() if q > 0}
    pf["history"].append({"date": d.strftime("%Y-%m-%d"), "filtre": filt, "btc_vs_moy100": round(btc_pct, 2),
                          "top": [s.replace("USDT", "") for s in top], "valeur": round(value, 4),
                          "frais": round(cost, 4), "cash_apres": cash, "positions": pf["qty"], "actions": actions})

for d in decision_days:
    ds = d.strftime("%Y-%m-%d")
    need = [p for p in PORTFOLIOS if ds not in processed(p) and (p == "quotidien" or d.dayofweek == 6)]
    if not need:
        continue
    if ds not in state["classements"]:
        rk = ranking(d)
        state["classements"][ds] = {r.symbol.replace("USDT", ""): round(float(r.pred), 5) for r in rk.itertuples()}
    rk = state["classements"][ds]
    filt = bool(btc.loc[d] > btc_sma.loc[d])
    btc_pct = float((btc.loc[d] / btc_sma.loc[d] - 1) * 100)
    top = [s + "USDT" for s in list(rk.keys())[:TOP_K]] if filt else []
    if state["btc_qty"] is None:
        state["btc_qty"] = INITIAL_CAPITAL * (1 - cost_rate) / float(C.loc[d, "BTCUSDT"])
    for p in need:
        rebalance(p, d, top, filt, btc_pct)
    log(f"{ds} : filtre {'+' if filt else '-'} | top {', '.join(s.replace('USDT', '') for s in top) or 'CASH'} | "
        + " | ".join(f"{p} {state[p]['history'][-1]['valeur']:.2f}" for p in need))
STATE_PATH.write_text(json.dumps(state, indent=1), encoding="utf-8")

# ======================= EXPORT POUR L'APPLICATION =======================
def curve(p):
    hist = state[p]["history"]
    if not hist:
        return pd.Series(dtype=float)
    end = min(last_day, PAPER_END)
    out = {}
    for i, h in enumerate(hist):
        d0 = pd.Timestamp(h["date"])
        d1 = pd.Timestamp(hist[i + 1]["date"]) - pd.Timedelta(days=1) if i + 1 < len(hist) else end
        qty = pd.Series(h["positions"], dtype=float).reindex(C.columns).fillna(0.0)
        for day in C.loc[d0:d1].index:
            out[day] = h["cash_apres"] + float((qty * C.loc[day]).fillna(0).sum())
    return pd.Series(out).sort_index()

def max_dd(s):
    return float((s / s.cummax() - 1).min() * 100) if len(s) else 0.0

def r2(x, n=2):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), n)

curves = {p: curve(p) for p in PORTFOLIOS}
ref = next((c for c in curves.values() if len(c)), pd.Series(dtype=float))
btc_curve = (state["btc_qty"] * C.loc[ref.index, "BTCUSDT"]) if (len(ref) and state["btc_qty"]) else pd.Series(dtype=float)

# IC hebdomadaire : classement du dimanche vs rendement réel des 7 jours suivants
weekly_ic = []
for ds, rk in sorted(state["classements"].items()):
    d = pd.Timestamp(ds); d7 = d + pd.Timedelta(days=7)
    if d.dayofweek == 6 and d7 <= last_day and d7 <= PAPER_END:
        syms = [s + "USDT" for s in rk]
        real = np.log(C.loc[d7, syms].values / C.loc[d, syms].values)
        weekly_ic.append({"dimanche": ds, "ic": r2(pd.Series(list(rk.values())).corr(pd.Series(real), method="spearman"), 3)})
ic_vals = [w["ic"] for w in weekly_ic if w["ic"] is not None]
ic_mean = float(np.mean(ic_vals)) if ic_vals else None

def week_reports():
    reps = []
    k = 1
    while True:
        s0 = PAPER_START + pd.Timedelta(weeks=k - 1); s1 = s0 + pd.Timedelta(weeks=1)
        if s1 > min(last_day, PAPER_END) or not len(ref):
            break
        row = {"semaine": k, "du": s0.strftime("%Y-%m-%d"), "au": s1.strftime("%Y-%m-%d")}
        for p in PORTFOLIOS:
            c = curves[p]
            row[p] = r2((c.loc[s1] / c.loc[s0] - 1) * 100) if (s0 in c.index and s1 in c.index) else None
        row["btc"] = r2((C.loc[s1, "BTCUSDT"] / C.loc[s0, "BTCUSDT"] - 1) * 100)
        h = next((x for x in state["hebdomadaire"]["history"] if x["date"] == s0.strftime("%Y-%m-%d")), None)
        row["top_hebdo"] = h["top"] if h else []
        row["filtre_hebdo"] = h["filtre"] if h else None
        dh = [x for x in state["quotidien"]["history"] if s0 <= pd.Timestamp(x["date"]) < s1]
        row["jours_cash_quotidien"] = sum(1 for x in dh if not x["filtre"])
        row["ic"] = next((w["ic"] for w in weekly_ic if w["dimanche"] == row["du"]), None)
        reps.append(row); k += 1
    return reps

def verdict(p):
    c = curves[p]
    weeks = min(TEST_WEEKS, int((min(last_day, PAPER_END) - PAPER_START).days // 7)) if len(c) else 0
    if not len(c):
        return {"semaines": 0, "criteres": [], "score": 0, "statut": "pas commencé",
                "texte": f"Le test commence avec la clôture du dimanche {PAPER_START.date()}."}
    v, b = float(c.iloc[-1]), float(btc_curve.iloc[-1])
    crit = [{"nom": "Rendement meilleur que le Bitcoin", "ok": v > b},
            {"nom": "Classements de l'IA utiles (IC moyen > 0)", "ok": bool(ic_mean is not None and ic_mean > 0)},
            {"nom": "Pire baisse plus faible que le Bitcoin", "ok": max_dd(c) > max_dd(btc_curve)}]
    score = sum(x["ok"] for x in crit)
    if weeks < TEST_WEEKS:
        statut, texte = "en cours", f"En cours : {weeks}/{TEST_WEEKS} semaines. Critères remplis pour l'instant : {score}/3."
    elif score == 3:
        statut, texte = "validé", "VALIDÉ : les 3 critères sont remplis sur 6 mois."
    elif score == 2:
        statut, texte = "à prolonger", "À PROLONGER : 2 critères sur 3, résultat insuffisant pour conclure."
    else:
        statut, texte = "non validé", "NON VALIDÉ : la stratégie ne tient pas en conditions réelles."
    return {"semaines": weeks, "criteres": crit, "score": score, "statut": statut, "texte": texte}

def portfolio_block(p):
    pf = state[p]; c = curves[p]; hist = pf["history"]
    if not hist:
        return {"nom": PORTFOLIOS[p], "commence": False}
    last = hist[-1]
    px = C.loc[min(last_day, PAPER_END)]
    holdings = []
    for s, q in pf["qty"].items():
        val = q * float(px[s])
        holdings.append({"crypto": s.replace("USDT", ""), "quantite": q, "valeur": r2(val), "prix": float(px[s])})
    value = float(c.iloc[-1]) if len(c) else last["valeur"]
    for h in holdings:
        h["poids"] = r2(h["valeur"] / value * 100, 1) if value else 0
    # Résultat du dernier jour
    day = {}
    if len(c) >= 2:
        d1, d0 = c.index[-1], c.index[-2]
        prev = next((x for x in reversed(hist) if pd.Timestamp(x["date"]) <= d0), None)
        contrib = []
        if prev:
            for s, q in prev["positions"].items():
                r = float(C.loc[d1, s] / C.loc[d0, s] - 1) * 100
                contrib.append({"crypto": s.replace("USDT", ""), "variation": r2(r), "gain": r2(q * (C.loc[d1, s] - C.loc[d0, s]))})
        day = {"date": d1.strftime("%Y-%m-%d"), "variation_pct": r2((c.iloc[-1] / c.iloc[-2] - 1) * 100),
               "variation_usdt": r2(c.iloc[-1] - c.iloc[-2]),
               "btc_pct": r2((C.loc[d1, "BTCUSDT"] / C.loc[d0, "BTCUSDT"] - 1) * 100), "details": contrib}
    nxt = (pd.Timestamp(last["date"]) + pd.Timedelta(days=1 if p == "quotidien" else 7))
    return {
        "nom": PORTFOLIOS[p], "commence": True, "valeur": r2(value),
        "rendement_pct": r2((value / INITIAL_CAPITAL - 1) * 100),
        "btc_valeur": r2(btc_curve.iloc[-1]) if len(btc_curve) else None,
        "btc_rendement_pct": r2((btc_curve.iloc[-1] / INITIAL_CAPITAL - 1) * 100) if len(btc_curve) else None,
        "pire_baisse_pct": r2(max_dd(c)), "btc_pire_baisse_pct": r2(max_dd(btc_curve)),
        "frais_total": r2(sum(h["frais"] for h in hist)),
        "transactions": sum(len(h["actions"]) for h in hist),
        "decisions": len(hist), "decisions_cash": sum(1 for h in hist if not h["filtre"]),
        "cash": r2(pf["cash"]), "positions": sorted(holdings, key=lambda x: -x["valeur"]),
        "signal": {"date_calcul": last["date"], "applicable_le": (pd.Timestamp(last["date"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                   "prochain": nxt.strftime("%Y-%m-%d") if nxt < PAPER_END else None,
                   "filtre": last["filtre"], "btc_vs_moy100": last["btc_vs_moy100"], "top": last["top"],
                   "actions": last["actions"], "valeur": r2(last["valeur"])},
        "jour": day,
        "courbe": [[d.strftime("%Y-%m-%d"), r2(v)] for d, v in c.items()],
        "historique": [{"date": h["date"], "filtre": h["filtre"], "top": h["top"], "valeur": r2(h["valeur"]),
                        "frais": r2(h["frais"], 3), "actions": [a for a in h["actions"] if a["majeur"]]} for h in hist[-60:]],
        "verdict": verdict(p),
    }

preview = None
if not any(state[p]["history"] for p in PORTFOLIOS):
    d = max(available)
    rk = ranking(d)
    preview = {"date": d.strftime("%Y-%m-%d"), "filtre": bool(btc.loc[d] > btc_sma.loc[d]),
               "btc_vs_moy100": r2((btc.loc[d] / btc_sma.loc[d] - 1) * 100),
               "classement": [s.replace("USDT", "") for s in rk["symbol"].head(5)]}

latest_rank_day = max(state["classements"]) if state["classements"] else None
app = {
    "genere_le": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    "donnees_jusqu_au": last_day.strftime("%Y-%m-%d"),
    "test": {"debut": PAPER_START.strftime("%Y-%m-%d"), "fin": PAPER_END.strftime("%Y-%m-%d"), "semaines": TEST_WEEKS,
             "capital": INITIAL_CAPITAL, "frais_pct": FEE_RATE * 100, "slippage_pct": SLIPPAGE_RATE * 100,
             "top_k": TOP_K, "filtre_jours": TREND_SMA, "termine": bool(last_day >= PAPER_END)},
    "apercu": preview,
    "classement": ({"date": latest_rank_day, "scores": state["classements"][latest_rank_day]} if latest_rank_day else None),
    "portefeuilles": {p: portfolio_block(p) for p in PORTFOLIOS},
    "btc_courbe": [[d.strftime("%Y-%m-%d"), r2(v)] for d, v in btc_curve.items()],
    "semaines": week_reports(),
    "ic": {"moyen": r2(ic_mean, 3) if ic_mean is not None else None, "semaines": weekly_ic},
}
APP_JSON.write_text(json.dumps(app, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
log(f"Application mise à jour : {APP_JSON}")
