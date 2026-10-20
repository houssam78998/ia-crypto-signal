"""Crypto v5 — l'IA choisit directement la répartition (test en argent fictif).

- Le filtre Bitcoin (BTC au-dessus de sa moyenne 100 jours) décide si on est investi ou en cash, comme la v4.
- Quand on est investi, un réseau de neurones donne directement les poids des 16 cryptos.
  Il a été entraîné pour maximiser le rapport gain / risque après frais (et non pour prédire un classement).
- 5 réseaux (graines différentes), poids moyennés ; réentraînement tous les 3 mois, uniquement sur le passé.
- Données : data/panel.csv.gz, mis à jour chaque nuit par le moteur v4 (lu seulement, jamais modifié).
- Portefeuille fictif de 1 000 USDT, rééquilibré chaque jour à l'ouverture suivante (frais 0,10 % + 0,05 %).
"""
import os, json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch, torch.nn as nn

torch.set_num_threads(2)
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PANEL = REPO / "data" / "panel.csv.gz"
MODEL_DIR, STATE_PATH = HERE / "modeles", HERE / "etat.json"
OUT = REPO / "docs" / "data" / "crypto_v5.json"
V4_APP = REPO / "docs" / "data" / "app.json"
FEE, SLIP, CAPITAL = 0.001, 0.0005, 1000.0
TREND_SMA, EMBARGO, RETRAIN_DAYS = 100, 9, 91
N_SEEDS, WIN, BATCH, MAX_EP, PAT = 5, 60, 64, 300, 25
PAPER_START = pd.Timestamp(os.environ.get("V5_PAPER_START", "2026-10-20"))
PAPER_END = PAPER_START + pd.Timedelta(weeks=26)
MAJOR = 0.05            # un ajustement de moins de 5 % du portefeuille est « facultatif »


def build(panel):
    O = panel.pivot(index="date", columns="symbol", values="open").sort_index()
    C = panel.pivot(index="date", columns="symbol", values="close").sort_index()
    QV = panel.pivot(index="date", columns="symbol", values="quote_volume").sort_index()
    idx = pd.date_range(C.index.min(), C.index.max(), freq="D")
    O, C, QV = O.reindex(idx), C.reindex(idx), QV.reindex(idx)
    logC = np.log(C); r1 = logC.diff(); raw = {}
    for k in (1, 3, 7, 14, 30, 60, 90): raw[f"ret_{k}"] = logC - logC.shift(k)
    for k in (7, 30, 90): raw[f"vol_{k}"] = r1.rolling(k, min_periods=k).std()
    for k in (20, 50, 100, 200): raw[f"sma_{k}"] = C / C.rolling(k, min_periods=k).mean() - 1
    raw["dist_haut_90"] = C / C.rolling(90, min_periods=90).max() - 1
    raw["volume_7_30"] = np.log(QV.rolling(7, min_periods=7).mean() / QV.rolling(30, min_periods=30).mean())
    raw["sharpe_30"] = raw["ret_30"] / (raw["vol_30"] * np.sqrt(30))
    valid = C.notna()
    for f in raw.values(): valid &= f.notna()
    ranked = {n: f.where(valid).rank(axis=1, pct=True) - 0.5 for n, f in raw.items()}
    absf = {"abs_vol_30": np.log(raw["vol_30"]), "abs_ret_30": raw["ret_30"], "abs_sma_100": raw["sma_100"]}
    btc = C["BTCUSDT"]; n_valid = valid.sum(axis=1)
    market = pd.DataFrame({"mkt_btc_ret_30": np.log(btc / btc.shift(30)), "mkt_btc_sma_100": btc / btc.rolling(100).mean() - 1,
                           "mkt_breadth_50": (C > C.rolling(50, min_periods=50).mean()).where(valid).sum(axis=1) / n_valid.replace(0, np.nan),
                           "mkt_vol_30": r1.where(valid).mean(axis=1).rolling(30).std()}, index=idx)
    X = np.nan_to_num(np.stack([ranked[n].values for n in ranked] + [absf[n].values for n in absf], axis=2).astype(np.float64))
    M = market.values.astype(np.float64)
    V = valid.values & (n_valid.values[:, None] >= 8)
    R = (O.shift(-2) / O.shift(-1) - 1).values.astype(np.float64)
    ok_day = ~np.isnan(M).any(axis=1) & (V.sum(axis=1) >= 8)
    R = np.where(V, np.nan_to_num(R), 0.0)
    trend = (btc > btc.rolling(TREND_SMA).mean()).values
    return dict(idx=idx, O=O, C=C, X=X, M=M, V=V, R=R, ok=ok_day, trend=trend, btc=btc, coins=list(C.columns))


class Alloc(nn.Module):
    def __init__(self, d, dm):
        super().__init__()
        self.coin = nn.Sequential(nn.Linear(d + dm, 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1))
        self.cash = nn.Sequential(nn.Linear(dm, 16), nn.GELU(), nn.Linear(16, 1))     # présent pour compatibilité, non utilisé

    def forward(self, x, m, v):
        mm = m[:, None, :].expand(-1, x.shape[1], -1)
        s = self.coin(torch.cat([x, mm], -1)).squeeze(-1).masked_fill(~v, -1e9)
        return torch.softmax(s, dim=1)


def norm(D, mu, sd, mmu, msd):
    Xs = torch.tensor(((D["X"] - mu) / sd).clip(-8, 8), dtype=torch.float32)
    Ms = torch.tensor(np.nan_to_num((D["M"] - mmu) / msd).clip(-8, 8), dtype=torch.float32)
    return Xs, Ms


def train(D, hi):
    days = np.where((D["ok"] & D["trend"])[250:hi])[0] + 250
    cut = int(len(days) * 0.85); tr, va = days[:cut - EMBARGO], days[cut:]
    mu = D["X"][tr].reshape(-1, D["X"].shape[2]).mean(0); sd = D["X"][tr].reshape(-1, D["X"].shape[2]).std(0) + 1e-6
    mmu, msd = D["M"][tr].mean(0), D["M"][tr].std(0) + 1e-6
    Xs, Ms = norm(D, mu, sd, mmu, msd)
    Vt, Rt = torch.tensor(D["V"]), torch.tensor(D["R"], dtype=torch.float32)

    def loss(model, ds):
        x, m, v, r = Xs[ds], Ms[ds], Vt[ds], Rt[ds]
        w = model(x.reshape(-1, x.shape[-2], x.shape[-1]), m.reshape(-1, m.shape[-1]), v.reshape(-1, v.shape[-1])).reshape(v.shape)
        turn = torch.cat([w[:, :1].abs().sum(-1), (w[:, 1:] - w[:, :-1]).abs().sum(-1)], dim=1)
        net = (w * r).sum(-1) - (FEE + SLIP) * turn
        return -(net.mean(1) / (net.std(1) + 1e-4)).mean()
    states = []
    for seed in range(N_SEEDS):
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        model = Alloc(D["X"].shape[2], D["M"].shape[1]); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        best, state, bad = np.inf, None, 0
        for ep in range(MAX_EP):
            model.train(); st = rng.integers(0, len(tr) - WIN, BATCH)
            opt.zero_grad(); l = loss(model, np.stack([tr[s:s + WIN] for s in st])); l.backward(); opt.step()
            if ep % 5 == 4:
                model.eval()
                with torch.no_grad(): vl = float(loss(model, va[None, :]))
                if vl < best - 1e-4: best, bad, state = vl, 0, {k: t.clone() for k, t in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= PAT // 5: break
        states.append(state)
    return {"states": states, "mu": mu, "sd": sd, "mmu": mmu, "msd": msd}


def get_model(D):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    last_known = D["idx"][-3]                                     # dernier jour dont le rendement est connu
    files = sorted(MODEL_DIR.glob("modele_cutoff_*.pt"))
    if files:
        cut = pd.Timestamp(files[-1].stem.split("_")[-1])
        if (last_known - cut).days < RETRAIN_DAYS:
            return torch.load(files[-1], weights_only=False), cut
    hi = int(np.searchsorted(D["idx"], last_known - pd.Timedelta(days=EMBARGO)))
    pack = train(D, hi)
    path = MODEL_DIR / f"modele_cutoff_{last_known.date()}.pt"
    torch.save(pack, path)
    for f in files:                                               # garder seulement le modèle en service
        f.unlink()
    print("Nouveau modèle entraîné :", path.name)
    return pack, last_known


def weights_at(D, pack, i):
    Xs, Ms = norm(D, pack["mu"], pack["sd"], pack["mmu"], pack["msd"])
    v = torch.tensor(D["V"][i:i + 1]); acc = np.zeros(len(D["coins"]))
    for st in pack["states"]:
        m = Alloc(D["X"].shape[2], D["M"].shape[1]); m.load_state_dict(st); m.eval()
        with torch.no_grad(): acc += m(Xs[i:i + 1], Ms[i:i + 1], v).numpy()[0]
    w = pd.Series(acc / len(pack["states"]), index=D["coins"])
    return w if D["trend"][i] else w * 0.0


def _j(o):
    return o.item() if hasattr(o, "item") else str(o)


def short(sym):
    return sym.replace("USDT", "")


def main():
    panel = pd.read_csv(PANEL, parse_dates=["date"])
    D = build(panel); idx = D["idx"]; O, C = D["O"], D["C"]
    pack, cutoff = get_model(D)
    S = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {"cash": CAPITAL, "qty": {}, "historique": [], "courbe": [],
                                                                           "dernier": None, "frais": 0.0, "transactions": 0}
    cost = FEE + SLIP
    # décisions à la clôture du jour d, exécutées à l'ouverture de d+1 (seulement si cette ouverture est connue)
    for i, d in enumerate(idx):
        if d < PAPER_START or d >= PAPER_END or (S["dernier"] and d <= pd.Timestamp(S["dernier"])):
            continue
        if i + 1 >= len(idx) or np.isnan(O.iloc[i + 1]).all():
            break
        w = weights_at(D, pack, i)
        px = O.iloc[i + 1]
        qty = pd.Series(S["qty"], dtype=float).reindex(D["coins"]).fillna(0.0)
        pos = (qty * px).fillna(0.0); equity = S["cash"] + pos.sum()
        target = (w * equity).where(px.notna(), 0.0); delta = target - pos
        fees = delta.abs().sum() * cost; target = target * (1 - fees / equity) if equity > 0 else target
        new_qty = (target / px).fillna(0.0)
        S["cash"] = float(equity - fees - target.sum()); S["qty"] = {k: float(v) for k, v in new_qty.items() if v > 0}
        S["frais"] += float(fees); S["transactions"] += int((delta.abs() > equity * 0.01).sum())
        S["historique"].append({"date": str(d.date()), "filtre": bool(D["trend"][i]),
                                "poids": {short(k): round(float(v) * 100, 1) for k, v in w[w > 0.005].sort_values(ascending=False).items()},
                                "valeur": round(float(equity - fees), 2)})
        S["dernier"] = str(d.date())
    # valorisation à chaque clôture connue
    qty = pd.Series(S["qty"], dtype=float).reindex(D["coins"]).fillna(0.0)
    started = S["dernier"] is not None
    if started:
        last_c = C.dropna(how="all").index[-1]
        val = float(S["cash"] + (qty * C.loc[last_c]).fillna(0.0).sum())
        if not S["courbe"] or S["courbe"][-1][0] != str(last_c.date()):
            S["courbe"].append([str(last_c.date()), round(val, 2)])
        else:
            S["courbe"][-1][1] = round(val, 2)
    else:
        val = CAPITAL
    STATE_PATH.write_text(json.dumps(S, indent=1, default=_j))

    # signal pour demain (décision à la dernière clôture disponible)
    i_last = int(np.where(~np.isnan(C.values).all(axis=1))[0][-1])
    w_now = weights_at(D, pack, i_last)
    px_last = C.iloc[i_last]
    pos_now = (qty * px_last).fillna(0.0); eq_now = S["cash"] + pos_now.sum() if started else CAPITAL
    actions = []
    for sym in D["coins"]:
        tgt = float(w_now[sym] * eq_now); cur = float(pos_now[sym]) if started else 0.0; dlt = tgt - cur
        if abs(dlt) >= 0.5:
            actions.append({"crypto": short(sym), "action": "ACHETER" if dlt > 0 else "VENDRE", "usdt": round(abs(dlt), 2),
                            "prix": float(px_last[sym]), "majeur": abs(dlt) >= MAJOR * eq_now})
    actions.sort(key=lambda a: -a["usdt"])
    btc = D["btc"]; sma = btc.rolling(TREND_SMA).mean()
    curve = S["courbe"]

    def rebase(points):
        if not points or not curve:
            return []
        dd = {p[0]: p[1] for p in points}; dates_ = [c[0] for c in curve]
        base = next((dd[x] for x in dates_ if x in dd), None)
        return [[x, round(dd[x] / base * CAPITAL, 2) if x in dd and base else None] for x in dates_] if base else []
    v4c, btcc = [], []
    if V4_APP.exists():
        app = json.loads(V4_APP.read_text())
        v4c = rebase(app["portefeuilles"]["quotidien"].get("courbe", []))
        btcc = rebase(app.get("btc_courbe", []))

    def sharpe(vals):
        v = np.array([x for x in vals if x is not None], dtype=float)
        if len(v) < 10: return None
        r = np.diff(np.log(v)); return float(r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else 0.0

    def dd(vals):
        v = np.array([x for x in vals if x is not None], dtype=float)
        return float((v / np.maximum.accumulate(v) - 1).min() * 100) if len(v) else 0.0
    vv = [c[1] for c in curve]; v4v = [c[1] for c in v4c]; bv = [c[1] for c in btcc]
    last_day = C.dropna(how="all").index[-1]
    crit = []
    if started and v4v and v4v[-1] and bv and bv[-1]:
        crit = [{"nom": "La v5 rapporte plus que la v4 quotidienne", "ok": vv[-1] > v4v[-1]},
                {"nom": "Meilleur rapport gain / risque (Sharpe) que la v4 quotidienne", "ok": (sharpe(vv) or 0) > (sharpe(v4v) or 0)},
                {"nom": "La v5 rapporte plus que le Bitcoin gardé", "ok": vv[-1] > bv[-1]}]
    n_ok = sum(c["ok"] for c in crit)
    finished = last_day >= PAPER_END - pd.Timedelta(days=1)
    statut = ("validé" if n_ok == 3 else "à prolonger" if n_ok == 2 else "non validé") if finished and crit else ("en cours" if started else "pas commencé")
    out = {
        "genere_le": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "donnees_jusqu_au": str(last_day.date()),
        "test": {"debut": str(PAPER_START.date()), "fin": str(PAPER_END.date()), "capital": CAPITAL, "frais_pct": FEE * 100,
                 "slippage_pct": SLIP * 100, "filtre_jours": TREND_SMA, "modele_cutoff": str(cutoff.date()),
                 "jours_ecoules": max(0, min(182, (last_day - PAPER_START).days)), "jours_total": 182},
        "commence": started,
        "signal": {"date_calcul": str(idx[i_last].date()), "applicable_le": str((idx[i_last] + pd.Timedelta(days=1)).date()),
                   "filtre": bool(D["trend"][i_last]), "btc_vs_moy100": round(float((btc.iloc[i_last] / sma.iloc[i_last] - 1) * 100), 1),
                   "poids": [{"crypto": short(k), "pct": round(float(v) * 100, 1)} for k, v in w_now[w_now > 0.005].sort_values(ascending=False).items()],
                   "actions": actions},
        "valeur": round(val, 2), "rendement_pct": round((val / CAPITAL - 1) * 100, 2), "cash": round(S["cash"], 2) if started else CAPITAL,
        "positions": [{"crypto": short(k), "quantite": float(q), "valeur": round(float(q * C.loc[last_day, k]), 2),
                       "poids": round(float(q * C.loc[last_day, k]) / val * 100, 1)} for k, q in qty.items() if q > 0] if started else [],
        "frais_total": round(S["frais"], 2), "transactions": S["transactions"],
        "pire_baisse_pct": round(dd(vv), 2), "v4_pire_baisse_pct": round(dd(v4v), 2), "btc_pire_baisse_pct": round(dd(bv), 2),
        "v4_rendement_pct": round((v4v[-1] / CAPITAL - 1) * 100, 2) if v4v and v4v[-1] else None,
        "btc_rendement_pct": round((bv[-1] / CAPITAL - 1) * 100, 2) if bv and bv[-1] else None,
        "courbe": curve, "v4_courbe": v4c, "btc_courbe": btcc, "historique": S["historique"][-60:],
        "verdict": {"statut": statut, "criteres": crit,
                    "texte": f"Verdict définitif le {PAPER_END.date()}. 3 critères sur 3 = VALIDÉ, 2 = À PROLONGER, 0 ou 1 = NON VALIDÉ."},
        "backtest": {"v5": 83.7, "v4": 54.9, "btc": 17.6, "periode": "2021–2026"},
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=_j))
    print(f"v5 : {'investi' if out['signal']['filtre'] else 'cash'} | valeur {out['valeur']} | poids {out['signal']['poids'][:4]}")


if __name__ == "__main__":
    main()
