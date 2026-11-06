"""Fabrique le zip « ia_signaux » à donner : code + tous les modèles entraînés (aucun entraînement ici).

    python outils/fabriquer_paquet.py            -> dist/ia_signaux.zip
"""
import json, shutil, sys, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
DIST = REPO / "dist"; BUILD = DIST / "build"
sys.path.insert(0, str(REPO))
from ia_signaux.commun import AE, WINDOW, bougies_4h   # noqa: E402


def normalisation_matieres():
    """Moyenne / écart de l'autoencodeur sur la période d'apprentissage (avant 2017), comme le moteur matières."""
    base = pd.read_csv(REPO / "matieres" / "donnees" / "horaire_base.csv.gz", parse_dates=["ts"])
    for m in ["SI", "LE", "HG"]:
        dest = REPO / "matieres" / "modeles" / m / "normalisation.json"
        if dest.exists():
            continue
        h = base[base["root"] == m].copy()
        for c in ["open", "high", "low", "close", "volume"]:
            h[c] = pd.to_numeric(h[c], errors="coerce")
        b = bougies_4h(h.dropna(subset=["open", "close"]), instrument=True)
        v = b["vol_30"].values; lv = np.log1p(b["vol"].values)
        lvz = (lv - pd.Series(lv).rolling(180, min_periods=60).mean().values) / pd.Series(lv).rolling(180, min_periods=60).std().values
        F = np.stack([np.clip(b["r"].values / v, -8, 8), np.clip(b["rng"].values / v, 0, 20) / 4, np.nan_to_num(np.clip(lvz, -5, 5))], axis=1).astype(np.float32)
        idx = np.arange(WINDOW, len(b)); t = b.index[idx]
        WIN = np.stack([F[i - WINDOW + 1:i + 1] for i in idx])
        ae = AE(); ae.load_state_dict(torch.load(REPO / "matieres" / "modeles" / m / "autoencodeur.pt", map_location="cpu")); ae.eval()
        with torch.no_grad():
            Z = np.concatenate([ae(torch.tensor(WIN[i:i + 4096]))[0].numpy() for i in range(0, len(WIN), 4096)])
        tm = np.asarray(t < "2017-01-01")
        dest.write_text(json.dumps({"moyenne": Z[tm].mean(0).tolist(), "ecart": Z[tm].std(0).tolist(), "fenetres": int(tm.sum())}))
        print("normalisation", m, int(tm.sum()), "fenêtres")


def main():
    normalisation_matieres()
    if BUILD.exists():
        shutil.rmtree(BUILD)
    pk = BUILD / "ia_signaux"
    shutil.copytree(REPO / "ia_signaux", pk, ignore=shutil.ignore_patterns("__pycache__", "modeles"))
    M = pk / "modeles"
    v4 = sorted((REPO / "data" / "modeles").glob("modele_cutoff_*.pt"))[-1]
    v5 = sorted((REPO / "crypto_v5" / "modeles").glob("modele_cutoff_*.pt"))[-1]
    (M / "v4").mkdir(parents=True); shutil.copy(v4, M / "v4")
    (M / "v5").mkdir(parents=True); shutil.copy(v5, M / "v5")
    for d in (REPO / "crypto_v2" / "modeles").iterdir():
        dd = M / "crypto_v2" / d.name; dd.mkdir(parents=True)
        for f in d.iterdir():
            shutil.copy(f, dd)
    for d in (REPO / "matieres" / "modeles").iterdir():
        dd = M / "matieres" / d.name; dd.mkdir(parents=True)
        for f in d.iterdir():
            shutil.copy(f, dd)
    for f in ["GUIDE.md", "exemple.py", "requirements.txt", "verifier.py"]:
        shutil.copy(REPO / "ia_signaux_docs" / f, BUILD / f)
    out = DIST / "ia_signaux.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(BUILD.rglob("*")):
            if f.is_file():
                z.write(f, Path("ia_signaux_paquet") / f.relative_to(BUILD))
    print(out, round(out.stat().st_size / 1e6, 1), "Mo")


if __name__ == "__main__":
    main()
