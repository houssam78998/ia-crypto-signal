"""Télécharge les derniers modèles v4 et v5 depuis le dépôt GitHub public (ils sont réentraînés sur GitHub :
v4 tous les 28 jours, v5 tous les 3 mois). Les robots Crypto v2 et matières premières ne changent pas."""
import requests
from .commun import MODELES

DEPOT = "houssam78998/ia-crypto-signal"
DOSSIERS = {"v4": "data/modeles", "v5": "crypto_v5/modeles"}


def mettre_a_jour_modeles():
    faits = []
    for nom, chemin in DOSSIERS.items():
        r = requests.get(f"https://api.github.com/repos/{DEPOT}/contents/{chemin}", timeout=30)
        r.raise_for_status()
        for f in r.json():
            if f["name"].startswith("modele_cutoff_") and f["name"].endswith(".pt"):
                dest = MODELES / nom / f["name"]
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(requests.get(f["download_url"], timeout=60).content)
                    faits.append(f"{nom}/{f['name']}")
    return faits or ["déjà à jour"]
