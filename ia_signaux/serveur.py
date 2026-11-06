"""Petit serveur web pour utiliser les modèles depuis n'importe quelle application (JavaScript, Android, iOS, PHP…).

    pip install fastapi uvicorn
    uvicorn ia_signaux.serveur:app --host 0.0.0.0 --port 8000

Puis, depuis l'application : GET http://ADRESSE:8000/crypto_v2/achat_seul  (réponse en JSON)
"""
from fastapi import FastAPI, HTTPException
from . import signal_v4, signal_v5, signal_crypto_v2, signaux_crypto_v2, signal_matiere

app = FastAPI(title="ia_signaux", version="1.0.0")


def _ok(f, *a, **k):
    try:
        return f(*a, **k)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))


@app.get("/v4")
def v4():
    return _ok(signal_v4)


@app.get("/v5")
def v5():
    return _ok(signal_v5)


@app.get("/crypto_v2/{version}")
def crypto_v2_tout(version: str):
    return _ok(signaux_crypto_v2, version)


@app.get("/crypto_v2/{version}/{symbole}")
def crypto_v2_un(version: str, symbole: str, position: int | None = None):
    return _ok(signal_crypto_v2, symbole.upper(), version, position)


@app.get("/matieres/{marche}")
def matiere(marche: str, position: int | None = None):
    return _ok(signal_matiere, marche.upper(), position)
