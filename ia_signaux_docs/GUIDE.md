# ia_signaux : guide d'intégration

Ce paquet contient **tous les modèles déjà entraînés** du projet « Signal IA », et le code qui calcule leur signal du jour.
Il n'entraîne rien : il charge les modèles et lit les données de marché publiques.

> ⚠️ **Outil de recherche, pas un conseil financier.** Ces modèles sont **encore en test en argent fictif** (verdicts
> en avril 2027). Les résultats passés ne garantissent rien. Teste d'abord en argent fictif dans ton application.

---

## 1. Ce qu'il y a dans le paquet

```
ia_signaux_paquet/
├── ia_signaux/                 ← le paquet Python à copier dans ton projet
│   ├── __init__.py             ← les fonctions à appeler
│   ├── journalier.py           ← crypto v4 et v5 (bougies journalières)
│   ├── crypto_v2.py            ← robots crypto v2
│   ├── matieres.py             ← robots matières premières
│   ├── commun.py               ← données Binance, autoencodeur, indicateurs
│   ├── mise_a_jour.py          ← télécharge les nouveaux modèles v4 / v5
│   ├── serveur.py              ← serveur web (pour une application non Python)
│   └── modeles/                ← LES MODÈLES ENTRAÎNÉS
│       ├── v4/modele_cutoff_….pt        5 réseaux (PyTorch) + normalisation
│       ├── v5/modele_cutoff_….pt        5 réseaux (PyTorch) + normalisation
│       ├── crypto_v2/<CRYPTO>/          3 robots PPO par version (.zip) + autoencodeur.pt + normalisation.json
│       └── matieres/<SI|LE|HG>/         3 robots PPO (.zip) + autoencodeur.pt + normalisation.json
├── exemple.py                  ← affiche tous les signaux du jour
├── verifier.py                 ← compare avec l'application officielle
└── requirements.txt
```

| Modèle | Ce qu'il fait | Marchés | Rythme | Données |
|---|---|---|---|---|
| **v4** | Classe 16 cryptos (hausse attendue sur 7 jours). Si Bitcoin est au-dessus de sa moyenne 100 jours : **top 3 à 33 %** chacune, sinon **100 % USDT** | 16 cryptos Binance | 1 fois par jour, après la clôture de 00:00 UTC | Binance, gratuit |
| **v5** | L'IA donne directement le **% à mettre sur chaque crypto**. Même filtre Bitcoin | 16 cryptos | 1 fois par jour | Binance, gratuit |
| **Crypto v2, achat seulement** | 3 robots votent **ACHETER** ou **DEHORS** | XRP, XLM, SOL, DOT, AVAX | 1 fois par jour (bougie 00:00–04:00 UTC) | Binance, gratuit |
| **Crypto v2, achat et vente** | 3 robots votent **ACHETER**, **DEHORS** ou **VENDRE** (pari à la baisse) | ETH, XRP, ADA, DOGE, SOL, LTC, DOT, AVAX | 1 fois par jour | Binance, gratuit |
| **Matières premières** | 3 robots votent **ACHETER** ou **DEHORS** | Argent (SI), bovins (LE), cuivre (HG) | toutes les 6 bougies de 4 h (environ 1 fois par jour de bourse) | Databento, payant (quelques centimes) |

---

## 2. Installation (5 minutes)

Python **3.10 à 3.12**.

```bash
cd ia_signaux_paquet
pip install torch --index-url https://download.pytorch.org/whl/cpu    # version légère, sans carte graphique
pip install -r requirements.txt
python exemple.py           # affiche les signaux du jour
python verifier.py          # vérifie qu'ils sont identiques à l'application officielle
```

Pour l'utiliser dans ton projet, copie le dossier `ia_signaux/` (avec `modeles/`) à côté de ton code.

---

## 3. Utilisation en Python

```python
from ia_signaux import signal_v4, signal_v5, signal_crypto_v2, signaux_crypto_v2, signal_matiere

# Crypto v4 : top 3
s = signal_v4()
s["portefeuille"]          # {'BNBUSDT': 33.33, 'BTCUSDT': 33.33, 'TRXUSDT': 33.33}  ou  {'USDT': 100.0}
s["classement"]            # les 16 cryptos triées par score de l'IA
s["filtre_bitcoin_positif"]

# Crypto v5 : répartition
signal_v5()["portefeuille"]     # {'BTCUSDT': 21.4, 'ETHUSDT': 17.5, ...}  ou  {'USDT': 100.0}

# Crypto v2 : une crypto
signal_crypto_v2("XRPUSDT", "achat_seul")
# {'signal': 'ACHETER', 'position': 1, 'votes': [0, 1, 1], 'change': False,
#  'bougie_de_decision': '2026-11-06 00:00:00', 'prix': 1.2072, 'prochaine_decision': '2026-11-07 04:00:00', ...}

# Crypto v2 : toutes les cryptos d'une version, avec la part conseillée (parts égales)
signaux_crypto_v2("achat_vente")

# Matières premières (clé Databento nécessaire)
import os; os.environ["DATABENTO_API_KEY"] = "ta_clé"      # ne jamais l'écrire dans le code publié
signal_matiere("SI")
```

### La position des robots (important pour Crypto v2 et les matières premières)

Les robots tiennent compte de leur position actuelle (déjà acheté ou non) pour voter. Deux façons de faire :

1. **Ne rien donner** (par défaut) : la fonction **rejoue tous les votes depuis le début du test officiel**, donc elle
   donne exactement le même signal que l'application.
2. **Donner ta position réelle**, par exemple `signal_crypto_v2("XRPUSDT", "achat_seul", position=1)` si tu détiens déjà
   du XRP. C'est le bon choix si ton application gère ses propres positions.

### Quand appeler les fonctions ?

| Modèle | Moment conseillé (UTC) | Exécution de l'ordre |
|---|---|---|
| v4 / v5 | après 00:30 UTC | au prix d'ouverture du jour |
| Crypto v2 | après 04:05 UTC | juste après le vote (prix de la bougie de 04:00) |
| Matières | le soir, après 22:00 UTC (les données CME arrivent avec plusieurs heures de retard) | au prochain prix disponible |

Appeler plus souvent ne change rien : la décision ne change qu'une fois par jour.

---

## 4. Application non Python (JavaScript, Android, iOS, PHP…) : le serveur web

Le plus simple est de faire tourner les modèles sur un petit serveur Python et de lire le résultat en JSON :

```bash
pip install fastapi uvicorn
uvicorn ia_signaux.serveur:app --host 0.0.0.0 --port 8000
```

| Adresse | Réponse |
|---|---|
| `GET /v4` | classement + portefeuille v4 |
| `GET /v5` | répartition v5 |
| `GET /crypto_v2/achat_seul` | les 5 signaux « achat seulement » |
| `GET /crypto_v2/achat_vente` | les 8 signaux « achat et vente » |
| `GET /crypto_v2/achat_seul/XRPUSDT?position=1` | une crypto, avec ta position |
| `GET /matieres/SI` | argent (clé Databento sur le serveur) |
| `GET /docs` | page de test interactive |

Exemple JavaScript :
```js
const r = await fetch("http://MON_SERVEUR:8000/crypto_v2/achat_seul");
const signaux = await r.json();      // [{symbole: "XRPUSDT", signal: "ACHETER", votes: [0,1,1], ...}, ...]
```

Pour héberger le serveur, n'importe quel petit serveur Linux suffit (1 Go de mémoire), par exemple un VPS, Render ou Railway.
Le premier appel prend environ 10 à 30 secondes (chargement des modèles et des données), les suivants sont rapides.

---

## 5. Mettre à jour les modèles

- **v4** est réentraîné tous les **28 jours** et **v5** tous les **3 mois** sur le dépôt GitHub du projet. Pour récupérer
  les nouveaux fichiers :
  ```python
  from ia_signaux import mettre_a_jour_modeles
  mettre_a_jour_modeles()          # télécharge seulement les modèles qui manquent
  ```
  Le paquet utilise toujours le modèle le plus récent de son dossier.
- **Crypto v2** et **matières premières** ne changent pas pendant le test.

---

## 6. Règles pour passer de vrais ordres (si tu le fais)

- Binance refuse les ordres de moins de **5 USDT**.
- Frais à prévoir : environ **0,1 %** par ordre, plus le glissement de prix.
- « VENDRE » dans Crypto v2 achat et vente = **vente à découvert**. C'est impossible sur Binance au comptant : il faut les
  contrats à terme (futures), très risqués. Les résultats publiés sont fictifs.
- Les montants de l'application officielle sont calculés pour 1 000 $. Applique les **pourcentages** à ton propre capital.

---

## 7. Résultats à connaître (honnêtement)

- Test historique 2024–2026 des robots Crypto v2 « achat seulement » : 5 cryptos validées sur 15, mais **aucun avantage
  en moyenne** sur les 15. XRP est le meilleur, peut-être par chance.
- « Achat et vente » a surtout gagné en pariant à la baisse sur des altcoins qui chutaient : cela dépend beaucoup de la
  période.
- v5 : son avantage sur la v4 dans l'historique n'est pas prouvé statistiquement (t = 1,52).
- Les résultats en direct (argent fictif) sont dans l'application : https://houssam78998.github.io/ia-crypto-signal/
- Code complet, moteurs et rapports : https://github.com/houssam78998/ia-crypto-signal
