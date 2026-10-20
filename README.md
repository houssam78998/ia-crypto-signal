# Signal IA (crypto + matières premières)

Application mobile (Android et iPhone) qui affiche chaque jour le signal de l'IA crypto v4. Elle suit deux portefeuilles **fictifs** de 1 000 USDT pendant 26 semaines : un avec un signal quotidien, l'autre avec un signal hebdomadaire.

**Argent fictif. Aucun ordre réel n'est passé. Outil de recherche, pas un conseil financier.**

## Fonctionnement
- Chaque nuit à 00:40 UTC, GitHub Actions lance `engine/run.py`, qui :
  1. met à jour les bougies journalières Binance de 16 cryptos ;
  2. calcule le classement de l'IA (mêmes indicateurs et même modèle que la v4 testée en walk-forward, réentraîné tous les 28 jours, uniquement sur le passé) ;
  3. met à jour les deux portefeuilles fictifs (frais 0,10 % + glissement 0,05 %) ;
  4. écrit `docs/data/app.json`.
- L'application (`docs/`) est publiée par GitHub Pages et lit ce fichier.

## Installation (une seule fois)
1. **Settings → Pages** : *Source* = « Deploy from a branch », *Branch* = `main`, dossier `/docs`, puis **Save**.
2. **Actions** : si un bandeau le demande, cliquez sur « I understand my workflows, go ahead and enable them ». Ouvrez ensuite **Signal quotidien → Run workflow**. Le premier lancement télécharge l'historique et prend 5 à 15 minutes.
3. Sur le téléphone, ouvrez `https://VOTRE-NOM.github.io/ia-crypto-signal/`, puis :
   - **Android (Chrome)** : menu ⋮ → **Ajouter à l'écran d'accueil** (ou « Installer l'application ») ;
   - **iPhone (Safari)** : bouton Partager → **Sur l'écran d'accueil**.

## Calendrier du test
- Début : clôture du dimanche 18 octobre 2026. Le premier signal apparaît le lundi 19 au matin.
- Fin : dimanche 18 avril 2027 (26 semaines), avec le verdict VALIDÉ / À PROLONGER / NON VALIDÉ selon 3 critères fixés à l'avance.

## À savoir
- Le bouton « C'est fait » est enregistré seulement sur le téléphone où vous l'avez touché.
- Si GitHub désactive la tâche automatique, faute d'activité sur le dépôt, il envoie un e-mail. Il suffit alors de cliquer sur « Enable workflow » dans l'onglet Actions.
- Ne modifiez pas les réglages de `engine/run.py` pendant les 26 semaines.

## Matières premières (ajouté le 20 octobre 2026)
- Onglet **« Matières premières »** en haut de l'application : argent, bovins et cuivre, **1 000 $ fictifs chacun**.
- Robots v11 (apprentissage par renforcement + autoencodeur), 3 par marché, décision à la majorité : **ACHETER** ou **DEHORS** (jamais de vente à découvert).
- Chaque soir de semaine à 22:30 UTC, `.github/workflows/matieres.yml` lance `matieres/run.py` (données CME Databento, secret `DATABENTO_API_KEY`) et écrit `docs/data/matieres.json`.
- Test du 20 octobre 2026 au 20 avril 2027. Verdict : VALIDÉ (3 critères sur 3), À PROLONGER (2), NON VALIDÉ (0 ou 1).
- Le moteur crypto (`engine/run.py`) et son test ne sont pas modifiés.

## Crypto v5 — répartition choisie par l'IA (ajouté le 20 octobre 2026)
- Bouton **« v5 Répartition »** dans la partie Crypto. Portefeuille fictif séparé de 1 000 USDT, du 20 octobre 2026 au 20 avril 2027.
- Le filtre Bitcoin décide investi / cash (comme la v4) ; 5 réseaux de neurones donnent directement les poids des 16 cryptos (entraînés pour maximiser le rapport gain / risque après frais).
- `.github/workflows/crypto_v5.yml` lance `crypto_v5/run.py` chaque jour à 01:20 UTC. Il lit `data/panel.csv.gz` (mis à jour par la v4) sans le modifier. Réentraînement tous les 3 mois.
- Verdict : la v5 doit battre la v4 quotidienne (rendement et Sharpe) et le Bitcoin. Le test v4 n'est pas modifié.
