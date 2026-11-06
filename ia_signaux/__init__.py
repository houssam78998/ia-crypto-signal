"""ia_signaux — modèles déjà entraînés du projet « Signal IA » (crypto v4, v5, v2 et matières premières).

    from ia_signaux import signal_v4, signal_v5, signal_crypto_v2, signaux_crypto_v2, signal_matiere

Aucun entraînement : ce paquet charge seulement les modèles et calcule le signal du jour.
Outil de recherche, pas un conseil financier. Les modèles sont encore en test (verdicts en avril 2027).
"""
from .journalier import signal_v4, signal_v5, panel_journalier
from .crypto_v2 import signal_crypto_v2, signaux_crypto_v2, VERSIONS as VERSIONS_CRYPTO_V2
from .matieres import signal_matiere, MARCHES
from .mise_a_jour import mettre_a_jour_modeles

__version__ = "1.0.0"
__all__ = ["signal_v4", "signal_v5", "signal_crypto_v2", "signaux_crypto_v2", "signal_matiere", "mettre_a_jour_modeles",
           "panel_journalier", "VERSIONS_CRYPTO_V2", "MARCHES"]
