"""
Einstiegspunkt der Dash-Applikation.

Beim Start wird zuerst geprüft ob der Datensatz und ein trainiertes Modell
vorhanden sind. Falls nicht, werden sie automatisch heruntergeladen bzw. trainiert.
Danach startet die eigentliche Web-App.
"""

import logging
import sys
from pathlib import Path

# app/ zum Suchpfad hinzufügen damit die Imports in den Modulen funktionieren
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "app"))

import dash
import dash_bootstrap_components as dbc

from callbacks import register_callbacks
from data_handler import load_training_data
from layout import build_layout
from model import get_or_train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def initialize_app() -> dict:
    """Lädt Daten und Modell beim Start.

    Beim ersten Aufruf: Datensatz herunterladen + Modell trainieren.
    Danach: vorhandene Dateien laden.
    """
    logger.info("Initialisierung: Lade Trainingsdaten...")
    df_train = load_training_data()

    logger.info("Initialisierung: Lade/trainiere Modell...")
    _, metrics = get_or_train_model(df_train)

    logger.info(f"Initialisierung abgeschlossen. Accuracy: {metrics['accuracy']:.4f}")
    return metrics


def create_app() -> dash.Dash:
    """Erstellt die Dash-App mit Layout und Callbacks."""
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title="Pinguin-Klassifikator",
        suppress_callback_exceptions=True,
    )
    app.layout = build_layout()
    register_callbacks(app)
    return app


if __name__ == "__main__":
    initialize_app()
    app = create_app()
    logger.info("App startet auf http://0.0.0.0:8050")
    app.run(host="0.0.0.0", port=8050, debug=False)
