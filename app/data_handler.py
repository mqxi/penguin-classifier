"""
Datenverwaltung für den Pinguin-Klassifikator.

Kümmert sich um das Laden des Trainingsdatensatzes, das Speichern neuer
Beobachtungen und das Zusammenführen beider Quellen für das Retraining.
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Pfade – BASE_DIR ist der Projektordner (eine Ebene über /app)
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
TRAINING_CSV = DATA_DIR / "penguins.csv"
OBSERVATIONS_CSV = DATA_DIR / "new_observations.csv"

# Direkter CSV-Download vom offiziellen palmerpenguins-GitHub
DATASET_URL = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"

FEATURE_COLUMNS = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "island",
    "sex",
]

OBSERVATION_COLUMNS = FEATURE_COLUMNS + ["predicted_species", "corrected_species", "is_corrected", "timestamp"]


def download_dataset() -> None:
    """Lädt den Palmer Penguins Datensatz herunter und speichert ihn lokal."""
    logger.info(f"Lade Palmer Penguins Datensatz herunter von {DATASET_URL} ...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(DATASET_URL, timeout=30)
        response.raise_for_status()

        df = pd.read_csv(pd.io.common.StringIO(response.text))

        # Spaltennamen vereinheitlichen (Leerzeichen raus, alles klein)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Prüfen ob alle erwarteten Spalten da sind
        needed = FEATURE_COLUMNS + ["species"]
        missing_cols = set(needed) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Erwartete Spalten fehlen im Datensatz: {missing_cols}")

        df = df[needed]

        # Geschlechter normalisieren: "male" -> "Male" etc., ungültige Einträge raus
        if "sex" in df.columns:
            df["sex"] = df["sex"].str.strip().str.capitalize()
            df = df[df["sex"].isin(["Male", "Female"]) | df["sex"].isna()]

        df.to_csv(TRAINING_CSV, index=False)
        logger.info(f"Datensatz gespeichert: {TRAINING_CSV} ({len(df)} Zeilen)")

    except Exception as e:
        logger.error(f"Fehler beim Herunterladen des Datensatzes: {e}")
        raise


def reload_dataset() -> None:
    """Löscht den lokalen Datensatz und lädt ihn frisch herunter.

    Nützlich wenn der Quelldatensatz auf GitHub aktualisiert wurde.
    """
    if TRAINING_CSV.exists():
        TRAINING_CSV.unlink()
        logger.info("Lokaler Datensatz gelöscht – lade neu herunter...")
    download_dataset()


def load_training_data() -> pd.DataFrame:
    """Gibt den Trainingsdatensatz zurück. Lädt ihn herunter falls noch nicht vorhanden."""
    if not TRAINING_CSV.exists():
        download_dataset()
    try:
        df = pd.read_csv(TRAINING_CSV)
        logger.info(f"Trainingsdaten geladen: {len(df)} Zeilen")
        return df
    except Exception as e:
        logger.error(f"Fehler beim Laden der Trainingsdaten: {e}")
        raise


def load_observations() -> pd.DataFrame:
    """Gibt die bisher gespeicherten neuen Beobachtungen zurück.

    Erstellt die CSV-Datei mit den richtigen Spalten, falls sie noch nicht existiert.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not OBSERVATIONS_CSV.exists():
        pd.DataFrame(columns=OBSERVATION_COLUMNS).to_csv(OBSERVATIONS_CSV, index=False)
        return pd.DataFrame(columns=OBSERVATION_COLUMNS)
    try:
        df = pd.read_csv(OBSERVATIONS_CSV)
        if "is_corrected" in df.columns:
            df["is_corrected"] = df["is_corrected"].fillna(False).astype(bool)
        if "corrected_species" in df.columns:
            df["corrected_species"] = df["corrected_species"].fillna("").astype(str)
        return df
    except Exception as e:
        logger.error(f"Fehler beim Laden der Beobachtungen: {e}")
        return pd.DataFrame(columns=OBSERVATION_COLUMNS)


def save_observation(input_dict: dict, predicted_species: str, corrected_species: str | None = None) -> None:
    """Hängt einen neuen klassifizierten Datenpunkt an new_observations.csv an."""
    try:
        new_row = {
            **input_dict,
            "predicted_species": predicted_species,
            "corrected_species": corrected_species if corrected_species else "",
            "is_corrected": corrected_species is not None,
            "timestamp": datetime.now().isoformat(),
        }
        df_existing = load_observations()
        df_updated = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)
        df_updated.to_csv(OBSERVATIONS_CSV, index=False)
        logger.info(f"Neue Beobachtung gespeichert ({predicted_species}{f' → korrigiert: {corrected_species}' if corrected_species else ''})")
    except Exception as e:
        logger.error(f"Fehler beim Speichern der Beobachtung: {e}")
        raise


def correct_last_observation(corrected_species: str) -> None:
    """Setzt corrected_species und is_corrected für die zuletzt gespeicherte Beobachtung."""
    try:
        df = load_observations()
        if df.empty:
            raise ValueError("Keine Beobachtungen vorhanden.")
        df.loc[df.index[-1], "corrected_species"] = corrected_species
        df.loc[df.index[-1], "is_corrected"] = True
        df.to_csv(OBSERVATIONS_CSV, index=False)
        logger.info(f"Letzte Beobachtung korrigiert: {corrected_species}")
    except Exception as e:
        logger.error(f"Fehler beim Korrigieren der Beobachtung: {e}")
        raise


def count_species_samples(species: str) -> int:
    """Zählt wie viele Beobachtungen mit diesem Artlabel (korrigiert oder predicted) gespeichert sind."""
    df_obs = load_observations()
    if df_obs.empty:
        return 0
    corrected = df_obs[df_obs["is_corrected"] == True]["corrected_species"]
    predicted = df_obs[df_obs["is_corrected"] != True]["predicted_species"]
    all_labels = pd.concat([corrected, predicted])
    return int((all_labels == species).sum())


def get_combined_data() -> pd.DataFrame:
    """Kombiniert Trainingsdaten und neue Beobachtungen für das Retraining."""
    df_train = load_training_data()
    df_obs = load_observations()

    if df_obs.empty:
        return df_train

    # Korrigiertes Label hat Vorrang; sonst predicted_species
    df_obs = df_obs.copy()
    if "is_corrected" in df_obs.columns and "corrected_species" in df_obs.columns:
        df_obs["species"] = df_obs.apply(
            lambda r: r["corrected_species"] if r["is_corrected"] and r["corrected_species"] else r.get("predicted_species", ""),
            axis=1,
        )
    elif "predicted_species" in df_obs.columns:
        df_obs["species"] = df_obs["predicted_species"]

    cols = FEATURE_COLUMNS + ["species"]
    df_obs = df_obs[[c for c in cols if c in df_obs.columns]]

    df_combined = pd.concat([df_train, df_obs], ignore_index=True)
    logger.info(f"Kombinierter Datensatz: {len(df_combined)} Zeilen")
    return df_combined
