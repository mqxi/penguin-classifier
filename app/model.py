"""
ML-Pipeline für den Pinguin-Klassifikator.

Enthält alles rund ums Modell: Aufbau der sklearn-Pipeline, Training,
Evaluation und Vorhersage. Das trainierte Modell wird als .pkl gespeichert
damit es beim nächsten Start nicht neu trainiert werden muss.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "model" / "classifier.pkl"

NUMERIC_FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CATEGORICAL_FEATURES = ["island", "sex"]
TARGET = "species"
SPECIES_CLASSES = ["Adelie", "Chinstrap", "Gentoo"]


def _build_pipeline() -> Pipeline:
    """Baut die vollständige sklearn-Pipeline auf.

    Numerische Features: Median-Imputation + Standardisierung
    Kategorische Features: Modus-Imputation + One-Hot-Encoding
    Klassifikator: Random Forest mit class_weight='balanced' wegen ungleicher Klassenverteilung
    """
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_transformer, NUMERIC_FEATURES),
        ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",  # wichtig wegen Adelie-Überrepräsentation
        )),
    ])
    return pipeline


def train_model(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    """Trainiert das Modell und gibt Pipeline + Evaluationsmetriken zurück."""
    df_clean = df.dropna(subset=[TARGET]).copy()
    # Klassen mit weniger als 2 Samples rausfiltern – stratifizierter Split braucht mind. 2
    class_counts = df_clean[TARGET].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    dropped = set(class_counts[class_counts < 2].index)
    if dropped:
        logger.warning(f"Klassen mit zu wenig Samples für Training ignoriert: {dropped}")
    df_clean = df_clean[df_clean[TARGET].isin(valid_classes)]

    classes = sorted(df_clean[TARGET].unique().tolist())

    X = df_clean[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df_clean[TARGET]

    logger.info(f"Starte Training mit {len(X)} Datenpunkten, Klassen: {classes}")

    # 80/20 Split, stratifiziert damit alle Klassen im Test-Set vertreten sind
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluation auf dem Test-Set
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    # Zusätzlich 5-fache Kreuzvalidierung für stabilere Schätzung
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")

    metrics = {
        "accuracy": round(acc, 4),
        "f1_weighted": round(f1, 4),
        "confusion_matrix": cm,
        "cv_mean": round(cv_scores.mean(), 4),
        "cv_std": round(cv_scores.std(), 4),
        "n_train": len(X_train),
        "n_total": len(X),
    }

    logger.info(
        f"Training abgeschlossen – Accuracy: {acc:.4f} | F1: {f1:.4f} | CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
    )
    return pipeline, metrics


def save_model(pipeline: Pipeline) -> None:
    """Speichert die trainierte Pipeline als classifier.pkl."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    logger.info(f"Modell gespeichert unter {MODEL_PATH}")


def load_model() -> Pipeline | None:
    """Lädt die gespeicherte Pipeline, falls vorhanden. Gibt None zurück wenn nicht."""
    if not MODEL_PATH.exists():
        return None
    try:
        pipeline = joblib.load(MODEL_PATH)
        logger.info(f"Modell geladen von {MODEL_PATH}")
        return pipeline
    except Exception as e:
        logger.error(f"Fehler beim Laden des Modells: {e}")
        return None


def get_or_train_model(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    """Lädt das Modell falls vorhanden, trainiert es sonst neu.

    Im Fall eines geladenen Modells werden die Metriken auf den
    übergebenen Daten neu berechnet (ohne Retraining).
    """
    pipeline = load_model()

    if pipeline is not None:
        # Metriken neu berechnen ohne das Modell neu zu trainieren
        df_clean = df.dropna(subset=[TARGET]).copy()
        known_classes = list(pipeline.classes_)
        df_clean = df_clean[df_clean[TARGET].isin(known_classes)]
        X = df_clean[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        y = df_clean[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        cm = confusion_matrix(y_test, y_pred, labels=known_classes)
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")

        metrics = {
            "accuracy": round(acc, 4),
            "f1_weighted": round(f1, 4),
            "confusion_matrix": cm,
            "cv_mean": round(cv_scores.mean(), 4),
            "cv_std": round(cv_scores.std(), 4),
            "n_train": len(X_train),
            "n_total": len(X),
        }
        return pipeline, metrics

    # Kein gespeichertes Modell gefunden – neu trainieren
    pipeline, metrics = train_model(df)
    save_model(pipeline)
    return pipeline, metrics


def retrain_model(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    """Trainiert das Modell auf den übergebenen Daten neu und überschreibt die .pkl."""
    pipeline, metrics = train_model(df)
    save_model(pipeline)
    return pipeline, metrics


def predict(input_dict: dict) -> dict:
    """Klassifiziert einen einzelnen Datenpunkt und gibt Art + Wahrscheinlichkeiten zurück.

    Args:
        input_dict: Die sechs Features als Dict (bill_length_mm, bill_depth_mm,
                    flipper_length_mm, body_mass_g, island, sex).

    Returns:
        Dict mit 'species' (str), 'confidence' (float) und 'probabilities' (dict).
    """
    pipeline = load_model()
    if pipeline is None:
        raise RuntimeError("Kein trainiertes Modell gefunden. Bitte zuerst trainieren.")

    df_input = pd.DataFrame([input_dict])

    try:
        proba = pipeline.predict_proba(df_input)[0]
        classes = pipeline.classes_

        prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}

        predicted_species = classes[np.argmax(proba)]
        confidence = float(np.max(proba))

        return {
            "species": predicted_species,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
        }

    except Exception as e:
        logger.error(f"Vorhersage fehlgeschlagen: {e}")
        raise
