"""Unit-Tests für ML-Pipeline: predict() und retrain_model()."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Projektpfad hinzufügen
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    """Erzeugt Trainings-DataFrame mit 30 Zeilen (10 je Klasse).

    Mindestgröße für stratifizierten 80/20-Split und 5-fache Kreuzvalidierung.
    """
    # 10× Adelie (Torgersen, kurzer Schnabel, flache Tiefe)
    adelie = {
        "bill_length_mm": [39.1, 38.2, 40.1, 42.0, 37.8, 41.1, 38.6, 39.5, 40.3, 41.8],
        "bill_depth_mm":  [18.7, 18.1, 17.5, 17.0, 18.3, 17.6, 18.0, 17.9, 18.2, 17.3],
        "flipper_length_mm": [181, 185, 183, 188, 180, 184, 182, 186, 181, 187],
        "body_mass_g":    [3750, 3900, 3700, 4000, 3650, 3800, 3725, 3850, 3775, 3950],
        "island":  ["Torgersen"] * 10,
        "sex":     ["Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"],
        "species": ["Adelie"] * 10,
    }
    # 10× Chinstrap (Dream, mittlerer Schnabel)
    chinstrap = {
        "bill_length_mm": [46.5, 45.0, 48.0, 47.2, 46.1, 49.0, 45.8, 47.5, 46.9, 48.3],
        "bill_depth_mm":  [17.9, 18.0, 17.3, 18.2, 17.6, 18.5, 17.1, 18.4, 17.8, 18.1],
        "flipper_length_mm": [192, 195, 190, 193, 191, 196, 189, 194, 192, 197],
        "body_mass_g":    [3800, 4200, 3600, 3750, 3650, 4100, 3500, 3900, 3700, 4050],
        "island":  ["Dream"] * 10,
        "sex":     ["Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male"],
        "species": ["Chinstrap"] * 10,
    }
    # 10× Gentoo (Biscoe, langer Schnabel, großer Körper)
    gentoo = {
        "bill_length_mm": [50.0, 51.3, 48.7, 49.8, 52.1, 50.5, 49.2, 51.8, 50.9, 48.4],
        "bill_depth_mm":  [15.2, 16.0, 15.1, 14.9, 15.8, 15.5, 14.7, 16.2, 15.3, 14.5],
        "flipper_length_mm": [210, 211, 209, 212, 208, 213, 207, 214, 210, 206],
        "body_mass_g":    [5200, 5500, 5100, 5350, 5450, 5250, 5000, 5600, 5150, 4950],
        "island":  ["Biscoe"] * 10,
        "sex":     ["Male", "Male", "Male", "Female", "Male", "Female", "Female", "Male", "Female", "Female"],
        "species": ["Gentoo"] * 10,
    }
    frames = [pd.DataFrame(adelie), pd.DataFrame(chinstrap), pd.DataFrame(gentoo)]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def trained_model(sample_df, tmp_path_factory):
    """Trainiert Modell auf Sample-Daten und gibt Pipeline zurück."""
    from model import save_model, train_model
    import model as m

    tmp_dir = tmp_path_factory.mktemp("model")
    m.MODEL_PATH = tmp_dir / "classifier.pkl"

    pipeline, metrics = train_model(sample_df)
    save_model(pipeline)
    return pipeline, metrics


class TestRetrain:
    """Tests für retrain_model()."""

    def test_retrain_returns_pipeline_and_metrics(self, sample_df, tmp_path):
        """retrain_model gibt Pipeline und Metriken zurück."""
        from model import retrain_model
        import model as m
        m.MODEL_PATH = tmp_path / "classifier.pkl"

        pipeline, metrics = retrain_model(sample_df)
        assert pipeline is not None
        assert "accuracy" in metrics
        assert "f1_weighted" in metrics
        assert "n_train" in metrics

    def test_retrain_accuracy_range(self, sample_df, tmp_path):
        """Accuracy liegt zwischen 0 und 1."""
        from model import retrain_model
        import model as m
        m.MODEL_PATH = tmp_path / "classifier.pkl"

        _, metrics = retrain_model(sample_df)
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_retrain_saves_model_file(self, sample_df, tmp_path):
        """Modell wird als Datei gespeichert."""
        from model import retrain_model
        import model as m
        m.MODEL_PATH = tmp_path / "classifier.pkl"

        retrain_model(sample_df)
        assert (tmp_path / "classifier.pkl").exists()

    def test_retrain_confusion_matrix_shape(self, sample_df, tmp_path):
        """Konfusionsmatrix hat Form (3, 3)."""
        from model import retrain_model
        import model as m
        m.MODEL_PATH = tmp_path / "classifier.pkl"

        _, metrics = retrain_model(sample_df)
        cm = metrics["confusion_matrix"]
        assert cm.shape == (3, 3)


class TestPredict:
    """Tests für predict()."""

    def test_predict_returns_correct_keys(self, trained_model):
        """predict() gibt dict mit species, confidence, probabilities zurück."""
        from model import predict
        result = predict({
            "bill_length_mm": 46.0, "bill_depth_mm": 17.8,
            "flipper_length_mm": 195.0, "body_mass_g": 3500.0,
            "island": "Dream", "sex": "Female",
        })
        assert "species" in result
        assert "confidence" in result
        assert "probabilities" in result

    def test_predict_species_is_valid(self, trained_model):
        """Vorhergesagte Art ist eine der trainierten Klassen."""
        from model import predict
        pipeline, _ = trained_model
        result = predict({
            "bill_length_mm": 50.0, "bill_depth_mm": 15.5,
            "flipper_length_mm": 210.0, "body_mass_g": 5200.0,
            "island": "Biscoe", "sex": "Male",
        })
        assert result["species"] in list(pipeline.classes_)

    def test_predict_confidence_range(self, trained_model):
        """Konfidenz liegt zwischen 0 und 1."""
        from model import predict
        result = predict({
            "bill_length_mm": 39.0, "bill_depth_mm": 18.5,
            "flipper_length_mm": 182.0, "body_mass_g": 3750.0,
            "island": "Torgersen", "sex": "Male",
        })
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_probabilities_sum_to_one(self, trained_model):
        """Klassenwahrscheinlichkeiten summieren sich zu 1."""
        from model import predict
        result = predict({
            "bill_length_mm": 48.0, "bill_depth_mm": 14.3,
            "flipper_length_mm": 210.0, "body_mass_g": 4600.0,
            "island": "Biscoe", "sex": "Female",
        })
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 1e-4

    def test_predict_all_species_in_probabilities(self, trained_model):
        """Alle trainierten Klassen sind in probabilities vorhanden."""
        from model import predict
        pipeline, _ = trained_model
        result = predict({
            "bill_length_mm": 45.0, "bill_depth_mm": 17.0,
            "flipper_length_mm": 195.0, "body_mass_g": 4200.0,
            "island": "Dream", "sex": "Male",
        })
        for species in list(pipeline.classes_):
            assert species in result["probabilities"]

    def test_predict_without_model_raises(self, tmp_path, monkeypatch):
        """predict() wirft RuntimeError wenn kein Modell vorhanden."""
        import model as m
        monkeypatch.setattr(m, "MODEL_PATH", tmp_path / "nonexistent.pkl")
        from model import predict
        with pytest.raises(RuntimeError, match="Kein trainiertes Modell"):
            predict({
                "bill_length_mm": 45.0, "bill_depth_mm": 17.0,
                "flipper_length_mm": 195.0, "body_mass_g": 4200.0,
                "island": "Dream", "sex": "Male",
            })
