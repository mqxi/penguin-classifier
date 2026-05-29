# Pinguin-Klassifikator

Interaktive Web-Applikation zur automatischen Artbestimmung von Pinguinen anhand morphologischer Messdaten (Palmer Penguins Dataset). Das Modell klassifiziert Adelie-, Chinstrap- und Gentoo-Pinguine auf Basis von Schnabellänge, Schnabeltiefe, Flossenlänge, Körpermasse, Insel und Geschlecht.

## Voraussetzungen

- Docker >= 20.10 und Docker Compose >= 2.0
- Alternativ: Python 3.11+, pip

## Installation (3 Schritte)

```bash
git clone <repository-url>
cd penguin-classifier
docker compose up --build
```

Die App ist anschließend unter [http://localhost:8050](http://localhost:8050) erreichbar.

### Lokale Installation (ohne Docker)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python app/app.py
```

## Projektstruktur

| Pfad | Beschreibung |
|---|---|
| `app/app.py` | Einstiegspunkt der Dash-Applikation |
| `app/layout.py` | UI-Layout: 3-Panel-Aufbau (Eingabe / Ergebnis / Visualisierung) |
| `app/callbacks.py` | Dash-Callbacks: Klassifizieren, Reset, Retraining |
| `app/model.py` | ML-Pipeline: Random Forest mit sklearn, Training und Vorhersage |
| `app/data_handler.py` | Datenverwaltung: Download, Laden, Speichern von Beobachtungen |
| `data/penguins.csv` | Trainingsdatensatz (wird beim ersten Start heruntergeladen) |
| `data/new_observations.csv` | Neue Beobachtungen (wird automatisch erstellt) |
| `model/classifier.pkl` | Trainiertes Modell (wird beim ersten Start erstellt) |
| `tests/test_model.py` | Unit-Tests für ML-Pipeline |
| `docs/CHANGELOG.md` | Änderungshistorie |
| `Dockerfile` | Container-Definition |
| `docker-compose.yml` | Compose-Konfiguration mit Volume-Mounts |
| `requirements.txt` | Python-Abhängigkeiten |

## Technologien

| Technologie | Version | Verwendungszweck |
|---|---|---|
| Dash | ≥ 2.14 | Web-Framework |
| Dash Bootstrap Components | ≥ 1.5 | UI-Komponenten |
| Plotly | ≥ 5.18 | Interaktive Visualisierungen |
| scikit-learn | ≥ 1.3 | Random Forest, Preprocessing-Pipeline |
| pandas | ≥ 2.0 | Datenverwaltung |
| joblib | ≥ 1.3 | Modell-Serialisierung |
| Python | 3.11 | Laufzeitumgebung |
| Docker | ≥ 20.10 | Containerisierung |

## Funktionsweise

1. **Eingabe**: Messdaten eines Pinguins werden in Panel ① eingegeben.
2. **Klassifizierung**: Der Random-Forest-Klassifikator gibt Art, Konfidenz und Klassenwahrscheinlichkeiten aus (Panel ②).
3. **Visualisierung**: Der neue Datenpunkt wird im Scatter-Plot eingetragen (Panel ③).
4. **Retraining**: Ab 10 neuen Beobachtungen kann das Modell auf den kombinierten Daten neu trainiert werden.

## Lizenz

MIT License – Copyright (c) 2026
