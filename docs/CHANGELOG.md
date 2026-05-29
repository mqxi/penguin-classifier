# Changelog

## 2026-05-27 – Projektstart

Ordnerstruktur angelegt, Libraries recherchiert. Dash-Doku durchgearbeitet, Callback-System
braucht etwas Einarbeitung.

`requirements.txt`, `Dockerfile`, `docker-compose.yml` angelegt – Container läuft durch.

`data_handler.py` und `model.py` implementiert. UCI-ZIP hatte verschachtelte Struktur,
auf direkten GitHub-Link von Allison Horst umgestiegen. sklearn-Pipeline steht:
Median/Modus-Imputation, One-Hot-Encoding, Random Forest. Erste Accuracy ~98 %.

---

## 2026-05-28 – Frontend, Callbacks, Tests

`layout.py`: 3-Panel-Bootstrap-Grid, Dropdowns, Eingabefelder. Proportionen erst nicht
gestimmt, mehrfach angepasst.

Callbacks implementiert – `State`/`Input` verwechselt, nach Debugging behoben. Scatter-Plot
eingebaut (Trainingsdaten + roter Stern für neuen Punkt).

Tests: `sample_df` mit nur 10 Zeilen war zu klein für stratifizierten Split + 5-fold CV,
auf 30 Zeilen erweitert. `confusion_matrix` lässt sich nicht in dcc.Store schreiben –
rausgefiltert.

UI: Emojis bei Panel-Überschriften entfernt, Footer fixiert, Plot-Höhe und Farben angepasst.

Retrain-Button immer sichtbar in Panel ③, lädt Datensatz neu von GitHub. `reload_dataset()`
ergänzt. Bug: `train_test_split()` wurde zweimal aufgerufen – behoben.

---

## 2026-05-29 – Review und Cleanup

Alles durchgetestet. Docker-Build sauber, alle 10 Tests grün. README und Dokumentation
fertiggeschrieben. Log-Statements auf f-Strings umgestellt.

Accuracy: 98,55 % | F1: 98,56 % | CV (5-fold): 98,84 % ± 0,58 %
