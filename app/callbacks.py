"""Dash-Callbacks: Klassifizieren, Zurücksetzen und Retraining."""

import logging

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dash import Input, Output, State, callback, html, no_update
import dash_bootstrap_components as dbc

from data_handler import (
    get_combined_data,
    load_observations,
    load_training_data,
    reload_dataset,
    save_observation,
)
from model import SPECIES_CLASSES, get_or_train_model, predict, retrain_model

logger = logging.getLogger(__name__)

# Farbzuordnung Arten
SPECIES_COLORS = {
    "Adelie": "#fb8c00",
    "Chinstrap": "#7b2d8b",
    "Gentoo": "#00897b",
}
SCIENTIFIC_NAMES = {
    "Adelie": "Pygoscelis adeliae",
    "Chinstrap": "Pygoscelis antarcticus",
    "Gentoo": "Pygoscelis papua",
}


def _build_scatter(df_train: pd.DataFrame, new_point: dict | None = None) -> go.Figure:
    """Erstellt den Scatter-Plot mit Trainings- und optionalem neuem Datenpunkt.

    Args:
        df_train: Trainingsdaten.
        new_point: Optionaler neuer Datenpunkt mit species.

    Returns:
        Plotly Figure.
    """
    df_plot = df_train.dropna(subset=["bill_length_mm", "flipper_length_mm", "species"]).copy()
    df_plot = df_plot[df_plot["species"].isin(SPECIES_CLASSES)]

    fig = px.scatter(
        df_plot,
        x="bill_length_mm",
        y="flipper_length_mm",
        color="species",
        color_discrete_map=SPECIES_COLORS,
        opacity=0.7,
        labels={
            "bill_length_mm": "Schnabellänge (mm)",
            "flipper_length_mm": "Flossenlänge (mm)",
            "species": "Art",
        },
        hover_data=["bill_depth_mm", "body_mass_g", "island", "sex"],
        title="",
    )
    # Punktgröße erhöhen
    fig.update_traces(marker=dict(size=8))

    # Neuen Datenpunkt hinzufügen
    if new_point is not None:
        fig.add_trace(go.Scatter(
            x=[new_point.get("bill_length_mm")],
            y=[new_point.get("flipper_length_mm")],
            mode="markers",
            marker=dict(symbol="star", size=15, color="#e53935", line=dict(width=1, color="#b71c1c")),
            name="Neuer Datenpunkt",
            hovertemplate=(
                f"<b>Neuer Datenpunkt</b><br>"
                f"Schnabellänge: {new_point.get('bill_length_mm')} mm<br>"
                f"Flossenlänge: {new_point.get('flipper_length_mm')} mm<br>"
                f"Schnabeltiefe: {new_point.get('bill_depth_mm')} mm<br>"
                f"Körpermasse: {new_point.get('body_mass_g')} g<br>"
                f"Insel: {new_point.get('island')}<br>"
                f"Geschlecht: {new_point.get('sex')}<br>"
                f"Vorhergesagte Art: <b>{new_point.get('predicted_species', '')}</b><extra></extra>"
            ),
        ))

    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            font=dict(size=13),
            itemsizing="constant",
        ),
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=13),
        xaxis=dict(
            gridcolor="#e0e0e0",
            title_font=dict(size=13),
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            gridcolor="#e0e0e0",
            title_font=dict(size=13),
            tickfont=dict(size=12),
        ),
    )
    return fig


def _build_result_content(result: dict) -> list:
    """Erstellt den Inhalt des Ergebnis-Panels.

    Args:
        result: Vorhersage-Dict mit species, confidence, probabilities.

    Returns:
        Liste von Dash-Komponenten.
    """
    species = result["species"]
    confidence = result["confidence"]
    probs = result["probabilities"]
    sci_name = SCIENTIFIC_NAMES.get(species, "")
    color = SPECIES_COLORS.get(species, "#333")

    components = [
        # Ergebnis-Badge
        html.Div(
            style={
                "border": f"2px solid #2e7d32",
                "borderRadius": "8px",
                "padding": "12px 16px",
                "marginBottom": "14px",
                "backgroundColor": "#f1f8e9",
                "textAlign": "center",
            },
            children=[
                html.Div(species, style={"fontSize": "1.8rem", "fontWeight": "bold", "color": color}),
                html.Div(
                    html.Em(sci_name),
                    style={"fontSize": "0.85rem", "color": "#555", "marginTop": "2px"},
                ),
            ],
        ),

        # Konfidenz-Balken
        html.Div([
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "marginBottom": "4px"},
                children=[
                    html.Span("Konfidenz", style={"fontWeight": "600", "fontSize": "0.85rem", "color": "#444"}),
                    html.Span(f"{confidence:.1%}", style={"fontWeight": "bold", "fontSize": "0.85rem", "color": "#2e7d32"}),
                ],
            ),
            html.Div(
                style={"backgroundColor": "#e8f5e9", "borderRadius": "4px", "height": "10px", "overflow": "hidden"},
                children=html.Div(
                    style={
                        "width": f"{confidence * 100:.1f}%",
                        "backgroundColor": "#43a047",
                        "height": "100%",
                        "transition": "width 0.5s ease",
                    }
                ),
            ),
        ], style={"marginBottom": "14px"}),

        # Klassenwahrscheinlichkeiten
        html.Div("Klassenwahrscheinlichkeiten", style={"fontWeight": "600", "fontSize": "0.85rem", "color": "#444", "marginBottom": "8px"}),
    ]

    for cls in SPECIES_CLASSES:
        p = probs.get(cls, 0.0)
        cls_color = SPECIES_COLORS.get(cls, "#999")
        components.append(
            html.Div(style={"marginBottom": "6px"}, children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "fontSize": "0.8rem", "marginBottom": "2px"},
                    children=[
                        html.Span(cls, style={"color": cls_color, "fontWeight": "600"}),
                        html.Span(f"{p:.1%}", style={"color": "#555"}),
                    ],
                ),
                html.Div(
                    style={"backgroundColor": "#f0f0f0", "borderRadius": "3px", "height": "7px", "overflow": "hidden"},
                    children=html.Div(
                        style={"width": f"{p * 100:.1f}%", "backgroundColor": cls_color, "height": "100%"}
                    ),
                ),
            ])
        )

    return components


def _build_metrics_box(metrics: dict) -> list:
    """Erstellt die kompakte Modellperformanz-Box.

    Args:
        metrics: Metriken-Dict mit accuracy, f1_weighted, n_train.

    Returns:
        Liste von Dash-Komponenten.
    """
    if not metrics:
        return [html.Span("Metriken werden geladen...", style={"color": "#999"})]

    acc = metrics.get("accuracy", 0)
    f1 = metrics.get("f1_weighted", 0)
    n = metrics.get("n_train", 0)

    return [
        html.Div("Modellperformanz", style={"fontWeight": "bold", "marginBottom": "6px", "fontSize": "0.82rem", "color": "#444"}),
        html.Div(
            style={"display": "flex", "gap": "14px", "flexWrap": "wrap"},
            children=[
                html.Div([
                    html.Span("Accuracy ", style={"color": "#777"}),
                    html.Span(f"{acc:.1%}", style={"fontWeight": "bold", "color": "#1a3a5c"}),
                ]),
                html.Div([
                    html.Span("F1 (gew.) ", style={"color": "#777"}),
                    html.Span(f"{f1:.1%}", style={"fontWeight": "bold", "color": "#1a3a5c"}),
                ]),
                html.Div([
                    html.Span("Trainings-N ", style={"color": "#777"}),
                    html.Span(str(n), style={"fontWeight": "bold", "color": "#1a3a5c"}),
                ]),
            ],
        ),
    ]


def register_callbacks(app) -> None:
    """Registriert alle Dash-Callbacks.

    Args:
        app: Dash-Applikations-Instanz.
    """

    @app.callback(
        Output("scatter-plot", "figure"),
        Output("store-metrics", "data"),
        Output("model-metrics-box", "children"),
        Input("scatter-plot", "id"),  # Einmaliger Trigger beim Seitenaufruf
    )
    def initialize_plot_and_metrics(_):
        """Befüllt Scatter-Plot und Metriken-Store beim ersten Seitenaufruf."""
        try:
            df_train = load_training_data()
            _, metrics = get_or_train_model(df_train)
            fig = _build_scatter(df_train)
            metrics_box = _build_metrics_box(metrics)
            # confusion_matrix ist kein JSON-serialisierbarer Typ – entfernen für Store
            metrics_serializable = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
            return fig, metrics_serializable, metrics_box
        except Exception as e:
            logger.error(f"Initialisierungsfehler: {e}")
            return go.Figure(), {}, []

    @app.callback(
        Output("result-content", "children"),
        Output("model-metrics-box", "children"),
        Output("scatter-plot", "figure"),
        Output("input-error", "children"),
        Output("store-new-point", "data"),
        Input("btn-classify", "n_clicks"),
        State("input-bill-length", "value"),
        State("input-bill-depth", "value"),
        State("input-flipper-length", "value"),
        State("input-body-mass", "value"),
        State("input-island", "value"),
        State("input-sex", "value"),
        State("store-metrics", "data"),
        prevent_initial_call=True,
    )
    def classify(n_clicks, bill_length, bill_depth, flipper_length, body_mass, island, sex, stored_metrics):
        """Klassifiziert den Datenpunkt bei Button-Klick."""
        missing = []
        if bill_length is None: missing.append("Schnabellänge")
        if bill_depth is None: missing.append("Schnabeltiefe")
        if flipper_length is None: missing.append("Flossenlänge")
        if body_mass is None: missing.append("Körpermasse")
        if not island: missing.append("Insel")
        if not sex: missing.append("Geschlecht")

        df_train = load_training_data()

        if missing:
            error_msg = f"Bitte alle Felder ausfüllen. Fehlend: {', '.join(missing)}."
            fig = _build_scatter(df_train)
            metrics_box = _build_metrics_box(stored_metrics or {})
            return no_update, metrics_box, fig, error_msg, no_update

        input_dict = {
            "bill_length_mm": float(bill_length),
            "bill_depth_mm": float(bill_depth),
            "flipper_length_mm": float(flipper_length),
            "body_mass_g": float(body_mass),
            "island": island,
            "sex": sex,
        }

        try:
            result = predict(input_dict)
        except Exception as e:
            logger.error(f"Vorhersagefehler: {e}")
            return (
                html.P(f"Fehler bei der Klassifizierung: {e}", style={"color": "#c62828"}),
                no_update, no_update, "", no_update,
            )

        try:
            save_observation(input_dict, result["species"])
        except Exception as e:
            logger.warning(f"Beobachtung konnte nicht gespeichert werden: {e}")

        new_point = {**input_dict, "predicted_species": result["species"]}
        fig = _build_scatter(df_train, new_point=new_point)
        result_children = _build_result_content(result)
        metrics_box = _build_metrics_box(stored_metrics or {})

        return result_children, metrics_box, fig, "", new_point

    @app.callback(
        Output("input-bill-length", "value"),
        Output("input-bill-depth", "value"),
        Output("input-flipper-length", "value"),
        Output("input-body-mass", "value"),
        Output("input-island", "value"),
        Output("input-sex", "value"),
        Input("link-reset", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_inputs(n_clicks):
        """Setzt alle Eingabefelder zurück."""
        return None, None, None, None, None, None

    @app.callback(
        Output("model-metrics-box", "children", allow_duplicate=True),
        Output("retrain-status", "children"),
        Output("store-metrics", "data"),
        Output("scatter-plot", "figure", allow_duplicate=True),
        Input("btn-retrain", "n_clicks"),
        prevent_initial_call=True,
    )
    def retrain(n_clicks):
        """Lädt Trainingsdaten neu herunter und trainiert das Modell komplett neu.

        Bestehende Beobachtungen fließen ebenfalls in das Retraining ein.
        """
        try:
            # Datensatz frisch von GitHub laden (überschreibt penguins.csv)
            reload_dataset()
            # Neu geladene Trainingsdaten + gespeicherte Beobachtungen kombinieren
            df_combined = get_combined_data()
            _, metrics = retrain_model(df_combined)

            metrics_box = _build_metrics_box(metrics)
            n = metrics.get("n_total", 0)
            status = html.Span(
                f"✓ Neu trainiert auf {n} Datenpunkten (Datensatz wurde neu geladen).",
                style={"color": "#2e7d32"},
            )
            metrics_serializable = {k: v for k, v in metrics.items() if k != "confusion_matrix"}

            # Plot mit frisch geladenem Datensatz aktualisieren
            df_train = load_training_data()
            fig = _build_scatter(df_train)

            return metrics_box, status, metrics_serializable, fig
        except Exception as e:
            logger.error(f"Retraining-Fehler: {e}")
            return no_update, html.Span(f"Fehler beim Retraining: {e}", style={"color": "#c62828"}), no_update, no_update
