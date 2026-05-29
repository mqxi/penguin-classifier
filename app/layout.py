"""UI-Layout der Dash-Applikation: 3-Panel-Aufbau."""

import subprocess

import dash_bootstrap_components as dbc
from dash import dcc, html


def _get_last_commit_date() -> str:
    """Liest das Datum des letzten Git-Commits aus."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ad", "--date=format:%d.%m.%Y"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""

# Farbpalette
COLORS = {
    "primary": "#1a3a5c",
    "accent": "#2e6da4",
    "background": "#f0f4f8",
    "panel_bg": "#ffffff",
    "panel_border": "#c8d8e8",
    "adelie": "#fb8c00",
    "chinstrap": "#7b2d8b",
    "gentoo": "#00897b",
    "new_point": "#e53935",
    "success": "#2e7d32",
    "gray_bg": "#f5f5f5",
}

# Wissenschaftliche Namen
SCIENTIFIC_NAMES = {
    "Adelie": "Pygoscelis adeliae",
    "Chinstrap": "Pygoscelis antarcticus",
    "Gentoo": "Pygoscelis papua",
}

PANEL_STYLE = {
    "backgroundColor": COLORS["panel_bg"],
    "border": f"1px solid {COLORS['panel_border']}",
    "borderRadius": "8px",
    "padding": "20px",
    "height": "100%",
}

HEADER_STYLE = {
    "color": COLORS["primary"],
    "fontWeight": "bold",
    "fontSize": "1.1rem",
    "marginBottom": "16px",
    "borderBottom": f"2px solid {COLORS['accent']}",
    "paddingBottom": "8px",
}

LABEL_STYLE = {
    "fontWeight": "600",
    "color": COLORS["primary"],
    "fontSize": "0.85rem",
    "marginBottom": "4px",
    "marginTop": "10px",
}

INPUT_STYLE = {
    "width": "100%",
    "padding": "8px 10px",
    "border": f"1px solid {COLORS['panel_border']}",
    "borderRadius": "4px",
    "fontSize": "0.9rem",
    "color": "#333",
}


def build_input_panel() -> dbc.Col:
    """Erstellt Panel ①: Dateneingabe."""
    return dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.H5("Dateneingabe", style=HEADER_STYLE),

                # Numerische Eingabefelder
                html.Div("Schnabellänge (mm)", style=LABEL_STYLE),
                dcc.Input(
                    id="input-bill-length",
                    type="number",
                    placeholder="z. B. 45.0",
                    min=30, max=65, step=0.1,
                    style=INPUT_STYLE,
                    debounce=False,
                ),

                html.Div("Schnabeltiefe (mm)", style=LABEL_STYLE),
                dcc.Input(
                    id="input-bill-depth",
                    type="number",
                    placeholder="z. B. 18.0",
                    min=13, max=22, step=0.1,
                    style=INPUT_STYLE,
                    debounce=False,
                ),

                html.Div("Flossenlänge (mm)", style=LABEL_STYLE),
                dcc.Input(
                    id="input-flipper-length",
                    type="number",
                    placeholder="z. B. 195",
                    min=170, max=235, step=1,
                    style=INPUT_STYLE,
                    debounce=False,
                ),

                html.Div("Körpermasse (g)", style=LABEL_STYLE),
                dcc.Input(
                    id="input-body-mass",
                    type="number",
                    placeholder="z. B. 3700",
                    min=2700, max=6300, step=10,
                    style=INPUT_STYLE,
                    debounce=False,
                ),

                html.Div("Insel", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="input-island",
                    options=[
                        {"label": "Torgersen", "value": "Torgersen"},
                        {"label": "Biscoe", "value": "Biscoe"},
                        {"label": "Dream", "value": "Dream"},
                    ],
                    placeholder="Insel auswählen...",
                    clearable=True,
                    style={"fontSize": "0.9rem"},
                ),

                html.Div("Geschlecht", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="input-sex",
                    options=[
                        {"label": "Männlich", "value": "Male"},
                        {"label": "Weiblich", "value": "Female"},
                    ],
                    placeholder="Geschlecht auswählen...",
                    clearable=True,
                    style={"fontSize": "0.9rem"},
                ),

                # Fehlermeldung
                html.Div(
                    id="input-error",
                    style={"color": "#c62828", "fontSize": "0.85rem", "marginTop": "8px", "minHeight": "20px"},
                ),

                # Klassifizieren-Button
                html.Div(style={"marginTop": "16px"}),
                dbc.Button(
                    "Klassifizieren",
                    id="btn-classify",
                    color="primary",
                    className="w-100",
                    style={"backgroundColor": COLORS["accent"], "borderColor": COLORS["accent"], "fontWeight": "bold"},
                ),

                # Reset-Link
                html.Div(
                    html.A(
                        "Eingaben zurücksetzen",
                        id="link-reset",
                        href="#",
                        style={"color": COLORS["accent"], "fontSize": "0.85rem", "textDecoration": "underline"},
                    ),
                    style={"textAlign": "center", "marginTop": "10px"},
                ),
            ]),
            style=PANEL_STYLE,
            className="h-100",
        ),
        width=3,
    )


def build_result_panel() -> dbc.Col:
    """Erstellt Panel ②: Klassifizierungsergebnis."""
    return dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.H5("Klassifizierungsergebnis", style=HEADER_STYLE),

                # Ergebnisbereich (initial: Platzhaltertext)
                html.Div(
                    id="result-content",
                    children=html.P(
                        "Noch keine Klassifizierung durchgeführt.",
                        style={"color": "#888", "fontStyle": "italic", "marginTop": "20px", "textAlign": "center"},
                    ),
                ),

                # Modellperformanz-Box (immer sichtbar)
                html.Div(
                    id="model-metrics-box",
                    style={
                        "backgroundColor": COLORS["gray_bg"],
                        "borderRadius": "6px",
                        "padding": "10px 14px",
                        "marginTop": "20px",
                        "fontSize": "0.82rem",
                        "color": "#555",
                    },
                ),
            ]),
            style=PANEL_STYLE,
            className="h-100",
        ),
        width=4,
    )


def build_visualization_panel() -> dbc.Col:
    """Erstellt Panel ③: Visualisierung."""
    return dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.H5("Visualisierung", style=HEADER_STYLE),

                dcc.Graph(
                    id="scatter-plot",
                    config={"displayModeBar": False},
                    style={"height": "460px"},
                ),

                # Retrain-Button – immer sichtbar, lädt Trainingsdaten neu und trainiert neu
                html.Hr(style={"borderColor": COLORS["panel_border"], "margin": "12px 0"}),
                dbc.Button(
                    "🔄 Neu trainieren",
                    id="btn-retrain",
                    color="secondary",
                    outline=True,
                    className="w-100",
                    style={"fontSize": "0.85rem"},
                ),
                html.Div(
                    "Lädt den aktuellen Datensatz neu herunter und trainiert das Modell von Grund auf.",
                    style={"fontSize": "0.75rem", "color": "#8a9bb0", "marginTop": "5px", "textAlign": "center"},
                ),

                # Retrain-Statusmeldung
                html.Div(id="retrain-status", style={"fontSize": "0.82rem", "marginTop": "8px", "color": "#555"}),
            ]),
            style=PANEL_STYLE,
            className="h-100",
        ),
        width=5,
    )


def build_layout() -> html.Div:
    """Erstellt das vollständige App-Layout.

    Returns:
        Root-Div mit Header und 3-Panel-Grid.
    """
    return html.Div(
        style={"backgroundColor": COLORS["background"], "minHeight": "100vh", "fontFamily": "Arial, sans-serif", "paddingBottom": "36px"},
        children=[
            # Store für Modell-Metriken (session-basiert)
            dcc.Store(id="store-metrics"),
            dcc.Store(id="store-new-point"),

            # Header
            html.Div(
                style={
                    "backgroundColor": COLORS["primary"],
                    "color": "white",
                    "padding": "16px 30px",
                    "marginBottom": "20px",
                    "boxShadow": "0 2px 4px rgba(0,0,0,0.2)",
                },
                children=[
                    html.H3(
                        "🐧 Pinguin-Klassifikator",
                        style={"margin": 0, "fontWeight": "bold", "display": "inline"},
                    ),
                    html.Span(
                        " – Automatische Artbestimmung anhand morphologischer Messdaten",
                        style={"fontSize": "0.9rem", "opacity": "0.8", "marginLeft": "10px"},
                    ),
                ],
            ),

            # 3-Panel-Grid
            dbc.Container(
                dbc.Row(
                    [
                        build_input_panel(),
                        build_result_panel(),
                        build_visualization_panel(),
                    ],
                    className="g-3 align-items-start",
                ),
                fluid=True,
                style={"paddingLeft": "20px", "paddingRight": "20px"},
            ),

            # Footer
            _build_footer(),
        ],
    )


def _build_footer() -> html.Footer:
    """Erstellt die dezente Fußzeile mit Metadaten."""
    date = _get_last_commit_date()
    date_part = f" · {date}" if date else ""
    return html.Footer(
        html.Div(
            [
                html.Span("Erstellt von "),
                html.Span("Maximilian Engler", style={"fontWeight": "600"}),
                html.Span(date_part),
                html.Span(" · Portfolio – Kurs DLMDSPMLSD01_D | Phase 2: Erarbeitungsphase"),
            ],
            style={"textAlign": "center"},
        ),
        style={
            "position": "fixed",
            "bottom": 0,
            "left": 0,
            "right": 0,
            "padding": "6px 20px",
            "backgroundColor": COLORS["background"],
            "borderTop": f"1px solid {COLORS['panel_border']}",
            "color": "#8a9bb0",
            "fontSize": "0.75rem",
            "letterSpacing": "0.02em",
            "zIndex": 1000,
        },
    )
