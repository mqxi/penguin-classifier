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

                # Korrektur-Panel (initial versteckt, erscheint nach Klassifizierung)
                html.Div(
                    id="correction-panel",
                    style={"display": "none", "marginTop": "16px"},
                    children=[
                        html.Hr(style={"borderColor": COLORS["panel_border"], "margin": "0 0 12px 0"}),
                        html.Div(
                            "Vorhersage falsch?",
                            style={"fontWeight": "600", "fontSize": "0.85rem", "color": "#444", "marginBottom": "8px"},
                        ),
                        dcc.Dropdown(
                            id="correction-species-dropdown",
                            options=[
                                {"label": "Adelie", "value": "Adelie"},
                                {"label": "Chinstrap", "value": "Chinstrap"},
                                {"label": "Gentoo", "value": "Gentoo"},
                                {"label": "Neue Art...", "value": "__new__"},
                            ],
                            placeholder="Korrekte Art auswählen...",
                            clearable=True,
                            style={"fontSize": "0.9rem", "marginBottom": "8px"},
                        ),
                        # Freitextfeld für neue Art – nur sichtbar wenn "__new__" gewählt
                        dcc.Input(
                            id="correction-new-species-input",
                            type="text",
                            placeholder="Artbezeichnung eingeben...",
                            style={**INPUT_STYLE, "display": "none", "marginBottom": "8px"},
                            debounce=False,
                        ),
                        dbc.Button(
                            "Korrektur speichern",
                            id="btn-save-correction",
                            color="warning",
                            outline=True,
                            size="sm",
                            className="w-100",
                            style={"fontSize": "0.85rem"},
                        ),
                        html.Div(id="correction-status", style={"fontSize": "0.82rem", "marginTop": "8px"}),
                    ],
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

                # Modal: Hinweis neue Art
                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Neue Art gespeichert")),
                        dbc.ModalBody(id="modal-new-species-body"),
                        dbc.ModalFooter(
                            dbc.Button("Schließen", id="modal-close", className="ms-auto", n_clicks=0)
                        ),
                    ],
                    id="modal-new-species-info",
                    is_open=False,
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


def _build_header(current_page: str = "main") -> html.Div:
    """Erstellt den App-Header mit Navigation."""
    nav_link = (
        html.A(
            "← Zurück zur App",
            href="/",
            style={"color": "rgba(255,255,255,0.85)", "fontSize": "0.85rem", "textDecoration": "none", "marginLeft": "auto"},
        )
        if current_page == "info"
        else html.A(
            "Modell & Funktionsweise",
            href="/info",
            style={"color": "rgba(255,255,255,0.85)", "fontSize": "0.85rem", "textDecoration": "none", "marginLeft": "auto"},
        )
    )
    return html.Div(
        style={
            "backgroundColor": COLORS["primary"],
            "color": "white",
            "padding": "16px 30px",
            "marginBottom": "20px",
            "boxShadow": "0 2px 4px rgba(0,0,0,0.2)",
            "display": "flex",
            "alignItems": "center",
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
            nav_link,
        ],
    )


def build_main_content() -> html.Div:
    """Erstellt den Hauptinhalt mit Header, 3-Panel-Grid und Footer."""
    return html.Div([
        _build_header(current_page="main"),
        # Lade-Overlay – wird beim Retraining eingeblendet
        html.Div(
            id="retrain-overlay",
            style={"display": "none"},
            children=html.Div(
                style={
                    "position": "fixed",
                    "top": 0, "left": 0, "right": 0, "bottom": 0,
                    "backgroundColor": "rgba(240,244,248,0.75)",
                    "zIndex": 999,
                    "display": "flex",
                    "flexDirection": "column",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "backdropFilter": "blur(2px)",
                },
                children=[
                    dbc.Spinner(color="primary", size="lg"),
                    html.Div(
                        "Modell wird trainiert...",
                        style={"marginTop": "16px", "color": "#1a3a5c", "fontWeight": "600", "fontSize": "1rem"},
                    ),
                ],
            ),
        ),
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
        _build_footer(),
    ])


def build_info_page() -> html.Div:
    """Erstellt die Info-Seite mit Modell- und Bedienungshinweisen."""
    card_style = {**PANEL_STYLE, "marginBottom": "16px"}

    return html.Div([
        _build_header(current_page="info"),
        dbc.Container([

            # Das Modell
            dbc.Card(dbc.CardBody([
                html.H5("Das Modell", style=HEADER_STYLE),
                html.P([
                    "Eingesetzt wird ein ", html.Strong("Random Forest Classifier"), " (scikit-learn, 100 Bäume, ",
                    html.Code("class_weight='balanced'"), ") – robust gegenüber kleinen Datensätzen und gut geeignet "
                    "für mehrklassige Klassifikation mit gemischten Feature-Typen.",
                ]),
                html.P([
                    "Trainiert auf dem ", html.Strong("Palmer Penguins Datensatz"),
                    " (344 Datenpunkte, 3 Arten). Die Pipeline verarbeitet numerische und kategorische Features getrennt: "
                    "Median-Imputation + Standardisierung für Maße, Modus-Imputation + One-Hot-Encoding für Insel und Geschlecht.",
                ]),
                html.Table([
                    html.Thead(html.Tr([html.Th("Metrik"), html.Th("Wert")])),
                    html.Tbody([
                        html.Tr([html.Td("Accuracy (Test-Set)"), html.Td("98,55 %")]),
                        html.Tr([html.Td("F1-Score (gewichtet)"), html.Td("98,56 %")]),
                        html.Tr([html.Td("Kreuzvalidierung (5-fold)"), html.Td("98,84 % ± 0,58 %")]),
                    ]),
                ], className="table table-sm table-bordered", style={"fontSize": "0.88rem", "marginTop": "10px"}),
            ]), style=card_style),

            # Verwendete Features
            dbc.Card(dbc.CardBody([
                html.H5("Verwendete Features", style=HEADER_STYLE),
                html.Table([
                    html.Thead(html.Tr([html.Th("Feature"), html.Th("Einheit"), html.Th("Typischer Bereich")])),
                    html.Tbody([
                        html.Tr([html.Td("Schnabellänge"), html.Td("mm"), html.Td("32 – 60")]),
                        html.Tr([html.Td("Schnabeltiefe"), html.Td("mm"), html.Td("13 – 22")]),
                        html.Tr([html.Td("Flossenlänge"), html.Td("mm"), html.Td("172 – 235")]),
                        html.Tr([html.Td("Körpermasse"), html.Td("g"), html.Td("2700 – 6300")]),
                        html.Tr([html.Td("Insel"), html.Td("–"), html.Td("Torgersen, Biscoe, Dream")]),
                        html.Tr([html.Td("Geschlecht"), html.Td("–"), html.Td("Male, Female")]),
                    ]),
                ], className="table table-sm table-bordered", style={"fontSize": "0.88rem"}),
                html.P(
                    "Fehlende Werte werden automatisch durch Median (numerisch) bzw. häufigsten Wert (kategorisch) ersetzt – "
                    "unvollständige Eingaben können trotzdem klassifiziert werden.",
                    style={"fontSize": "0.85rem", "color": "#555", "marginTop": "10px", "marginBottom": 0},
                ),
            ]), style=card_style),

            # Bedienung
            dbc.Card(dbc.CardBody([
                html.H5("Bedienung", style=HEADER_STYLE),
                html.Ol([
                    html.Li("Messwerte im linken Panel eingeben (alle sechs Felder ausfüllen)."),
                    html.Li([html.Strong("Klassifizieren"), " klicken – Ergebnis erscheint im mittleren Panel mit Art, Konfidenz und Klassenwahrscheinlichkeiten."]),
                    html.Li("Scatter-Plot im rechten Panel zeigt den neuen Datenpunkt (roter Stern) im Vergleich zu den Trainingsdaten."),
                    html.Li(["Vorhersage falsch? Im Korrektur-Bereich die richtige Art auswählen und ", html.Strong("Korrektur speichern"), " klicken."]),
                    html.Li(["Nach mehreren Beobachtungen: ", html.Strong("🔄 Neu trainieren"), " lädt den Originaldatensatz frisch von GitHub und trainiert das Modell auf allen gesammelten Daten neu."]),
                ], style={"fontSize": "0.9rem", "lineHeight": "1.8"}),
            ]), style=card_style),

            # Neue Arten & Korrektur
            dbc.Card(dbc.CardBody([
                html.H5("Vorhersage-Korrektur & neue Arten", style=HEADER_STYLE),
                html.P(
                    'Nach jeder Klassifizierung erscheint ein Korrektur-Panel. Bekannte Arten können direkt aus dem Dropdown '
                    'gewählt werden. Über "Neue Art..." lässt sich eine eigene Artbezeichnung eingeben.',
                    style={"fontSize": "0.9rem"},
                ),
                html.Div(
                    style={"backgroundColor": "#fff3e0", "borderRadius": "6px", "padding": "10px 14px"},
                    children=html.P([
                        html.Strong("Hinweis: "),
                        "Für eine neue Art werden mindestens ",
                        html.Strong("15 Samples"),
                        " empfohlen, bevor ein Retraining stabile Ergebnisse liefert. "
                        "Das System erlaubt das Speichern auch mit weniger Samples und informiert per Dialog über den aktuellen Stand.",
                    ], style={"margin": 0, "fontSize": "0.88rem", "color": "#e65100"}),
                ),
                html.P(
                    "Korrigierte Labels fließen beim nächsten Retraining bevorzugt als Zielwert ein – "
                    "fehlerhafte Modellvorhersagen verbessern das Modell also langfristig.",
                    style={"fontSize": "0.88rem", "color": "#555", "marginTop": "10px", "marginBottom": 0},
                ),
            ]), style=card_style),

            # Datenquelle
            dbc.Card(dbc.CardBody([
                html.H5("Datenquelle", style=HEADER_STYLE),
                html.P([
                    "Gorman KB, Williams TD, Fraser WR (2014). ",
                    html.Em("Ecological sexual dimorphism and environmental variability within a community of Antarctic penguins."),
                    " PLoS ONE 9(3):e90081.",
                ], style={"fontSize": "0.88rem"}),
                html.P([
                    "Datensatz bereitgestellt von Allison Horst via ",
                    html.A("palmerpenguins (GitHub)", href="https://github.com/allisonhorst/palmerpenguins", target="_blank"),
                    ".",
                ], style={"fontSize": "0.88rem", "marginBottom": 0}),
            ]), style=card_style),

        ], fluid=True, style={"paddingLeft": "20px", "paddingRight": "20px", "maxWidth": "860px"}),
        _build_footer(),
    ])


def build_layout() -> html.Div:
    """Erstellt das vollständige App-Layout mit Routing.

    Returns:
        Root-Div mit Stores, dcc.Location und page-content.
    """
    return html.Div(
        style={"backgroundColor": COLORS["background"], "minHeight": "100vh", "fontFamily": "Arial, sans-serif", "paddingBottom": "36px"},
        children=[
            dcc.Location(id="url", refresh=False),
            dcc.Store(id="store-metrics"),
            dcc.Store(id="store-new-point"),
            dcc.Store(id="store-retrain-fig"),
            html.Div(id="page-content"),
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
