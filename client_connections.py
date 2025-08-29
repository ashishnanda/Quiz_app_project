# app.py
import warnings
warnings.filterwarnings("ignore")

import uuid
import pandas as pd
import networkx as nx

import dash
from dash import html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc

from dash.dash_table import Format  # for numeric display in DataTable
from pyvis.network import Network


# ----------------------------- Demo data (replace with your SQL/Cosmos load) -----------------------------
def demo_connections_df() -> pd.DataFrame:
    rows = [
        dict(client_id="C001", client_name="Jamie Rose", primary_ace="Greta Clark",
             est_aum=2_170_000, est_nw=56_750_000,
             connection_name="Adam Wiles", connection_type="School Board",
             organization="NYU; Red Cross of New York", score=4.83, is_client=True,
             role="Secretary", since_year=2009, city="Norwalk, CT"),

        dict(client_id="C001", client_name="Jamie Rose", primary_ace="Greta Clark",
             est_aum=2_170_000, est_nw=56_750_000,
             connection_name="Marion Bishop", connection_type="Board (Corporate)",
             organization="Adobe Systems", score=4.66, is_client=False,
             role="Director", since_year=2017, city="NYC"),

        dict(client_id="C001", client_name="Jamie Rose", primary_ace="Greta Clark",
             est_aum=2_170_000, est_nw=56_750_000,
             connection_name="Steven Walker", connection_type="School",
             organization="NYU", score=4.61, is_client=False,
             role="", since_year=2003, city="NYC"),

        dict(client_id="C001", client_name="Jamie Rose", primary_ace="Greta Clark",
             est_aum=2_170_000, est_nw=56_750_000,
             connection_name="John Phillips", connection_type="Employer",
             organization="Champion Health", score=4.60, is_client=False,
             role="VP", since_year=2014, city="Boston"),

        dict(client_id="C001", client_name="Jamie Rose", primary_ace="Greta Clark",
             est_aum=2_170_000, est_nw=56_750_000,
             connection_name="Martin Reed", connection_type="Board (Charitable)",
             organization="Red Cross of New York", score=4.58, is_client=False,
             role="Chair", since_year=2015, city="NYC"),
    ]
    return pd.DataFrame(rows)


# ----------------------------- Helpers -----------------------------
def build_client_options(df: pd.DataFrame):
    if df.empty or not {"client_id", "client_name"}.issubset(df.columns):
        return []
    d = df[["client_id", "client_name"]].drop_duplicates().astype(str)
    return [{"value": v, "label": f"{n} ({v})"} for v, n in zip(d.client_id, d.client_name)]


def make_header():
    return dbc.Row(
        [
            dbc.Col(html.H4("Visualizing client connections", className="mb-0"), md=8),
            dbc.Col(
                html.Div(
                    [
                        html.Span("Data status: "),
                        dbc.Badge("3rd party / AI-driven", color="secondary", className="me-1"),
                        dbc.Badge("Advisor-confirmed", color="success"),
                    ],
                    className="text-end",
                ),
                md=4,
            ),
        ],
        align="center", className="mb-2",
    )


def make_filters(df: pd.DataFrame):
    client_options = build_client_options(df)
    return dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Client"),
                                dcc.Dropdown(
                                    id="client-dd",
                                    options=client_options,
                                    value=client_options[0]["value"] if client_options else None,
                                    clearable=False,
                                ),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Search connections"),
                                dcc.Input(
                                    id="search-box",
                                    type="text",
                                    debounce=True,
                                    placeholder="name / org / type…",
                                    className="form-control",
                                ),
                            ],
                            md=5,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Min score"),
                                dcc.Slider(
                                    id="min-score", min=0, max=5, step=0.1, value=4.3,
                                    marks={i: str(i) for i in range(0, 6)},
                                ),
                            ],
                            md=3,
                        ),
                    ],
                    className="g-3",
                )
            ]
        ),
        className="mb-3 shadow-sm",
    )


def make_client_stats():
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5(id="client-name", className="mb-1"),
                html.Div([html.Span("Primary ACE: "), html.B(id="client-ace")]),
                html.Div(
                    [
                        html.Span("Estimated AUM: "), html.B(id="client-aum"),
                        html.Span(" • Estimated Net Worth: "), html.B(id="client-nw"),
                    ]
                ),
            ]
        ),
        className="mb-2 shadow-sm",
    )


def make_table():
    columns = [
        {"name": "Name", "id": "connection_name"},
        {"name": "Connection Type", "id": "connection_type"},
        {"name": "Organization", "id": "organization"},
        {"name": "Score", "id": "score", "type": "numeric",
         "format": Format(precision=2, scheme=Format.Scheme.fixed)},
        {"name": "Client?", "id": "is_client"},
        {"name": "Role", "id": "role"},
        {"name": "Since", "id": "since_year"},
    ]
    return dash_table.DataTable(
        id="conn-table",
        columns=columns,
        data=[],
        page_size=10,
        sort_action="native",
        filter_action="none",
        row_selectable="single",
        style_table={"height": "420px", "overflowY": "auto"},
        style_cell={"fontSize": 13, "padding": "8px"},
        style_header={"fontWeight": "bold"},
        selected_rows=[],
    )


def make_graph_card():
    return dbc.Card(
        dbc.CardBody([html.H6("Connections graph", className="mb-2"), html.Div(id="graph-container")]),
        className="shadow-sm",
        style={"minHeight": "520px"},
    )


# ----------------------------- Graph builder -----------------------------
def build_pyvis_graph_html(df_for_client: pd.DataFrame, focus_selection: str | None) -> str:
    """
    Build an ego graph centered at client_name from df_for_client (one client subset).
    """
    if df_for_client.empty:
        return "<div style='padding:1rem'>No data</div>"

    focal = str(df_for_client.iloc[0]["client_name"])
    G = nx.Graph()
    G.add_node(focal, kind="client")

    for _, r in df_for_client.iterrows():
        node_label = str(r["connection_name"])
        G.add_node(
            node_label,
            kind=("client" if bool(r.get("is_client", False)) else "connection"),
            title=f"{r.get('connection_type','')}<br>{r.get('organization','')}",
        )
        G.add_edge(focal, node_label,
                   weight=float(r.get("score", 0) or 0),
                   label=str(r.get("role", "")))

    net = Network(height="520px", width="100%", bgcolor="#ffffff", font_color="#333333",
                  notebook=False, directed=False)
    net.barnes_hut(gravity=-20000, central_gravity=0.3, spring_length=180, spring_strength=0.015)

    # Nodes
    for n, attrs in G.nodes(data=True):
        size = 28 if attrs["kind"] == "client" else 16
        border = 4 if attrs["kind"] == "client" else 1
        color = "#CC0000" if attrs["kind"] == "client" else "#8fbff6"
        if focus_selection and n == focus_selection:
            size, border, color = 26, 4, "#f6a623"
        net.add_node(n, label=n, title=attrs.get("title", n), size=size,
                     borderWidth=border, color=color)

    # Edges
    for s, t, attrs in G.edges(data=True):
        width = 1 + (attrs.get("weight", 0) or 0)  # map score -> width
        net.add_edge(s, t, value=width,
                     title=f"Score {attrs.get('weight','')}; {attrs.get('label','')}",
                     width=width)

    # Valid JSON (not JS) for options; or comment this out to use defaults
    net.set_options("""
    {
      "nodes": { "shape": "dot" },
      "interaction": { "hover": true },
      "physics": { "stabilization": true }
    }
    """)

    # Return HTML string robustly across pyvis versions
    try:
        return net.generate_html(notebook=False)
    except Exception:
        from pathlib import Path
        import tempfile
        tmp = Path(tempfile.gettempdir()) / f"pyvis_{uuid.uuid4().hex}.html"
        net.write_html(tmp, notebook=False)
        return tmp.read_text(encoding="utf-8")


# ----------------------------- App layout -----------------------------
df = demo_connections_df()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Client Connections"

app.layout = dbc.Container(
    [
        make_header(),
        make_filters(df),
        dbc.Row(
            [
                dbc.Col([make_client_stats(), dbc.Card(dbc.CardBody([make_table()]))], md=7),
                dbc.Col([make_graph_card()], md=5),
            ],
            className="g-3",
        ),
        dcc.Store(id="store-all", data=df.to_dict("records")),
    ],
    fluid=True,
)


# ----------------------------- Callbacks -----------------------------
@app.callback(
    Output("conn-table", "data"),
    Output("client-name", "children"),
    Output("client-ace", "children"),
    Output("client-aum", "children"),
    Output("client-nw", "children"),
    Input("store-all", "data"),
    Input("client-dd", "value"),
    Input("search-box", "value"),
    Input("min-score", "value"),
)
def update_table(all_rows, client_id, query, min_score):
    df_all = pd.DataFrame(all_rows or [])
    d = df_all[df_all["client_id"] == client_id].copy()

    if d.empty:
        return [], "", "", "", ""

    # text search
    if query:
        q = str(query).lower()
        mask = (
            d["connection_name"].str.lower().str.contains(q, na=False)
            | d["organization"].str.lower().str.contains(q, na=False)
            | d["connection_type"].str.lower().str.contains(q, na=False)
        )
        d = d[mask]

    # score filter
    if min_score is not None:
        d = d[d["score"] >= float(min_score)]

    # boolean pretty
    if "is_client" in d.columns:
        d["is_client"] = d["is_client"].map({True: "Yes", False: "No"})

    # header
    r0 = df_all[df_all["client_id"] == client_id].iloc[0]
    client_name = r0.get("client_name", "")
    ace = r0.get("primary_ace", "")
    aum = f"${float(r0.get('est_aum', 0)):,.0f}"
    nw = f"${float(r0.get('est_nw', 0)):,.0f}"

    d = d.sort_values("score", ascending=False)
    return d.to_dict("records"), client_name, ace, aum, nw


@app.callback(
    Output("graph-container", "children"),
    Input("conn-table", "data"),
    Input("conn-table", "selected_rows"),
)
def update_graph(table_rows, selected_rows):
    try:
        d = pd.DataFrame(table_rows or [])
        if d.empty:
            return html.Div("No connections found", style={"padding": "1rem"})
        focus = None
        if selected_rows and len(selected_rows) > 0:
            focus = str(d.iloc[selected_rows[0]]["connection_name"])
        html_str = build_pyvis_graph_html(d, focus)
        return html.Iframe(
            srcDoc=html_str,
            style={"width": "100%", "height": "520px", "border": "0"},
            sandbox="allow-scripts allow-same-origin",
        )
    except Exception as e:
        return html.Pre(f"Graph failed: {e}", style={"color": "crimson", "whiteSpace": "pre-wrap"})


# ----------------------------- Main -----------------------------
if __name__ == "__main__":
    app.run_server(debug=True)