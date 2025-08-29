# app.py
import warnings
warnings.filterwarnings("ignore")

import uuid
from typing import List, Dict

import pandas as pd
import networkx as nx

import dash
from dash import html, dcc, Input, Output, State, dash_table, ctx, ALL
import dash_bootstrap_components as dbc
from dash.dash_table import Format

from pyvis.network import Network


# =======================================
# 0) DEMO DATA  (replace with your loader)
# =======================================
def demo_connections_df() -> pd.DataFrame:
    """
    Long-form table. One row = (client, organization, connection).
    Columns used by the UI & graph:
      client_id, client_name, client_is_fav, primary_ace, est_aum, client_est_nw
      connection_name, connection_is_client, connection_est_nw
      connection_type, organization, score
      client_org_relation, conn_org_relation
    """
    rows: List[Dict] = [
        # -------- Client 1: Jamie Rose (fav) --------
        dict(client_id="C001", client_name="Jamie Rose", client_is_fav=1, primary_ace="Greta Clark",
             est_aum=2_170_000, client_est_nw=56_750_000,
             connection_name="Adam Wiles", connection_is_client=True, connection_est_nw=12_300_000,
             connection_type="School Board",
             organization="NYU; Red Cross of New York",
             score=4.83,
             client_org_relation="MBA 2003; Donor/Chair",
             conn_org_relation="Secretary; Volunteer"),

        dict(client_id="C001", client_name="Jamie Rose", client_is_fav=1, primary_ace="Greta Clark",
             est_aum=2_170_000, client_est_nw=56_750_000,
             connection_name="Marion Bishop", connection_is_client=False, connection_est_nw=4_600_000,
             connection_type="Board (Corporate)",
             organization="Adobe Systems",
             score=4.66,
             client_org_relation="Independent Director",
             conn_org_relation="Director"),

        dict(client_id="C001", client_name="Jamie Rose", client_is_fav=1, primary_ace="Greta Clark",
             est_aum=2_170_000, client_est_nw=56_750_000,
             connection_name="Steven Walker", connection_is_client=False, connection_est_nw=2_150_000,
             connection_type="School",
             organization="NYU",
             score=4.61,
             client_org_relation="Alumnus",
             conn_org_relation="Alumnus"),

        dict(client_id="C001", client_name="Jamie Rose", client_is_fav=1, primary_ace="Greta Clark",
             est_aum=2_170_000, client_est_nw=56_750_000,
             connection_name="John Phillips", connection_is_client=False, connection_est_nw=3_900_000,
             connection_type="Employer",
             organization="Champion Health",
             score=4.60,
             client_org_relation="Advisor to CEO",
             conn_org_relation="VP, Ops"),

        # -------- Client 2: Ava Patel (fav) --------
        dict(client_id="C002", client_name="Ava Patel", client_is_fav=1, primary_ace="Daniel Ortiz",
             est_aum=4_500_000, client_est_nw=84_200_000,
             connection_name="Rohan Mehta", connection_is_client=True, connection_est_nw=21_000_000,
             connection_type="Employer",
             organization="Nexon Biotech; IIT Bombay Alumni Association",
             score=4.74,
             client_org_relation="CEO; Alumnus",
             conn_org_relation="CFA; Alumnus"),

        dict(client_id="C002", client_name="Ava Patel", client_is_fav=1, primary_ace="Daniel Ortiz",
             est_aum=4_500_000, client_est_nw=84_200_000,
             connection_name="Grace Lin", connection_is_client=False, connection_est_nw=5_400_000,
             connection_type="Board (Charitable)",
             organization="Girls Who Code",
             score=4.52,
             client_org_relation="Patron/Board",
             conn_org_relation="Chapter Lead"),

        dict(client_id="C002", client_name="Ava Patel", client_is_fav=1, primary_ace="Daniel Ortiz",
             est_aum=4_500_000, client_est_nw=84_200_000,
             connection_name="Leo Fischer", connection_is_client=False, connection_est_nw=7_200_000,
             connection_type="Club",
             organization="World Economic Forum",
             score=4.31,
             client_org_relation="Young Global Leader",
             conn_org_relation="Member"),

        # -------- Client 3: Liam Chen (not fav) --------
        dict(client_id="C003", client_name="Liam Chen", client_is_fav=0, primary_ace="Amelia Brown",
             est_aum=1_250_000, client_est_nw=32_600_000,
             connection_name="Noah Garcia", connection_is_client=False, connection_est_nw=1_900_000,
             connection_type="School",
             organization="Stanford University; IEEE",
             score=4.20,
             client_org_relation="MS CS 2010; Senior Member",
             conn_org_relation="PhD 2012; Member"),

        dict(client_id="C003", client_name="Liam Chen", client_is_fav=0, primary_ace="Amelia Brown",
             est_aum=1_250_000, client_est_nw=32_600_000,
             connection_name="Sophia Rossi", connection_is_client=True, connection_est_nw=9_800_000,
             connection_type="Employer",
             organization="QuantFlow Capital",
             score=4.45,
             client_org_relation="Partner",
             conn_org_relation="CFA; PM"),
    ]
    return pd.DataFrame(rows)


# =====================================================
# 1) UI BUILDING BLOCKS (favorites, filters, table, etc)
# =====================================================
COLOR_CLIENT = "#CC0000"     # red
COLOR_ORG = "#8fbff6"        # light blue
COLOR_CONN = "#2ca58d"       # teal

def money(x) -> str:
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return "-"

def build_client_options(df: pd.DataFrame):
    if df.empty:
        return []
    d = df[["client_id", "client_name"]].drop_duplicates().astype(str)
    return [{"value": v, "label": f"{n} ({v})"} for v, n in zip(d.client_id, d.client_name)]

def favorites_panel(df: pd.DataFrame) -> dbc.Card:
    favs = (
        df[df["client_is_fav"] == 1][["client_id", "client_name", "client_est_nw"]]
        .drop_duplicates()
        .sort_values("client_name")
    )

    items = []
    for _, r in favs.iterrows():
        items.append(
            dbc.ListGroupItem(
                [
                    html.Div(r["client_name"], className="fw-bold"),
                    html.Small(f"Net worth: {money(r['client_est_nw'])}", className="text-muted")
                ],
                id={"type": "fav-item", "value": r["client_id"]},
                action=True,
                n_clicks=0,
            )
        )

    body = dbc.CardBody(
        [
            html.H6("Favourites", className="mb-2"),
            dbc.ListGroup(items) if items else html.Small("No favourites yet"),
        ]
    )
    return dbc.Card(body, className="shadow-sm")

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
                                    id="search-box", type="text", debounce=True,
                                    placeholder="name / org / type…", className="form-control",
                                ),
                            ],
                            md=5,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Min score"),
                                dcc.Slider(
                                    id="min-score", min=0, max=5, step=0.1, value=4.0,
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
        {"name": "Client?", "id": "connection_is_client"},
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

def legend_box():
    dot = lambda c: html.Span(style={"display": "inline-block", "width": "10px", "height": "10px",
                                     "borderRadius": "50%", "background": c, "marginRight": "6px"})
    return html.Div(
        [
            html.Small([dot(COLOR_CLIENT), "Client"], className="me-3"),
            html.Small([dot(COLOR_ORG), "Organization"], className="me-3"),
            html.Small([dot(COLOR_CONN), "Connection"], className="me-3"),
        ],
        className="mb-2",
    )

def make_graph_card():
    return dbc.Card(
        dbc.CardBody(
            [
                html.H6("Connections graph", className="mb-2"),
                legend_box(),
                html.Div(id="graph-container"),
            ]
        ),
        className="shadow-sm",
        style={"minHeight": "560px"},
    )


# =======================================
# 2) GRAPH: Tripartite builder (on selection)
# =======================================
def build_tripartite_graph_html(row: pd.Series) -> str:
    """
    Build a tripartite graph for a SINGLE table row:
      - Top: Client node (level 0)
      - Middle: Organization node(s) (level 1; split by ';')
      - Bottom: Connection node (level 2)
    Edge labels show client_org_relation and conn_org_relation.
    """
    # Safety
    if row is None or row.empty:
        return "<div style='padding:1rem'>No selection</div>"

    client = str(row["client_name"])
    conn = str(row["connection_name"])
    orgs = [o.strip() for o in str(row["organization"]).split(";") if o.strip()]

    client_nw = row.get("client_est_nw", 0)
    conn_nw = row.get("connection_est_nw", 0)
    conn_is_client = bool(row.get("connection_is_client", False))

    client_rel = str(row.get("client_org_relation", ""))
    conn_rel = str(row.get("conn_org_relation", ""))

    # Graph
    G = nx.Graph()

    # Nodes with explicit levels for hierarchical layout
    G.add_node(client, kind="client", level=0,
               title=f"<b>{client}</b><br>Net worth: {money(client_nw)}")

    for o in orgs:
        G.add_node(o, kind="org", level=1, title=f"{o}")

    conn_title = f"<b>{conn}</b><br>Net worth: {money(conn_nw)}<br>Is client: {'Yes' if conn_is_client else 'No'}"
    G.add_node(conn, kind="connection", level=2, title=conn_title)

    # Edges
    for o in orgs:
        G.add_edge(client, o, label=client_rel, title=client_rel)
        G.add_edge(conn, o, label=conn_rel, title=conn_rel)

    # PyVis network
    net = Network(height="520px", width="100%", bgcolor="#ffffff", font_color="#333333",
                  notebook=False, directed=False)

    # Hierarchical layout (top→bottom)
    net.set_options("""
    {
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "UD",
          "sortMethod": "directed",
          "nodeSpacing": 200,
          "levelSeparation": 160
        }
      },
      "interaction": { "hover": true },
      "physics": { "enabled": false }
    }
    """)

    # Add nodes
    for n, attrs in G.nodes(data=True):
        kind = attrs["kind"]
        level = int(attrs.get("level", 1))
        color = COLOR_CLIENT if kind == "client" else (COLOR_CONN if kind == "connection" else COLOR_ORG)
        size = 28 if kind == "client" else (18 if kind == "connection" else 16)
        border = 4 if kind == "client" else 1

        net.add_node(n, label=n, title=attrs.get("title", n),
                     color=color, size=size, borderWidth=border, level=level)

    # Add edges with labels
    for s, t, attrs in G.edges(data=True):
        net.add_edge(s, t, label=attrs.get("label", ""), title=attrs.get("title", ""))

    # Return HTML
    try:
        return net.generate_html(notebook=False)
    except Exception:
        from pathlib import Path
        import tempfile
        tmp = Path(tempfile.gettempdir()) / f"pyvis_{uuid.uuid4().hex}.html"
        net.write_html(tmp, notebook=False)
        return tmp.read_text(encoding="utf-8")


# ======================
# 3) APP INITIALIZATION
# ======================
df = demo_connections_df()

# ---- per your screenshot ----
dash_port = 8891
app = dash.Dash(
    requests_pathname_prefix=f"/user/ashish.nanda@ubs.com/proxy/{dash_port}/",
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)
app.title = "Client Connections"


# =========
# 4) LAYOUT
# =========
app.layout = dbc.Container(
    [
        make_header(),

        # Whole page split: left favourites, right main content
        dbc.Row(
            [
                dbc.Col(favorites_panel(df), md=3),

                dbc.Col(
                    [
                        make_filters(df),
                        dbc.Row(
                            [
                                dbc.Col([make_client_stats(),
                                         dbc.Card(dbc.CardBody([make_table()]))], md=7),
                                dbc.Col([make_graph_card()], md=5),
                            ],
                            className="g-3",
                        ),
                        dcc.Store(id="store-all", data=df.to_dict("records")),
                    ],
                    md=9,
                ),
            ],
            className="g-3",
        ),
    ],
    fluid=True,
)


# ============
# 5) CALLBACKS
# ============
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

    # client header
    r0 = df_all[df_all["client_id"] == client_id].iloc[0]
    client_name = r0.get("client_name", "")
    ace = r0.get("primary_ace", "")
    aum = money(r0.get("est_aum", 0))
    nw = money(r0.get("client_est_nw", 0))

    # sort
    d = d.sort_values("score", ascending=False)

    return d.to_dict("records"), client_name, ace, aum, nw


# Click on a connection row → render tripartite graph
@app.callback(
    Output("graph-container", "children"),
    Input("conn-table", "data"),
    Input("conn-table", "selected_rows"),
)
def update_graph(table_rows, selected_rows):
    d = pd.DataFrame(table_rows or [])
    if d.empty:
        return html.Div("No connections found", style={"padding": "1rem"})

    if not selected_rows:
        return html.Div("Pick a connection in the table to render the graph.", style={"padding": "1rem"})

    try:
        sel = d.iloc[selected_rows[0]]
        html_str = build_tripartite_graph_html(sel)
        return html.Iframe(
            srcDoc=html_str,
            style={"width": "100%", "height": "520px", "border": "0"},
            sandbox="allow-scripts allow-same-origin",
        )
    except Exception as e:
        return html.Pre(f"Graph failed: {e}", style={"color": "crimson", "whiteSpace": "pre-wrap"})


# Favourites panel → set client dropdown
@app.callback(
    Output("client-dd", "value"),
    Input({"type": "fav-item", "value": ALL}, "n_clicks"),
    State({"type": "fav-item", "value": ALL}, "id"),
    prevent_initial_call=True,
)
def fav_click(n_clicks_list, id_list):
    if not n_clicks_list or not id_list:
        raise dash.exceptions.PreventUpdate
    # Identify which favourite was clicked last
    triggered = ctx.triggered_id
    if not triggered:
        raise dash.exceptions.PreventUpdate
    return triggered["value"]


# =========
# 6) MAIN
# =========
if __name__ == "__main__":
    # When using requests_pathname_prefix, pick the same port in your proxy URL
    app.run_server(host="0.0.0.0", port=dash_port, debug=True)