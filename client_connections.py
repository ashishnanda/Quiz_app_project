# app.py
import warnings
warnings.filterwarnings("ignore")

import uuid
from typing import List, Dict

import pandas as pd
import networkx as nx

import dash
from dash import html, dcc, Input, Output, State, dash_table, ALL, ctx   # <- ctx imported
import dash_bootstrap_components as dbc
from dash.dash_table import Format
from pyvis.network import Network

# ----------------------------------
# Colors for legend / node categories
# ----------------------------------
COLOR_CLIENT = "#CC0000"     # red
COLOR_ORG    = "#8fbff6"     # light blue
COLOR_CONN   = "#2ca58d"     # teal

def money(x) -> str:
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return "-"

# =========================================================
# 0) DEMO DATA (each org in a separate row with org_type)
# =========================================================
def demo_connections_df() -> pd.DataFrame:
    """
    Long-form: one row per (client, connection, organization).
    Columns:
      client_id, client_name, client_is_fav, primary_ace, est_aum, client_est_nw
      connection_name, connection_est_nw
      organization, org_type, score
      client_org_relation, conn_org_relation
    """
    rows: List[Dict] = [
        # ---------- Client 1: Jamie Rose (fav) ----------
        # Jamie & Adam share NYU and Red Cross as separate rows
        dict(client_id="C001", client_name="Jamie Rose", client_is_fav=1, primary_ace="Greta Clark",
             est_aum=2_170_000, client_est_nw=56_750_000,
             connection_name="Adam Wiles", connection_est_nw=12_300_000,
             organization="NYU", org_type="School", score=4.83,
             client_org_relation="MBA 2003", conn_org_relation="Alumnus"),
        dict(client_id="C001", client_name="Jamie Rose", client_is_fav=1, primary_ace="Greta Clark",
             est_aum=2_170_000, client_est_nw=56_750_000,
             connection_name="Adam Wiles", connection_est_nw=12_300_000,
             organization="Red Cross of New York", org_type="Philanthropy", score=4.83,
             client_org_relation="Donor/Chair", conn_org_relation="Secretary"),

        # Jamie & Marion share Adobe (corporate board)
        dict(client_id="C001", client_name="Jamie Rose", client_is_fav=1, primary_ace="Greta Clark",
             est_aum=2_170_000, client_est_nw=56_750_000,
             connection_name="Marion Bishop", connection_est_nw=4_600_000,
             organization="Adobe Systems", org_type="Organisation", score=4.66,
             client_org_relation="Independent Director", conn_org_relation="Director"),

        # Jamie & Steven share NYU
        dict(client_id="C001", client_name="Jamie Rose", client_is_fav=1, primary_ace="Greta Clark",
             est_aum=2_170_000, client_est_nw=56_750_000,
             connection_name="Steven Walker", connection_est_nw=2_150_000,
             organization="NYU", org_type="School", score=4.61,
             client_org_relation="Alumnus", conn_org_relation="Alumnus"),

        # Jamie & John share Champion Health (employer)
        dict(client_id="C001", client_name="Jamie Rose", client_is_fav=1, primary_ace="Greta Clark",
             est_aum=2_170_000, client_est_nw=56_750_000,
             connection_name="John Phillips", connection_est_nw=3_900_000,
             organization="Champion Health", org_type="Organisation", score=4.60,
             client_org_relation="Advisor to CEO", conn_org_relation="VP, Ops"),

        # ---------- Client 2: Ava Patel (fav) ----------
        # Ava & Rohan share Nexon Biotech and IIT Bombay Alumni Association (2 rows)
        dict(client_id="C002", client_name="Ava Patel", client_is_fav=1, primary_ace="Daniel Ortiz",
             est_aum=4_500_000, client_est_nw=84_200_000,
             connection_name="Rohan Mehta", connection_est_nw=21_000_000,
             organization="Nexon Biotech", org_type="Organisation", score=4.74,
             client_org_relation="CEO", conn_org_relation="CFO"),
        dict(client_id="C002", client_name="Ava Patel", client_is_fav=1, primary_ace="Daniel Ortiz",
             est_aum=4_500_000, client_est_nw=84_200_000,
             connection_name="Rohan Mehta", connection_est_nw=21_000_000,
             organization="IIT Bombay Alumni Association", org_type="School", score=4.30,
             client_org_relation="Alumnus", conn_org_relation="CFA; Alumnus"),

        dict(client_id="C002", client_name="Ava Patel", client_is_fav=1, primary_ace="Daniel Ortiz",
             est_aum=4_500_000, client_est_nw=84_200_000,
             connection_name="Grace Lin", connection_est_nw=5_400_000,
             organization="Girls Who Code", org_type="Philanthropy", score=4.52,
             client_org_relation="Patron/Board", conn_org_relation="Chapter Lead"),

        dict(client_id="C002", client_name="Ava Patel", client_is_fav=1, primary_ace="Daniel Ortiz",
             est_aum=4_500_000, client_est_nw=84_200_000,
             connection_name="Leo Fischer", connection_est_nw=7_200_000,
             organization="World Economic Forum", org_type="Organisation", score=4.31,
             client_org_relation="Young Global Leader", conn_org_relation="Member"),

        # ---------- Client 3: Liam Chen (not fav) ----------
        dict(client_id="C003", client_name="Liam Chen", client_is_fav=0, primary_ace="Amelia Brown",
             est_aum=1_250_000, client_est_nw=32_600_000,
             connection_name="Noah Garcia", connection_est_nw=1_900_000,
             organization="Stanford University", org_type="School", score=4.20,
             client_org_relation="MS CS 2010", conn_org_relation="PhD 2012"),
        dict(client_id="C003", client_name="Liam Chen", client_is_fav=0, primary_ace="Amelia Brown",
             est_aum=1_250_000, client_est_nw=32_600_000,
             connection_name="Noah Garcia", connection_est_nw=1_900_000,
             organization="IEEE", org_type="Organisation", score=4.10,
             client_org_relation="Senior Member", conn_org_relation="Member"),

        dict(client_id="C003", client_name="Liam Chen", client_is_fav=0, primary_ace="Amelia Brown",
             est_aum=1_250_000, client_est_nw=32_600_000,
             connection_name="Sophia Rossi", connection_est_nw=9_800_000,
             organization="QuantFlow Capital", org_type="Organisation", score=4.45,
             client_org_relation="Partner", conn_org_relation="PM"),
    ]
    return pd.DataFrame(rows)

# ======================
# 1) UI COMPONENTS
# ======================
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
        [html.H6("Favourites", className="mb-2"),
         dbc.ListGroup(items) if items else html.Small("No favourites yet")]
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
    # one row per connection, aggregated fields
    columns = [
        {"name": "Name", "id": "connection_name"},
        {"name": "Organizations", "id": "orgs_concat"},
        {"name": "Org Types", "id": "org_types_concat"},
        {"name": "Score", "id": "score", "type": "numeric",
         "format": Format(precision=2, scheme=Format.Scheme.fixed)},
        {"name": "Connection Net Worth", "id": "connection_net_worth"},
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

# =======================================================
# 2) GRAPH: Tripartite builder from a subset of rows
# =======================================================
def build_tripartite_graph_html_from_subset(rows: pd.DataFrame) -> str:
    """
    rows: all rows for a chosen (client, connection) across multiple organizations.
    This renders:
      Level 0: client
      Level 1: each organization node in `rows['organization']`
      Level 2: connection
    Edges:
      client --org--> organization  (label=client_org_relation)
      connection --org--> organization (label=conn_org_relation)
    """
    if rows.empty:
        return "<div style='padding:1rem'>No selection</div>"

    client = rows.iloc[0]["client_name"]
    client_nw = rows.iloc[0]["client_est_nw"]
    connection = rows.iloc[0]["connection_name"]
    connection_nw = rows.iloc[0]["connection_est_nw"]

    G = nx.Graph()

    # Client node (top)
    G.add_node(
        client, kind="client", level=0,
        title=f"<b>{client}</b><br>Net worth: {money(client_nw)}"
    )

    # Connection node (bottom)
    G.add_node(
        connection, kind="connection", level=2,
        title=f"<b>{connection}</b><br>Net worth: {money(connection_nw)}"
    )

    # Organization nodes (middle) + edges with labels
    for _, r in rows.iterrows():
        org = r["organization"]
        org_title = f"{org}<br><i>{r.get('org_type','')}</i>"
        G.add_node(org, kind="org", level=1, title=org_title)
        G.add_edge(client, org,
                   label=str(r.get("client_org_relation", "")),
                   title=str(r.get("client_org_relation", "")))
        G.add_edge(connection, org,
                   label=str(r.get("conn_org_relation", "")),
                   title=str(r.get("conn_org_relation", "")))

    # PyVis network (hierarchical UD)
    net = Network(height="520px", width="100%", bgcolor="#ffffff", font_color="#333333",
                  notebook=False, directed=False)
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

    for n, attrs in G.nodes(data=True):
        kind = attrs["kind"]
        level = int(attrs.get("level", 1))
        color = COLOR_CLIENT if kind == "client" else (COLOR_CONN if kind == "connection" else COLOR_ORG)
        size = 28 if kind == "client" else (18 if kind == "connection" else 16)
        border = 4 if kind == "client" else 1
        net.add_node(n, label=n, title=attrs.get("title", n),
                     color=color, size=size, borderWidth=border, level=level)

    for s, t, attrs in G.edges(data=True):
        net.add_edge(s, t, label=attrs.get("label", ""), title=attrs.get("title", ""))

    try:
        return net.generate_html(notebook=False)
    except Exception:
        from pathlib import Path
        import tempfile
        tmp = Path(tempfile.gettempdir()) / f"pyvis_{uuid.uuid4().hex}.html"
        net.write_html(tmp, notebook=False)
        return tmp.read_text(encoding="utf-8")

# ======================
# 3) APP INIT (proxy)
# ======================
df = demo_connections_df()

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
                        dcc.Store(id="store-client-rows"),  # detailed rows after filter (per client)
                    ],
                    md=9,
                ),
            ],
            className="g-3",
        ),
    ],
    fluid=True,
)

# ======================
# 5) CALLBACKS
# ======================
@app.callback(
    Output("conn-table", "data"),
    Output("client-name", "children"),
    Output("client-ace", "children"),
    Output("client-aum", "children"),
    Output("client-nw", "children"),
    Output("store-client-rows", "data"),
    Input("store-all", "data"),
    Input("client-dd", "value"),
    Input("search-box", "value"),
    Input("min-score", "value"),
)
def update_table(all_rows, client_id, query, min_score):
    df_all = pd.DataFrame(all_rows or [])
    detailed = df_all[df_all["client_id"] == client_id].copy()

    if detailed.empty:
        return [], "", "", "", "", []

    # search across connection/organization/type
    if query:
        q = str(query).lower()
        mask = (
            detailed["connection_name"].str.lower().str.contains(q, na=False)
            | detailed["organization"].str.lower().str.contains(q, na=False)
            | detailed["org_type"].str.lower().str.contains(q, na=False)
        )
        detailed = detailed[mask]

    # score filter
    if min_score is not None:
        detailed = detailed[detailed["score"] >= float(min_score)]

    # header values from any row for this client
    r0 = df_all[df_all["client_id"] == client_id].iloc[0]
    client_name = r0.get("client_name", "")
    ace = r0.get("primary_ace", "")
    aum = money(r0.get("est_aum", 0))
    nw = money(r0.get("client_est_nw", 0))

    if detailed.empty:
        return [], client_name, ace, aum, nw, []

    # ---- Aggregate to one row per connection ----
    agg = (
        detailed.groupby("connection_name")
        .agg(
            orgs_concat=("organization", lambda s: "; ".join(pd.unique(s.astype(str)))),
            org_types_concat=("org_type", lambda s: "; ".join(pd.unique(s.astype(str)))),
            score=("score", "max"),  # or "mean"
            connection_est_nw=("connection_est_nw", "first"),
        )
        .reset_index()
    )
    agg["connection_net_worth"] = agg["connection_est_nw"].apply(money)
    agg = agg.drop(columns=["connection_est_nw"])

    # sort by score desc
    agg = agg.sort_values("score", ascending=False)

    return agg.to_dict("records"), client_name, ace, aum, nw, detailed.to_dict("records")

# Graph: click a connection row → use store-client-rows (detailed) to build tripartite
@app.callback(
    Output("graph-container", "children"),
    Input("conn-table", "data"),
    Input("conn-table", "selected_rows"),
    State("store-client-rows", "data"),
)
def update_graph(table_rows, selected_rows, detailed_rows):
    d_table = pd.DataFrame(table_rows or [])
    if d_table.empty:
        return html.Div("No connections found", style={"padding": "1rem"})
    if not selected_rows:
        return html.Div("Pick a connection in the table to render the graph.", style={"padding": "1rem"})

    try:
        sel_name = d_table.iloc[selected_rows[0]]["connection_name"]
        detailed = pd.DataFrame(detailed_rows or [])
        subset = detailed[detailed["connection_name"] == sel_name].copy()
        if subset.empty:
            return html.Div("No organization rows for this connection.", style={"padding": "1rem"})
        html_str = build_tripartite_graph_html_from_subset(subset)
        return html.Iframe(
            srcDoc=html_str,
            style={"width": "100%", "height": "520px", "border": "0"},
            sandbox="allow-scripts allow-same-origin",
        )
    except Exception as e:
        return html.Pre(f"Graph failed: {e}", style={"color": "crimson", "whiteSpace": "pre-wrap"})

# Favourites → set dropdown client
@app.callback(
    Output("client-dd", "value"),
    Input({"type": "fav-item", "value": ALL}, "n_clicks"),
    State({"type": "fav-item", "value": ALL}, "id"),
    prevent_initial_call=True,
)
def fav_click(n_clicks_list, id_list):
    if not n_clicks_list or not id_list:
        raise dash.exceptions.PreventUpdate
    triggered = ctx.triggered_id
    if not triggered:
        raise dash.exceptions.PreventUpdate
    return triggered["value"]

# =========
# 6) MAIN
# =========
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=dash_port, debug=True)