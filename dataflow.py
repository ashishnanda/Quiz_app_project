import yaml
import pandas as pd
from sqlalchemy import create_engine
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import datetime

def load_config(path="config.yml"):
    """Load database URI and SQL snippets from YAML."""
    with open(path) as f:
        return yaml.safe_load(f)

def get_engine(db_uri):
    """Create a SQLAlchemy engine."""
    return create_engine(db_uri)

def run_query(engine, sql, **params):
    """Run a single SQL snippet, substituting params like run_date."""
    rendered = sql.replace("{{run_date}}", params["run_date"])
    return pd.read_sql(rendered, engine).iloc[0]["value"]

def fetch_wx_metrics(engine, queries, run_date):
    """Fetch all required WX metrics in one shot."""
    m = {}
    for key, sql in queries["wx_dossier"].items():
        m[key] = run_query(engine, sql, run_date=run_date)
    return m

def make_kpi_cards(metrics):
    """Return a row of simple KPI cards."""
    cards = []
    definitions = {
        "ingestion_count":      "Rows ingested",
        "raw_mapping_count":    "Raw mappings",
        "duplicate_mapping_count":"Dup. in mappings",
        "filtered_count":       "Passed is_current='Y'",
        "loaded_count":         "Loaded to master"
    }
    for metric, label in definitions.items():
        cards.append(
            html.Div([
                html.H4(f"{metrics[metric]:,}"),
                html.P(label)
            ], className="kpi-card")
        )
    return html.Div(cards, className="kpi-row")

def make_funnel_chart(metrics):
    """Build a 5-step funnel from the metrics."""
    stages = [
        ("Ingested",     metrics["ingestion_count"]),
        ("Mapped",       metrics["raw_mapping_count"]),
        ("Deduped",      metrics["raw_mapping_count"] - metrics["duplicate_mapping_count"]),
        ("Filtered",     metrics["filtered_count"]),
        ("Loaded",       metrics["loaded_count"])
    ]
    fig = go.Figure(go.Funnel(
        y=[s[0] for s in stages],
        x=[s[1] for s in stages],
        textinfo="value+percent initial"
    ))
    return dcc.Graph(figure=fig)

def make_duplicates_table(engine, run_date):
    """Show the actual duplicate mapping rows for inspection."""
    sql = """
    SELECT dossier_id, COUNT(DISTINCT internal_client_id) AS cnt
    FROM wx_mapping
    WHERE run_date = '{run_date}'
    GROUP BY dossier_id
    HAVING COUNT(DISTINCT internal_client_id) > 1
    LIMIT 50;
    """.format(run_date=run_date)
    df = pd.read_sql(sql, engine)
    return html.Table([
        html.Thead(html.Tr([html.Th(c) for c in df.columns])),
        html.Tbody([
            html.Tr([html.Td(df.iloc[i][c]) for c in df.columns])
            for i in range(len(df))
        ])
    ], className="dup-table")

# ────────────────────────────────────────────────────────────────────────────────

# Load config and initialize at import time:
cfg    = load_config("config.yml")
engine = get_engine(cfg["database"]["uri"])

app = dash.Dash(__name__)
app.layout = html.Div([
    html.H2("WX – Dossier Ingestion Dashboard"),
    html.Div([
        html.Label("Run date:"),
        dcc.DatePickerSingle(
            id="run-date-picker",
            date=datetime.date.today()
        )
    ]),
    html.Div(id="kpi-container"),
    html.Div(id="funnel-container"),
    html.H3("Duplicate Mappings Sample"),
    html.Div(id="dup-table-container")
], style={"width":"80%", "margin":"auto"})

@app.callback(
    [Output("kpi-container", "children"),
     Output("funnel-container", "children"),
     Output("dup-table-container", "children")],
    [Input("run-date-picker", "date")]
)
def update_dash(run_date):
    # Convert run_date to YYYY-MM-DD string if needed
    rd = run_date[:10] if isinstance(run_date, str) else run_date.isoformat()

    # 1. Fetch metrics
    metrics = fetch_wx_metrics(engine, cfg["sql_queries"], rd)

    # 2. Build components
    kpis   = make_kpi_cards(metrics)
    funnel = make_funnel_chart(metrics)
    dup_tbl = make_duplicates_table(engine, rd)

    return kpis, funnel, dup_tbl

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)