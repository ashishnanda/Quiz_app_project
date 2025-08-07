from dash.dependencies import Input, Output, ALL
from dash import callback_context, html

@app.callback(
    Output("fa-model-card-content", "children"),
    Input("prospect-search", "value"),
    Input({"type": "prospect-card", "index": ALL}, "n_clicks"),
)
def render_fa_model_card(search_value, all_clicks):
    # --- 1) figure out which prospect is selected ---
    ctx = callback_context
    triggered = ctx.triggered[0]["prop_id"]
    # your existing logic to get `selected_name`, then:
    data = get_prospect_data(selected_name)
    fit_model = data.get("fit_model")  # or however you pull it

    # --- 2) handle no-data case ---
    if not fit_model:
        return html.Div("No connections found")

    # --- 3) build the header ---
    header = html.Div(
      html.Span("FA model", style={"fontWeight": "bold", "fontSize": "18px"})
    )

    # --- 4) build the list of connection entries ---
    entries = []
    for conn in fit_model:
        block = []

        # Name + badge
        block.append(
          html.Div([
            html.Strong(conn["name"]),
            source_badge(conn["entity_type"])
          ])
        )

        block.append(html.Br())

        # Best-fit indicator
        if conn.get("best_fit_indicator") == 1:
            block.append(
              html.Div(
                html.Span("Best Fit Client", style={"fontWeight": "bold"})
              )
            )

        # Net worth
        block.append(
          html.Div([
            html.Span("Net Worth: "),
            html.Span(conn.get("networth", "—"), style={"fontWeight": "bold"})
          ])
        )

        # Intermediaries
        mids = conn.get("intermediaries", {})
        block.append(
          html.Div([
            html.Span("Intermediaries: "),
            html.Span(
              ", ".join(f"{k} ({v})" for k, v in mids.items()),
              style={"fontWeight": "bold"}
            )
          ])
        )

        # Lead FA
        if conn.get("lead_financial_advisor"):
            block.append(
              html.Div([
                html.Span("Lead Financial Advisor: "),
                html.Span(conn["lead_financial_advisor"], style={"fontWeight": "bold"})
              ])
            )

        # Separator
        block.append(html.Hr())

        entries.append(html.Div(block))

    container = html.Div(
      entries,
      style={
        "maxHeight": "300px",
        "overflow": "auto",
        "paddingRight": "10px"
      }
    )

    # --- 5) return header + container ---
    return [header, container]