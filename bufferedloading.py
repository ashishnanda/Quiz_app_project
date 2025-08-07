from dash.dependencies import Input, Output, ALL
from dash import html, dcc, callback_context

@app.callback(
    Output("client-graph-card-content", "children"),
    Input("prospect-search", "value"),
    Input({"type": "prospect-card", "index": ALL}, "n_clicks"),
)
def render_client_graph_card(search_value, all_clicks):
    # figure out which prospect is selected
    ctx = callback_context
    triggered = ctx.triggered[0]["prop_id"]
    # your existing logic to derive `selected_name`, `fit_model`, `id_networth`, etc.
    if not selected_name:
        return html.Div("Select a prospect to view details.")

    # load data, model, whatever
    data = get_prospect_data(selected_name)
    if not data or not fit_model:
        return html.Div("No data available for this prospect.")

    # 1) build the header
    header = html.Span(
      "Client Relationship Graph",
      style={"fontWeight": "bold", "fontSize": "18px"}
    )

    # 2) build the slow-to-compute graph
    fig = plot_bipartite_graph(
      fit_model=fit_model,
      input_node_network=id_networth,
      name=selected_name
    )
    graph = dcc.Graph(figure=fig, config={"displayModeBar": False})

    # 3) build the FA & summary sections
    fa_section = html.Div([
      html.Div([html.Span("CRM FA: ", style={"fontWeight": "bold"}),
                html.Span(main_fa if main_fa else "—")]),
      html.Br(),
      html.Div([html.Span("Best Fit FA: ", style={"fontWeight": "bold"}),
                html.Span(lead_fa if lead_fa else "—")]),
      html.Br(),
      html.Div([html.Span("Best Fit Relationship: ", style={"fontWeight": "bold"}),
                html.Span(ai_relationship_summary if ai_relationship_summary else "—")]),
    ], style={"marginTop": "10px"})

    # 4) return all of it in a list
    return [
      header,
      html.Div(graph, style={"marginTop": "15px"}),
      fa_section
    ]