import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table

def create_filter_options(option_list):
    return [{'label': 'All', 'value': 'All'}] + [{'label': opt, 'value': opt} for opt in sorted(list(option_list))]

def create_multi_filter_options(option_list):
    return [{'label': 'All', 'value': 'All'}] + [{'label': opt, 'value': opt} for opt in sorted(list(option_list))]

def _ensure_children_list(x):
    # Accept a single component or a list of components; return a flat list
    if x is None:
        return []
    if isinstance(x, list) or isinstance(x, tuple):
        out = []
        for item in x:
            # if caller accidentally passed nested lists, flatten one level
            if isinstance(item, (list, tuple)):
                out.extend(item)
            else:
                out.append(item)
        return out
    return [x]


def create_kpi_card(title: str, kpi_id: str, color: str, width: int = 4, md_width: int = 6, children=None) -> dbc.Col:
    icon_map = {
        "primary": "bi-bar-chart",
        "success": "bi-graph-up-arrow",
        "info": "bi-info-circle",
        "danger": "bi-exclamation-triangle",
        "warning": "bi-lightning",
        "secondary": "bi-currency-dollar",
        "dark": "bi-pie-chart"
    }
    icon = icon_map.get(color, "bi-bar-chart")
    # Colorful gradient variants for KPI cards
    # Map to nicer gradient backgrounds (CSS classes are also provided in theme.css)
    variant = color if color in ("primary", "success", "info", "danger", "warning", "secondary", "dark") else "primary"
    # Icon should be white on colorful backgrounds
    icon_style = {"fontSize": "1.25rem", "color": "#ffffff"}

    inner = children if children is not None else html.Div(html.H3(id=kpi_id, className="kpi-value"))
    inner_children = _ensure_children_list(inner)
    return dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.Div(html.I(className=f"bi {icon}", style=icon_style), className="me-3"),
                    html.Div(html.H6(title, style={"margin": 0, "color": "rgba(255,255,255,0.95)", "fontWeight": 600})),
                ], className="d-flex align-items-center mb-2"),
                html.Div(className="graph-panel", children=inner_children)
            ], style={"padding": "0.6rem 0.75rem"}),
            class_name=f"shadow-sm border-0 rounded-3 kpi-card {variant}"
        ),
        lg=width, md=md_width, sm=12, class_name="mb-4"
    )

def create_graph_card(graph_id: str, title: str = None, width: int = 6, lg_width: int = None, height: int = 360, children=None) -> dbc.Col:
    # wrap the graph in a .graph-panel for improved inner contrast on dark theme
    # If a `children` element is provided, render it inside the card instead of the default dcc.Graph
    if children is not None:
        card_inner = children
    else:
        graph = dcc.Graph(id=graph_id, config={"displayModeBar": False, "responsive": True}, style={"height": f"{height}px"})
        card_inner = html.Div(graph, className="graph-panel")

    inner_children = _ensure_children_list(card_inner)
    card_content = []
    if title:
        card_content.append(html.H6(title, className="card-title mb-2 text-secondary fw-semibold"))
    card_content.extend(inner_children)
    lg_col_width = lg_width if lg_width is not None else width
    return dbc.Col(
        dbc.Card(
            dbc.CardBody(card_content, style={"padding": "0.75rem 1rem"}),
            class_name="shadow-sm border-0 rounded-3"
        ),
        lg=lg_col_width, md=width, sm=12, class_name="mb-3"
    )

def create_datatable_card(table_id: str, title: str, width: int = 6, lg_width: int = None, children=None) -> dbc.Col:
    lg_col_width = lg_width if lg_width is not None else width
    # If children provided, render them inside the card body instead of the default DataTable
    if children is not None:
        body_children = children
    else:
        body_children = dash_table.DataTable(
            id=table_id,
            style_cell={
                'textAlign': 'left',
                'backgroundColor': 'transparent',
                'color': '#e6eefc',
                'fontSize': '0.95rem',
                'border': 'none',
                'padding': '0.5rem 0.25rem'
            },
            style_header={
                'backgroundColor': 'transparent',
                'color': 'var(--muted-300)',
                'fontWeight': 'bold',
                'border': 'none',
                'fontSize': '1rem'
            },
            page_size=10,
            sort_action='native',
            style_table={'overflowX': 'auto', 'borderRadius': '8px', 'boxShadow': '0 6px 18px rgba(2,8,23,0.6)', 'height': '340px'}
        )

    body_children_list = _ensure_children_list(body_children)

    return dbc.Col(
        dbc.Card(
            dbc.CardBody([html.H6(title, className="card-title mb-2 text-secondary fw-semibold")] + body_children_list, style={"padding": "1rem 0.75rem"}),
            class_name="shadow-sm"
        ),
        lg=lg_col_width, md=width, sm=12, class_name="mb-3"
    )
