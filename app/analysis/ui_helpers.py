from typing import List, Dict, Any
import dash_bootstrap_components as dbc
from dash import html


def render_recommendations(recs: List[Dict[str, Any]], accordion_id: str = "recs-accordion", severity_whitelist: List[str] = None):
    """
    Render structured recommendations into a Bootstrap Accordion.

    Defensive behavior:
    - Accepts a variety of `recs` shapes (None, list of dicts, list of strings).
    - Skips entries that are completely malformed.
    - If an entry is missing a `text` field, uses a placeholder title "Recommendation"
      and shows "Details not available." in the body instead of crashing.
    - Ensures local variables are always defined to avoid UnboundLocalError.

    Expected rec item shape (preferred):
      {"text": "message", "severity": "info"|"warning"|"critical"}

    Returns:
      - `dbc.Accordion` with one item per valid recommendation, or
      - `dbc.Card` with an explanatory message if there are no valid recommendations.
    """
    severity_map = {
        'info': 'secondary',
        'warning': 'warning',
        'critical': 'danger'
    }

    # Fast path: empty or None input -> friendly placeholder card
    if not recs:
        return dbc.Card(dbc.CardBody(html.P("No recommendations at this time.")))

    # Optional severity filtering (expect list like ['critical','warning'])
    if severity_whitelist:
        try:
            whitelist = set([s.lower() for s in severity_whitelist if isinstance(s, str)])
            recs = [r for r in recs if (isinstance(r, dict) and str(r.get('severity','info')).lower() in whitelist) or (not isinstance(r, dict) and 'info' in whitelist)]
        except Exception:
            # If filter parsing fails, ignore it and continue with original recs
            pass

    # Prepare normalized items first so we can sort by severity priority
    normalized_items = []
    severity_priority = {'critical': 0, 'warning': 1, 'info': 2}

    for idx, r in enumerate(recs):
        # Defensive normalization of the recommendation entry
        if r is None:
            # Skip explicit None entries
            continue

        # If entry is a plain string or number, coerce to text
        sev = 'info'
        title_text = None
        text = ""
        if not isinstance(r, dict):
            try:
                text = str(r)
            except Exception:
                # Skip entries that cannot be stringified
                continue
        else:
            # r is a dict: attempt to extract fields safely
            try:
                raw_text = r.get('text', None)
            except Exception:
                raw_text = None

            # If raw_text is None or empty, use placeholder for title/body
            if raw_text is None or (isinstance(raw_text, str) and raw_text.strip() == ""):
                text = ""  # body will show a "Details not available" message
                title_text = "Recommendation"
            else:
                # Ensure text is a string
                try:
                    text = str(raw_text)
                except Exception:
                    text = ""

            try:
                sev = str(r.get('severity', 'info')).lower()
            except Exception:
                sev = 'info'

        # Ensure severity is valid
        if sev not in severity_map:
            sev = 'info'
        color = severity_map.get(sev, 'secondary')

        # Title: truncate long titles, fall back to generic if empty
        if title_text is None:
            if isinstance(text, str) and text:
                title_text = text if len(text) < 120 else text[:120] + '...'
            else:
                title_text = "Recommendation"

        # Body: show badge + text (or fallback message)
        body_text = text if text else "Details not available."
        body = html.Div([
            dbc.Badge(sev.upper(), color=color, className='me-2'),
            html.P(body_text)
        ])

        # Collect for sorting and later rendering
        normalized_items.append({
            'priority': severity_priority.get(sev, 2),
            'sev': sev,
            'color': color,
            'title': title_text,
            'body': body,
            'idx': idx,
        })

    # If no valid items were assembled, return a friendly placeholder
    if not normalized_items:
        return dbc.Card(dbc.CardBody(html.P("No recommendations at this time.")))

    # Sort items: critical -> warning -> info, then by original order
    normalized_items.sort(key=lambda it: (it['priority'], it['idx']))

    # Build AccordionItems in sorted order
    items = []
    for it in normalized_items:
        try:
            items.append(dbc.AccordionItem(it['body'], title=it['title'], item_id=f"rec-{it['idx']}"))
        except Exception:
            continue

    return dbc.Accordion(items, id=accordion_id, start_collapsed=True)
