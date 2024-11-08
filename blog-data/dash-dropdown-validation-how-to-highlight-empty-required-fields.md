---
title: "DASH Dropdown Validation: How to Highlight Empty Required Fields?"
date: '2024-11-08'
id: 'dash-dropdown-validation-how-to-highlight-empty-required-fields'
---

```python
import dash
import dash_html_components as html
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

select = dbc.Select(
    id="dropdown-required",
    required=True,
    options=[
        {'label': '', 'value': ''},
        {'label': 'Heated', 'value': 'Heated'},
        {'label': 'Ambient', 'value': 'Ambient'},
        {'label': 'N/A', 'value': 'N/A'}],
)

app.layout = html.Div([select])

@app.callback(
    Output("dropdown-required", "required"),
    Input("dropdown-required", "value")
)
def set_dropdown_required(value):
    res = (
        True if value is None or value == ""
        else False
    )
    return res

app.run_server(debug=True)
```

This code implements the solution provided in the Stack Overflow answer, creating a Dash app with a `dbc.Select` dropdown and a callback function to dynamically set the `required` attribute based on the selected value. The CSS styling for the `:required` pseudo-class is added to `assets/style.css` file to outline the dropdown with a red border when it's empty.

