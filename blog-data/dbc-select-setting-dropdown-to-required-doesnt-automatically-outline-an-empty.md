---
title: "dbc select setting dropdown to required doesn't automatically outline an empty?"
date: "2024-12-13"
id: "dbc-select-setting-dropdown-to-required-doesnt-automatically-outline-an-empty"
---

Okay so here's the deal you're facing a classic problem with Dash Bootstrap Components or dbc for short right the dropdown doesn't automatically outline when it's set to required and left empty it's a minor but annoying thing I've seen it countless times myself I've even debugged that exact problem late at night fuelled by too much coffee and the faint hum of my server room back in the day

It's like the browser is saying "you told me it's required but you never said it should scream at the user with a red outline when it's empty" which it kinda is I guess Anyway let's break it down in a way that'll get you unstuck I mean nobody likes having a form that's all polite and quiet about missing fields it's almost passive aggressive right

Here's the thing dbc components they rely on underlying HTML elements and browser behavior for form validation when you set a dropdown to required in dbc it adds the HTML `required` attribute to the underlying select element so the browser *should* handle the outline but the problem is that the default styling for browsers is not uniform

For instance some browsers might add a red outline when a required input is empty and invalid but some might not add anything until you try to submit the form and a few might only add the outline after you click into the input field and then click away I know right it's a mess and that is the whole point of frontend work isn't it just a mess of different browser behaviors

So you need to add some custom CSS to force the outline when the required select element is invalid

First let's look at a basic example of what your dbc select might look like in python code

```python
import dash
import dash_bootstrap_components as dbc
from dash import html

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(
    [
        dbc.Label("Select an Option", html_for="dropdown-example"),
        dbc.Select(
            id="dropdown-example",
            options=[
                {"label": "Option 1", "value": "1"},
                {"label": "Option 2", "value": "2"},
                {"label": "Option 3", "value": "3"},
            ],
            required=True,
        ),
        html.Div(id="output")
    ],
    style={'width': '500px', 'margin': '50px auto'}
)


@app.callback(
    dash.Output('output', 'children'),
    dash.Input('dropdown-example', 'value'),
    prevent_initial_call=True
)
def update_output(value):
    return f"Selected value: {value}" if value else "No value selected"


if __name__ == '__main__':
    app.run_server(debug=True)
```

This is the most basic dropdown example with a required option set but that doesn't enforce the outline on some browsers we have this problem every day it feels like no one cares if the required fields are actually required or not so we need to do this ourselves

Okay so here's where the CSS magic comes in we're targeting the select element specifically when it's both `required` and `:invalid` here's the CSS snippet to add

```css
select:required:invalid {
    border-color: red;
    outline: 2px solid red;
    box-shadow: 0 0 5px red;
}
```

Let me break this down for those who might be new to this stuff it says "Hey browser whenever you see a select element that's marked required AND its current value is considered invalid aka empty then give it a red border and a red outline and a little red shadow cause you know why not make sure it's obvious"

This CSS works because when a select is `required` and you haven't chosen anything its current value is indeed `invalid` from the browser's perspective

Now the issue is how to incorporate this into a Dash app you got a few options the simplest is to add this CSS to your app's external stylesheets this means we'll have to make some adjustments to the code provided before and you can create a separate `style.css` file in the same directory where your python code is located and add the snippet and call it this way

```python
import dash
import dash_bootstrap_components as dbc
from dash import html

external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    "style.css"
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    [
        dbc.Label("Select an Option", html_for="dropdown-example"),
        dbc.Select(
            id="dropdown-example",
            options=[
                {"label": "Option 1", "value": "1"},
                {"label": "Option 2", "value": "2"},
                {"label": "Option 3", "value": "3"},
            ],
            required=True,
        ),
        html.Div(id="output")
    ],
    style={'width': '500px', 'margin': '50px auto'}
)


@app.callback(
    dash.Output('output', 'children'),
    dash.Input('dropdown-example', 'value'),
    prevent_initial_call=True
)
def update_output(value):
    return f"Selected value: {value}" if value else "No value selected"


if __name__ == '__main__':
    app.run_server(debug=True)

```

With this setup when you refresh your page and have the dropdown empty you'll see that sweet red outline you've been craving (or more accurately needing for a decent looking form)

You might wonder why not use CSS frameworks' built in classes for this and yes you could probably dig through Bootstrap's or other frameworks documentation and find some classes that *might* do the same but sometimes it's quicker and more precise to target the specific element using basic CSS selectors like this you know keep it simple

Sometimes when people are dealing with these issues they tend to overthink it and try some complex solutions when it can be fixed with a few lines of CSS I once saw a dev who created a whole custom component and then did like a hundred hours of work on it to have the same thing done with just 5 lines of CSS and a single `required` attribute now that was a disaster that made me question life that was a hard week for him but he learned his lesson

By the way here is a bit of a tech joke related to the whole thing  why did the CSS dev breakup with the HTML dev? Because they kept arguing about style sheets and the HTML dev always said the code was required

If you are interested in learning more about the how the underlying form validation works you can checkout the HTML specification specifically the parts on form controls and validation it might seem boring at first but it is actually quite important or if you are more into learning CSS I would suggest reading the "CSS: The Definitive Guide" by Eric Meyer that's a classic and also look into the W3C website and read their documentation about selectors and CSS specificities

This should get you on the right track I know debugging little quirks like this is annoying and wastes time but that's part of the fun I guess now go forth and make some beautifully outlined dropdowns and keep coding
