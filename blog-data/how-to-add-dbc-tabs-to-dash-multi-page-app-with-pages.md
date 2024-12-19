---
title: "how to add dbc tabs to dash multi page app with pages?"
date: "2024-12-13"
id: "how-to-add-dbc-tabs-to-dash-multi-page-app-with-pages"
---

Alright so you’re wrestling with dash multi-page apps and wanting to throw in some dbc tabs right Been there done that got the t-shirt and probably spilled coffee on it too a few times

It’s a bit of a sticky situation at first glance combining dash's routing mechanism with dbc’s tab component isn’t exactly plug and play It feels like trying to fit a square peg into a round hole but don’t worry we can definitely make this work I've spent way more hours than I'd like to admit debugging similar shenanigans

Basically what we're talking about here is needing a way for a page in a dash multipage app to render inside of a dbc Tab component and not break your routing setup It’s all about making sure the right bits get rendered in the right places at the right times

Let's say a few years back I was working on this internal data dashboard thingy for some totally forgettable company They wanted all these different views like trends user analytics whatever Each view was a page in my dash app And they also wanted this slick tabbed navigation so that users can switch between those views without needing to reload the whole page This was a pain in the rear back then because the documentation at the time was not great if i'm being honest

I ended up diving deep into the dash callback system and a little bit of javascript jiggery-pokery to make it work So trust me on this it can be done and without requiring any arcane incantations or anything like that

Here’s the gist of how to do this in a way that makes some semblance of sense and won't leave you crying into your keyboard

The key idea here is that our main app layout will contain the `dbc.Tabs` component along with a `html.Div` which will be the placeholder for the page content And we will control the content of the placeholder using callbacks which can read the current url from the `dcc.Location` component

First the basic layout structure which you should have already:

```python
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], use_pages=True)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dbc.NavbarSimple(
        brand="My Awesome Dashboard",
        brand_href="#",
    ),
    dbc.Tabs(
        id="tabs",
        active_tab="tab-1",
        children=[
            dbc.Tab(label="Tab 1", tab_id="tab-1"),
            dbc.Tab(label="Tab 2", tab_id="tab-2"),
            dbc.Tab(label="Tab 3", tab_id="tab-3"),
        ],
    ),
    html.Div(id="page-content"),
])


if __name__ == '__main__':
    app.run_server(debug=True)
```

Notice that I've used dummy ids but it's best that you use descriptive ids that you can remember and understand when you work with more complex things. The app itself is pretty basic and as you can see I've got a `dbc.Tabs` element and a `html.Div` that will hold my pages. I'm just going to write pages inside this folder as I normally do. This is the standard setup from dash multi page docs

Ok so here's where the magic happens We need a callback that will listen to the tab selection and update the `page-content` div with the correct content of the page corresponding to the selected tab. We are going to use the `active_tab` property from the `dbc.Tabs` element to determine the current selection:

```python
from dash import Output, Input, callback, html, page_registry

@callback(Output("page-content", "children"), Input("tabs", "active_tab"))
def render_page_content(tab):
    for page in page_registry.values():
        if page["name"] == tab:
            return page["layout"]
    return html.Div(f"Page not found with tab {tab}")

```

This function looks at the `page_registry` that dash creates automatically for multipage apps. It basically checks if the `page["name"]` is the same as the `active_tab` and if it is, it will output the layout of that page. Otherwise it will return that it can't find that page. Very simple straightforward way of making this work.

Now it's time to configure the pages. In order to make them integrate with the `tab_id` correctly we need to give each of our pages a `name` that matches with the `tab_id` in the `dbc.Tabs` element from the main app layout. We just need to change the `name` in each of the pages we have. A small example of one page:

```python
# file: pages/tab1.py
import dash
from dash import html

dash.register_page(__name__, path="/tab1", name="tab-1")

layout = html.Div("This is the content of Tab 1")
```

And that's pretty much it you just need to replicate this structure for your others tab pages and the callbacks will take care of switching the layouts.

You can think about this system as a simple router of sorts for the tab components and that's basically how it's done.

A few things I learned the hard way on this:

First make sure your `tab_id` values on the `dbc.Tabs` are always the same as the name of your pages in the `dash.register_page()` function. Spelling mistakes can and will bring your whole app crashing down. Don’t ask me how I know this.

Second don't try to over complicate the routing logic or that you try to use the `dcc.Location` element in your pages too much. It can lead to all kinds of inconsistencies that are hard to debug. Just keep it simple like I've shown you. It makes life way easier.

Third you'll want to dive deeper into how callbacks actually work they are the lifeblood of Dash apps. It is better to really understand the documentation rather than randomly copy pasting stuff. I can't stress this enough. Check out the Dash documentation directly it’s got everything you need there just search for it in the docs.

And lastly use consistent id naming. It helps so much when you’re working with a lot of moving parts like when you're trying to build a real app. I've seen more than one dev spend hours because of a typo in an id it's not pretty trust me. Also if you have more complex situations you may need to do more conditional rendering. That's another can of worms for another day though.

If you want to get really deep on dash I recommend reading the official documentation and also searching for some books on the subject, there are some good books on programming with python that will touch dash and its capabilities and that could help you develop your expertise further. You can also look for academic papers but there are few of them regarding dash as it is not a particularly old technology.

Anyway I hope this was helpful to you If you are still facing any issues feel free to throw more questions and I will gladly answer more of them. I’ve been doing this for a while now you know. I'm pretty sure I've seen every single edge case out there. Actually I'm sure there are many edge cases I haven't seen I'm just kidding, it’s my bad sense of humor you know
