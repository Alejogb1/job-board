---
title: "How to Testing Python code that produces Altair graphs?"
date: "2024-12-15"
id: "how-to-testing-python-code-that-produces-altair-graphs"
---

alright, so you're hitting that classic wall of testing visualizations, huh? i’ve been there, staring at the monitor wondering how to assert that my beautiful altair chart is actually, well, beautiful *and* correct. it's not like checking a simple integer return value. it’s a visual thing, and testing visuals programmatically is a different beast.

first off, let's clear the air, testing altair outputs directly like we might with, say, a math function is a headache. you can't just `assert chart == some_expected_chart`. altair charts are complex objects that are essentially a spec for a visualization. they are not an image, not a pixel-perfect thing. so, we need a different strategy.

my experience started back when i was building this data dashboard, the one for monitoring server utilization. i got so deep into the charting that i skipped the tests, and, well, you can guess. one friday evening, the dashboard decided to display some nonsense and i was stuck debugging till late. that's when i learned i needed actual tests.

what i settled on is this approach: focus on the *data* and the chart *specification*. altair is all about taking data and turning it into a json specification that a renderer, like vega-lite, consumes. that json specification is our gold mine for testing.

so instead of testing images you will want to test the structure of the json or data it uses.

here's a simplified example of what i mean. let's say you have this basic altair chart:

```python
import altair as alt
import pandas as pd

def create_scatter_chart(df):
    chart = alt.Chart(df).mark_circle().encode(
        x='x',
        y='y',
        color='category'
    )
    return chart

data = {'x': [1, 2, 3, 4, 5],
        'y': [2, 3, 5, 4, 7],
        'category': ['a', 'b', 'a', 'b', 'a']}

df = pd.DataFrame(data)

chart = create_scatter_chart(df)
```

now, instead of trying to verify the *rendered* chart, we will look at the generated json spec. we can access the chart's json using the `.to_json()` method and then compare its parts. something like this:

```python
def test_scatter_chart_data():
    data = {'x': [1, 2, 3, 4, 5],
        'y': [2, 3, 5, 4, 7],
        'category': ['a', 'b', 'a', 'b', 'a']}
    df = pd.DataFrame(data)
    chart = create_scatter_chart(df)
    chart_json = chart.to_json()

    assert chart_json['data']['values'] == data # check correct data used
    assert chart_json['mark'] == 'circle'  # ensure it's the mark we expected
    assert chart_json['encoding']['x']['field'] == 'x' # check field
    assert chart_json['encoding']['y']['field'] == 'y'
    assert chart_json['encoding']['color']['field'] == 'category' #check encodings
```

notice how we are directly accessing the internal structure of the json specification? this means we can check, that the `mark` is correct, that the data is indeed the expected one and also that the encodings are correctly mapped to their data columns. this is much more reliable and less flaky than trying to compare rendered images. for me, this was a breakthrough moment. i remember when i was still trying to compare images using some library, but the differences in anti-aliasing or text rendering could fail my tests even when the image looked visually identical.

you can also take this to the extreme and start comparing full json specs with expected specs. in such case you would have a reference chart and then assert that generated chart is the same. this might come with it's own drawbacks as specs can get very lengthy and any minimal change would fail your tests even if visually is the same.

also i would recomend not to test the full spec, but rather focus on the aspects you care the most and what are crucial for your visualization, such as the encodings, marks, data columns used. this would also make tests less prone to failing to any minor change in specification.

now, if we move a bit further, imagine you're generating a more complex chart. let's say you have some data transformations in your chart. for example lets create a chart that calculates a rolling mean, we can also try testing it:

```python
import altair as alt
import pandas as pd


def create_rolling_mean_chart(df, window=3):
    chart = alt.Chart(df).transform_window(
    rolling_mean = 'mean(y)',
    frame = [-window,0],
    sort = [alt.SortField('x')],
    ).mark_line().encode(
        x='x:O',
        y='rolling_mean:Q',
    )
    return chart

data = {'x': [1, 2, 3, 4, 5, 6],
        'y': [2, 3, 5, 4, 7, 9]}

df = pd.DataFrame(data)
chart = create_rolling_mean_chart(df)
```
now we want to test that it generates the correct transform in the specification. again, we go straight to the json.

```python
def test_rolling_mean_chart():
    data = {'x': [1, 2, 3, 4, 5, 6],
        'y': [2, 3, 5, 4, 7, 9]}
    df = pd.DataFrame(data)
    chart = create_rolling_mean_chart(df)
    chart_json = chart.to_json()
    transform = chart_json['transform'][0]

    assert transform['window'] == [{'op':'mean', 'field':'y','as':'rolling_mean'}]
    assert transform['frame'] == [-3,0]
    assert transform['sort'] == [{'field':'x'}]
    assert chart_json['mark'] == 'line'
    assert chart_json['encoding']['x']['field'] == 'x'
    assert chart_json['encoding']['y']['field'] == 'rolling_mean'
```

we are again examining the structure inside the json specification and checking that the `transform_window` method we used has created the expected json specification. this allows to check not just that the data is in place, but also that any data manipulation you are making is properly translated into the specification.

now, i know sometimes things get tricky, especially with interactive charts. in those cases i try to test the event handlers or actions as much as possible, not so much the final rendering. it’s hard to test that a hover tooltip is indeed the tooltip and not something else but you can test that when hovering a certain element it triggers the action you expect. the key is still the structure of json, always.

and before i forget, one thing that is also helpful, is to always keep an eye on the altair documentation, it has gotten better lately and i would say is pretty well written. it explains all the json specifications and how altair maps to it.

also, for more theoretical knowledge on data visualization i find “the grammar of graphics” by wilkinson a great resource, it explains many of the ideas that altair is based on.

last but not least, regarding testing, i've found "pragmatic programmer" a great book that covers testing in a general context. you might think that testing visualization is a hard problem, but it's still software at the end of the day, so everything the book explains applies, and should help you build more robust visualization tools.

it’s more about how you approach testing, and less about the chart itself. and no, don't try to pixel compare the images of the charts, i mean, unless you *really* want to spend a lot of time dealing with image differences, that's where it all started for me, and trust me, it's a rabbit hole. i spent more time tweaking the tests than debugging the actual charting code (that was ironic, the irony is always the best test).
