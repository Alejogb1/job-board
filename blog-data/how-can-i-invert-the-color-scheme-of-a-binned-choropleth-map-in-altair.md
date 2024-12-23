---
title: "How can I invert the color scheme of a binned choropleth map in Altair?"
date: "2024-12-23"
id: "how-can-i-invert-the-color-scheme-of-a-binned-choropleth-map-in-altair"
---

Alright, let's tackle this color inversion issue with Altair choropleth maps. I’ve definitely been down this road before, particularly when dealing with visualizations that needed to adhere to specific client branding guidelines or to effectively highlight different aspects of a dataset. It’s not always as straightforward as one might initially hope, but it's certainly achievable with a little understanding of Altair's encoding and scale mechanisms.

The core challenge, as I see it, isn’t about fundamentally altering how Altair handles colors, but rather about manipulating the *mapping* between data values and the color palette we're using. When we talk about inverting a color scheme, what we usually mean is swapping the order of colors within the chosen scale. Rather than starting with the lightest color for the smallest values and proceeding to the darkest for larger values, we want to reverse that.

Now, Altair typically uses a continuous color scale for choropleth maps, even if the data is binned. These continuous scales, such as `'viridis'` or `'magma'`, have a defined sequence of colors from one extreme to another. Inverting them requires a shift in the encoding process, effectively telling Altair to associate the lowest data bin with the *end* of the color scale and the highest bin with the *beginning*.

The most common way to accomplish this is by leveraging the `scale` parameter within the color encoding. Specifically, we can provide a new range of colors (reversed) or modify the existing one with `.reverse()`. Let's dive into some practical examples to illustrate these approaches.

First, let’s imagine a scenario where I’m working with population density data across different regions, binned into, say, five categories. Normally, the highest density would appear darkest on the map. If we want the opposite, we’ll need to invert the color mapping.

**Example 1: Reversing a Named Scale**

Let’s start with a reversed standard scale. I recall a project where the standard `viridis` scale just didn't visually resonate with the stakeholders. So, I needed to invert it. Here's how you might do it. Remember, this code assumes you already have a `geojson` object and your `data` dataframe structured correctly for a choropleth (with a column, say 'bins', used for coloring). I'll focus on the critical color scale encoding:

```python
import altair as alt
import pandas as pd

# Sample Data (replace with your actual data)
data = pd.DataFrame({
    'region': ['A', 'B', 'C', 'D', 'E', 'F'],
    'bins': [1, 2, 3, 4, 5, 2],
    'other_values': [10, 20, 30, 40, 50, 25]
})

# Sample GeoJSON (replace with your actual GeoJSON)
geojson = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature", "id": "A", "properties": {}},
        {"type": "Feature", "id": "B", "properties": {}},
         {"type": "Feature", "id": "C", "properties": {}},
         {"type": "Feature", "id": "D", "properties": {}},
         {"type": "Feature", "id": "E", "properties": {}},
        {"type": "Feature", "id": "F", "properties": {}}
    ]
}

chart = alt.Chart(data).mark_geoshape(stroke="black").encode(
    alt.Color(
        "bins:O",
        scale=alt.Scale(scheme="viridis", reverse=True)
    ),
    tooltip=['region', 'bins','other_values']
).transform_lookup(
    lookup='region',
    from_=alt.LookupData(data=geojson, key='id')
)

chart
```

In this snippet, we use the `reverse=True` argument within the `alt.Scale` definition. This simple addition is the key to flipping the `viridis` color scheme. Now, the smallest `bins` values will be associated with the dark end of the `viridis` palette, and the largest values with the light end.

**Example 2: Manually Reversing a Scale**

Sometimes, named scales won’t suffice. Maybe the client provides a specific color palette that needs to be applied in reverse. Here’s where manually reversing a list of colors comes into play. Suppose we are given the colors `['#e0f2f7', '#b2ebf2', '#80deea', '#4dd0e1', '#26c6da']` and want to apply these in reverse for the same population density visualization.

```python
import altair as alt
import pandas as pd

# Sample Data (replace with your actual data)
data = pd.DataFrame({
    'region': ['A', 'B', 'C', 'D', 'E', 'F'],
    'bins': [1, 2, 3, 4, 5, 2],
    'other_values': [10, 20, 30, 40, 50, 25]
})

# Sample GeoJSON (replace with your actual GeoJSON)
geojson = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature", "id": "A", "properties": {}},
        {"type": "Feature", "id": "B", "properties": {}},
         {"type": "Feature", "id": "C", "properties": {}},
         {"type": "Feature", "id": "D", "properties": {}},
         {"type": "Feature", "id": "E", "properties": {}},
        {"type": "Feature", "id": "F", "properties": {}}
    ]
}
color_scheme = ['#e0f2f7', '#b2ebf2', '#80deea', '#4dd0e1', '#26c6da']


chart = alt.Chart(data).mark_geoshape(stroke="black").encode(
    alt.Color(
        "bins:O",
        scale=alt.Scale(range=color_scheme[::-1])
    ),
    tooltip=['region', 'bins','other_values']
).transform_lookup(
    lookup='region',
    from_=alt.LookupData(data=geojson, key='id')
)
chart
```

Here, we manually specify the `range` argument within the `alt.Scale` using the color scheme in reverse. The `[::-1]` notation efficiently reverses the order of the list. This way we can ensure the color ramp follows the specific brand or design guideline, even if the default scales from Altair don’t match. I've used this technique countless times when dealing with branded maps.

**Example 3: Inverting a Custom Binned Scale**

Now, let's consider a slightly more complex case. What if our data isn't pre-binned but we want to apply specific color bins? This happens more frequently than you might think. Let’s say we want to bin our data into a set of categories and then invert the provided colors.

```python
import altair as alt
import pandas as pd
import numpy as np

# Sample Data (replace with your actual data)
data = pd.DataFrame({
    'region': ['A', 'B', 'C', 'D', 'E', 'F','G','H'],
    'values': [10, 25, 120, 35, 75, 20, 110, 13],
     'other_values': [10, 20, 30, 40, 50, 25, 60, 70]

})

# Sample GeoJSON (replace with your actual GeoJSON)
geojson = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature", "id": "A", "properties": {}},
        {"type": "Feature", "id": "B", "properties": {}},
         {"type": "Feature", "id": "C", "properties": {}},
         {"type": "Feature", "id": "D", "properties": {}},
         {"type": "Feature", "id": "E", "properties": {}},
         {"type": "Feature", "id": "F", "properties": {}},
         {"type": "Feature", "id": "G", "properties": {}},
         {"type": "Feature", "id": "H", "properties": {}}
    ]
}
# define breaks and labels
breaks = [0, 30, 60, 90, 120, np.inf]
labels = ['Low', 'Low-Mid','Mid', 'Mid-High', 'High']
color_scheme = ['#e0f2f7', '#b2ebf2', '#80deea', '#4dd0e1', '#26c6da']

chart = alt.Chart(data).mark_geoshape(stroke="black").encode(
    alt.Color(
        field="binned_values",
        type="ordinal",
        scale=alt.Scale(
            domain=labels,
            range=color_scheme[::-1]
        )
    ),
    tooltip=['region','values', 'other_values']

).transform_lookup(
    lookup='region',
    from_=alt.LookupData(data=geojson, key='id')
).transform_bin(
    as_ = "binned_values",
    field = "values",
    bin=alt.Bin(extent=[min(data['values']), max(data['values'])],
        maxbins=5,
    )
).transform_calculate(
    calculate=f"if(datum.binned_values == null, null, {labels})[round((datum.binned_values - {min(data['values'])})* {len(labels)}/({max(data['values'])-min(data['values'])}))] ",
    as_="binned_values"
)

chart
```

This example shows that using `transform_bin` and `transform_calculate` we can bin the data on the fly to create an ordinal scale based on categories and then reverse the colors. This allows for very specific customizations, which I've found to be a necessity for advanced data visualization.

**Technical Resources**

For a deeper dive into Altair’s capabilities, I recommend exploring the official Altair documentation. The sections on encoding channels and scales are particularly insightful. Further, the book “Interactive Data Visualization for the Web” by Scott Murray provides an excellent foundation for understanding data visualization principles. For a theoretical understanding of color scales, the work of Cynthia Brewer on colorbrewer2.org is indispensable. In general, the research on visual perception and effective color use is essential for avoiding common pitfalls in creating visualizations that can be correctly interpreted.

In conclusion, inverting the color scheme of a binned choropleth map in Altair is achieved by manipulating the scale definition. Whether you use `reverse=True`, manually reverse a list of colors, or even bin and label data on the fly, the principle is about controlling the mapping of data to color. I hope that my explanation and the provided code snippets provide a solid starting point and that you find this information useful. It's a nuanced topic, but mastering it is crucial for creating effective and visually appealing data visualizations.
