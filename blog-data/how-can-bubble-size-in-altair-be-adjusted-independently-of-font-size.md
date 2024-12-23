---
title: "How can bubble size in Altair be adjusted independently of font size?"
date: "2024-12-23"
id: "how-can-bubble-size-in-altair-be-adjusted-independently-of-font-size"
---

,  It's a common frustration when you’re visualizing data with Altair and bubble charts, finding that the bubble size is tied to the font size, especially when you want more control over the visual hierarchy. I’ve bumped into this exact scenario more than a few times during data analysis projects, particularly when working on geographic visualizations where you might want large bubbles to indicate high values, regardless of the label size. The key here lies in understanding how Altair generates these charts and then leveraging its encoding capabilities to break that connection.

The core issue stems from Altair’s default behavior of scaling bubble sizes based on an underlying data value which, when you introduce text labels, can inadvertently scale based on the font size of that label in some default rendering contexts, which is almost never what you want. To decouple them, we need to explicitly define the bubble size using a separate data field and encoding, effectively overriding any automatic scaling that might occur based on the font or text.

Let’s get into the mechanics of it. Rather than allowing the default behavior to dictate the bubble size based on text labels, we'll use the `size` encoding channel. We will map a numeric data field to this channel. This field will dictate the size of your bubbles. Importantly, this doesn’t have to be the same data that determines the text labels.

Here’s the first working snippet to illustrate. Imagine we have a dataset of cities with their populations and a 'label' for city name:

```python
import altair as alt
import pandas as pd

data = pd.DataFrame({
    'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
    'population': [8_300_000, 4_000_000, 2_700_000, 2_300_000, 1_600_000],
    'label': ['NY', 'LA', 'CHI', 'HOU', 'PHX'],
    'normalized_population': [83,40,27,23,16]
})

chart = alt.Chart(data).mark_circle().encode(
    x=alt.X('population', title='Population'),
    y=alt.Y('normalized_population', title='Normalized Population'),
    size=alt.Size('normalized_population', title='Bubble Size', scale=alt.Scale(range=[50, 500])),
    tooltip=['city', 'population']
).properties(
    title='City Populations'
)

chart_with_text = chart.mark_text(
    align='left', dx=7
).encode(
    text='label',
).properties(
    title='City Populations with Labels'
)

combined_chart = (chart+chart_with_text).resolve_scale(size='independent')


combined_chart
```
In this code, we first create a basic scatter plot with circle marks. Crucially, the `size` encoding is explicitly set to map the 'normalized\_population' field to the bubble size. I also set the range of the size using a scale. Then we create a separate mark for text and use the `text` encoding to display our labels. Finally we `resolve_scale` the `size` independently. You can clearly see the bubbles are based on the `normalized_population`, entirely disconnected from the text label size.

Now let's look at a slight variation which demonstrates an alternative approach to creating bubbles using `mark_point`, which is also a valid approach. Here, I'm going to use a slightly different data set, focusing on an example from one of my previous projects related to product sales, for a better illustration of real-world cases.

```python
import altair as alt
import pandas as pd

data = pd.DataFrame({
    'product': ['Product A', 'Product B', 'Product C', 'Product D', 'Product E'],
    'sales': [1200, 800, 2000, 500, 1500],
     'profit':[100,70,150,40,110],
    'label': ['A', 'B', 'C', 'D', 'E'],
    'scaled_sales': [12,8,20,5,15]

})


chart_points = alt.Chart(data).mark_point(
    filled=True,
    opacity=0.8
).encode(
    x=alt.X('profit', title='Profit'),
    y=alt.Y('sales', title='Sales'),
    size=alt.Size('scaled_sales', title='Sales Size', scale=alt.Scale(range=[100,1000])),
    tooltip=['product', 'sales']
).properties(
    title='Product Sales Analysis'
)


chart_points_text = chart_points.mark_text(
    align='left', dx=7
).encode(
    text='label',
).properties(
    title='Product Sales Analysis with labels'
)


combined_chart_points = (chart_points+chart_points_text).resolve_scale(size='independent')

combined_chart_points
```
Here, we use `mark_point` with the `filled` option and customize the opacity. Again, we use `size` to map our `scaled_sales` values. Crucially, the size is based on this numeric field while the text labels are controlled separately. In my experience, this approach offers more direct control over the appearance of the bubbles. We decouple the size from the font size using the resolve\_scale which tells Altair to use independent scales for the size encoding.

Finally, let’s tweak this approach to demonstrate the creation of a geographical chart with independent control of the bubble size. Let's use a simple example with simulated location data.

```python
import altair as alt
import pandas as pd

data = pd.DataFrame({
    'location': ['Location A', 'Location B', 'Location C', 'Location D'],
    'latitude': [34.0522, 40.7128, 51.5074, 48.8566],
    'longitude': [-118.2437, -74.0060, -0.1278, 2.3522],
    'value': [100, 50, 200, 150],
    'normalized_value':[20,10,40,30],
    'label':['A', 'B','C','D']
})


chart_geo = alt.Chart(data).mark_circle(filled=True, opacity=0.7).encode(
    longitude='longitude:Q',
    latitude='latitude:Q',
    size=alt.Size('normalized_value', title='Value', scale=alt.Scale(range=[50, 500])),
    tooltip=['location', 'value']
).properties(
    title='Geographical Data Visualization'
)

chart_geo_text = chart_geo.mark_text(
    align='left', dx=7
).encode(
    text='label',
).properties(
    title='Geographical Data Visualization'
)


combined_chart_geo = (chart_geo + chart_geo_text).resolve_scale(size='independent')

combined_chart_geo
```
In this code, I'm mapping latitude and longitude to the X and Y axes.  We continue to decouple the size using a numeric field and use a separate mark for the text. The key takeaway is that the bubble sizes now respond to our explicit size mapping and not to the font sizes of the text labels. Resolving the scale independently ensures we have complete control over the visual hierarchy.

For deeper insight into data visualization principles and encoding best practices, I would recommend reading “The Visual Display of Quantitative Information” by Edward Tufte. It's a foundational text for anyone serious about data visualization. Also, explore the Vega-Lite grammar, upon which Altair is built. Understanding its underlying structure can be quite helpful. The official Vega-Lite documentation is an invaluable resource that has helped me many times. Furthermore, “Information Visualization: Perception for Design” by Colin Ware will offer valuable guidance on human perception principles that will help design effective visualizations.

To sum up, adjusting bubble size independently of font size in Altair boils down to explicitly mapping a numeric field to the `size` encoding channel, and ensuring the scales are resolved independently of each other. By using the `.resolve_scale(size='independent')` call, the size is decoupled from the text labels and their rendering characteristics. This technique provides the flexibility needed to create accurate and informative visualizations where size conveys meaningful quantitative data without being tied to text label sizes. It is a common task, and by mastering this, you’ll greatly improve the clarity and effectiveness of your visualizations.
