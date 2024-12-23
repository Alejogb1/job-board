---
title: "How to add footnotes to layered Altair charts?"
date: "2024-12-16"
id: "how-to-add-footnotes-to-layered-altair-charts"
---

, let's talk about layering altair charts and footnotes – something that's come up more than a few times in my experience. I recall one project, specifically, where I was visualizing complex geospatial data, and simple labels weren't sufficient; we needed context, sources, and explanations. Altair, while excellent for declarative charting, doesn’t offer a direct, built-in footnote mechanism. So, we had to get a bit creative.

The core challenge, as you likely know, is that Altair focuses on declarative specification of visual encodings. Footnotes, in the conventional sense, fall outside this model. They're more annotation than core data visualization. What we're essentially trying to do is overlay a supplementary visual element that is tied to specific parts of the chart but is not part of the data being charted.

My approach generally involves a combination of text marks and careful coordinate specification. Instead of trying to ‘add’ footnotes directly to the chart layers, I construct them separately, much like you’d build another layer, but with a specific purpose: annotation. This means calculating the necessary x and y coordinates to place the footnotes and references relative to the primary chart's spatial domain.

One common method is to use the `concat` or `vconcat` operators in Altair. This isn't strictly "layering," in the sense that a chart is overlaid upon another within the same coordinate system. Instead, we're creating visually separate components and then joining them vertically or horizontally. This allows us to position the 'footnote' section below or beside the main chart.

Now, let’s look at a few concrete examples, starting with a basic implementation.

**Example 1: Simple Vertical Concatenation**

Let's assume we have a simple scatter plot and want to add a single footnote below it.

```python
import altair as alt
import pandas as pd

data = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [2, 4, 1, 3, 5]
})

chart = alt.Chart(data).mark_point().encode(
    x='x:Q',
    y='y:Q'
)

footnote_text = 'Source: Example Dataset'

footnote = alt.Chart(
    pd.DataFrame({'text': [footnote_text]})
).mark_text(
    align='left',
    baseline='top',
    fontSize=10
).encode(
    text='text:N',
    x=alt.value(10),
    y=alt.value(10)
)

final_chart = alt.vconcat(chart, footnote).resolve_scale(y='independent')

final_chart
```

In this initial example, we construct the primary chart as usual. The footnote itself is created as a separate `Chart` with only a text encoding. I've used `alt.value` to specify an offset in pixel coordinates from the top left corner of the composite chart. Finally, `alt.vconcat` joins the scatter plot and the footnote vertically. Importantly, I've used `.resolve_scale(y='independent')` to prevent the y-axis from trying to include the footnote's coordinates. This is crucial because we are using pixel values for annotation and want the chart to scale according to the actual data.

**Example 2: Multiple Footnotes and Reference Markers**

The first example provided a basic framework, but it often needs improvement. Here’s how we can approach a scenario with multiple footnotes, each associated with different parts of the primary chart, using a marker (e.g., an asterisk) within the chart itself.

```python
import altair as alt
import pandas as pd

data = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [2, 4, 1, 3, 5],
    'marker': ['', '*', '', '**', '*']
})

chart = alt.Chart(data).mark_point().encode(
    x='x:Q',
    y='y:Q',
    text='marker:N'
)

text_mark = chart.mark_text(
    align='left',
    baseline='bottom',
    dy=-5
).encode(text='marker:N')

chart = chart + text_mark

footnotes_data = pd.DataFrame({
    'marker': ['*', '**'],
    'text': ['Reference to Point 1, Reference to Point 2']
})

footnotes = alt.Chart(footnotes_data).mark_text(
    align='left',
    baseline='top',
    fontSize=10
).encode(
    text='marker:N + " - " + text:N',
    x=alt.value(10),
    y=alt.value(10)
)


final_chart = alt.vconcat(chart, footnotes).resolve_scale(y='independent')
final_chart

```
Here, I've introduced a 'marker' column within the initial dataframe. I use this column to add text markers adjacent to points on the chart using a second mark_text layer. We then define `footnotes_data` and use the 'marker' and 'text' columns to produce the footnote text. Finally, the same `alt.vconcat` mechanism joins the chart and the reference texts.

**Example 3: Dynamic Positioning of Footnotes**

Fixed pixel coordinates for footnotes, as shown in the first two examples, might not work well when the chart size is variable. A more robust approach involves dynamically positioning the footnote text relative to the chart's overall dimensions. For this, we need to extract chart size and scale information and use it to define dynamic locations.

```python
import altair as alt
import pandas as pd
import numpy as np

data = pd.DataFrame({
    'x': np.arange(10),
    'y': np.random.rand(10)
})

chart = alt.Chart(data).mark_line().encode(
    x='x:Q',
    y='y:Q'
)

footnote_text = 'Source: Example Dataset, note some variance'

# Getting the x scale and width for dynamic positioning
# This is an approach based on an understanding of vega-lite specification.

# Since the actual scaling is handled in vega-lite, we can approximate by assuming uniform linear scales
xmin = data['x'].min()
xmax = data['x'].max()
width = 500  # Setting a fixed width, but this could be dynamic

footnote_x = xmin
footnote_y = 0  # Place it at the bottom of the chart

footnote = alt.Chart(
    pd.DataFrame({'text': [footnote_text]})
).mark_text(
    align='left',
    baseline='top',
    fontSize=10
).encode(
    text='text:N',
    x=alt.value(footnote_x),
    y=alt.value(footnote_y)
)

final_chart = alt.vconcat(chart, footnote).resolve_scale(y='independent')
final_chart

```
In this final example, rather than relying on a fixed y-offset, we are setting the y value to '0' to have the footnote at the 'bottom', and x-coordinate is set to the minimum x-value of the data. The x value could be varied to accommodate more space for multiple foot notes. I’ve hard-coded the width to 500 for the purpose of example but it can be controlled by the overall figure dimensions, making the annotation more consistent across different chart sizes. While pixel locations can be used in simple cases, understanding how the scales work in Altair and Vega-lite can provide a more flexible annotation solution.

In my experience, these techniques have proven useful for generating informative and context-rich visualizations. For further information, I recommend delving into the vega-lite documentation, as altair is a high-level python API over it and understanding vega-lite can provide a deeper understanding of how to customize it. Specifically, refer to the sections on *mark properties*, *layered charts*, and *concatenation*. Also, the "The Grammar of Graphics" by Leland Wilkinson provides the theoretical grounding for many of the concepts employed in Altair, and a deeper understanding of this material is invaluable. While I have not included any links here, these are the sources I would direct you to for deeper insight into crafting complex visualizations. You may also find the Altair documentation (which is often updated with examples) very useful in your research.
