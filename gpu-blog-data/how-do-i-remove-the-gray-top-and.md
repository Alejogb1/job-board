---
title: "How do I remove the gray top and right lines in Altair visualizations?"
date: "2025-01-30"
id: "how-do-i-remove-the-gray-top-and"
---
The default rendering of Altair charts includes thin, gray lines along the top and right edges of the plotting area, representing the outer bounds of the coordinate system. These lines, while subtle, are often undesirable in publication-quality visuals or when integrating plots into broader web applications. Removing them requires specific configuration of the chart's view and axis properties.

The core of the solution involves manipulating the `view` and axis settings within the Altair specification. The `view` property controls the overall appearance of the chart canvas, while axis properties manage the appearance of the x and y axes. Specifically, setting `stroke=None` within the `view` configuration and also within axis `domain` specifications is necessary to achieve the desired outcome of removing the top and right borders. I encountered this exact situation during a project visualizing sensor data a few months back where the default lines created unnecessary clutter when embedded in a dashboard.

To elaborate, Altair, when compiled to Vega-Lite, uses SVG elements for rendering the visualizations. The gray lines we observe are the borders or domains of the coordinate system rendered as strokes on these SVG paths. Thus, we must instruct Vega-Lite to not render these strokes. We achieve this in Altair by explicitly setting the `stroke` property to `None` within the appropriate JSON path. This property, under the hood, instructs the renderer to skip drawing any line at that location.

Let's examine a straightforward example of a scatter plot with these lines present:

```python
import altair as alt
import pandas as pd

data = {'x': [1, 2, 3, 4, 5], 'y': [2, 4, 1, 3, 5]}
df = pd.DataFrame(data)

chart = alt.Chart(df).mark_point().encode(
    x='x:Q',
    y='y:Q'
)

chart.show()
```

This code snippet generates a basic scatter plot. The default appearance will include the gray top and right borders. To remove them, I would adjust the code as follows, leveraging the `configure_view` and `configure_axis` properties:

```python
import altair as alt
import pandas as pd

data = {'x': [1, 2, 3, 4, 5], 'y': [2, 4, 1, 3, 5]}
df = pd.DataFrame(data)


chart = alt.Chart(df).mark_point().encode(
    x='x:Q',
    y='y:Q'
).configure_view(
    stroke=None
).configure_axis(
    domain=False
)

chart.show()
```

In this modification, `configure_view(stroke=None)` targets the general canvas boundaries, removing all external strokes. Simultaneously, `configure_axis(domain=False)` explicitly removes the axis domain lines. The `domain=False` setting is preferred here over `domain=None` due to the way Vega-Lite processes boolean settings regarding rendering of these lines. This effectively eradicates the unwanted top and right lines. This approach, when coupled with careful label placement, significantly cleaned up the plots in the sensor project, improving legibility.

The `configure_axis` method can also be used to target individual axes. For example, to remove only the right border (while maintaining the bottom), I would specifically disable the x axis domain:

```python
import altair as alt
import pandas as pd

data = {'x': [1, 2, 3, 4, 5], 'y': [2, 4, 1, 3, 5]}
df = pd.DataFrame(data)

chart = alt.Chart(df).mark_point().encode(
    x='x:Q',
    y='y:Q'
).configure_view(
    stroke=None
).configure_axis(
    x=alt.Axis(domain=False)
)


chart.show()
```

This snippet retains the y-axis domain line (the left axis line), removing only the right. In more complicated situations, one can selectively target the axis of interest using this method. I've found this fine-grained approach indispensable when needing to convey complex information with different axes using slightly different visual cues.

A related issue frequently arises when combining charts. If one embeds a chart in another using the `&` or `|` operators, these borders may unexpectedly reappear if the constituent charts have different internal default padding or view configurations. Ensuring each chart in the combined composition has the appropriate `view` and axis configurations is essential to produce a uniform and clean combined result. For instance, when dealing with stacked bar charts, ensuring that both the inner bar series and combined chart have the strokes set to None prevents the appearance of unwanted borders.

Several helpful resources further explain the relevant concepts. Official Altair documentation offers comprehensive details on the `configure_view`, `configure_axis`, and individual axis `domain` properties. Additionally, exploring Vega-Lite specifications directly is informative, as Altair charts are translated to this lower level. The Vega-Lite documentation details the SVG rendering mechanics, and specific stroke and domain settings. Reading examples from the Altair gallery can also provide practical demonstrations of such customization. Moreover, examining examples provided with the Vega Editor often proves insightful, especially when debugging complex chart interactions. Finally, numerous publicly available notebooks on platforms like Kaggle and GitHub can illustrate real-world scenarios and common practices. These examples, coupled with the foundational knowledge of stroke settings, provide a complete understanding to tackle these kinds of border-related issues.
