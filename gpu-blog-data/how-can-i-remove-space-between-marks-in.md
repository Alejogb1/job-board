---
title: "How can I remove space between marks in an Altair visualization?"
date: "2025-01-30"
id: "how-can-i-remove-space-between-marks-in"
---
The core challenge when eliminating whitespace between marks in Altair visualizations stems from the interplay of default padding, axis configuration, and mark properties. Having spent considerable time fine-tuning various chart types for data presentation in financial dashboards, I've found that controlling these elements is critical to achieving tightly packed, visually dense graphics. Specifically, the default behavior often introduces spacing that, while beneficial for readability in many situations, becomes undesirable when aiming for minimal separation.

The fundamental issue revolves around several Altair configurations, primarily related to scale settings, the `spacing` property of some mark types, and, indirectly, axis properties. In essence, Altair attempts to create a reasonable visual separation between data points to prevent overlap and improve legibility. However, situations arise where we want these marks to touch or sit immediately adjacent to each other. To resolve this, we must override these defaults through precise specification in the Altair encoding. The specific approach varies slightly depending on the type of mark used (e.g., bars, rects, circles).

Here is a breakdown of techniques and three examples demonstrating common scenarios. I'll avoid direct linkage of sources, but be advised that the official Altair documentation and example galleries are indispensable. I regularly consult both when exploring new visual encodings.

**Technique 1: Direct `spacing` control for bars**

For bar charts, Altair offers a specific `spacing` property directly within the mark definition. This directly controls the gap between bars along the categorical axis. The default value is typically a positive number that introduces separation. Setting it to zero, or even a negative value for overlap, is the core of this technique.

```python
import altair as alt
import pandas as pd

# Sample Data
data = {'category': ['A', 'B', 'C', 'D'], 'value': [10, 20, 15, 25]}
df = pd.DataFrame(data)

# Visualization
chart = alt.Chart(df).mark_bar(spacing=0).encode(
    x='category:N',
    y='value:Q'
)

chart.show()
```

In this first example, we use the pandas library to create a simple dataframe. We then construct an Altair `Chart` object, utilizing `mark_bar` to signify bars. Crucially, `spacing=0` is added inside the `mark_bar` declaration. This property explicitly sets the distance between adjacent bars to zero, effectively eliminating the default gaps. Finally, `encode` is used to assign the appropriate data fields to the x and y axes, respectively.  The `show()` call will display the chart. This approach is the most straightforward when dealing with bar charts. The crucial element is the `spacing` parameter, directly attached to the `mark_bar` function.

**Technique 2: Manipulating scales and axes for `rect` marks**

For more complex scenarios such as heatmaps, utilizing `mark_rect` and manipulating scales and axis properties is crucial. Here, it's not the mark-specific `spacing` we control, but the scale and axis configuration to make the rectangles directly touch each other. We ensure the ranges align perfectly with the data intervals.

```python
import altair as alt
import pandas as pd
import numpy as np

# Sample Data: Create a 10x10 matrix
data = np.random.rand(10, 10)
df = pd.DataFrame(data)
df = df.stack().reset_index()
df.columns = ['y', 'x', 'value']

# Visualization
chart = alt.Chart(df).mark_rect().encode(
    x=alt.X('x:O',  axis=None),
    y=alt.Y('y:O',  axis=None),
    color='value:Q'
).properties(width=300, height=300)
chart.configure_scale(bandPaddingInner=0, bandPaddingOuter=0).show()
```

Here, we create a 10x10 data matrix, reshape it into a suitable dataframe format for altair, and build a `mark_rect` chart. The crucial changes here involve disabling axes (axis=None) to eliminate any default spacing these add, and then the `configure_scale` declaration, specifically setting `bandPaddingInner` and `bandPaddingOuter` to 0.  These configurations eliminate the padding between rectangles at the level of scale configuration. This is the primary method for this kind of mark. We also set the width and height using the `properties` call which allows us to size the overall figure. This example illustrates the importance of scale configuration in rect visualizations.

**Technique 3: Customizing scale with ordinal domains for circular marks (minimal gap)**

While circles do not have direct "spacing", achieving a tightly packed circle distribution requires attention to the underlying ordinal scale. Specifically, it's necessary to explicitly specify the domain to ensure the data points are placed close enough, that they touch or very nearly touch. We will use a limited size example for clarity.

```python
import altair as alt
import pandas as pd
import numpy as np

# Sample Data: Limited number of circles along an ordinal scale
data = {'category': ['A', 'B', 'C', 'D'], 'value': np.random.rand(4)}
df = pd.DataFrame(data)

# Visualization
chart = alt.Chart(df).mark_circle(size=200).encode(
    x=alt.X('category:N', scale=alt.Scale(domain=['A','B','C','D']), axis=None),
    y=alt.Y('value:Q', axis=None)

).properties(width=400, height=200)
chart.configure_view(strokeWidth=0).show()

```

In this example, we present a series of circles along an ordinal axis. We use a small random value for the y axis to illustrate different vertical placement, but the key point is setting the x axis scale with `domain` to include all categories and using `axis=None`. The domain array ensures that the marks are placed directly next to each other along the categorical axis. `size` within `mark_circle` defines the size of the circles themselves. The `configure_view` with `strokeWidth=0` removes any outline that the viewbox would otherwise add. We have minimized the space between the circles to ensure minimal separation, almost touching.

In conclusion, manipulating whitespace in Altair visualizations involves a blend of understanding mark-specific parameters like spacing, carefully adjusting scales through properties like `bandPaddingInner` and `bandPaddingOuter`, and, indirectly, axis configurations. These techniques, combined with the ability to specify the exact domain of scales, provide comprehensive control over visual layout, enabling the creation of precise, densely packed charts. I strongly recommend studying the official documentation and example gallery, focusing specifically on scale configurations for each mark type. Experimenting with these options is the best path to mastering fine-grained visual adjustments in Altair.
