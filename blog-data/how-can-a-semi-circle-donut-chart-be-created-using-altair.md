---
title: "How can a semi-circle donut chart be created using Altair?"
date: "2024-12-23"
id: "how-can-a-semi-circle-donut-chart-be-created-using-altair"
---

Alright, let's tackle this semi-circle donut chart challenge with Altair. I’ve certainly bumped into similar visualization needs during various data explorations, especially when space is a premium or when a standard pie chart feels…excessive. The trick lies in manipulating Altair's arc marks and careful data transformations, which I've streamlined over time through a few different project contexts. Let's get into the practicalities.

Firstly, the essence of a semi-circle donut chart involves visualizing data as arcs covering only 180 degrees instead of the full 360. Altair, being declarative, doesn't inherently offer a 'semi-circle' chart type. Instead, we achieve it by calculating and applying start and end angles for our arc marks. This requires a specific data structure and encoding, which is where understanding the nuances becomes essential.

The core challenge I've consistently encountered is that Altair expects angle data to be in radians, not degrees. Many data sources provide angles in degrees, so the conversion becomes a necessary step. Beyond this, ensuring that the arcs are properly positioned, have appropriate widths (to create the donut effect), and are color-coded to represent data categories, requires careful configuration. Also, we must avoid having the chart look crammed or disproportionate by fine-tuning parameters like the radius.

Let’s dive into three illustrative code snippets, starting with the foundational logic and building on top of it for more refinement:

**Example 1: A Basic Semi-Circle Donut Chart**

This code will produce a rudimentary semi-circle donut. We’ll deal with colors and labels in subsequent examples.

```python
import altair as alt
import pandas as pd
import numpy as np

data = pd.DataFrame({
    'category': ['A', 'B', 'C'],
    'value': [40, 30, 30]
})

total = data['value'].sum()
data['angle'] = data['value'] / total * np.pi
data['startAngle'] = data['angle'].cumsum() - data['angle']
data['endAngle'] = data['angle'].cumsum()

chart = alt.Chart(data).mark_arc(innerRadius=50, outerRadius=100).encode(
    theta=alt.Theta(field="angle", type="quantitative"),
    startAngle=alt.StartAngle("startAngle"),
    endAngle=alt.EndAngle("endAngle")
).properties(title='Basic Semi-Circle Donut Chart')

chart.show()
```

Here, we compute each category's angle as a proportion of the total, convert it to radians (multiplying by *pi*), then we compute start and end angles by accumulating this new angle. We then encode these computed values using the *startAngle* and *endAngle* attributes. This is the basic mechanism for creating arc-based charts in Altair when working with angles. Note the use of *innerRadius* and *outerRadius* to generate the donut hole.

**Example 2: Adding Colors and a Background Arc**

A semi-circle donut chart looks cleaner with distinct colors and a background arc, especially when you have data categories. This example builds upon the basic logic by adding categorical coloring and a background 'empty' arc.

```python
import altair as alt
import pandas as pd
import numpy as np

data = pd.DataFrame({
    'category': ['A', 'B', 'C'],
    'value': [40, 30, 30]
})

total = data['value'].sum()
data['angle'] = data['value'] / total * np.pi
data['startAngle'] = data['angle'].cumsum() - data['angle']
data['endAngle'] = data['angle'].cumsum()

background_data = pd.DataFrame({'startAngle': [0], 'endAngle': [np.pi]})

background = alt.Chart(background_data).mark_arc(innerRadius=50, outerRadius=100, color='lightgray').encode(
    startAngle = 'startAngle:Q',
    endAngle='endAngle:Q'
)

chart = alt.Chart(data).mark_arc(innerRadius=50, outerRadius=100).encode(
    theta=alt.Theta(field="angle", type="quantitative"),
    startAngle=alt.StartAngle("startAngle"),
    endAngle=alt.EndAngle("endAngle"),
    color='category:N'
).properties(title='Semi-Circle Donut Chart with Colors and Background')


combined_chart = (background + chart)

combined_chart.show()
```

Here, a *background* chart is added and positioned behind the arc chart, representing the overall shape. I've found that this helps viewers grasp that the display is actually half of a circle. Note the explicit encoding of data type (Q = Quantitative, N = Nominal) in Altair. This is crucial for correct interpretation and avoids errors.

**Example 3: Adding Labels and Adjusting Parameters**

Often, a semi-circle donut chart without clear labels is not useful for presentation purposes. Let’s integrate labels and fine-tune some visual parameters.

```python
import altair as alt
import pandas as pd
import numpy as np

data = pd.DataFrame({
    'category': ['A', 'B', 'C'],
    'value': [40, 30, 30]
})

total = data['value'].sum()
data['angle'] = data['value'] / total * np.pi
data['startAngle'] = data['angle'].cumsum() - data['angle']
data['endAngle'] = data['angle'].cumsum()
data['midAngle'] = (data['startAngle'] + data['endAngle']) / 2

background_data = pd.DataFrame({'startAngle': [0], 'endAngle': [np.pi]})

background = alt.Chart(background_data).mark_arc(innerRadius=50, outerRadius=100, color='lightgray').encode(
    startAngle = 'startAngle:Q',
    endAngle='endAngle:Q'
)

chart = alt.Chart(data).mark_arc(innerRadius=50, outerRadius=100).encode(
    theta=alt.Theta(field="angle", type="quantitative"),
    startAngle=alt.StartAngle("startAngle"),
    endAngle=alt.EndAngle("endAngle"),
    color='category:N'
).properties(title='Semi-Circle Donut Chart with Labels')

text = alt.Chart(data).mark_text(radius=120, align='center', dy=-5).encode(
    x = alt.value(100),
    y = alt.value(100),
    text='category:N',
    angle = 'midAngle:Q'
)

combined_chart = (background + chart + text).configure_view(strokeWidth=0).properties(width=400, height=250)
combined_chart.show()

```

Here, we generate a *text* chart to add labels that are placed a bit outside the donut. A *midAngle* field is generated to position the labels properly. The radius is also increased here to move labels outside the donut and *configure_view* is added to remove the default border and *properties* are used to limit the chart size.

Key takeaway is that each 'layer' you see being built upon in the code examples – the background arc, the data arc marks, and the text labels – are all combined together to create the overall semi-circle chart that we want. This is part of the beauty of Altair, as it allows modular composition of charts in a very readable format.

For further exploration into advanced chart customizations with Altair, I would recommend checking out the official documentation; it's a treasure trove of information and examples. Specifically for understanding the underlying grammar of graphics, I would suggest Hadley Wickham's *ggplot2: Elegant Graphics for Data Analysis*, while it’s about R’s ggplot2 package, the principles it presents regarding layered graphics and visual encoding transfer directly to Altair. Also, *Interactive Data Visualization for the Web* by Scott Murray is an excellent, broader resource for creating data visualizations with JavaScript, and many of the concepts covered are foundational for understanding what libraries like Altair are doing under the hood.

Working with Altair can sometimes seem tricky, especially when dealing with custom charts, but breaking down the visualization into simpler components (like calculating angles, layering chart types, setting text attributes, etc.) simplifies the process. This systematic, composable, approach, while requiring some initial setup, provides great flexibility when tackling various data display challenges.
