---
title: "How can altair facet plots be annotated with metrics?"
date: "2025-01-30"
id: "how-can-altair-facet-plots-be-annotated-with"
---
Faceting with Altair provides an elegant means to visualize data subsets, but the default presentation often lacks context for comparative analysis beyond the visual. Specifically, incorporating summary metrics directly into the facet plots requires careful composition of Altair’s layering and transformation capabilities. I've faced this challenge frequently, particularly when analyzing sensor data across different locations, where presenting the mean and standard deviation within each facet greatly improves interpretability. My approach involves three primary techniques, each suited to different types of annotations.

First, calculating and displaying single scalar values within each facet, such as a mean or median, is achieved by creating a separate text layer, carefully aligned with the chart elements. The process starts with calculating the target metric via a `transform_aggregate` operation. Then, using that transformed data source, a text mark is created using the aggregated value along with appropriate X and Y coordinates. The layering ensures that the text appears on top of the plot. Precision in positioning these annotations is crucial; I usually employ a combination of relative and absolute coordinates to achieve the desired visual effect. This often involves trial-and-error, adjusting pixel offsets until the text sits comfortably within the space allocated to the plot, considering the plot's axes and the surrounding padding.

```python
import altair as alt
import pandas as pd

# Sample data (replace with your actual data)
data = pd.DataFrame({
    'location': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    'sensor_value': [10, 12, 15, 8, 9, 11, 13, 16, 14]
})


# Calculate the mean sensor value per location
mean_data = data.groupby('location').agg({'sensor_value':'mean'}).reset_index()
mean_data.rename(columns={'sensor_value':'mean_value'},inplace=True)


# The base scatter plot
base_chart = alt.Chart(data).mark_circle().encode(
    x = 'location:N',
    y = 'sensor_value:Q'
).properties(
    title = 'Sensor Values by Location with Mean Annotations'
)

# The text layer for mean annotation.
text_layer = alt.Chart(mean_data).mark_text(dy=-5, color='red').encode(
    x = 'location:N',
    y = alt.value(17),  # Adjust vertical position to place text above bars
    text = alt.Text('mean_value:Q', format='.1f')
)

# Combine layers
final_chart = (base_chart + text_layer).facet(
    column = 'location:N',
    spacing = 10
).configure_title(
    fontSize=16,
    anchor='middle'
)


final_chart
```

In this example, `mean_data` is derived from the initial dataframe and stores the average sensor reading for each location. The `text_layer` utilizes this, placing each mean reading directly above the scatter plot. The `dy` parameter in the mark_text function offsets the text slightly above the calculated position, preventing overlap with data points. The `y = alt.value(17)` is adjusted to accommodate the maximum sensor value, preventing clipping of the mean text.

Second, for annotations such as statistical ranges (e.g., standard deviation), a similar but more complex layered approach is required. Instead of single values, we're now working with two values defining the range which needs to be visualised. This means using a rule mark along with associated error bars can enhance interpretation. Again, we calculate the summary statistics using a transform operation. In practice, I often prefer using upper and lower limit markers instead of error bars to improve clarity, especially when the dataset is noisy. These markers are constructed using rule marks, taking care to provide correct x-values relative to the base plot and precise y-values representing the upper and lower bounds. The placement of these error ranges is often done using absolute values, in order to prevent re-scaling issues due to the ranges themselves or any faceting,

```python
import altair as alt
import pandas as pd
import numpy as np

# Sample data (replace with your actual data)
data = pd.DataFrame({
    'location': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    'sensor_value': [10, 12, 15, 8, 9, 11, 13, 16, 14]
})

# Calculate mean and standard deviation per location
summary_data = data.groupby('location').agg(
    mean_value = pd.NamedAgg(column='sensor_value',aggfunc='mean'),
    std_value = pd.NamedAgg(column='sensor_value',aggfunc='std')
).reset_index()

summary_data['upper_bound'] = summary_data['mean_value'] + summary_data['std_value']
summary_data['lower_bound'] = summary_data['mean_value'] - summary_data['std_value']


# Base scatter plot
base_chart = alt.Chart(data).mark_circle().encode(
    x='location:N',
    y='sensor_value:Q'
).properties(
    title = 'Sensor Values by Location with SD Range'
)


# Error bar layer
error_layer = alt.Chart(summary_data).mark_rule(color='red',strokeDash=[3,3]).encode(
    x = 'location:N',
    y = 'lower_bound:Q',
    y2 = 'upper_bound:Q'
)

# Combine and facet the chart
final_chart = (base_chart + error_layer).facet(
    column='location:N',
    spacing = 10
).configure_title(
    fontSize=16,
    anchor='middle'
)

final_chart
```

Here, `summary_data` now includes the mean, standard deviation, and upper/lower bounds calculated from the original sensor readings. The `error_layer` draws a dotted red rule between the calculated upper and lower bounds for each location, which can be interpreted as one standard deviation around the mean. The `strokeDash` styling improves the visual clarity of this element.

Third, and finally, overlaying a histogram or density plot of the metric values alongside the facet plot is another strategy. This type of annotation provides a global context to the distribution of values within each facet. It's typically implemented using a combination of a vertical rule indicating the mean or median along with the density layer. These secondary plots use a shared vertical axis, but are located such that they do not interfere with the primary plots. This is often implemented by specifying the same scale for the y-axis, but not including it as part of the facetting. To do this, it's important to perform all transformations to your data beforehand and to construct the various layers and charts to combine before applying the facetting.

```python
import altair as alt
import pandas as pd

# Sample data
data = pd.DataFrame({
    'location': ['A'] * 10 + ['B'] * 10 + ['C'] * 10,
    'sensor_value': [10, 12, 15, 8, 9, 11, 13, 16, 14, 12,
                     20, 22, 25, 18, 19, 21, 23, 26, 24, 22,
                     30, 32, 35, 28, 29, 31, 33, 36, 34, 32]
})

# Calculate mean sensor value per location
mean_data = data.groupby('location').agg({'sensor_value':'mean'}).reset_index()
mean_data.rename(columns={'sensor_value':'mean_value'},inplace=True)


# Base Scatter Plot
base_chart = alt.Chart(data).mark_circle().encode(
    x = 'location:N',
    y = 'sensor_value:Q'
).properties(
    title = 'Sensor Values with Density and Mean'
)


# Density plot for global sensor value distribution
density_layer = alt.Chart(data).transform_density(
    density='sensor_value',
    as_=['sensor_value', 'density'],
).mark_area(opacity=0.5,color='lightblue').encode(
    x=alt.X('sensor_value:Q', scale = alt.Scale(zero=False)),
    y=alt.Y('density:Q',axis = None),
).properties(
  width = 150,
  height=175
)


# Vertical mean marker on density plot
mean_marker = alt.Chart(mean_data).mark_rule(color='red').encode(
    x=alt.X('mean_value:Q'),
    size = alt.value(2)
)

density_plot = density_layer + mean_marker


# Combining the plots

final_chart = alt.concat(
    base_chart,
    density_plot,
    spacing=20
).resolve_scale(
    y='shared'
).facet(
    column = 'location:N'
).configure_title(
    fontSize=16,
    anchor='middle'
)

final_chart
```

Here, a density plot and mean marker are created and then placed to the right of the faceted plot using `alt.concat()`. Note, that the density plot was created *before* faceting was applied, but has a shared y-scale.

In summary, annotating facet plots in Altair with summary metrics requires careful planning of layers and data transformations. The choice of method—single scalar text labels, range-based visual aids, or overlaid distributions—depends entirely on the specific data and the communication goal. Key to successful annotation is precise positioning using relative and absolute coordinate offsets, along with clear layering to maintain visual hierarchy. I've found these techniques to greatly enhance my data visualizations, allowing for easier and more insightful interpretation. Recommended reading includes the Altair documentation on `transform_aggregate`, `mark_text`, `mark_rule`, `transform_density`, and layered chart composition. Furthermore, consider seeking out examples relating to statistical annotation of graphs within visualization packages such as Matplotlib, as often the concepts overlap.
