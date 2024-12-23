---
title: "How can a layered violin plot and stripplot be created using Altair?"
date: "2024-12-23"
id: "how-can-a-layered-violin-plot-and-stripplot-be-created-using-altair"
---

 Creating a layered visualization using Altair, specifically combining a violin plot and a stripplot, is something I've had to implement more than a few times over the years, often when analyzing complex, multi-dimensional datasets. The goal is usually to show both the distribution of data (the violin plot) and the individual data points themselves (the stripplot), allowing for a more nuanced understanding.

Now, Altair, being a declarative visualization library, makes this relatively straightforward, though there are some nuances you need to keep in mind. The key is to use `alt.layer()` to combine the two plots, ensuring that the aesthetics, like axes, scales, and data encoding, align correctly. Without that, you could easily end up with an overlapping mess that's hard to decipher.

I'll break down the process and share some code examples that should get you started. Before we jump into the code, though, a brief point about best practices. While layering is powerful, it can sometimes lead to visualizations that are visually busy, so remember to evaluate whether a layered approach is actually conveying the necessary information clearly. Sometimes separate, simpler plots might be more effective. Also, be sure your chosen encoding channel (e.g., x or y) for categorical values is appropriate and ordered reasonably for human readability. With that said, let's dive in.

First, let's consider a basic case. I recall working on a project where we were analyzing the response time of different application servers under varying load conditions. We had load level as a categorical variable and the actual response time values for each level as our continuous variable, which is a typical scenario where a violin plot combined with a stripplot comes in handy.

Here’s the first example using sample data generated with pandas and numpy:

```python
import altair as alt
import pandas as pd
import numpy as np

np.random.seed(42)
data = pd.DataFrame({
    'load_level': np.repeat(['low', 'medium', 'high'], 50),
    'response_time': np.random.normal(loc=[10, 15, 20], scale=3, size=150)
})

violin = alt.Chart(data).mark_area(opacity=0.6, interpolate='monotone').encode(
    alt.X('load_level:N'),
    alt.Y('response_time:Q', scale=alt.Scale(zero=False)),
    alt.Y2('response_time:Q', scale=alt.Scale(zero=False)),
).transform_density(
    density='response_time',
    as_=['response_time', 'density'],
    groupby=['load_level']
).transform_calculate(
    mid='datum.density/2',
    lower='datum.density/-2',
    upper='datum.density/2',
).mark_area(orient='horizontal').encode(
  alt.Y('response_time:Q', scale=alt.Scale(zero=False)),
  alt.X('lower:Q', title='Density', axis=None),
  alt.X2('upper:Q'),
  alt.Color('load_level:N',legend=None)
)


strip = alt.Chart(data).mark_circle(color='black', opacity=0.4).encode(
    alt.X('load_level:N'),
    alt.Y('response_time:Q',  scale=alt.Scale(zero=False)),
    alt.Color('load_level:N', legend=None),
).transform_jitter(
    'response_time', as_ = 'response_time'
)

chart = alt.layer(violin, strip).resolve_scale(
    y='independent'
)

chart.display()

```

In this snippet, I’m first generating some sample data with pandas and numpy to simulate response times under different load levels. Then, I create two individual charts: `violin` and `strip`, using Altair's declarative API. The violin plot utilizes the `transform_density` method to calculate the probability density and the `transform_calculate` method to compute the lower and upper boundaries for the plot's area, creating a classic violin-plot shape. Note the `orient='horizontal'` specification and the transformation to obtain the 'x' positions for plotting. This differs from the way the vertical area was constructed. The stripplot is simply a scatter plot using the  `mark_circle()` method. Finally, `alt.layer()` combines them. I've used  `resolve_scale(y='independent')` to ensure both the violin plot's y axis scaling based on density and the stripplot’s y axis scaling based on the response time can be accommodated without having to force a common scale. It's critical, otherwise they may not align properly and would have the incorrect vertical offset. This simple example showcases how to encode data with jitter for improved readability.

Now, let's look at a slightly more complex example. Say I was analyzing sales data across various product categories and geographical regions. We might want a violin-stripplot combination that also includes some visual differentiation based on a secondary grouping. I’ve had projects like this where a combination of two categorizations is important.

Here is another code example, showcasing a grouping dimension:

```python
import altair as alt
import pandas as pd
import numpy as np

np.random.seed(42)
categories = ['Electronics', 'Books', 'Clothing']
regions = ['North', 'South', 'East', 'West']
data = pd.DataFrame({
    'category': np.random.choice(categories, size=300),
    'region': np.random.choice(regions, size=300),
    'sales': np.random.normal(loc=100, scale=30, size=300) + np.random.randint(-40, 40, size=300)
})

violin = alt.Chart(data).mark_area(opacity=0.6, interpolate='monotone').encode(
    alt.X('category:N'),
    alt.Y('sales:Q', scale=alt.Scale(zero=False)),
    alt.Y2('sales:Q', scale=alt.Scale(zero=False)),
    alt.Color('region:N')
).transform_density(
    density='sales',
    as_=['sales', 'density'],
    groupby=['category', 'region']
).transform_calculate(
    mid='datum.density/2',
    lower='datum.density/-2',
    upper='datum.density/2',
).mark_area(orient='horizontal').encode(
  alt.Y('sales:Q', scale=alt.Scale(zero=False)),
  alt.X('lower:Q', title='Density', axis=None),
  alt.X2('upper:Q'),
  alt.Color('region:N')
)


strip = alt.Chart(data).mark_circle(opacity=0.4).encode(
    alt.X('category:N'),
    alt.Y('sales:Q', scale=alt.Scale(zero=False)),
    alt.Color('region:N'),
).transform_jitter(
    'sales', as_ = 'sales'
)

chart = alt.layer(violin, strip).resolve_scale(
    y='independent'
)


chart.display()

```

Here, I've added a ‘region’ column, and that's also added as an encoding for the violin plot and the strip plot. Note that in this case the violin plots and stripplots are being colored by this new column, demonstrating the flexible way you can combine these encodings to express additional data dimensions and relationships. Again, `resolve_scale(y='independent')` ensures each chart's vertical scaling is correctly applied.

For the final example, let's look at a situation that involves custom colors and a more sophisticated layout. Suppose, in one of my projects, I had to represent the performance of different algorithms across several test cases. To ensure visual clarity, we had specific colors for each algorithm, and I made it a multi-chart arrangement using Altair’s `concat` functionality.

Here's that example:

```python
import altair as alt
import pandas as pd
import numpy as np

np.random.seed(42)
algorithms = ['Algo A', 'Algo B', 'Algo C']
tests = [f'Test {i}' for i in range(1, 6)]
data = pd.DataFrame({
    'algorithm': np.repeat(algorithms, 30*len(tests)),
    'test': np.tile(np.repeat(tests, 30), len(algorithms)),
    'performance': np.random.normal(loc=[10, 15, 12], scale=2, size=30*len(tests)*len(algorithms)) + np.random.randint(-2, 2, size=30*len(tests)*len(algorithms))
})

color_scale = alt.Scale(
    domain=algorithms,
    range=['#67a9cf', '#ef8a62', '#a6d854'] #specific color codes for the algorithms
)

charts = []
for test in tests:
  test_data = data[data['test'] == test]

  violin = alt.Chart(test_data).mark_area(opacity=0.6, interpolate='monotone').encode(
        alt.X('algorithm:N'),
        alt.Y('performance:Q', scale=alt.Scale(zero=False),  axis = None),
        alt.Y2('performance:Q', scale=alt.Scale(zero=False)),
        alt.Color('algorithm:N', scale=color_scale, legend=None)
    ).transform_density(
        density='performance',
        as_=['performance', 'density'],
        groupby=['algorithm']
    ).transform_calculate(
        mid='datum.density/2',
        lower='datum.density/-2',
        upper='datum.density/2',
    ).mark_area(orient='horizontal').encode(
      alt.Y('performance:Q', scale=alt.Scale(zero=False), axis=None),
      alt.X('lower:Q', title='Density', axis=None),
      alt.X2('upper:Q'),
      alt.Color('algorithm:N', scale=color_scale, legend=None)
    )

  strip = alt.Chart(test_data).mark_circle(opacity=0.4).encode(
        alt.X('algorithm:N'),
        alt.Y('performance:Q',  scale=alt.Scale(zero=False), title = 'performance'),
        alt.Color('algorithm:N', scale=color_scale, legend=None)
    ).transform_jitter(
        'performance', as_ = 'performance'
    )

  chart = alt.layer(violin, strip).resolve_scale(
      y='independent'
  ).properties(title=test)

  charts.append(chart)

concat_chart = alt.concat(*charts, columns=len(tests))
concat_chart.display()

```

In this example, I've introduced a custom color scale, specified with specific hex color codes. This helps ensure that the colors used in the visualization are consistent with a brand or reporting style. The main point, here, however, is that now I’m looping over the values of ‘test’ to create separate layered violin/stripplots and composing them using `alt.concat()` to display them in a single, well-organized grid. This showcases how you can combine layered plots with other Altair capabilities to achieve more elaborate visualizations.

To deepen your understanding further, I suggest exploring these resources. For a solid understanding of declarative visualization, consult "The Grammar of Graphics" by Leland Wilkinson; it’s foundational. For a deeper dive into Altair, its online documentation is exhaustive and provides numerous examples. Finally, for a practical treatment of visual design principles in data visualization, consider "Information Visualization: Perception for Design" by Colin Ware. These texts should provide a solid grounding in both theory and implementation.

These examples provide a starting point for creating effective layered violin and stripplots in Altair. Remember that visualization is an iterative process; you might have to adjust your designs based on the specific details of your data and the insights you're trying to convey. The power of Altair lies in its expressiveness and declarativeness, enabling you to create precise and visually compelling plots.
