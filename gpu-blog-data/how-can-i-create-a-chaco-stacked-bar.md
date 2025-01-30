---
title: "How can I create a Chaco stacked bar plot with distinct colors for each bar and segment?"
date: "2025-01-30"
id: "how-can-i-create-a-chaco-stacked-bar"
---
Stacked bar plots, commonly used for visualizing compositional data, present unique challenges in ensuring both clarity and aesthetic appeal. In my experience developing data visualizations for geological survey analysis, precisely controlling the color palette across multiple bars and segments is crucial to convey the nuances in sediment composition. Achieving this in Chaco, a Python plotting library designed for interactive data exploration, necessitates a careful understanding of its data structures and rendering pipeline.

The core principle revolves around leveraging Chaco's capability to accept lists of colors for its various plot elements. Unlike simpler plotting libraries where you might specify a single color for an entire bar, Chaco permits the association of distinct colors with individual segments of a stacked bar and further allows for distinct colors of the bars themselves, where each bar often represents a distinct category. We need to build a suitable data structure that maps categories and sub-categories to colors, and then configure the plotting components accordingly.

Let’s break down the implementation. First, consider the underlying data. You'll typically have a dataset that describes the composition of different entities, each entity containing multiple subcategories with associated values. A common format is a dictionary, where keys represent the entities (or bars), and the values are dictionaries themselves, detailing subcategories and their corresponding numerical values. For example:

```python
data = {
    "Sample A": {"Sandstone": 40, "Shale": 35, "Limestone": 25},
    "Sample B": {"Sandstone": 60, "Shale": 20, "Limestone": 20},
    "Sample C": {"Sandstone": 30, "Shale": 50, "Limestone": 20},
}
```

The critical step is to create a *consistent* mapping of subcategories to colors. This mapping must be applied uniformly across all bars to ensure a coherent visual representation. A standard dictionary can handle this as well:

```python
category_colors = {
    "Sandstone": "lightcoral",
    "Shale": "cadetblue",
    "Limestone": "mediumseagreen",
}
```

Now, let's construct a Chaco plot. We will begin with a simplified version, focusing on setting per-segment colors within a single stack, then enhance it in subsequent examples.

```python
# Example 1: Basic stacked bar with per-segment colors for a single bar
from chaco.api import create_plot, add_default_grids, StackedBarPlot
from enable.component import Component
from enable.api import ComponentEditor
from traits.api import HasTraits, Instance
from traitsui.api import View, Item

class PlotContainer(HasTraits):
    plot = Instance(Component)

    def __init__(self, data, category_colors, **kwargs):
        super(PlotContainer, self).__init__(**kwargs)
        x = [0]  # Single bar position
        y_arrays = []
        colors = []
        labels = []

        for category, value in data.items():
            y_arrays.append([value])
            colors.append(category_colors[category])
            labels.append(category)

        plot = create_plot(
            (x, y_arrays),
            type=StackedBarPlot,
            orientation='v',
            padding=10,
            bar_width=0.6
            )

        plot.plots["StackedBarPlot 0"].color = colors
        plot.plots["StackedBarPlot 0"].bar_labels = labels

        add_default_grids(plot)
        self.plot = plot


    view = View(
        Item('plot', editor=ComponentEditor(), show_label=False),
        width = 600,
        height = 400,
        resizable=True
    )


if __name__ == "__main__":
    # Sample data
    sample_data = {
        "Sandstone": 40,
        "Shale": 35,
        "Limestone": 25,
    }

    sample_category_colors = {
       "Sandstone": "lightcoral",
       "Shale": "cadetblue",
       "Limestone": "mediumseagreen",
    }
    
    container = PlotContainer(data=sample_data, category_colors = sample_category_colors)
    container.configure_traits()
```

In this simplified example, only one bar is plotted. The *y_arrays* list now contains a single list which represents the stack of values to represent as segments. The `colors` list holds the colors that will be applied to the segments, ensuring the appropriate color for each geological type. In the main section, the sample data and category colors are defined and used for creating and showing the plot.

To visualize multiple bars, each with its own composition, the process extends by iterating over the samples. This will necessitate building lists of `x_values`, and building the `y_arrays` to be a list of lists where each inner list holds the values for the segments in a bar.

```python
# Example 2: Stacked bar plot with distinct bar colors
from chaco.api import create_plot, add_default_grids, StackedBarPlot
from enable.component import Component
from enable.api import ComponentEditor
from traits.api import HasTraits, Instance
from traitsui.api import View, Item
import numpy as np

class PlotContainer(HasTraits):
    plot = Instance(Component)

    def __init__(self, data, category_colors, bar_colors, **kwargs):
        super(PlotContainer, self).__init__(**kwargs)
        
        x_values = np.arange(len(data)) # positions for bars
        y_arrays = [] # data to plot
        bar_labels = []
        
        for sample_name, sample_data in data.items():
            sample_values = []
            for category in category_colors.keys(): # ensures consistent sub-category order.
                sample_values.append(sample_data.get(category, 0)) # zero if category not in sample
            y_arrays.append(sample_values)
            bar_labels.append(sample_name)


        plot = create_plot(
            (x_values, y_arrays),
            type=StackedBarPlot,
            orientation='v',
            padding=10,
            bar_width=0.6
            )
        
        # apply colors to segments (each bar segment)
        plot.plots["StackedBarPlot 0"].color = list(category_colors.values())
        
        # apply the bar labels
        plot.plots["StackedBarPlot 0"].bar_labels = bar_labels
        
        # apply colors to each bar
        plot.plots["StackedBarPlot 0"].bar_fill_color = bar_colors
        
        add_default_grids(plot)
        self.plot = plot

    view = View(
        Item('plot', editor=ComponentEditor(), show_label=False),
        width = 600,
        height = 400,
        resizable=True
    )

if __name__ == "__main__":
    # Sample data with multiple bars
    sample_data = {
        "Sample A": {"Sandstone": 40, "Shale": 35, "Limestone": 25},
        "Sample B": {"Sandstone": 60, "Shale": 20, "Limestone": 20},
        "Sample C": {"Sandstone": 30, "Shale": 50, "Limestone": 20},
    }

    sample_category_colors = {
        "Sandstone": "lightcoral",
        "Shale": "cadetblue",
        "Limestone": "mediumseagreen",
    }
    
    sample_bar_colors = ["lightgrey", "whitesmoke", "gainsboro"]

    container = PlotContainer(data=sample_data, category_colors = sample_category_colors, bar_colors = sample_bar_colors)
    container.configure_traits()

```

This updated example iterates through our `data` dictionary. For each sample, it extracts values of each category, and appends them to the `y_arrays`. Critically, in this example we also build the labels for each bar, apply colors to each of the segments and then finally set the fill color for each bar using the `bar_fill_color` property. This allows for distinct visual separation of bars in addition to distinct segmentation by category. The main section now defines multiple samples, and a list of colors to assign to the bars.

Finally, to handle scenarios with potentially missing subcategories, ensure you use `.get()` with a default value of 0. In the second example, if a sample does not have a specific category, the code will treat the value as `0`. In case, the order of the categories in each sample is not consistent, you would need to iterate through the `category_colors` dictionary to ensure consistency, as done above.

In terms of resources, I would recommend exploring the official Chaco documentation. While not as extensive as some other plotting libraries, it offers precise details on the `StackedBarPlot` and relevant components. Additionally, examining the source code can provide insights into how the plot rendering functions work. Books specializing in scientific visualization with Python, particularly those addressing the Enthought Tool Suite (ETS), are beneficial.

This combination of knowledge, combined with careful construction of the data structures and a clear understanding of Chaco's properties, allows for generating highly customized stacked bar plots, precisely controlling all elements, including the distinct colors of the bars and their segments. These practices, honed through practical experience, enable one to generate effective visual tools for analytical and exploratory purposes.
