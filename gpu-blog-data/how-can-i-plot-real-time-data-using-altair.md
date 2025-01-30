---
title: "How can I plot real-time data using Altair in Python?"
date: "2025-01-30"
id: "how-can-i-plot-real-time-data-using-altair"
---
The primary challenge when plotting real-time data with Altair stems from its declarative nature, designed for static charts rather than continuous updates. Altair renders charts based on a provided data frame and encoding, making it ill-suited to handle an ongoing stream of data. A typical approach involves continuously updating and re-rendering the chart, often using an external library to manage the dynamic data stream and presentation.

The fundamental problem is that Altair creates visualizations by describing them rather than drawing them imperatively on a canvas. When data changes, the chart must be entirely redefined; it doesn't allow for incremental modifications like those found in imperative plotting systems. This means that simply altering the data source after the chart has been rendered doesn't automatically update the visual output. Consequently, a loop or periodic process is necessary, continually feeding new data to the Altair specification and then re-rendering the chart. This re-rendering involves a translation of the specification into a JSON structure which is rendered client-side by either Vega-Lite or a server-side renderer.

A robust strategy to achieve "real-time" plotting with Altair relies on a combination of an iterative data acquisition loop, regular updates to a Pandas DataFrame, and a mechanism to trigger a re-render of the chart. Consider a scenario involving streaming sensor readings, such as temperature. Instead of directly modifying a chart object, a new chart object must be constructed with the current state of the data each time.

Let's examine three illustrative examples. I often prefer starting with a simplified version before diving into something more complex, as this makes debugging and problem identification easier.

**Example 1: Basic Line Chart Update with Simulated Data**

This example showcases a basic dynamic line chart.  I've opted to use the `ipywidgets` library to visualize the chart as output within a Jupyter environment. Since real-time data is simulated, this approach can easily be adapted to any data stream. The key here is constructing a new chart object every time and updating the output object.

```python
import altair as alt
import pandas as pd
import time
from IPython.display import display, clear_output
import ipywidgets as widgets

# Initialize an empty DataFrame
data = pd.DataFrame({'time': [], 'temperature': []})
chart_output = widgets.Output()

def update_plot(data):
    chart = alt.Chart(data).mark_line().encode(
        x='time:T',
        y='temperature:Q'
    )
    with chart_output:
      clear_output(wait=True)
      display(chart)

display(chart_output)

# Simulation loop
start_time = time.time()
for i in range(100):
  current_time = time.time() - start_time
  new_temp = 20 + (i * 0.1) + (i % 5) * 2 # Simulated temperature variation
  data = pd.concat([data, pd.DataFrame({'time': [current_time], 'temperature': [new_temp]})], ignore_index=True)
  update_plot(data)
  time.sleep(0.1)
```

In this example, `ipywidgets` manages the display. The `update_plot` function constructs the Altair chart. Note how we do not modify a single chart object; we constantly create new ones using the latest data. `clear_output(wait=True)` within the `with chart_output` statement is crucial for removing the previous chart rendering and is necessary to avoid having the charts accumulate within the output cell. The simulation loop generates a synthetic temperature reading over time, updating the dataframe. The delay introduced by `time.sleep` manages the visual update frequency, simulating real-time data.

**Example 2: Adding a Dynamic Title**

Often, displaying the current value alongside the real-time data enhances understanding. I often use dynamic chart titles for this. This example builds on the previous one, adding a dynamic title that displays the latest temperature.

```python
import altair as alt
import pandas as pd
import time
from IPython.display import display, clear_output
import ipywidgets as widgets

# Initialize an empty DataFrame
data = pd.DataFrame({'time': [], 'temperature': []})
chart_output = widgets.Output()


def update_plot(data):
    latest_temp = data['temperature'].iloc[-1] if not data.empty else 'N/A'
    chart = alt.Chart(data, title=f"Temperature: {latest_temp:.2f}°C").mark_line().encode(
        x='time:T',
        y='temperature:Q'
    )
    with chart_output:
      clear_output(wait=True)
      display(chart)


display(chart_output)

# Simulation loop
start_time = time.time()
for i in range(100):
  current_time = time.time() - start_time
  new_temp = 20 + (i * 0.1) + (i % 5) * 2 # Simulated temperature variation
  data = pd.concat([data, pd.DataFrame({'time': [current_time], 'temperature': [new_temp]})], ignore_index=True)
  update_plot(data)
  time.sleep(0.1)
```
In this example, we added `title` to the Altair Chart constructor to display the latest temperature. The use of an f-string in the title declaration provides a mechanism for formatting the output. I used `data['temperature'].iloc[-1]` to extract the most recent temperature value before building the chart. The conditional within that statement avoids errors if data is empty, during initialization for instance. Note the use of a format specifier `:.2f` to truncate the floating-point number to two decimal places, for a less cluttered chart. This also provides a sense of the dynamic nature of the data.

**Example 3: Implementing a Scrolling Chart**

Often, as data is continuously streamed, it is not desirable to display all previous data. Instead, a scrolling chart can be implemented that focuses on a specific "window" of data. This provides context and reduces visual clutter. Here, we'll modify the previous examples to limit the data displayed to the last 30 time points.

```python
import altair as alt
import pandas as pd
import time
from IPython.display import display, clear_output
import ipywidgets as widgets

# Initialize an empty DataFrame
data = pd.DataFrame({'time': [], 'temperature': []})
chart_output = widgets.Output()
window_size = 30


def update_plot(data):
    if len(data) > window_size:
      data = data.tail(window_size)
    latest_temp = data['temperature'].iloc[-1] if not data.empty else 'N/A'
    chart = alt.Chart(data, title=f"Temperature: {latest_temp:.2f}°C").mark_line().encode(
        x='time:T',
        y='temperature:Q'
    )
    with chart_output:
      clear_output(wait=True)
      display(chart)


display(chart_output)

# Simulation loop
start_time = time.time()
for i in range(100):
  current_time = time.time() - start_time
  new_temp = 20 + (i * 0.1) + (i % 5) * 2 # Simulated temperature variation
  data = pd.concat([data, pd.DataFrame({'time': [current_time], 'temperature': [new_temp]})], ignore_index=True)
  update_plot(data)
  time.sleep(0.1)
```

This version introduces the `window_size` variable, initialized to 30. The `if len(data) > window_size` condition checks if the current data size exceeds this window. If so, `data.tail(window_size)` slices the Pandas DataFrame, keeping only the last 30 rows, and thus providing a rolling window effect.

For resources, I would recommend focusing on the core documentation for Altair, Pandas, and any widget library you choose (such as ipywidgets). I also often find it useful to explore tutorials on declarative programming as it can help clarify some core principles behind using Altair. Experimenting with smaller examples like the ones I have provided is often more helpful than delving into overly complex code when learning these techniques. Finally, studying best practices for UI development, particularly in context of Jupyter environments, can dramatically improve how your application renders data.
