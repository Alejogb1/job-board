---
title: "jupyter notebook plot multiple graphs cell?"
date: "2024-12-13"
id: "jupyter-notebook-plot-multiple-graphs-cell"
---

Okay so you're asking about plotting multiple graphs in a single Jupyter Notebook cell right Been there done that plenty of times Let me tell you it's a common thing people stumble on especially if you're moving from a single graph setup to needing to visualize more data at once

I remember back in my early days I was working on a project involving sensor data analysis and I had like five different sensor readings that I needed to compare simultaneously It was a real headache trying to figure out how to get all that onto one page using Matplotlib in Jupyter I started just chucking plot() after plot() and of course it was a complete mess They were either overlapping or just not appearing at all it was like a train wreck of a visualization

So the key here is understanding that Matplotlib and pandas plotting tools provide ways to define multiple subplots within a single figure This is where `plt.subplots` becomes your best friend It lets you create a grid of axes where each axe is a separate space to render a plot

Let’s start with a basic example using Matplotlib's `pyplot` interface you’ll see this a lot

```python
import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create a figure and a set of subplots
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

# Plot on the first subplot
axes[0].plot(x, y1, label='Sine')
axes[0].set_title('Sine Wave')
axes[0].legend()

# Plot on the second subplot
axes[1].plot(x, y2, label='Cosine', color='red')
axes[1].set_title('Cosine Wave')
axes[1].legend()


plt.tight_layout() # This is super important to prevent overlapping labels
plt.show()
```

Okay lets break this down The `plt.subplots(nrows=2, ncols=1, figsize=(8, 6))` is the important line here We're making a figure which is like the canvas and then we create a grid of subplots This case two rows and one column meaning two plots stacked vertically and then the figsize just sets the size of the overall figure `axes` now is not just a single thing it’s a 2d numpy array where each element in the array is a subplot in our case `axes[0]` is the first plot and `axes[1]` is the second plot then when we plot our data we're targeting each axes array element in the next step

`axes[0].plot(x, y1)` is like plotting on the first axis the same as `axes[1].plot(x, y2)` on the second axis  It’s pretty straightforward once you get the hang of it Then those `set_title` methods are just setting the titles for each axis we add the legends and finally that `plt.tight_layout()` prevents axis overlapping issues because if you do not use this they tend to overlap and it's a mess

Now what if you have multiple things to plot inside the same axes that was a problem for me too when I wanted to visualize different values of a single sensor say the current and voltage Here is how you do it

```python
import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x)/2

# Create a figure and one subplot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot multiple lines on the same subplot
ax.plot(x, y1, label='Sine')
ax.plot(x, y2, label='Cosine', linestyle='--') # Added a style
ax.plot(x,y3, label='Half Sine', color = 'green')

ax.set_title('Multiple Curves on One Plot')
ax.legend()
plt.show()
```

In this example instead of subplots we have a single subplot which is the default setup of matplotlib's `plt.subplots` function but we pass multiple plot calls to this single axis as you can see there is only one ax we use `ax.plot` several times and each time we plot the data but they are on the same axis and that makes it convenient to compare different values on the same chart

Now lets talk about using Pandas DataFrame that also works quite well when we have data that is already in a dataframe format

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample data as a Pandas DataFrame
data = {
    'x': np.linspace(0, 10, 100),
    'Sine': np.sin(np.linspace(0, 10, 100)),
    'Cosine': np.cos(np.linspace(0, 10, 100)),
    'Tangent' : np.tan(np.linspace(0,10,100))
}

df = pd.DataFrame(data)


# Directly using DataFrame's plot method with subplots
df.plot(x='x', subplots=True, figsize=(10, 8))

plt.tight_layout()
plt.show()

```

Here I generated a DataFrame containing all the data and the interesting part is `df.plot(x='x', subplots=True, figsize=(10, 8))` Pandas has this amazing built-in `plot` function for dataframes and when you set `subplots=True` it automatically generates multiple plots for all the columns in your dataframe (except the index column and our specific x-axis column) This is very useful if you have data already in Pandas format

The cool thing is that if you have different column groups you can also specify a `layout` param or specify the columns to be plotted with `y` this is extremely versatile and powerful stuff

So to summarize a few key things to remember

*   **`plt.subplots()` is your friend**: Use it to create a grid of subplots when you need multiple separate plots in one cell
*   **`ax.plot()` vs `plt.plot()`**: Use axes objects if you are doing multiple subplots using subplots and plt.plot if you have a single axes and want to make multiple plot lines.
*   **`plt.tight_layout()`**: Always use it to prevent label overlapping issues its a time saver and will save you hours of debugging in the long run
*   **Pandas `df.plot()`**: It is an efficient way to plot data from Pandas DataFrames

For learning more about the underlying mechanics I'd recommend diving into the Matplotlib documentation It’s a gold mine for plot customization Also look into some material regarding data visualization for principles of effective plotting if you are new to it A good book to start is "Fundamentals of Data Visualization" by Claus O. Wilke also some more advanced books on the topic would be "The Visual Display of Quantitative Information" by Edward R. Tufte

Remember that every chart has a purpose avoid plot clutter and always plot with intention you might also wanna look into various charting libraries like seaborn or plotly they are built on top of Matplotlib so understanding the fundamentals is useful when learning these advanced tools

Oh and before I forget there’s this one situation I had where I spent like half a day trying to debug a plot issue and it turns out the code was actually fine I had forgotten that I had commented out the plt.show() command it's like leaving the house but forgetting your keys a complete face palm moment right? so don’t forget your keys everyone

I hope this helps let me know if you have any more questions
