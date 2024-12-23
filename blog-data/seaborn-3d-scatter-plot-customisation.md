---
title: "seaborn 3d scatter plot customisation?"
date: "2024-12-13"
id: "seaborn-3d-scatter-plot-customisation"
---

 so seaborn 3d scatter plots right yeah I've been there a few times believe me customization can be a bit of a rabbit hole especially when you're trying to get it exactly right This is one of those things where you think "oh it's just a scatter plot how hard can it be" and then you find yourself three hours deep wrestling with matplotlib axes objects

So lemme break it down based on what I've seen over the years my own struggles and hopefully steer you clear of some pitfalls I definitely fell into back in the day

First off yeah seaborn doesn't directly have a 3D scatter plot function You're gonna need matplotlib for the 3D aspect Seaborn is awesome for general statistical plotting but for that extra dimension matplotlib is your tool I think the seaborn documentation hints at this in the background

I remember once trying to do some analysis of a simulation I had run involving a bunch of particles I had position data in X Y and Z and I figured a 3D scatter plot was the way to go I started just throwing seaborn at the problem and it was obviously the wrong tool

I tried for like 2 hours to find some secret keyword argument or some magic function but of course no that was not it because seaborn does not have the native functions to make 3D plots

 so how do you actually create the 3D scatter plot You'll want to create a matplotlib figure and axes with projection='3d' first then use matplotlib's scatter function to create the actual points

Here's a basic example to get us started

```python
import matplotlib.pyplot as plt
import numpy as np

# Sample data
np.random.seed(42)
x = np.random.rand(100)
y = np.random.rand(100)
z = np.random.rand(100)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection='3d')

ax.scatter(x, y, z)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.title("Basic 3D Scatter Plot")
plt.show()
```

Simple enough right You're creating some random points and visualizing them the crucial part is using fig.add_subplot(projection='3d') that tells matplotlib you want a 3D axes object

Now the real customization work begins Things like marker size color shape labels that's where the real effort goes I once had to visualize protein folding data and believe me the default settings are not suitable for this kind of job you would just see overlapping blobs it looked like spaghetti and not like science

So lets talk about marker size and color I usually find myself needing to adjust these when the data becomes too noisy or dense or when I want some points to have emphasis

Here's how you would do it

```python
import matplotlib.pyplot as plt
import numpy as np

# Sample Data
np.random.seed(42)
x = np.random.rand(100)
y = np.random.rand(100)
z = np.random.rand(100)
sizes = np.random.rand(100) * 50 # Make size dependant on a variable
colors = np.random.rand(100) # make the color dependent on a variable

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection='3d')

scatter_plot = ax.scatter(x, y, z, s=sizes, c=colors, cmap='viridis', alpha=0.6)
# Add a colorbar to show color mapping
cbar = fig.colorbar(scatter_plot)
cbar.set_label('Color Value')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.title("Custom Size and Color Scatter Plot")
plt.show()
```

You'll see a couple of things here first `s=sizes` for variable marker sizes and `c=colors` for variable colors using a colormap `cmap='viridis'` is often a good default but feel free to explore other colormaps you can actually go deep down on matplotlib's colormap documentation if you want to have the exact colormap you need for a given plot you can even make your own colormaps

I have to say at this point if you are dealing with large data sets you are going to start struggling with performance at this point the most simple thing you can do is reduce the number of data points but you can also explore other more advanced visualization options you can't just draw all the points all the time without the code taking too long to render

Now let's talk about the labels they can be so helpful for providing context but the defaults usually make it hard for the user to see what exactly you are plotting at a single glance This part I struggled for long I used to have a plot and a report I was presenting and after asking many questions people were not understanding the plot I had to redo all of them again that was a big lesson for me make sure the labels are readable

And also let's talk about the legend on 3D scatter plots it's a little more nuanced than on 2D plots you need to create a proxy artist for each type of marker you want in the legend you cant directly do it like in 2D scatter plots because these markers are actually 3D objects

Here is an example with the labels and the legend and some other settings

```python
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Sample Data
np.random.seed(42)
x = np.random.rand(100)
y = np.random.rand(100)
z = np.random.rand(100)
categories = np.random.choice(['A', 'B', 'C'], size=100) # Generate random labels
colors = {'A': 'red', 'B': 'blue', 'C': 'green'}
marker_size = 40
marker_alpha = 0.8 # Transparancy

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')

for category in ['A','B','C']:
    indices = categories == category
    ax.scatter(x[indices], y[indices], z[indices], c=colors[category], marker='o', label=category, s=marker_size, alpha=marker_alpha)

ax.set_xlabel('X-axis - Component 1')
ax.set_ylabel('Y-axis - Component 2')
ax.set_zlabel('Z-axis - Component 3')
ax.set_title('Categorical 3D Scatter Plot')

ax.legend(loc='upper left')
ax.view_init(elev=20, azim=30) # change the default view to see it from a better angle
plt.show()
```

A few things are going on here

*   We iterate over the unique categories and plot the corresponding data points with different colors
*   We use ax.legend() to display the legend
*  We use `ax.view_init` to make the visualization better
*   Labels are more detailed

I remember once spending a whole day trying to make sense of a plot and it turned out all I needed was to slightly adjust the view angle using `ax.view_init` it was a game changer for that data presentation

Also a word of caution When dealing with complex data it is helpful to rotate the view and zoom in or out to try to get a better idea of the data layout you are plotting Also sometimes the data might be completely useless if you only use a 3D visualization consider also having 2D visualizations or projections to help understand different perspectives of the data

For further reading I would highly recommend "Python Data Science Handbook" by Jake VanderPlas it goes deep into Matplotlib and it is a bible for any data science person Also if you are struggling with 3D objects and linear algebra I suggest looking into "Linear Algebra and Its Applications" by Gilbert Strang understanding transformations and coordinate systems is key to making good 3D plots

And that's pretty much it when it comes to seaborn 3D plots remember seaborn doesn't directly give 3D scatter plots so you have to use matplotlib but this is not such a bad thing because matplotlib gives you so much control over the final look of your plots And it should work as it works for me I once had to fight with my colleague because he though python was not good enough and after showing him this code he was convinced that python is amazing
