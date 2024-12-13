---
title: "2d alpha shape concave hull problem in python?"
date: "2024-12-13"
id: "2d-alpha-shape-concave-hull-problem-in-python"
---

Alright so you're wrestling with 2D alpha shapes trying to get a concave hull in Python I've been there man trust me it's not a walk in the park been doing this for like a decade now and geometric stuff can be a real pain in the neck especially when you start playing with parameters like alpha

So the alpha shape thing is essentially trying to generalize convex hulls but with a parameter right The alpha value controls how tightly the shape hugs the points A big alpha gives you something closer to the convex hull while a small alpha lets you generate more concave shapes that really follow the point distribution This is useful for a ton of stuff like outlining clusters finding boundaries of irregular shaped data or even for some basic image processing tasks

Now in my own experience I first bumped into this back when I was working on a project visualizing geo-spatial data remember we needed to represent the boundary of a set of reported sighting locations for some unusual bird migrations and a convex hull was just not going to cut it It made the regions look all clunky and just not accurate So that's when I fell down the rabbit hole of alpha shapes It took me a good week or so to really wrap my head around it and even then there were still some corner cases that kept popping up

Let's talk code for a second I'm assuming you're somewhat familiar with Python and some common scientific libraries like NumPy and SciPy If not I'd highly recommend checking out the official SciPy documentation and Numerical Recipes 3rd Edition really good stuff there

Okay so the core idea is this triangulation is the key player We use Delaunay triangulation to connect all the points and then we selectively remove edges based on the alpha value Any edge that's longer than the alpha value we remove that's roughly how it works And its why you need some good libraries to help you out

Here's a basic example using SciPy's Delaunay which honestly does a lot of the heavy lifting

```python
import numpy as np
from scipy.spatial import Delaunay

def alpha_shape(points, alpha):
    """
    Compute the alpha shape of a set of 2D points.

    Args:
        points: A numpy array of shape (n, 2) representing 2D points.
        alpha: The alpha value.

    Returns:
        A list of edges that form the alpha shape.
    """

    tri = Delaunay(points)
    edges = set()

    for simplex in tri.simplices:
        for i in range(3):
            p1 = points[simplex[i]]
            p2 = points[simplex[(i + 1) % 3]]

            distance = np.linalg.norm(p1 - p2)

            if distance <= alpha:
                edges.add(frozenset((tuple(p1), tuple(p2))))
    
    return list(edges)

if __name__ == '__main__':
    points = np.array([
        [0, 0], [0, 1], [1, 0], [1, 1],
        [0.5, 0.5], [0.2, 0.2], [0.8, 0.8]
        ])

    alpha = 0.6
    result_edges = alpha_shape(points, alpha)
    print("Alpha shape edges:")
    for edge in result_edges:
        print(edge)
```

Now that code snippet gets you the edges but not directly the boundary points That’s because alpha shapes are actually often made up of a lot of non-connected edges This is a super simple way to get the basic edges though

A good next step would be to visualize this right A picture is worth a thousand words and in this case it’s also worth a lot of debugging time using matplotlib

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay

def alpha_shape(points, alpha):
    tri = Delaunay(points)
    edges = set()
    for simplex in tri.simplices:
      for i in range(3):
          p1 = points[simplex[i]]
          p2 = points[simplex[(i + 1) % 3]]

          distance = np.linalg.norm(p1 - p2)
          if distance <= alpha:
              edges.add(frozenset((tuple(p1), tuple(p2))))
    
    return list(edges)


if __name__ == '__main__':
  points = np.array([
        [0, 0], [0, 1], [1, 0], [1, 1],
        [0.5, 0.5], [0.2, 0.2], [0.8, 0.8],
        [0.1, 0.9],[0.9, 0.1],
    ])

  alpha = 0.6
  edges = alpha_shape(points, alpha)
  plt.figure()
  plt.scatter(points[:, 0], points[:, 1], label='Points')
  for edge in edges:
    p1, p2 = list(edge)
    x1, y1 = p1
    x2, y2 = p2
    plt.plot([x1, x2], [y1, y2], color='red')
  
  plt.legend()
  plt.xlabel('X-axis')
  plt.ylabel('Y-axis')
  plt.title('2D Alpha Shape Visualization')
  plt.grid(True)
  plt.show()
```

This example shows you how to plot the points and the resulting edges of the alpha shape it's basic but it works Remember that playing around with the alpha value is key to finding the perfect fit for your data There’s a lot of art in that parameter selection by trial and error I've got a few scars from that I can tell you

Now that we have the edges we're halfway to forming the polygon That is not a polygon but a collection of edges To make a proper polygon you’ll often have to collect all edges and put them into an ordered sequence this means having to identify edge connections which is not easy

And remember the concave hull will probably have more than one polygon so sometimes you need a strategy to identify these separate components which usually can be done by connecting edges that share the same nodes

A more complex problem and something I’ve faced is when your data is super noisy and you've got some outliers The alpha shape can get really sensitive to these outliers so you often need some kind of pre-processing like noise reduction or outlier removal before applying it Or even you might want to run this alpha shape over different subsets of the data

Sometimes using just one alpha value will produce multiple polygons which is a pain too and is not ideal for some applications There are a bunch of different strategies you can use to solve this but all depend on the data you are using

Here's another example with a bit more going on it combines using matplotlib patches which can help create the polygon by combining edges

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import Delaunay

def alpha_shape_polygon(points, alpha):
    tri = Delaunay(points)
    edges = set()
    for simplex in tri.simplices:
      for i in range(3):
          p1 = points[simplex[i]]
          p2 = points[simplex[(i + 1) % 3]]
          distance = np.linalg.norm(p1 - p2)
          if distance <= alpha:
              edges.add(frozenset((tuple(p1), tuple(p2))))
    
    return edges

def find_boundary_points(edges):
  graph = {}
  for edge in edges:
    p1, p2 = list(edge)
    graph.setdefault(p1, []).append(p2)
    graph.setdefault(p2, []).append(p1)

  boundary_points = []
  visited = set()
  for point in graph:
        if point in visited:
            continue

        path = []
        current_node = point
        while current_node not in visited:
            visited.add(current_node)
            path.append(current_node)
            neighbors = graph[current_node]
            if len(neighbors) == 1:
                current_node = neighbors[0]
            else:
                for neighbor in neighbors:
                  if neighbor not in visited:
                    current_node = neighbor
                    break
                else:
                    break
        if len(path) > 2:
            boundary_points.append(path)
  return boundary_points

if __name__ == '__main__':
  points = np.array([
      [0, 0], [0, 1], [1, 0], [1, 1],
      [0.5, 0.5], [0.2, 0.2], [0.8, 0.8],
      [0.1, 0.9],[0.9, 0.1],
      [3, 3],[3, 4],[4, 3],[4,4], [3.5,3.5]
    ])
  alpha = 0.6

  edges = alpha_shape_polygon(points, alpha)
  boundary_points = find_boundary_points(edges)
  
  plt.figure()
  plt.scatter(points[:, 0], points[:, 1], label='Points')

  for poly_points in boundary_points:
    polygon = Polygon(poly_points, closed = True, facecolor = 'none', edgecolor='blue')
    plt.gca().add_patch(polygon)


  plt.legend()
  plt.xlabel('X-axis')
  plt.ylabel('Y-axis')
  plt.title('2D Alpha Shape Polygon Visualization')
  plt.grid(True)
  plt.show()
```

This shows a more complete example of how to get the polygon and not just the edges note however that there could be still problems with how the points are connected this is because in the real world that you might get edges that are overlapping or you might get edges that are disconnected that is a big issue and this code does not solve those issues but should give you a general start.

One really big issue with alpha shapes is the parameter selection of alpha the value will massively impact the results I've once spent like 3 days just trying to figure out the right alpha parameter so the result would not be wrong

It's a good exercise to write code that can estimate what's the best alpha value for a dataset for example you could write some function that calculates the edge lengths of the delaunay and gets some percentage threshold from that to estimate the optimal alpha value for some basic cases that's usually what I try to do before just trying alpha values randomly

So there you have it my slightly grumpy but experienced take on 2D alpha shapes in Python Been there done that and have the t-shirt The important thing is to keep experimenting keep debugging and don't be afraid to go deep into those libraries and really really important to understand the math behind it you will need to I assure you.

Now go forth and conquer those concave hulls and remember the alpha is a fickle beast.
