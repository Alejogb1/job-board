---
title: "How do the Cantor pairing function and Morton encoding compare?"
date: "2024-12-23"
id: "how-do-the-cantor-pairing-function-and-morton-encoding-compare"
---

Okay, let's delve into the fascinating intersection of the Cantor pairing function and Morton encoding, two techniques I've encountered quite a bit in my years working on spatial data structures and algorithms. These methods, while both serving the purpose of mapping multi-dimensional data into a single dimension, approach the problem from very different angles, each with its own trade-offs. My experiences have shown that understanding these nuances is crucial for choosing the best approach in various scenarios.

Let's start with the Cantor pairing function. If I recall correctly, my initial brush with this method was when I was working on a system for indexing user activity on a virtual map. We needed a way to uniquely identify a user's location at a given time, given its x and y coordinates on the map grid, as well as a timestamp. The Cantor pairing function offered an elegant, albeit somewhat less spatially-aware solution. The basic premise, as originally developed by Georg Cantor, is to map two natural numbers, `k1` and `k2`, to a unique natural number, `k`. It's defined as:

`pairing(k1, k2) = ((k1 + k2) * (k1 + k2 + 1) // 2) + k2`

where `//` represents integer division. This function is bijective, which means that every pair `(k1, k2)` maps to a unique `k`, and every `k` has a unique `(k1, k2)` pair. This is incredibly useful, as it allows you to, without loss of information, represent a 2-dimensional point in one dimension. The beauty of the Cantor pairing function is its simplicity, but that very simplicity often hides its limitations. For instance, the generated single-dimensional representation doesn't necessarily respect spatial proximity between points, and the single value grows rapidly as the input numbers increase. I remember one particular case where we needed to store coordinates for objects using 32-bit integers. With Cantor pairing, it quickly became apparent that we were going to overflow, and we eventually opted for a different approach for this scenario.

Here's a simple python implementation, demonstrating its use:

```python
def cantor_pairing(k1, k2):
  return ((k1 + k2) * (k1 + k2 + 1) // 2) + k2

def cantor_unpairing(k):
  w = int((((8 * k) + 1) ** 0.5 - 1) / 2)
  t = (w * w + w) // 2
  y = k - t
  x = w - y
  return x, y

#example
x, y = 5, 10
paired_value = cantor_pairing(x, y)
print(f"paired value for ({x},{y}): {paired_value}")

x_reconstructed, y_reconstructed = cantor_unpairing(paired_value)
print(f"unpaired value for {paired_value}: ({x_reconstructed}, {y_reconstructed})")
```

Now, let’s switch gears to Morton encoding (also known as Z-order curve). My first exposure to this approach was during my time working on a geospatial indexing system, specifically quadtrees. Morton encoding takes a very different approach. Instead of performing a mathematical calculation, it interleaves the bits of the input numbers. Imagine you have a coordinate (x, y), both representable as binary numbers. The Morton code is created by interleaving bits from x and y, starting with the least significant bit. This results in a single number where the bits alternate between the x and y components. The result is a single value that, unlike the cantor pairing function, tends to maintain spatial proximity, which I discovered during a frustrating debugging session where points plotted with this encoding tended to cluster visually, and it was really helpful. Points close together in the 2D space tend to be close in their morton encoding, that makes things like range queries very efficient.

The interleaving nature of Morton encoding also lends itself well to hierarchical data structures, such as quadtrees and octrees, which is a point i encountered a lot. The resulting ordering allows for efficient spatial indexing and searching because the single dimension representation preserves 2D locality to a fair degree. It’s worth noting that while better than Cantor pairing, the spatial proximity isn’t perfectly preserved, and there will always be edges in your index.

Here's a python implementation of morton encoding for 2D coordinates, demonstrating the bit manipulation:

```python
def morton_encode_2d(x, y):
  result = 0
  for i in range(32):  # assuming 32-bit integers
      mask = 1 << i
      if x & mask:
          result |= (1 << (2 * i))
      if y & mask:
          result |= (1 << (2 * i + 1))
  return result

def morton_decode_2d(morton_code):
    x = 0
    y = 0
    for i in range(32):
        mask = 1 << (2*i)
        if morton_code & mask:
            x |= (1 << i)
        mask = 1 << (2 * i + 1)
        if morton_code & mask:
            y |= (1 << i)
    return x, y

#example
x, y = 5, 10
morton_value = morton_encode_2d(x, y)
print(f"Morton encoded value for ({x},{y}): {morton_value}")

x_reconstructed, y_reconstructed = morton_decode_2d(morton_value)
print(f"Morton decoded value for {morton_value}: ({x_reconstructed}, {y_reconstructed})")
```

Finally, lets show an example in a slightly different way, representing the morton encoding with another common way, to make it clear there isn't only one algorithm implementation. Here we use bit manipulation in a slightly more compact manner

```python
def morton_encode_2d_compact(x, y):
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F
    x = (x | (x << 2)) & 0x3333333333333333
    x = (x | (x << 1)) & 0x5555555555555555

    y = (y | (y << 16)) & 0x0000FFFF0000FFFF
    y = (y | (y << 8)) & 0x00FF00FF00FF00FF
    y = (y | (y << 4)) & 0x0F0F0F0F0F0F0F0F
    y = (y | (y << 2)) & 0x3333333333333333
    y = (y | (y << 1)) & 0x5555555555555555

    return x | (y << 1)

def morton_decode_2d_compact(morton_code):
    x = morton_code & 0x5555555555555555;
    y = morton_code >> 1 & 0x5555555555555555;

    x = (x | (x >> 1)) & 0x3333333333333333;
    x = (x | (x >> 2)) & 0x0F0F0F0F0F0F0F0F;
    x = (x | (x >> 4)) & 0x00FF00FF00FF00FF;
    x = (x | (x >> 8)) & 0x0000FFFF0000FFFF;
    x = (x | (x >> 16)) & 0x00000000FFFFFFFF;


    y = (y | (y >> 1)) & 0x3333333333333333;
    y = (y | (y >> 2)) & 0x0F0F0F0F0F0F0F0F;
    y = (y | (y >> 4)) & 0x00FF00FF00FF00FF;
    y = (y | (y >> 8)) & 0x0000FFFF0000FFFF;
    y = (y | (y >> 16)) & 0x00000000FFFFFFFF;
    return x, y

#example
x, y = 5, 10
morton_value = morton_encode_2d_compact(x, y)
print(f"Compact Morton encoded value for ({x},{y}): {morton_value}")

x_reconstructed, y_reconstructed = morton_decode_2d_compact(morton_value)
print(f"Compact Morton decoded value for {morton_value}: ({x_reconstructed}, {y_reconstructed})")
```

In practice, my team and I have found that the choice between Cantor and Morton encoding really hinges on the specific application. If the primary concern is to simply map tuples of natural numbers to unique single values without spatial considerations, the Cantor pairing function might be sufficient and simpler to implement. However, if spatial locality is a crucial factor, such as when building spatial indexes, the Morton encoding approach, despite its increased implementation complexity, generally provides much better performance for most spatial search and indexing purposes. Also, I encountered cases where the performance difference was significant when trying to do range queries. Morton encoding tends to have fewer false positives, making these queries more efficient.

For those looking to dive deeper into this area, I'd recommend researching material focused on spatial databases and spatial indexing. Papers covering quadtrees and octrees are quite helpful in understanding the practical applications of morton encoding. The book "Spatial Database: With Application to GIS" by Philippe Rigaux, Michel Scholl, and Agnes Voisard provides a very good overview of spatial indexing and related concepts. Additionally, the foundational work by Gaston Gonnet and Ricardo Baeza-Yates in their book "Handbook of Algorithms and Data Structures" contains excellent treatment of encoding methods in general. I found these sources especially valuable during my learning process, and i hope that is useful for others too.
