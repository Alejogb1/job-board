---
title: "How to Quickly Find the Highest Point in a Histogram"
date: '2024-11-08'
id: 'how-to-quickly-find-the-highest-point-in-a-histogram'
---

```python
import matplotlib.pyplot as plt
from numpy.random import randn

hdata = randn(500)
y, x, _ = plt.hist(hdata)

print(y.max())
```
