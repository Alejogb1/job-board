---
title: "Need a Weightless Object in Drake? Here's How!"
date: '2024-11-08'
id: 'need-a-weightless-object-in-drake-here-s-how'
---

```python
from pydrake.multibody.tree import SpatialInertia
from pydrake.multibody.tree import RigidBody

# Create a massless spatial inertia
massless_inertia = SpatialInertia(0.0, [0.0, 0.0, 0.0], [[0.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0]])

# Create a dummy body with the massless inertia
dummy_body = RigidBody(massless_inertia)
```
