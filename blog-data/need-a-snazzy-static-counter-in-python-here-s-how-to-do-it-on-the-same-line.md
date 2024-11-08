---
title: "Need a snazzy static counter in Python? Here's how to do it on the same line!"
date: '2024-11-08'
id: 'need-a-snazzy-static-counter-in-python-here-s-how-to-do-it-on-the-same-line'
---

```python
import time

for i in range(100):
    time.sleep(1)
    print(f"line >>> {i:03d}", end="\r")
```
