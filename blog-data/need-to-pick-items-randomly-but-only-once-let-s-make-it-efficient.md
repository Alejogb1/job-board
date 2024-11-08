---
title: "Need to pick items randomly, but only once? Let's make it efficient!"
date: '2024-11-08'
id: 'need-to-pick-items-randomly-but-only-once-let-s-make-it-efficient'
---

```python
import random

lst = [i for i in range(50800)]
random.shuffle(lst)

def random_int(lst, size):
    drawn, lst = lst[0:size],lst[size:]
    return drawn, lst

for i in range(50800//400):
    drawn, lst = random_int(lst, 400)
    # do something with drawn
```
