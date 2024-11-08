---
title: "Need Help Maximizing the Sum of Sub-Matrices in Python?"
date: '2024-11-08'
id: 'need-help-maximizing-the-sum-of-sub-matrices-in-python'
---

```python
def seanMatrix(matrix):
    n = len(matrix) // 2
    return sum(max(matrix[i][j], matrix[i][~j], matrix[~i][j], matrix[~i][~j])
               for i in range(n)
               for j in range(n))
```
