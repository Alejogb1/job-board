---
title: "Building the Biggest Rectangle: How to Use Line Segments to Maximize Area"
date: '2024-11-08'
id: 'building-the-biggest-rectangle-how-to-use-line-segments-to-maximize-area'
---

```python
def all_but(lst, pos):
    return lst[0:pos]+lst[pos+1:]

def find_sets_with_len(segs, l):
    for i in range(0, len(segs)):
        val = segs[i]
        if (val == l):
            yield [val], all_but(segs, i)
        if (val < l):
            for soln, rest in find_sets_with_len(all_but(segs, i), l - val):
                yield [val]+soln, rest

def find_rect(segs, l1, l2):
    for side1, rest1 in find_sets_with_len(segs, l1):
        for side2, rest2 in find_sets_with_len(rest1, l1):
            for side3, rest3 in find_sets_with_len(rest2, l2):
                return [side1, side2, side3, rest3]

def make_rect(segs):
    tot_len = sum(segs)
    if (tot_len %2) == 0:
        opt_len=tot_len/4
        for l in range(opt_len, 0, -1):
            sides = find_rect(segs, l, tot_len/2-l)
            if sides is not None:
                print(sides)
                return sides
    print("Can't find any solution")

make_rect([4,2,4,4,6,8])
```
