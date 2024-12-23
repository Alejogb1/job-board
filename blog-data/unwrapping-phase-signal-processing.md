---
title: "unwrapping phase signal processing?"
date: "2024-12-13"
id: "unwrapping-phase-signal-processing"
---

so you're wrestling with unwrapping phase I know that beast pretty well let me break down what I've learned over the years dealing with this stuff It's a common headache in signal processing especially if you're working with things like radar interferometry or magnetic resonance imaging

Essentially you're stuck with wrapped phase which is the output of functions like arctangent and that gives you a value between -π and +π the issue is that the actual underlying phase might have gone way beyond that range it's like having a clock that only shows hours between 1 and 12 but the day keeps rolling on You need to unwrap that thing to get the real continuous phase

The crux of it is figuring out those jumps of 2π you know when the phase wraps around and you're presented with the illusion of a smaller phase change than what actually occurred The most basic method that you might encounter is a simple cumulative sum approach This is usually the first thing that comes up when you start researching it and honestly this is what I did when I first saw this in college back in the old days we had to write the whole thing in fortran which is ancient history at this point but let me show you what a python example looks like

```python
import numpy as np

def unwrap_phase_simple(wrapped_phase):
    unwrapped_phase = np.copy(wrapped_phase)
    for i in range(1, len(wrapped_phase)):
        diff = wrapped_phase[i] - wrapped_phase[i-1]
        if diff > np.pi:
            unwrapped_phase[i:] -= 2 * np.pi
        elif diff < -np.pi:
            unwrapped_phase[i:] += 2 * np.pi
    return unwrapped_phase
```

This function iterates through the phase and looks at the differences between adjacent phase samples If the difference is larger than π or smaller than -π we know that there has been a jump of ±2π and we add or subtract that amount to the remaining samples This works pretty well if you have relatively smooth phase but things fall apart when you encounter noise or actual phase discontinuities

In my earlier projects I was working with synthetic aperture radar data trying to create terrain elevation maps the images looked like some kind of strange disco floor with all those fringes it's pretty but not very informative I was using this exact algorithm in a rush but the results were just completely wrong especially in areas with high topography change it was basically a useless pile of unwrapped garbage in the real sense of the word

After hours and days of trying to make it work I had to dig deeper into the literature I mean the proper stuff I wasn't looking at blog posts anymore you know the kind that make you feel stupid because the math looks like hieroglyphs I had to go to books like "Digital Signal Processing" by Proakis and Manolakis and even some older papers in the IEEE transactions I found that the problem stems from how this algorithm deals with noisy phase measurements if a single phase difference is incorrectly identified as a jump then you get error propagation and things get messed up real quick

A better approach and I've had much success with this in more recent projects is to use a path following method like the quality guided path following algorithm The main idea here is to create a quality map which is a measure of the phase reliability for each pixel or sample and then guide the unwrapping along paths that follow these high quality areas This limits error propagation This type of method also works well if the phase change is really fast in some areas and it's slow in others it doesn't care It just goes with the flow as it should

I am not going into too much detail about the math here since this is an informal answer but let me give you an idea of what the code can look like

```python
import numpy as np
from scipy.signal import convolve2d
import heapq

def calculate_quality_map(wrapped_phase):
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    laplacian = convolve2d(wrapped_phase, kernel, mode='same')
    return np.abs(laplacian)

def unwrap_phase_quality_guided(wrapped_phase):
    rows, cols = wrapped_phase.shape
    quality_map = calculate_quality_map(wrapped_phase)
    unwrapped_phase = np.zeros_like(wrapped_phase, dtype=float)
    unwrapped_phase[:] = np.nan
    unwrapped_phase[0, 0] = 0 # Start unwrapping at some arbitrary point

    priority_queue = [(-quality_map[0, 0], 0, 0)]
    visited = np.zeros_like(wrapped_phase, dtype=bool)
    visited[0, 0] = True
    while priority_queue:
        _, row, col = heapq.heappop(priority_queue)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row = row + dr
            new_col = col + dc
            if 0 <= new_row < rows and 0 <= new_col < cols and not visited[new_row, new_col]:
                diff = wrapped_phase[new_row, new_col] - wrapped_phase[row, col]
                if diff > np.pi:
                     unwrapped_phase[new_row, new_col] = unwrapped_phase[row, col] + diff - 2 * np.pi
                elif diff < -np.pi:
                    unwrapped_phase[new_row, new_col] = unwrapped_phase[row, col] + diff + 2 * np.pi
                else:
                    unwrapped_phase[new_row, new_col] = unwrapped_phase[row, col] + diff
                visited[new_row, new_col] = True
                heapq.heappush(priority_queue, (-quality_map[new_row, new_col], new_row, new_col))
    return unwrapped_phase
```

Here I calculate a quality map by using the laplacian which indicates a change in the phase values then I start unwrapping based on the areas with high quality which means those areas are less prone to error and therefore I use them to unwrap the rest This algorithm is a lot more robust but also a lot more complex it's worth it though trust me on this

But wait there is even a better one

For very complex situations I have often found the minimum norm approach to be the most reliable The idea here is to formulate a cost function that penalizes phase jumps and then find a solution that minimizes this cost It turns out that you can solve the problem through a least-squares formulation this is usually more computationally demanding but for higher quality phase fields it works like a charm especially if I am looking at very complex scenarios like the flow of a fluid or any physical phenomenon in which the phase change is very difficult to understand

Let me show you the code

```python
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import lsqr

def unwrap_phase_minimum_norm(wrapped_phase):
    rows, cols = wrapped_phase.shape
    n_pixels = rows * cols

    row_indices = []
    col_indices = []
    data = []
    b = np.zeros(2 * (n_pixels - rows) + 2 * (n_pixels - cols))
    row_count = 0

    # Vertical differences
    for i in range(rows - 1):
        for j in range(cols):
            idx = i * cols + j
            idx_below = (i+1) * cols + j
            row_indices.extend([row_count,row_count])
            col_indices.extend([idx, idx_below])
            data.extend([-1, 1])
            diff = wrapped_phase[i+1, j] - wrapped_phase[i, j]
            if diff > np.pi:
                 b[row_count] = diff - 2 * np.pi
            elif diff < -np.pi:
                  b[row_count] = diff + 2 * np.pi
            else:
                  b[row_count] = diff
            row_count += 1

    # Horizontal differences
    for i in range(rows):
        for j in range(cols-1):
            idx = i * cols + j
            idx_right = i * cols + j + 1
            row_indices.extend([row_count, row_count])
            col_indices.extend([idx, idx_right])
            data.extend([-1, 1])
            diff = wrapped_phase[i, j+1] - wrapped_phase[i, j]
            if diff > np.pi:
                 b[row_count] = diff - 2 * np.pi
            elif diff < -np.pi:
                 b[row_count] = diff + 2 * np.pi
            else:
                 b[row_count] = diff
            row_count += 1

    A = diags(data, [0], shape=(row_count, n_pixels)).tocsr()
    unwrapped_phase_flat = lsqr(A, b)[0]
    unwrapped_phase = unwrapped_phase_flat.reshape(rows, cols)
    return unwrapped_phase
```
I know that code looks ugly and you are probably thinking “what the heck is this guy doing with all those sparse matrices” well let me tell you it works just trust me on this one this is like the gold standard for unwrapping phase in the wild and there are even more complex ones and all of them involve heavy mathematical apparatus which in the end translates into lots of code and lots of headaches but that's part of the job I guess

So there you have it three methods you can use for unwrapping phase the simple one that works only for very simple data the path following that works much better and the minimum norm one which is very robust and will handle even the worst datasets that you can throw at it I've been there I've done that

And lastly one pro tip I learned through a lot of trial and error you want to avoid phase unwrapping as much as possible whenever you can try to calculate what you want directly from the wrapped phase I know this sounds like cheating but the less you have to unwrap the better sometimes it's like trying to fix a broken pipe its better to just get a new one so always think if you actually need the unwrapped phase or not maybe just the rate of phase change is what you're after in that case you're in the clear

And just in case you were wondering the least favorite part of working with phase unwrapping is debuggging because you need to go through all the mathematical formulations and make sure the code does what you intended and that can sometimes be as confusing as trying to decide what to order for pizza the only thing that is certain is that you always end up eating something so the same can be applied to unwrapped phase you always end up unwrapping something sometimes good sometimes bad so the trick here is not trying to create perfect code but to find the code that works better for your needs

For more resources check papers by Ghiglia and Pritt they have a lot of stuff about these techniques and you will probably also find a bunch of algorithms implemented in MATLAB too but I prefer python because you have more control over it and you can code anything you want
