---
title: "segmented least squares algorithm dynamic programming?"
date: "2024-12-13"
id: "segmented-least-squares-algorithm-dynamic-programming"
---

Okay so you wanna talk segmented least squares and dynamic programming I've been around this block more times than I care to remember man this is old news but a goodie I guess let's dive in

Right off the bat segmented least squares it's about finding the best way to break a bunch of points into segments where each segment is fitted by a line or some other curve you're not dealing with one massive least squares fit that would likely be a horrific mess but a series of smaller much more manageable fits Now the cool part is how you're going to find the *optimal* segmentation and that's where dynamic programming comes in real handy it's like a cheat code but a legit one you know

I've seen so many folks try to brute force this I even had a go myself back in my university days a student back then we just started working in C++ and i was trying to implement this segmented least square thing on some random sensor readings of a robot arm It was terrible we were trying every possible combination like some madmen it was awful. Needless to say we didn't even get close to a useful solution it was basically a crash course in why dynamic programming exists. The runtime was literally days. I remember the professor mentioning in his last class about dynamic programming, how it was like a very efficient approach to solving complex problems by breaking them into simpler overlapping subproblems. The guy was right you know, but i didn't have enough coffee for it that day and skipped that part. It was a painful realization that the brute force way is a dead end.

So what is this dynamic programming all about? Well its approach is about building solutions incrementally. It stores and reuses solutions to subproblems instead of recomputing them from scratch every single time. Think of it like a memo if you will if you already calculated something no point doing it again. It’s really like caching in a way. That's what makes it so efficient it avoids all that redundant computation stuff we foolishly tried to do with brute force.

In the case of segmented least squares the subproblem is basically finding the best segmentation for the first `i` points given a certain number of segments. Then the key insight is that the solution for `i+1` points can be derived from the optimal solution for `i` points along with the cost of adding a new segment so to say.

Let's put some code to it shall we? I’ll give you a Python example it's easier for quick prototyping. Remember i have done this in C++ many years ago.

```python
import numpy as np

def cost(points):
    # Here you calculate the cost of fitting a line to the points
    # This is a simplified example you'd probably use a proper
    # least squares method
    if len(points) == 0:
        return 0
    x_values = np.array([p[0] for p in points])
    y_values = np.array([p[1] for p in points])
    
    if len(x_values) <= 1:
        return 0 #No cost for single or zero points
    
    
    #Simplified linear fit just for demonstration
    
    
    n = len(x_values)
    
    x_bar = np.mean(x_values)
    y_bar = np.mean(y_values)

    
    
    numerator = sum((x - x_bar) * (y - y_bar) for x,y in zip(x_values, y_values) )
    denominator = sum((x - x_bar)**2 for x in x_values)
    
    if denominator == 0:
       return float('inf')
    
    b = numerator / denominator

    a = y_bar - b * x_bar
    
    
    total_error = sum((y - (a + b*x) )**2 for x,y in zip(x_values,y_values))
    return total_error


def segmented_least_squares(points, max_segments, penalty):
    n = len(points)
    dp = np.full((n + 1, max_segments + 1), float('inf'))
    dp[0][0] = 0
    
    
    break_points = np.empty((n+1,max_segments+1), dtype=object)
    break_points[:] = None

    for i in range(1, n + 1):
      for j in range(max_segments + 1):
        
        for k in range(i):
         current_cost = dp[k][j-1] if j>0 else float('inf')
         
         
         segment_cost = cost(points[k:i])
         total_cost = current_cost + segment_cost + penalty
         
         if total_cost < dp[i][j]:
            dp[i][j] = total_cost
            if j > 0:
                break_points[i][j] = k
            else:
                break_points[i][j] = 0
            
            

    # Backtrack to find segmentation
    segmentation = []
    if max_segments > 0:
        
        current_index = n
        current_segments = max_segments
        
        while current_index > 0 and current_segments >0:
            
            segment_start = break_points[current_index][current_segments]
           
            segmentation.insert(0,(segment_start,current_index))
            current_index = segment_start
            current_segments -= 1

    return dp[n][max_segments], segmentation
```

That is basically it, this code will calculate the segmentation with the minimum cost. The `cost` function here is just a dummy, you will need your own more precise least squares fit and not that simple linear one. The `segmented_least_squares` implements the actual dp and the `break_points` matrix records the best breakpoints to allow the recovering of the segmentation

You might be wondering about the `penalty` parameter thats for controlling the complexity of the segmentation. If we go without it then we will get lots of segments. We can add a constant penalty per new segment that prevents creating an arbitrary number of segments which would result in overfitting, you know. You can try different penalty values to see the effects.

Let's say you want to visualize the segmented result then i'll also show you how to do that. Just some plotting. I really like plotting data it has something soothing in it. I have spent more time plotting data than i care to admit. I was trying to show off a segmented least squares fitting in a presentation and i just got stuck in plotting the data for 3 days. You don't want to make the same mistake trust me.

```python
import matplotlib.pyplot as plt

def plot_segments(points, segmentation):
    x_values = [p[0] for p in points]
    y_values = [p[1] for p in points]
    
    
    plt.scatter(x_values,y_values, label="data points", color='blue')
    
    
    for segment_start, segment_end in segmentation:
        segment = points[segment_start:segment_end]
        if len(segment) > 1 : # Only plot if there are two or more points
            x_seg = [p[0] for p in segment]
            y_seg = [p[1] for p in segment]
            

            
            
            x_bar = np.mean(x_seg)
            y_bar = np.mean(y_seg)
            
            
            
            numerator = sum((x - x_bar) * (y - y_bar) for x,y in zip(x_seg, y_seg) )
            denominator = sum((x - x_bar)**2 for x in x_seg)
            
            if denominator !=0:

                b = numerator / denominator
            
            
            
                a = y_bar - b * x_bar
                
                x_fit = np.linspace(min(x_seg), max(x_seg), 100)
                y_fit = a + b * x_fit
                plt.plot(x_fit, y_fit, color='red')
    plt.legend()
    plt.show()
```

This takes a segmentation from our function and plots it all nice using matplotlib library, you can make it as nice as you want of course. If you are to use this in a production enviroument consider using a better plotting library like plotly it gives a great level of interactivity.

Now if you want a more robust solution you'll probably want to use a proper least squares fit instead of this simple linear approximation. You can find good documentation on how to do this in "Numerical Recipes" by Press et al. it's a bit of a classic you will definitely stumble on it if you are to work in numerical algorithms. Another good one that i would recommend is the “Introduction to Algorithms” by Cormen et al. it covers the dynamic programming theory with great detail if you want to grasp the concepts.

So there you have it the segmented least squares with dynamic programming. It ain't rocket science but it's pretty powerful and definitely faster than whatever brute force approach you will find. I hope this helps with your segmented fitting adventures. Happy coding. Oh and why don't scientists trust atoms they make up everything hehe.
