---
title: "How can I implement 2D interval scheduling in Python?"
date: "2025-01-30"
id: "how-can-i-implement-2d-interval-scheduling-in"
---
2D interval scheduling, often encountered in resource allocation scenarios involving time and location or multiple resource types, presents a considerable expansion over its 1D counterpart. I’ve implemented solutions for this in various applications, ranging from optimizing delivery truck routes to scheduling manufacturing processes, and found that a clear understanding of the underlying data structures and algorithms is critical for an efficient and scalable solution. The key challenge stems from the need to consider overlapping intervals not just along a single dimension but across two, making direct application of simple greedy approaches typically used in 1D scheduling inadequate.

The core problem with 2D interval scheduling lies in determining which set of non-overlapping rectangles (represented by start and end points in both x and y dimensions) can be selected to maximize some objective function, frequently the number of selected intervals. This differs from the standard 1D scheduling where intervals overlap along a single axis. Brute-force solutions, testing all possible combinations, become quickly intractable even for a moderate number of intervals. Therefore, one needs to employ algorithmic strategies that systematically consider overlaps in both dimensions.

One common approach involves transforming the 2D problem into a series of 1D scheduling problems. Here's how it generally works: First, sort the rectangles based on their start time in one dimension (e.g., the x-dimension). Next, iterate through these rectangles, using each start x-value as a sweep line. For each sweep line position, create a set of “active” rectangles, meaning the rectangles that span this particular x value. On this “active” set, perform a greedy interval scheduling algorithm based on the other dimension (e.g., the y-dimension). This greedy step usually involves sorting the active rectangles based on their ending y-value and then selecting non-overlapping intervals greedily. The key advantage here is that by sweeping across one dimension, you effectively simplify the overlap checking to a standard 1D interval scheduling within a specific x coordinate, repeating the procedure along each x position. This method is not guaranteed to yield the *absolute* maximum of intervals selected globally but often provides a near-optimal, efficient approximation.

Let's demonstrate this concept with code examples. These assume that the intervals are represented as tuples `(x_start, x_end, y_start, y_end)`:

**Example 1: Basic Sweep Line Approach**

```python
def schedule_2d_intervals_sweep(intervals):
    """
    Schedules 2D intervals using a sweep line approach.

    Args:
        intervals: A list of tuples representing intervals (x_start, x_end, y_start, y_end).

    Returns:
        A list of selected intervals.
    """

    intervals.sort(key=lambda x: x[0]) # Sort by x-start

    selected_intervals = []
    x_positions = sorted(list(set([x[0] for x in intervals] + [x[1] for x in intervals]))) #All x positions

    for x in x_positions:
       active_intervals = [interval for interval in intervals if interval[0] <= x < interval[1]]

       if not active_intervals:
          continue
       active_intervals.sort(key=lambda x: x[3]) # Sort active set by y-end

       current_end_y = -float('inf')
       for interval in active_intervals:
          if interval[2] >= current_end_y:
             selected_intervals.append(interval)
             current_end_y = interval[3]
    
    
    return list(set(selected_intervals)) #Remove Duplicates due to sweep at multiple points

```

This function first sorts the intervals based on their x-start values and then iterates through all unique x-start and x-end values as sweep positions. It creates an "active" list of intervals overlapping that current sweep position. Within this list, it performs a standard 1D interval scheduling by sorting intervals based on their y-end and selecting non-overlapping ones. This function is relatively straightforward for implementation but might include some redundant intervals as we iterate across multiple x positions, hence the use of `set()` to remove duplicates. It provides an approximate solution, not necessarily the global maximum, but it’s efficient in practice.

**Example 2: Handling Overlaps with a Nested Loop**

This example attempts to further reduce redundancy and provide a slightly better result, although the worst-case time complexity is increased. Here, nested loops are utilized to select intervals which do not conflict with existing selections:

```python
def schedule_2d_intervals_nested(intervals):
    """
    Schedules 2D intervals with a nested loop.

    Args:
        intervals: A list of tuples representing intervals (x_start, x_end, y_start, y_end).

    Returns:
        A list of selected intervals.
    """
    intervals.sort(key=lambda x: x[0]) # Sort by x-start
    selected_intervals = []
    
    for interval in intervals:
      valid = True
      for selected in selected_intervals:
        if (not (interval[1] <= selected[0] or interval[0] >= selected[1] or interval[3] <= selected[2] or interval[2] >= selected[3])): #Check all overlaps
           valid = False
           break
      if valid:
        selected_intervals.append(interval)
    
    return selected_intervals
```

Here, intervals are first sorted according to their x-start value. Then, we iterate through each interval and compare to all existing selections. If a selected interval does not overlap with any existing selections it is added to the result list. This process reduces the redundancy of Example 1, but at the cost of increased time complexity because of the nested loop. This is not an optimal solution but a demonstration of a different perspective towards handling overlaps.

**Example 3: Using a custom class**

The below code demonstrates a simple class that can be created to keep track of interval data and be expanded further in terms of functionality, and also provides a similar nested loop solution.

```python
class Interval:
  def __init__(self, x_start, x_end, y_start, y_end):
    self.x_start = x_start
    self.x_end = x_end
    self.y_start = y_start
    self.y_end = y_end

  def overlaps(self, other):
    return not (self.x_end <= other.x_start or self.x_start >= other.x_end or self.y_end <= other.y_start or self.y_start >= other.y_end)


def schedule_2d_intervals_custom(interval_data):
    intervals = [Interval(*data) for data in interval_data]
    intervals.sort(key=lambda i: i.x_start)

    selected_intervals = []
    for interval in intervals:
      valid = True
      for selected in selected_intervals:
        if interval.overlaps(selected):
          valid = False
          break
      if valid:
        selected_intervals.append(interval)

    return [(i.x_start, i.x_end, i.y_start, i.y_end) for i in selected_intervals]
```

This example creates an interval class which contains the interval coordinates and an overlaps function. This custom class allows for more complex manipulation of the interval data. The rest of the function provides similar functionality to that of Example 2, but using the custom class. This approach can be more useful when one wants to associate different types of properties to each interval object.

For further learning, I recommend studying the concept of sweep-line algorithms more deeply. In a broader sense, computational geometry texts offer extensive coverage of techniques for dealing with overlapping geometric shapes. Books specifically on algorithms and data structures often have chapters or sections dedicated to scheduling problems which extend past the 1D case. Additionally, consider researching graph theory since interval scheduling can be represented as a graph problem where each interval is a vertex, and an edge connects two vertices if their intervals overlap. Exploring dynamic programming may also reveal optimization strategies applicable to 2D problems with more complex cost functions, despite its increased computational overhead. While I prefer to develop these algorithms from first principles, reviewing established libraries can provide valuable insights as well.

These three examples showcase slightly different approaches to the same problem. The sweep line approach (Example 1) offers a relatively simple and efficient implementation that may give non-optimal solutions. The nested loop approach (Example 2 and 3) offers a more deliberate selection method by checking against all existing selections, leading to less redundancy but at a higher computational cost.
