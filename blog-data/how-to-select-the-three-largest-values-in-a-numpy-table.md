---
title: "How to select the three largest values in a numpy table?"
date: "2024-12-14"
id: "how-to-select-the-three-largest-values-in-a-numpy-table"
---

alright, let's talk about grabbing those top three values from a numpy array. it's a common task, and thankfully, numpy makes it pretty straightforward. i've bumped into this countless times, especially when i was working on some image processing pipelines back in the day where i had to quickly identify the most prominent pixel intensities in some badly exposed jpegs, and some computer vision projects that involved finding the strongest feature detections. i'll break down a few methods, starting from the simplest and moving toward ones that offer a little more flexibility.

first, the most basic approach involves using `np.sort` and then slicing. this is usually my go to if i don't need any fancy stuff. `np.sort` sorts the entire array, and then it's just a matter of picking off the last three elements. here’s how it generally goes:

```python
import numpy as np

def get_top_three_sort(array):
    sorted_array = np.sort(array)
    return sorted_array[-3:]

# example usage
data = np.array([5, 2, 9, 1, 7, 6, 3, 8, 4])
top_three = get_top_three_sort(data)
print(top_three) # prints: [7 8 9]
```

this is efficient enough for smaller arrays, and it's super easy to read. the downside is that `np.sort` sorts the *entire* array, which can be wasteful if you’re dealing with large datasets and only care about the top few values. that extra sorting work might not matter for small arrays but can become a noticeable performance hit as the array size goes up. i vividly remember this when i was trying to analyze some sensor data from a robotics project; the raw sensor feeds where huge matrices and using sort on the whole dataset for just the top few values was unnecessary and computationally expensive

now, if you're looking for better performance with larger arrays, `np.partition` is your friend. this method is faster because it doesn't fully sort the array. instead, it arranges the array such that the k-th largest elements are in their correct positions. this is usually my pick if i am dealing with large datasets and i don't want the overhead of a full sort.

```python
import numpy as np

def get_top_three_partition(array):
    partitioned_array = np.partition(array, -3)
    return partitioned_array[-3:]


# example usage
data = np.array([5, 2, 9, 1, 7, 6, 3, 8, 4])
top_three = get_top_three_partition(data)
print(top_three) # prints: [8 7 9]  the order may vary
```

notice that the output is slightly different, the top values are there, but not necessarily in strictly sorted order. that's because `partition` just guarantees the k-th largest elements are in the correct place, not necessarily sorted among themselves. if you *need* the output to be sorted, you can sort just the last three values after partitioning, which is still quicker than sorting the entire array.

and now one more, sometimes you want to know the indices of the top values too. for this, we can combine `np.argpartition` with some indexing. `np.argpartition` gives you the *indices* that would partition the array. we grab the top three indices, then use these to get the corresponding values. i used this technique heavily during some of my previous machine learning projects, specifically to track where the models were looking at when doing some sort of heatmap visualization. it helped a lot with debugging the model's behaviour:

```python
import numpy as np

def get_top_three_indices_and_values(array):
    indices = np.argpartition(array, -3)[-3:]
    values = array[indices]
    return indices, values


# example usage
data = np.array([5, 2, 9, 1, 7, 6, 3, 8, 4])
indices, values = get_top_three_indices_and_values(data)
print("indices:", indices)  # prints: indices: [7 2 6] or similar, depending on numpy versions.
print("values:", values) # prints: values: [8 9 7]
```
again, the order of the indices might not be in descending order of the value. if you need that, you can sort these indices based on their corresponding values.
there's another approach you might see, it's using `np.argsort` which is the same idea as the `np.sort` but with indices it has the same weakness as `np.sort` that it does not scale well with bigger datasets. just like `np.sort` it sorts the whole data. i tend to skip using `argsort` when i'm not doing any kind of complex indexing of the original data.

now, a quick note about performance. if you're working with really large datasets, consider using libraries like numba or cython to further optimize these operations. numpy is already pretty fast, but you can get even better performance if you're willing to dive into those more advanced tools.

when it comes to resources, rather than just linking out to random blog posts, i'd recommend starting with the core numpy documentation. it's well-written and a great place to get the details on functions like `sort`, `partition`, `argpartition`, and `argsort`. specifically, make sure to check out the “sorting, searching, and counting” section. if you really want to learn how these things work under the hood, “numerical recipes: the art of scientific computing” by william h. press et al. is an absolute classic. it goes into the algorithms used in these sorts of routines, which is invaluable information for a tech person. it might be a bit of a heavy read, but it's worth it for understanding. i know it might sound like some ancient thing, but i swear it's a good book even today. you can also check out “python for data analysis” by wes mckinney for a more practical application of numpy within the data analysis context. it also serves as a great resource for using pandas, and they often go hand in hand.

oh, one more thing, i was once trying to debug some seriously slow code, and after some head-scratching, i realized i was calling `np.sort` inside a loop on every iteration. i felt incredibly dumb, but, hey, that's the life of a programmer, isn't it? always those silly things. i had a very long day, the code was slow and i was getting a headache, so i went home and had a long nap. i woke up and felt that the issue was related to how i was handling memory, i looked up `np.sort` again and then boom! i was so embarrassed at myself after that. i just went outside to get some air and it all clicked. i think that is how most debugging goes.
anyway, that is how you can select the top three values, simple, effective and not too crazy, use the approach that better fits your use case. good luck.
