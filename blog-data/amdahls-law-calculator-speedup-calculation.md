---
title: "amdahl's law calculator speedup calculation?"
date: "2024-12-13"
id: "amdahls-law-calculator-speedup-calculation"
---

Okay so you wanna talk about Amdahl's Law right Speedup calculations classic stuff Been there done that got the t-shirt and probably a few grey hairs from debugging performance bottlenecks over the years lets dive in

First off Amdahl's Law is your go to for figuring out how much faster your code can realistically get if you parallelize some of it it's not some magic bullet that'll make everything instantly zoom along it's a constraint a reality check that's why it's so useful

The basic idea is that you have parts of your program that can be sped up typically by running in parallel and parts that gotta run sequentially like serial and you can't touch them they are like the stubborn bits

The formula is pretty simple to remember: `Speedup = 1 / ( (1 - P) + (P / N) )` where `P` is the fraction of the code that can be parallelized and `N` is the number of processors or cores you're using

So `1 - P` is the serial portion the part that's stuck running sequentially it's always going to take up time no matter how many cores you throw at it The `P / N` part is the parallel portion spread over your multiple cores

I've seen this a million times in the wild you think you're gonna get like a 10x speedup by using 10 cores only to get something like 3x its like reality is hitting you with a very blunt object

Okay lets get to the code so you can see it in action I'll throw in a couple of different versions so you have some variety

```python
def amdahl_speedup(parallel_fraction, num_processors):
    """Calculates Amdahl's Law speedup.

    Args:
        parallel_fraction: Fraction of code that can be parallelized (0 to 1).
        num_processors: Number of processors or cores.

    Returns:
        Speedup factor.
    """
    if not 0 <= parallel_fraction <= 1:
        raise ValueError("Parallel fraction must be between 0 and 1")
    if num_processors <= 0:
       raise ValueError("Number of processors must be greater than 0")
    return 1 / ((1 - parallel_fraction) + (parallel_fraction / num_processors))

# Example Usage
p = 0.8 # 80% of code is parallelizable
n = 4  # 4 cores

speedup = amdahl_speedup(p, n)
print(f"Amdahl's Law Speedup: {speedup}")

```

This is just pure python no libraries needed just a function for calculating speedup you give it the parallel fraction and number of cores you get your result pretty straightforward right? Now let's say you're dealing with a slightly different scenario maybe you're doing some simulation work or image processing where you need to deal with big arrays or matrices we should move to using numpy its a good practice

```python
import numpy as np

def amdahl_speedup_np(parallel_fraction, num_processors):
     """Calculates Amdahl's Law speedup using NumPy.

    Args:
        parallel_fraction: Fraction of code that can be parallelized (0 to 1).
        num_processors: Number of processors or cores.

    Returns:
        Speedup factor as a NumPy array.
    """
     if not 0 <= parallel_fraction <= 1:
        raise ValueError("Parallel fraction must be between 0 and 1")
     if num_processors <= 0:
       raise ValueError("Number of processors must be greater than 0")
     return 1 / ((1 - parallel_fraction) + (parallel_fraction / num_processors))

# Example usage with NumPy arrays
parallel_fractions = np.array([0.5, 0.8, 0.9]) # Different parallel fractions
num_processors = np.array([2, 4, 8]) # Different core counts

speedups = amdahl_speedup_np(parallel_fractions, num_processors)
print(f"Amdahl's Law Speedups: {speedups}")
```

Now with numpy we can feed arrays of different values and calculate a lot of speedup values at once Its faster with numerical calculations but not for this example because it's just one single scalar formula we are using it to exemplify that is possible

And hey you know what sometimes you don’t even need to bother with the actual calculations just plotting the theoretical speedup curve is really helpful so here we go a plotting example

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_amdahl_speedup_curve(parallel_fraction):
    """Plots Amdahl's Law speedup curve for a given parallel fraction."""
    num_processors = np.arange(1, 101) # Cores from 1 to 100
    speedups = 1 / ((1 - parallel_fraction) + (parallel_fraction / num_processors))

    plt.plot(num_processors, speedups)
    plt.xlabel("Number of Processors")
    plt.ylabel("Speedup")
    plt.title(f"Amdahl's Law Speedup Curve (Parallel Fraction: {parallel_fraction})")
    plt.grid(True)
    plt.show()

# Example usage:
parallel_fraction = 0.7 # 70% of code can be parallelized
plot_amdahl_speedup_curve(parallel_fraction)

```

This one gives you a visual representation you can see how the speedup starts to plateau after a certain number of cores no matter how much you try it's actually good to see it like this because sometimes I don't believe the results of a single calculation I need a plot to see the big picture sometimes its just a feeling like I should not trust the single numeric result only and you now its that feeling its like a data scientist has that feeling we know the answer sometimes but we need the plot to be sure I don't know if that make sense for you but it does for me.

Now this is just Amdahl's Law remember it has it limitations for example it assumes that the parallelized part scales perfectly and this is almost never true in the real world you always have overhead communication between cores locking etc which reduces the speedup of your program and you should also think about Gustafson’s Law if you are scaling your problem size together with the number of cores that's for another conversation though

Also consider that sometimes your bottlenecks are not from the code running slowly but from other factors like disk IO or network latency you should always measure to see where your program spends most of it time before you start changing code for parallelization and remember that sometimes optimizing the code itself is better than just simply parallelizing like for example improving algorithms and data structures choices can yield a better performance increase than parallelization sometimes but you need to measure this always before starting any optimization

And speaking of measuring performance you should check out tools like perf or Linux's `time` command these are your friends always use them for benchmarking and profiling before and after changes I'm an expert in profiling and I still measure all changes to see what the reality looks like.

Resources? I recommend starting with "Computer Architecture A Quantitative Approach" by John L. Hennessy and David A. Patterson it covers a lot of the fundamentals and some theoretical and detailed discussion about parallelization techniques and architecture issues. Also “Parallel Programming” by Thomas Rauber and Gudula Rünger it’s more specialized in algorithms and coding for parallel systems but I think you should check them out it gives you more in depth knowledge into parallel programming techniques.

Oh and one more thing it’s a common mistake but please remember Amdahl’s law doesn’t say that using more cores will *always* make your program faster it just states that the speedup is bounded by the serial part so I mean parallelize wisely my friend!

And finally if you're finding this stuff tricky dont worry it's all part of the learning process debugging and optimizing are life skills for us software developers so don't get discouraged just keep on learning and practicing.
