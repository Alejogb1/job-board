---
title: "How to Solve Problems Efficiently with Dynamic Programming"
date: "2024-11-16"
id: "how-to-solve-problems-efficiently-with-dynamic-programming"
---

okay so you wanna hear about this crazy video i watched right it's all about this thing called "dynamic programming" sounds super intimidating i know but trust me it's way less scary than it sounds it's basically this clever way to solve problems by breaking them down into smaller bite-sized chunks and then cleverly reusing the answers to those smaller problems so you don't have to do all the work over and over again  think of it as being super organized and efficient it's like having a cheat sheet for your brain  the whole video is like a detective story unraveling the mystery of how to be awesome at problem-solving with code


the setup was pretty chill the guy in the video this total coding guru dude starts with this super simple example like finding the nth fibonacci number you know the sequence 1 1 2 3 5 8 and so on it's a classic comp sci intro problem  he’s like “look how inefficient this recursive solution is” and he shows this code  it's like watching a snail race a cheetah  it takes forever  he's got this totally exasperated expression on his face  that was a great visual cue seriously that face alone was worth the price of admission


then he drops the bomb the dynamic programming solution  he's all smiles now  it's like watching the cheetah finally win and he's super smug about it  another visual cue was this awesome graph he drew showing how the solution builds up step by step like building a pyramid of answers each level resting on the ones below it just incredibly satisfying to watch  i swear i could almost hear angels singing


one of the main ideas the video hammered home was memoization  it's basically caching the results of expensive function calls so you don't have to recalculate them every time you need them it's like having a super-efficient memory for your code  think of it like this imagine you're making a complicated cake you wouldn't want to bake each ingredient from scratch every time you need it you'd make a bunch of them ahead of time and store them for later use right that’s memoization in a nutshell


here’s a little python code snippet to illustrate memoization  


```python
cache = {}  # our super-efficient cake ingredient storage

def fib_memo(n):
    if n in cache:
        return cache[n] # if we have it use it! no need to bake again
    if n <= 1:
        return n
    else:
        result = fib_memo(n-1) + fib_memo(n-2)  # bake the cake from previous pieces
        cache[n] = result # store the result in the cache so we can reuse it later!
        return result

print(fib_memo(10)) # try this out! watch how much faster it is!
```


another key concept was tabulation  it’s like memoization's more organized cousin  instead of storing things in a dictionary  like in memoization  you use an array to store all the results  it’s like making a perfectly organized baking chart for every ingredient, step by step from start to finish you just build the array from the bottom up filling it out  it’s more systematic and can even sometimes be a little faster  


here’s how tabulation would look for the fibonacci sequence:


```python
def fib_tab(n):
    tab = [0] * (n + 1) # create our ingredient chart  lots of space for our cake's ingredients
    tab[1] = 1 # base case of the cake-making
    for i in range(2, n + 1):
        tab[i] = tab[i-1] + tab[i-2]  # systematically bake the cake ingredients
    return tab[n]

print(fib_tab(10)) # enjoy that perfectly ordered cake!
```


the video also touched on a third technique related to dynamic programming, which is often useful but I don't want to get bogged down in the details because i want to get to the main point  but the key here is that dynamic programming relies on an overlapping subproblems property that is there are repetitive calculations in your problem this means that you can solve it much faster by remembering the results of previous subproblems  many problems in computer science and other fields have this property for example you can find the shortest path between two points in a graph by remembering all shortest paths to all points along the way


here's a tiny example of this in action  imagine you're trying to find the shortest way to walk from your house to the coffee shop  one technique involves trying every possible route however what if you've already found the shortest path to a few intermediate locations along the way you can reuse those results as shortcuts  dynamic programming lets us do just that  if we can find the shortest way to every intermediate point then we can simply choose the route among those that leads to the coffee shop!  it's a huge time-saver


```python
# a super simplified example of shortest path using dynamic programming principles

# let's say we have distances between some locations
distances = {
    'home': {'a': 5, 'b': 10},
    'a': {'b': 3, 'coffee': 8},
    'b': {'coffee': 7}
}

# We want to find the shortest distance from home to coffee
shortest_distance = float('inf')  # this is equivalent to infinity - starts very large

# We 'iterate' through possible intermediate points
for intermediate in distances['home']:  # iterate through possible next steps from home
    total_distance = distances['home'][intermediate] + distances[intermediate]['coffee']  # calculate total distance to coffee shop
    shortest_distance = min(shortest_distance, total_distance)  # keep only shortest

print(f"Shortest distance from home to coffee shop: {shortest_distance}")
```


a key spoken cue in the video was when the instructor said “optimizing for efficiency is key” that really stuck with me  it's not just about getting the right answer; it's about getting the right answer in a reasonable amount of time—especially when you're dealing with big datasets or complex problems you can't have your computer chugging for hours or days to get a solution


the resolution of the video was pretty straightforward dynamic programming is a powerful technique for solving complex problems efficiently by breaking them down into smaller, overlapping subproblems and storing the results—that’s why it’s efficient you only compute each subproblem once and then reuse the result  it's all about clever reuse and organization it's not magic it's just well-structured, elegant code


so yeah that was the video in a nutshell  it was way more engaging than i expected  i went in thinking it'd be super dry and theoretical but it was surprisingly fun i still have to practice a lot to master dynamic programming but now i at least know the basics and i feel way more confident about tackling tricky problems  it’s like having a secret weapon in my coding arsenal  i'm excited to see how i can use it to solve even more challenging problems  plus i got to see that awesome exasperated face  so that’s always a bonus!
