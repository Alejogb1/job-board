---
title: "6.13 lab filter and sort a list python?"
date: "2024-12-13"
id: "613-lab-filter-and-sort-a-list-python"
---

Okay so you’re looking at filtering and sorting a list in Python right I've been there more times than I can count it's a bread-and-butter task really you see this pattern everywhere. Honestly I think I've written variations of this problem probably since Python 2 days back when we had those weird lambda functions everywhere and print statements that weren’t functions I'm getting flashbacks.

Okay so here’s the deal you've got a list you want to cherry-pick certain elements based on a condition then you want those cherry-picked elements nicely organized in some order typically ascending or descending numerical or alphabetical you know the drill.

Let’s break it down first the filtering part we’re going to be leaning on something called list comprehensions or filter functions because they're usually the easiest to understand and use in everyday python. I've seen some seriously overcomplicated versions of these back in my early days and I learned the hard way simplicity is usually king.

Here's a simple example let's say you’ve got a list of numbers and you only want the even ones. It’s classic I remember having to filter a list of port numbers back in the day to find available ones for a service. Good times.

```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = [num for num in numbers if num % 2 == 0]
print(even_numbers) # Output: [2, 4, 6, 8, 10]
```

See that `[num for num in numbers if num % 2 == 0]` that’s a list comprehension it iterates over your list `numbers` for each `num` it checks the condition `num % 2 == 0` that is does it have no remainder when divided by two only the numbers that satisfy this condition are placed in the new list `even_numbers`.

You could also achieve the same thing with `filter` function I don't personally like them that much I find list comprehensions usually more clear to read but if you're into a slightly more functional programming style it's your call I remember one project where I was trying to be cool and used a combination of filter map and reduce only to realize I created something nobody could understand after two weeks even me.

```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = list(filter(lambda num: num % 2 == 0, numbers))
print(even_numbers)  # Output: [2, 4, 6, 8, 10]
```

In this example we use the `filter` function with a `lambda` (an anonymous function) that returns `True` if the number is even and `False` otherwise it’s functionally identical to the comprehension above.

Okay now we filtered the list that's great but what about sorting well python's `sort` method is pretty useful for this and also the built-in `sorted` function. These can sort lists or any iterable in place or return a new sorted list respectively. You know what the difference is right one changes the original list the other doesn't. If you want to maintain the initial list unchanged then using `sorted` is the way to go otherwise use the list’s method `sort()`.

Let’s sort the list of even numbers we just created let’s say for some reason I need the even numbers in descending order that’s a thing I had to do for some data analysis I did involving some very weird and hard to explain patterns.

```python
even_numbers = [2, 4, 6, 8, 10]
even_numbers.sort(reverse=True) # Sorts in place
print(even_numbers) # Output: [10, 8, 6, 4, 2]

# Or using sorted which creates a new copy
even_numbers = [2, 4, 6, 8, 10]
sorted_even_numbers = sorted(even_numbers, reverse=True)
print(sorted_even_numbers) # Output: [10, 8, 6, 4, 2]
print(even_numbers) # Output: [2, 4, 6, 8, 10]
```

As you can see when you use `list.sort()` the list is modified in place and `sorted()` gives you a new copy the `reverse=True` argument tells it to sort in descending order by default its ascending. You can also provide a `key` function to these methods to make them sort by something other than the values themselves like we might want to sort complex objects based on some attribute or some function applied to its attributes. I remember I used that a lot while working with data coming from web apis I had to sort by timestamps and other stuff it can be pretty powerful.

Now let's put it all together to filter and sort at once here's an example that does everything all at once. I swear I had a project some time ago that looked a lot like this and this is literally how I solved it maybe with less comments.

```python
numbers = [1, 10, 3, 8, 5, 6, 9, 2, 7, 4]
filtered_sorted_numbers = sorted([num for num in numbers if num % 2 == 0], reverse=True)
print(filtered_sorted_numbers) # Output: [10, 8, 6, 4, 2]
```
First we are filtering numbers to only get even numbers then we sort those even numbers in descending order. Simple and efficient you don't need much more than that for these common cases. It's concise and readable which is what you want especially if you're coming back to your code after a few months.

Now it is important to remember that the order of operations here is important First the filtering is performed and then the sorting. You cannot sort a list before it is filtered if you want to sort it after being filtered that is like you cannot wash the dishes before you have eaten the food or you could do it but it would be pointless to the task.

Now for performance considerations for small lists it doesn't really matter which approach you take but when your dealing with big lists you have to take care of things because some algorithms and methods are more efficient than others in different scenarios. If you're dealing with very large datasets consider using libraries like NumPy or Pandas which are designed for efficient data manipulation I once wrote something that used a list when it should be an array and the performance was atrocious. It taught me that performance is something that is not good when it's not good enough.

Also if you're doing more complex sorting consider reading up on comparison sorting algorithms the best intro I've seen on the subject is actually from the "Introduction to Algorithms" book by Cormen et al although it's a bit academic it’s still a great resource to understand the underlying principles behind things like `sort()` and `sorted()`. Also consider the python documentation for `sort` which contains a lot of useful information.

And that’s pretty much all you need to know for filtering and sorting lists in Python if you have any specific cases that you need help with feel free to ask. If you’re asking me about parallel sorting algorithms well that's a whole other level and then you should be asking in supercomputing stackoverflow or something but that’s for another day. I always say that you should not try to overcomplicate a simple thing unless you’re really asking for it. Also remember premature optimization is the root of all evil (or something like that I don't know really).
