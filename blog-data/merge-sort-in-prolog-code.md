---
title: "merge sort in prolog code?"
date: "2024-12-13"
id: "merge-sort-in-prolog-code"
---

Alright so you're looking at merge sort in Prolog right Been there done that countless times I've had my fair share of wrestling with Prolog and sorting algorithms especially when I was first getting into declarative programming. It’s different from the imperative world of loops and assignments that I was used to. Let me tell you it was quite a shift.

So when you're tackling merge sort in Prolog you're really thinking about recursion and list manipulation which are basically the bread and butter of the language. You’re not going to be directly modifying arrays like you would in say C or Java. Instead you're defining relationships and how lists should be transformed.

First thing's first we need a way to divide a list into two roughly equal halves. Here’s how I typically do it. I've used variations of this in so many Prolog projects it's not even funny.

```prolog
split_list([], [], []).
split_list([X], [X], []).
split_list([X, Y | Rest], [X | Left], [Y | Right]) :-
    split_list(Rest, Left, Right).
```

Okay what’s happening here? `split_list([], [], []).` This is our base case when we encounter an empty list we return two empty lists. Makes sense right? Next up `split_list([X], [X], []).` This is the base case for single element lists one element we output the same single-element list in the left list and an empty list in the right list. Then the general case: `split_list([X, Y | Rest], [X | Left], [Y | Right]) :- split_list(Rest, Left, Right).` This is where the magic happens. We take the head of the list two elements `X` and `Y` and stick them into separate output lists `Left` and `Right` respectively and then recursively call `split_list` with `Rest`. This continues until we hit the base cases resulting in roughly equal lists.

Now that we have splitting down the next crucial step is merging the sorted lists into a single sorted list. Here’s the merge function I usually rely on I’ve tweaked it over the years but the core idea remains the same.

```prolog
merge([], Right, Right).
merge(Left, [], Left).
merge([X | Left], [Y | Right], [X | Merged]) :-
    X =< Y,
    merge(Left, [Y | Right], Merged).
merge([X | Left], [Y | Right], [Y | Merged]) :-
    X > Y,
    merge([X | Left], Right, Merged).
```

This is where you see the sorting logic taking place. `merge([], Right, Right).` and `merge(Left, [], Left).` are base cases: when either left or right lists are empty we just return the other list. Pretty straightforward. Now for the fun part.

`merge([X | Left], [Y | Right], [X | Merged]) :- X =< Y, merge(Left, [Y | Right], Merged).` If the head of the `Left` list `X` is less than or equal to the head of the `Right` list `Y` we put `X` at the front of the merged list and recursively call merge with the tail of the `Left` list and the entire `Right` list.
`merge([X | Left], [Y | Right], [Y | Merged]) :- X > Y, merge([X | Left], Right, Merged).` If the head of the `Left` list is bigger than the head of the `Right` list we put `Y` at the front and recursively call merge with the whole `Left` and the tail of the `Right`. This handles the sorting part of the algorithm placing the smaller element first in the result.

Now it is time for the main merge sort function that combines the split and merge logic. It’s a classic recursive dance of divide and conquer. Here's a pretty solid implementation:

```prolog
merge_sort([], []).
merge_sort([X], [X]).
merge_sort(List, Sorted) :-
    split_list(List, Left, Right),
    merge_sort(Left, SortedLeft),
    merge_sort(Right, SortedRight),
    merge(SortedLeft, SortedRight, Sorted).
```

Pretty clear right? `merge_sort([], []).` handles the base case of an empty list just returns an empty list. `merge_sort([X], [X]).` handles the base case of a single element returning the same list. The main logic `merge_sort(List, Sorted) :-` splits the input `List` using `split_list` and recursively calls `merge_sort` on the `Left` and `Right` halves. It then merges the sorted halves `SortedLeft` and `SortedRight` using `merge` and binds the result to `Sorted`.

These were the basic functions and now you have a pretty functional merge sort implementation. I remember in one of my early projects I was using a particularly complex set of data and the standard sort functions were taking forever. I switched to merge sort after I remembered the algorithms complexity was O(n log n) on average and the performance difference was night and day. That is to this day one of the fastest sorting implementations I have written and I have never looked back.

Now there are some things to keep in mind especially when working with Prolog. Prolog uses backtracking and this recursion style of programming can consume stack space. You might see stack overflows for very large lists that may mean you have to look into tail recursion optimization techniques and there are books on the subject like "The Art of Prolog" by Sterling and Shapiro and “Prolog Programming for Artificial Intelligence” by Bratko. These are both really good books and really helped me with Prolog. Tail recursion is a whole other topic for now lets just focus on the basics. You can avoid that problem with small enough lists that you are usually dealing with in most practical cases.

And remember this is just one way to do it. You can tweak it adapt it and there's no single right answer in programming just ways to achieve the same result. You could make it more efficient by changing the splitting or merge logic and the algorithm but the foundation is the same.

Oh and one more thing did you hear about the programmer who got stuck in the shower? He couldn't figure out how to exit the loop he was in infinite loop i think...

Anyway hope this helps if you have more questions hit me up.
