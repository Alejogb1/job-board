---
title: "Why does Prolog fail to halt when reversing a list?"
date: "2025-01-30"
id: "why-does-prolog-fail-to-halt-when-reversing"
---
The core reason Prolog can fail to halt when attempting to reverse a list stems from how it manages recursion in conjunction with a specific definition for list reversal, frequently involving an append operation. This interaction can easily lead to an infinite loop if not carefully structured. I've personally encountered this issue multiple times, particularly in early projects, and its resolution highlights critical aspects of Prolog programming.

The problem arises when we define list reversal using a recursive approach that repeatedly adds elements to the end of an initially empty list. This appending is often achieved using a predicate similar to `append/3`, where the intended list to be reversed acts as one argument to `append`, and another initially empty list serves as the accumulation point, and then each element of list to be reversed added at end of the accumulation list, via append. However, if the append predicate itself is not carefully constructed, or if it's employed in a way that generates infinite choice points, we can trigger a non-terminating computation. The fundamental challenge is that the straightforward implementation for appending can be left-recursive; in such a case, there are multiple ways a list can be broken down, which will cause a loop, where new goals may be created to satisfy append and then continue to generate new goals recursively, but never lead to the base case where the accumulator list is returned.

Let’s consider a simple but problematic definition of list reversal. We’ll assume that `append/3` is implemented such that the first list's length can be infinite if it's a variable. This can result in an infinite loop when appending to an accumulator list, as explained. Here's the problematic approach:

```prolog
% Incorrect reverse definition
reverse_incorrect([], []).
reverse_incorrect([H|T], R) :-
    reverse_incorrect(T, TR),
    append(TR, [H], R).


% Naive append implementation that can cause infinite loops
append([], L, L).
append([H|T], L, [H|Result]) :-
    append(T, L, Result).
```

In this flawed example, the base case for `reverse_incorrect/2` is correct: an empty list reverses to an empty list. The recursive case breaks the input list into head `H` and tail `T`, reverses the tail `T` into `TR`, and then attempts to append `H` to the end of `TR` using `append/3`.  If `append/3` is naively coded, without carefully controlling the variable input positions, then the recursive call to `append(T, L, Result)` can potentially generate an infinite number of ways for `T` and `L` to satisfy the required `Result` list.

Specifically, consider the query `?- reverse_incorrect([1, 2, 3], Result).`. The execution proceeds as follows:

1.  `reverse_incorrect([1,2,3], R)` calls `reverse_incorrect([2,3], TR1)`
2.  `reverse_incorrect([2,3], TR1)` calls `reverse_incorrect([3], TR2)`
3.  `reverse_incorrect([3], TR2)` calls `reverse_incorrect([], TR3)`
4.  `reverse_incorrect([], TR3)` unifies `TR3` with `[]`.
5.  `append([], [3], TR2)` is called, which unifies `TR2` with `[3]`
6.  `append([3], [2], TR1)` is called. At this point, Prolog can explore infinite ways for the first argument to be satisfied via recursion which would cause a loop.

This illustrates how the naive use of `append/3` can cause an infinite loop. Prolog backtracks and tries infinite ways to satisfy `append`, but they all fail to create a list where first argument is broken down.

The core issue is that our naive `append/3` implementation is left-recursive. It repeatedly breaks down the first list argument in the `append` clause before considering its base case, which can lead to infinite branching. The order of arguments and the manner in which arguments are unified is crucial in Prolog.

A common solution is to rewrite the append predicate such that it terminates deterministically and employ an accumulator in the reverse implementation. Here’s a more effective `append` implementation, followed by a correct `reverse` implementation:

```prolog
% Correct append implementation
append_correct([], L, L).
append_correct([H|T], L, [H|Result]) :-
    append_correct(T, L, Result).


% Correct reverse implementation using accumulator.
reverse_correct(List, Result) :-
    reverse_accumulator(List, [], Result).

reverse_accumulator([], Acc, Acc).
reverse_accumulator([H|T], Acc, Result) :-
    reverse_accumulator(T, [H|Acc], Result).
```

In this improved version, we've used the same logic for `append`. However, we aren't using append within the definition for reverse. Instead, we are using an accumulator.

The key insight is this accumulator `Acc` to accumulate the reversed part of list and it avoids using append. The initial accumulator is an empty list (`[]`). As we recursively traverse the input list, we prepend each element to the accumulator. When the input list is empty, the accumulator will have the reversed list. The base case for `reverse_accumulator` is when the input list is empty. In this case, we simply unify `Acc` with result.

For example, querying `?- reverse_correct([1, 2, 3], Result).` will unfold as follows:

1.  `reverse_correct([1,2,3], Result)` calls `reverse_accumulator([1,2,3], [], Result)`
2. `reverse_accumulator([1,2,3], [], Result)` calls `reverse_accumulator([2,3], [1], Result)`
3. `reverse_accumulator([2,3], [1], Result)` calls `reverse_accumulator([3], [2, 1], Result)`
4. `reverse_accumulator([3], [2, 1], Result)` calls `reverse_accumulator([], [3, 2, 1], Result)`
5. `reverse_accumulator([], [3, 2, 1], Result)` unifies `Result` with `[3,2,1]`

This method efficiently builds the reversed list without unnecessary recursion in `append` or infinite branching. In the accumulator version, `Acc` and Result are always unified, not generating any infinite looping opportunities, thus leading to termination.

Another important detail is to use the built-in append predicate, which employs the correct implementation of appending. In the below code, we can observe that it correctly reverses a list. Here is another example to show this in practice.

```prolog
% Example with using built-in append
reverse_with_built_in_append([], []).
reverse_with_built_in_append([H|T], R) :-
  reverse_with_built_in_append(T, TR),
  append(TR, [H], R).
```

This final example shows that a careful implementation for the recursive step within the reverse function is necessary. If we try to use the naive `append` within `reverse` it will cause a non-terminating query. However, using the built-in `append` implementation ensures that the program terminates as it does not lead to left-recursive problems. This built-in append will not produce infinite choice points during the append operation.

Key resources for deepening understanding of Prolog recursion and list manipulation include introductory books dedicated to Prolog programming, many of which contain chapters that delve into such common problems. I would recommend texts that focus on logic programming foundations and practical exercises. For a more formal treatment, resources on automated theorem proving are helpful to understand the logic of program execution. Additionally, online communities specific to logic programming (such as those focused on Prolog) frequently contain discussions and examples of these types of issues and their solutions. Carefully structured online tutorials and examples will also be helpful. Lastly, the documentation of SWI-Prolog, a popular implementation of Prolog, is an invaluable tool for understanding built-in predicates like `append/3`. They can help understand the correct use of standard predicates, leading to correct and efficient implementations, especially when working with recursion and list manipulation.
