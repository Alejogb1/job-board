---
title: "prolog trace how to use?"
date: "2024-12-13"
id: "prolog-trace-how-to-use"
---

Okay so you're asking about Prolog tracing right been there done that a few times tracing in Prolog can feel like navigating a maze with a blindfold especially when you're dealing with some real complex rules

Look I remember back in the day probably around '08 I was working on this automated theorem prover thing yeah early AI before it was cool and oh man the recursion was deep so so deep tracing was my only friend and a very temperamental friend at that we weren't always on the same page

The basic idea of tracing is to see how Prolog steps through the evaluation process this is super crucial when you are debugging a logic program because if a goal fails or succeeds in an unexpected way it's really hard to pinpoint the issue without tracing So think of it like this when Prolog tries to satisfy a goal it can either try to call another predicate execute a rule back track or succeed tracing shows you each of these actions in sequence and lets you see where the program got into trouble or where a variable got the wrong value

Now Prolog has a bunch of trace modes for different purposes the simplest and honestly most used one is `trace/0` this is the plain vanilla you turn it on and watch the world unfold

Here is a basic example of a small program lets say we are trying to define a parent child relationship:

```prolog
parent(john, mary).
parent(john, peter).
parent(susan, john).

ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).

```

If you are like me first thing is to test this right? So the typical query is `ancestor(susan, peter).` but lets say we are not sure how Prolog is evaluating this so you can fire up the trace before the query:

```prolog
trace.
ancestor(susan, peter).
```

You'll see output similar to this it depends on your Prolog system but roughly it will be something like:

```
   Call: (7) ancestor(susan, peter) ?
   Call: (8) parent(susan, peter) ?
   Fail: (8) parent(susan, peter) ?
   Call: (8) parent(susan, _G1210) ?
   Exit: (8) parent(susan, john) ?
   Call: (8) ancestor(john, peter) ?
   Call: (9) parent(john, peter) ?
   Exit: (9) parent(john, peter) ?
   Exit: (8) ancestor(john, peter) ?
   Exit: (7) ancestor(susan, peter) ?
```

Now I know what you are thinking this output looks like a bunch of gobbledygook but let me break it down. `Call` means Prolog is trying to satisfy a goal `Exit` means a goal was successful and `Fail` means it didn't work out. The numbers in parentheses are the depth of recursion this helps you follow the stack. The `?` means Prolog is waiting for your input to see if it should continue executing and you usually just press enter to continue and so on

When you get a complex predicate that has a lot of options I find it useful to trace specific predicates only. So if you want to only trace `ancestor` and not `parent` you can use `trace/1`:

```prolog
trace(ancestor/2).
ancestor(susan, peter).
```

This will then show you only the trace related to `ancestor/2`. This feature is like having a scalpel instead of a chainsaw you get way more fine-grained control. Sometimes its enough to find where the problem is.

Sometimes `trace` gives you way too much information. Imagine debugging a program that has hundreds or thousands of clauses it is just painful to go through every call right? In such cases it may be useful to look into `spy/1` or `nospy/1` this is super similar to breakpoints if you are used to procedural debugging. With `spy` you are essentially saying 'pause execution whenever Prolog enters or exists this particular predicate'

Here's the deal you pick the predicate you want to spy on and then you can step through it more carefully. It is like putting a magnifying glass on specific parts of your program and ignore the rest. So in the previous example if you do this:

```prolog
spy(ancestor/2).
ancestor(susan, peter).
```

The trace output will be similar to the previous one but you now have the control over when to proceed through the predicate calls because it will pause whenever the predicate you spied on is called or exits. This is my preferred way to debug programs once they become complex because sometimes you just want to check one single clause or condition and go to the next step in the debugger

Now lets talk about something a little less verbose and more about understanding how to see variable assignments. You can use the `debug/0` mode this mode is like trace but gives you a bit more information about the variable bindings so you can see what values they are taking during execution especially when unification gets complex

I was having a really hard time once with some Prolog program involving graph traversals and all this backtracking going on and it turned out the problem was that my variables were getting unified with the wrong values due to a missing base case. It was like trying to track a specific rabbit in a forest of rabbits. The debug mode helped me see which variable had the wrong values and at which point.

Here is an example say we have a program that checks if a list is sorted:

```prolog
sorted([]).
sorted([_]).
sorted([X, Y | Rest]) :-
    X =< Y,
    sorted([Y | Rest]).
```

And lets say you call it like `sorted([1,3,2]).` and it obviously fails because the list is not sorted now you want to know why. This is where `debug/0` shines

```prolog
debug.
sorted([1,3,2]).
```

This will give you a trace and variable bindings like this:

```
   Call: (7) sorted([1, 3, 2]) ?
   Call: (8) 1=<3 ?
   Exit: (8) 1=<3 ?
   Call: (8) sorted([3, 2]) ?
   Call: (9) 3=<2 ?
   Fail: (9) 3=<2 ?
   Fail: (8) sorted([3, 2]) ?
   Fail: (7) sorted([1, 3, 2]) ?
```

You can see that initially X = 1 and Y = 3 then in the next call X = 3 and Y=2. The line `Call: (9) 3=<2 ?` shows you that this is where the program is failing because 3 is not less or equal to 2. Boom problem solved!

Now you might be wondering is there any way to save this trace output into a file for later analysis well good thing you asked. Depending on your Prolog implementation you can usually redirect the trace to a file using the `tell/1` predicate for example using SWI-Prolog you can do the following:

```prolog
open('trace_output.txt', write, Stream),
tell(Stream),
trace.
ancestor(susan, peter).
told,
close(Stream).
```

This code will save the output in `trace_output.txt`. Now you have your debugging information saved and you can study it later if that helps in finding the problem

You know what debugging in Prolog is kind of like a really intricate puzzle box you know the answer is in there somewhere but you have to look at the tiny gears and levers to understand how everything works. It can be frustrating sometimes but when you finally figure it out its extremely rewarding. I mean I once spent three whole days debugging a program and the issue was a missing base case in a recursion I almost pulled my hair out but it made me a better programmer so I guess I should thank Prolog for that or not lol

Okay so to summarize what have we covered today tracing is your best friend in prolog. First we have the plain `trace/0` mode to see everything then we have the specific `trace/1` to trace predicates then we have `spy/1` to breakpoint on predicates finally we talked about `debug/0` to see variable bindings and lastly we saved the trace to a file for later inspection.

Now some resources to dive deeper I would definitely recommend 'Programming in Prolog' by Clocksin and Mellish it's like the bible for Prolog programmers and the tracing parts are very well explained also 'The Art of Prolog' by Sterling and Shapiro it has a more advanced view on debugging techniques. Also any good online course or tutorial that includes specific sections on debugging using tracing will do the trick. I also found that looking at open source projects using Prolog and trying to understand their code with a trace was a great way to learn.

Hope this helps you with your Prolog adventures. Happy debugging!
