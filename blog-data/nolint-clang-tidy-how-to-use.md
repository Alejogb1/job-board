---
title: "nolint clang tidy how to use?"
date: "2024-12-13"
id: "nolint-clang-tidy-how-to-use"
---

Okay so someone wants to know about `nolint clang-tidy` right I've been down this rabbit hole more times than I care to admit Lets break it down real simple no frills just the tech

First off `nolint` is not a command itself it's a directive a comment really It tells `clang-tidy` "hey ignore this specific piece of code don't bother me with your warnings" It's a way to silence noise when you know better than the static analyzer at least you think you do you sometimes don't but hey we all learn

I've used it extensively in a ton of projects from embedded systems where every cycle counts to high performance computing code where the compiler is my best friend and my worst enemy It's a valuable tool and a dangerous one too remember that great power and all that

Now why would you use `nolint` well imagine you have a situation where the code is perfectly valid logically and it runs fine it's not going to crash but `clang-tidy` throws some warning about it like "potential memory leak" or something Well sometimes those are false positives other times the code might be doing something low level that needs that kind of behavior Its best to be really sure its not a real issue before using it though sometimes the compiler just cant understand our genius its the compiler not you... well mostly not you

Lets look at some common use cases

**1 Suppressing specific warnings**

This is the most common scenario You have some code that triggers a particular warning and you want it to go away without rewriting perfectly good code I've been in situations where the cost of rewriting code to make the compiler happy is not worth the effort time is money in my world

```cpp
// clang-tidy warning: Possible integer overflow
int calculateSomething(int a, int b) {
    return a + b;  // NOLINT(hicpp-signed-bitwise)
}
```
Here you're specifically telling `clang-tidy` to ignore the `hicpp-signed-bitwise` check its a specific type of warning so you know exactly what is being suppressed be specific when using the `nolint` as it could mask other actual warnings

Sometimes it's good to explain why you are disabling the linter warnings so when you go back to the code a year later you're not just scratching your head I added a small note below

```cpp
// clang-tidy warning: Potential leak of object 'data'
void someFunction() {
   char* data = (char*)malloc(100);
   //Do something
   free(data); // NOLINT(cppcoreguidelines-no-malloc) freed elsewhere
}
```

In this example I am doing some specific memory allocation which is flagged by the linter It is freed somewhere else in a specific pattern this is very common in low level C programming The comment helps understand why the warning is suppressed

**2 Temporarily Disabling Checks**

There are times when you have to work on code that violates a rule and you are doing refactoring and don't want to be constantly bothered by `clang-tidy` Just keep in mind it's easy to forget to remove these later and end up with a lot of dead code

```cpp
void complexFunction() {
    // NOLINTBEGIN
    int a = 10;
    char* b = (char*)malloc(100);
   
    
    // lots of messy legacy code that clang-tidy would flag
    //...
    
    free(b);
    // NOLINTEND
}
```

The `NOLINTBEGIN` and `NOLINTEND` directives are used to disable all clang-tidy checks within that code block Use with caution this one is good to ignore large parts of messy legacy code you are refactoring but is a very wide net to cast you are disabling every single check be very careful with this one

**3 Suppressing errors only in certain lines**

Sometimes you want to suppress clang tidy checks just on one line, use `NOLINT` in the end of the line

```cpp
void functionWithError() {
	 int* ptr = nullptr;
	 *ptr = 10; // NOLINT(bugprone-null-dereference)
}
```

This allows you to have specific exceptions on one line only instead of multiple lines of code its the more elegant approach on a single line warning

**How do I know what checks to disable**

`clang-tidy` usually outputs the name of the check that's being violated for example  `hicpp-signed-bitwise` or `cppcoreguidelines-no-malloc` you'll see it in the output when running the `clang-tidy`

`clang-tidy` documentation is your friend you will find all the checks there its a really good resource https://clang.llvm.org/extra/clang-tidy/checks/list.html but there are other resources like the book "Effective Modern C++" by Scott Meyers its not about static analysis but having a deeper understanding of the language helps to avoid these types of issues all together sometimes

I've found that the checks are quite comprehensive and there are a lot of them this is why its good to be specific when disabling them instead of globally disabling the checks with the begin and end operators

**When should you NOT use nolint**

This is important `nolint` is not a get-out-of-jail-free card I've seen people use it as a crutch masking all the real problems with a bunch of `NOLINT` comments  Its never a good idea It's almost like having a bad habit and then you make a comment like it's a feature instead of a problem You know what I mean? no? you do you are like me I know you are.

If you are getting a lot of warnings it is better to fix the root cause of the problem rather than just suppressing all the warnings It could be a design problem an architectural mistake or just some bad programming practices sometimes the clang-tidy warnings are like little hints telling you to check the code with a fresh set of eyes

**My personal experience**

I had a real nightmare project with a huge legacy code base that used a lot of `malloc` and `free` directly It was a mess `clang-tidy` was throwing all kind of warnings `cppcoreguidelines-no-malloc` was the main culprit but since this was code that had been running for years and had a lot of manual memory management and a very peculiar style and pattern we used a lot of `NOLINT` but only after we had really reviewed the code and it was a good pattern it made sense for the specific implementation.

We even had to write custom `clang-tidy` checks to better deal with the custom memory management its no fun but it's good if you understand `clang-tidy` well enough to do that it was fun at the end honestly

So there you have it `nolint clang-tidy` in a nutshell use it wisely young padawan dont abuse the tool and most important understand why you are using it and document it so the next person who works with your code has a better time

Remember the goal is to produce clean safe and maintainable code the static analyzer is trying to help you achieve that even if you are a grumpy developer or think you are better than it just try to listen to it sometimes... mostly
