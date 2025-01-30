---
title: "Why am I getting a 'TypeError: call() takes 2 positional arguments but 3 were given' error?"
date: "2025-01-30"
id: "why-am-i-getting-a-typeerror-call-takes"
---
The "TypeError: call() takes 2 positional arguments but 3 were given" error, in my experience, most commonly arises when a method defined within a class in Python is invoked incorrectly, specifically when the *self* argument is unintentionally provided by the user. The *self* parameter, in essence, is Python’s mechanism for passing the instance itself to a method, but the user typically should not explicitly include it in the function call. This error indicates a mismatch between the method's expected signature and the arguments provided during its call. The Python interpreter automatically handles the *self* argument, and any attempt to manually pass an additional argument in its place triggers the TypeError.

Let's dissect this more thoroughly. Python methods within a class always receive the instance of the class as their first parameter, which, by convention, is named *self*. When defining a method like `def my_method(self, arg1)`, the user only ever needs to provide `arg1` when calling it; Python implicitly passes the instance, the object on which the method is being called, as *self*. The error you're encountering signifies that you’ve likely attempted to pass the *self* argument manually, leading to one too many positional arguments.

The root cause often resides in two scenarios. First, novice developers transitioning from other languages may misunderstand Python's implicit *self* handling and include it in their method calls, believing it is a necessary explicit parameter. Second, this can happen due to misunderstanding the difference between calling the method directly on an object instance and accessing it as an unbound function from the class itself.

To illustrate, let's look at some code examples.

**Example 1: The Correct Approach**

Consider this simple class:

```python
class MyClass:
    def __init__(self, value):
        self.value = value

    def add_value(self, increment):
        return self.value + increment


my_instance = MyClass(10)
result = my_instance.add_value(5)
print(result)  # Output: 15
```
Here, the `add_value` method takes two parameters in its definition: *self* and `increment`. When I create an instance, `my_instance`, and subsequently call `add_value(5)`, Python implicitly inserts `my_instance` as the *self* argument, allowing the method to operate on the specific instance's `value`. I am passing just the explicit `increment`, 5 in this case, and the method functions as designed. No TypeError occurs because the interpreter manages `self`.

**Example 2: The Erroneous Approach - Explicit *self***

Now, let's examine a scenario where the TypeError does arise:

```python
class MyClass:
    def __init__(self, value):
        self.value = value

    def add_value(self, increment):
        return self.value + increment


my_instance = MyClass(10)
result = MyClass.add_value(my_instance, 5) #Incorrect Call!
print(result) #Raises TypeError
```

In this version, I explicitly pass `my_instance` as the first argument to the `add_value` method. While the method's signature includes *self*, its purpose is to represent the instance *from which* the method is called. I am attempting to provide this instance explicitly *as an argument,* which is not how method calling should work. In doing so, the method now interprets my explicit `my_instance` as the first *positional* argument (intended for the *self* parameter), the second one `5` as `increment`, and Python is not expecting a third positional argument (that does not exist, but it is provided as the error clearly indicates). This mismatch triggers the "TypeError: call() takes 2 positional arguments but 3 were given" error. The error arises from incorrectly treating the method as a simple function, when it is indeed a method bound to a particular class instance.

**Example 3: The Erroneous Approach - Unbound Method**

Another source of this error is trying to call a method directly from the *class* itself without binding it to an instance:

```python
class MyClass:
    def __init__(self, value):
        self.value = value

    def add_value(self, increment):
        return self.value + increment


my_instance = MyClass(10)
result = MyClass.add_value(5) #Raises TypeError
print(result)
```

Here, I am not invoking `add_value` on an *instance* of `MyClass`. I'm trying to use `add_value` directly on the class `MyClass` itself, and in this case, since I'm attempting to treat it like a function rather than a method, I'm not passing an instance as the first argument, which Python uses to fill the `self` parameter. In this case, the method is said to be "unbound." It's still a method in that it has `self`, so when I provide it one positional argument, Python expects *two*, namely `self` and increment, not just increment. Python sees *one* passed explicit argument (5), and still expects its implicit *self*, and there are not 2 arguments passed explicitly. The interpreter then tries to pass `5` as *self*, and it does not fit the expectation (it's not a `MyClass` instance) and more importantly it sees that there should only be *two* parameters, and in a manner of speaking, it sees that the *user* passed an implied *third* parameter (where `self` would have been filled implicitly) so this case also raises the "TypeError: call() takes 2 positional arguments but 3 were given" error. While it is similar to example 2, it is different in the error source. The previous one explicitly passes the class *instance*, this one does not.

To rectify this, you must first create an instance of the class, and then call the method on that instance, thus providing Python with the implicit first argument, which is the self parameter.

In summary, the core issue is the confusion surrounding how Python manages the *self* parameter for methods within classes. The interpreter handles it implicitly when you correctly call a method on an instance of the class. Attempting to provide it explicitly as an argument, or by calling an unbound method directly on the class, results in the "TypeError: call() takes 2 positional arguments but 3 were given" error.  It’s a good practice to always verify how you're invoking your methods: on an object of the class, and not the class itself, unless you're dealing with class methods (which require different syntax).

For further reading, I recommend investigating resources that delve deeply into object-oriented programming in Python. Study the sections on class methods, instance methods, the concept of *self*, and how methods are called. Also, review sections on debugging techniques that cover error tracing. Pay close attention to documentation from the official Python website and other reputable educational sources. Focus on understanding the fundamental concepts before moving on to more intricate techniques.
