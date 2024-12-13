---
title: "class invariants in python?"
date: "2024-12-13"
id: "class-invariants-in-python"
---

Okay so class invariants in Python huh yeah I've been down this road before plenty of times it's one of those things that seems simple at first but then you start building more complex systems and it bites you in the rear more often than you'd like. Let me tell you my story.

So back in the day I was working on this data processing system a big one you know taking in tons of data reshaping it running it through a bunch of algorithms. We're talking terabytes a day. We had this class representing data points called `DataPoint` real straightforward stuff.  It had attributes like `timestamp` `value` and `sensor_id`. Now initially things were fine but then we started seeing weird anomalies in the output garbage data things that didn't make any sense.  Turns out we had bugs where sometimes `timestamp` would be in the future or `value` would be negative when it absolutely shouldn't be.  These were classic invariant violations.

I remember the sheer panic of debugging those production issues trying to trace where these values came from. We had no proper checks no invariant enforcement and the result was a chaotic mess and a few late nights fueled by copious amounts of caffeine. Lesson learned hard way.

So what's the core problem with Python? Well Python doesn't have built-in language level support for class invariants like some languages do. You won't find an `invariant` keyword or anything of that sort.  It relies more on programmer discipline and explicit checks. Now that isn’t always a bad thing it's just requires care.  Here's what I mean by programmer discipline.

You see I realized you needed to think about your class as something with rules that should always hold true. An invariant is a condition that must be true for all valid instances of that class at all times before and after method calls.  Like the `DataPoint` class I mentioned, for a valid data point `timestamp` must be a valid timestamp `value` should be within a specific valid range it can't be any random thing. So we had to explicitly check them in our classes.

Here's a typical approach with a bit of a simple `DataPoint` example. We should have been doing this from the start:

```python
import datetime

class DataPoint:
    def __init__(self, timestamp, value, sensor_id):
        self.timestamp = timestamp
        self.value = value
        self.sensor_id = sensor_id
        self._ensure_invariants()

    def _ensure_invariants(self):
        if not isinstance(self.timestamp, datetime.datetime):
            raise ValueError("Timestamp must be a datetime object")
        if self.timestamp > datetime.datetime.now():
            raise ValueError("Timestamp cannot be in the future")
        if not isinstance(self.value, (int, float)):
            raise ValueError("Value must be a number")
        if self.value < 0:
            raise ValueError("Value cannot be negative")
        if not isinstance(self.sensor_id, str):
             raise ValueError("Sensor ID must be a string")
```

This was one of the first things I did when that data disaster happened. The `_ensure_invariants` method is called at the end of the `__init__` ensuring that our invariants are valid on object initialization. After that initial fix we moved that `_ensure_invariants` method call to other places especially in setter methods. Let me show you.

```python
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        if not isinstance(new_value, (int, float)):
            raise ValueError("Value must be a number")
        if new_value < 0:
            raise ValueError("Value cannot be negative")
        self._value = new_value
        self._ensure_invariants()
```

This snippet demonstrates a `value` property with a setter and you see `_ensure_invariants` at the end of the setter. This ensures that whenever a value attribute changes we are always in a valid state. Now this was a good start.  We then had a bunch of other classes as well that represented different concepts of the system and we did the same thing all over the place.  You get the idea.  The key is to place invariant checks in the right places namely on object creation in the init function and any method where the state could be changed.

Now you might think this is a bit tedious having to write these `if` statements all over the place I know I did. And yeah it is a bit repetitive so we moved it a step forward. We had a function to validate type check on values to make sure we were consistent and reusable throughout the codebase. Something like this:

```python

def check_type_value(value, expected_type, condition_func = lambda x: True, error_message=None):
     if not isinstance(value, expected_type):
           raise ValueError(f"Expected type {expected_type} but got {type(value)}")
     if not condition_func(value):
         error_message_to_use = error_message or f"value does not match a valid state"
         raise ValueError(error_message_to_use)
     return True


class DataPointAdvanced:
    def __init__(self, timestamp, value, sensor_id):
      check_type_value(timestamp, datetime.datetime)
      check_type_value(value, (int, float), lambda x: x >= 0 , "Value cannot be negative")
      check_type_value(sensor_id, str)
      self.timestamp = timestamp
      self._value = value
      self.sensor_id = sensor_id

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
         check_type_value(new_value, (int, float), lambda x: x >= 0 , "Value cannot be negative")
         self._value = new_value
```

So here we extracted the logic to its own function. I know its not perfect but it's something that evolved over time. It made it slightly less tedious and more consistent to use. You see how the constructor uses the `check_type_value` function and also the setter. The lambda allows to add more conditional checks not just type checks and also a way to add a more explanatory error message if needed.

Now here is something funny not really funny like haha funny but funny like weird in hindsight. You see back in my early days I actually thought that it was faster to skip those kind of checks because they add a small overhead. I was chasing micro-optimizations and you know what they say about premature optimization. Yeah it bit me hard. The overhead of the checks are completely negligible compared to the cost of dealing with bad data. So that's a lesson that will always stick to me.

Another thing that is good to keep in mind is that you should write tests to assert that the invariants are being held in all the scenarios. So if you are mutating the value you test that they pass after that specific operation to make sure that you don't have a regression. This becomes crucial when the system grows and has multiple developers working on the codebase.

Now If you're looking for more resources on this topic you should check out some books. "Design by Contract" by Bertrand Meyer goes deep into the concept and theoretical aspects even if it focuses on Eiffel the concepts are transferrable. Also Martin Fowler's book "Refactoring" also has a lot of good insights and practical techniques that might help when developing these things. In this context also Robert Martin's “Clean Code” is useful to keep the codebase maintainable when we're applying these constraints everywhere.

So yeah class invariants in Python are a bit like a house built on a strong foundation. If you don't set those rules up early and rigorously things get messy and unstable real quick. It's not just about writing code it’s about making sure that your code does what it is supposed to do and remains in a consistent valid state. I've been there done that and I'm trying to save you from my mistakes and hopefully this has been helpful to you.
