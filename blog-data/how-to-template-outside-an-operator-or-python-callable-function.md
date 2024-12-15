---
title: "How to template outside an operator or python callable function?"
date: "2024-12-15"
id: "how-to-template-outside-an-operator-or-python-callable-function"
---

alright, so you're asking about templating outside the usual suspects in python, specifically not within a function or a class method, that's a pretty interesting problem, and i've definitely been there, many times. it's not something that comes up in beginner tutorials, that's for sure.

the classic case of templating, the one everyone learns first, involves using f-strings or `.format()` inside function definitions. it’s straightforward, variables come in, string goes out. but when you want to template something globally, outside the scope of any callable, things get less obvious, and the standard tools don’t quite cut it. it’s like trying to use a screwdriver to hammer a nail. it can be done, but not efficiently.

i remember one time, back in my early days when i was working on a clunky server configuration script, i needed to dynamically generate config files, but the logic to do so was far too complex to fit in just one function. i had configuration settings scattered across multiple files and wanted to use variable substitution to handle server specifics. i tried just dropping f-strings everywhere in the global scope, obviously that didn't work. that was a facepalm moment. i quickly learned that f-strings are evaluated at definition time, not at runtime like in php or javascript.

the challenge is python's evaluation order and how it deals with global scopes. when a python script is compiled, the global variables are evaluated, and the literal values that appear in those assignments are set in stone, they are not re-evaluated later. you can’t just put an `f'{some_var}'` out there at the top of the script and expect it to use the runtime value of `some_var`, because the f-string evaluates when python reads the file not during the execution.

so, what are the options? well, there are several different approaches, and which one is “best” really depends on your specific requirements. let's go through a few i've used in the past, and i think should be applicable for you:

**1. delayed evaluation using lambdas**

this is a simple and clean method i’ve found quite useful, it involves using lambda functions, yes, lambdas! these are anonymous functions that can capture the environment in which they’re defined.

```python
import datetime

some_var = "initial"
timestamp_var = lambda: datetime.datetime.now().isoformat()

template_string = lambda: f"variable value: {some_var}, time: {timestamp_var()}"

#later, during runtime
some_var = "new value"

print(template_string())
print(template_string()) #will have different timestamps
```

in this example, `template_string` isn’t a string; it’s a lambda function that returns a templated string. it’s not evaluated when defined but when you call it like a normal function, this provides the delayed evaluation effect we are looking for.
when you change `some_var` and call `template_string()` later on, you get the updated value. `timestamp_var` is also a lambda to demonstrate that it works for functions too which is quite neat.

the downside is that you end up having to write `()` everywhere you want to access the templated string, which might be fine in some situations but can be annoying. you have to be explicit every time you want the string value. it makes the intent clearer that you're not just referring to a simple variable, but, it is still a slight inconvenience.

**2. using string formatting with a dictionary**

this is more like a classic templating way, very similar to what you find in django or jinja2. you prepare a template string with placeholders and then a dictionary with the actual values.

```python
import datetime

some_var = "original value"
template_string = "the variable is {some_var} and current time is {timestamp}"

def generate_template():
    timestamp = datetime.datetime.now().isoformat()
    template_data = {
        "some_var": some_var,
        "timestamp": timestamp
    }
    return template_string.format(**template_data)

#later, during runtime
some_var = "changed value"
print(generate_template())
print(generate_template()) #will have different timestamps
```

here, `template_string` is a regular string but we're using `.format` with a dictionary to plug the values in the placeholders. `generate_template` function takes care of generating the values dynamically, this has the advantage that you have a function to control how you actually update the variables which can be useful in some complex scenarios.

it is explicit, it’s fairly easy to read, and it decouples the template from the data. the issue is that you are required to manage a function and you cannot use variables directly. it is a trade off of functionality and readability.

**3. template class**

this is my more advanced approach when things get really complex, you could create a templating class, which holds the template string and provides a method to render the string with variable substitution. it can handle complex scenarios such as optional values and nested variables. it’s overkill for simple scenarios but it scales very well.

```python
import datetime

class StringTemplate:
    def __init__(self, template_str):
        self.template_str = template_str
        self.variables = {}

    def set_variable(self, name, value):
        self.variables[name] = value

    def render(self):
        timestamp = datetime.datetime.now().isoformat()
        render_data = self.variables.copy()
        render_data["timestamp"] = timestamp
        return self.template_str.format(**render_data)


some_var = "initial value"
template = StringTemplate("the value is: {some_var}, time: {timestamp}")
template.set_variable("some_var",some_var)


#later, during runtime
some_var = "changed value"
template.set_variable("some_var",some_var)

print(template.render())
print(template.render()) #will have different timestamps
```

here `StringTemplate` class contains the logic for string templating, set variables dynamically and uses the render method to build the strings, i really like this as it encapsulates everything really neatly. and it does scale well to complicated scenarios.

so, which one is the way to go? it depends. for simple cases, lambdas are very quick to setup and they work very well for basic templating. for a bit more structured approach, the string formatting with a dictionary could be better. but, for complex requirements, a templating class is going to save you a lot of trouble in the long run. it's like choosing between a spanner and a socket set, it is based on what you are trying to achieve.

i wouldn’t recommend using `eval()` or `exec()` for templating, it's a security risk and makes your code hard to debug, never a good idea. it is the equivalent of trying to run a car without the wheels.

and here's a joke: why did the programmer quit his job? because he didn't get arrays!

for more detailed information on these concepts, i'd point you towards python language references; specifically "fluent python" by luciano ramalho, it goes into detail on language mechanics and also, if you want to dive a bit into the underpinnings of how templating engines work, then look into "compiler design in c" by allen i. holub, even though it’s c-based it touches the fundamental ideas. there is lots of good material out there, avoid blog posts unless it is something really specific. and always prefer to read the original documentation from python.org it always helps.

i hope this answer was helpful, and please feel free to ask if you have more questions.
