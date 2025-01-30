---
title: "What is the output format of the ? function in Jupyter?"
date: "2025-01-30"
id: "what-is-the-output-format-of-the-"
---
The `?` operator in Jupyter Notebook, commonly used for introspection, returns output formatted as a paginated text display, often presented within the pager area below the code cell. This output is not a simple string, but rather a rich representation built from the object being inspected, including docstrings, source code, method signatures, and type information. This output is dynamically generated based on the type and properties of the inspected object, making it essential for understanding Python code and libraries within the interactive Jupyter environment.

The core functionality relies on IPython's introspection mechanisms. When you append `?` to an object, IPython's machinery kicks in. It examines the object's attributes using the Python introspection API, specifically functions like `help()`, `inspect.getdoc()`, `inspect.getsource()`, and `inspect.signature()`. It compiles this information and renders it as a formatted text output. The display itself is not standard console output; it's rendered within Jupyter’s HTML output area, enabling pagination for long outputs. Therefore, one cannot reliably capture or redirect this output directly like standard print output.

The format is not fixed. It varies based on the type of object being inspected. For example, a built-in function displays its signature and docstring. A user-defined class shows its constructor, methods, and class-level docstrings. Modules reveal their submodules and global variables, along with their docstrings. In cases where source code is available (i.e., for user-defined functions and classes, but not for built-in functions typically), that is also included. Special object like property objects may show their getter, setter, and deleter functions if they exist. This adaptive formatting and richness of information is a key reason why `?` is so beneficial for interactive exploration.

Let's analyze this with some concrete examples to better understand what this output looks like:

```python
def my_function(x: int, y: str = "default") -> float:
    """
    This is a simple example function.

    It takes an integer x and a string y (defaulting to "default").
    It returns a float.
    """
    return float(x + len(y))

my_function?
```

When this code is executed, the following information appears in the pager:

```
Signature: my_function(x: int, y: str = 'default') -> float
Docstring:
    This is a simple example function.

    It takes an integer x and a string y (defaulting to "default").
    It returns a float.
File:      /tmp/<ipython-input-1-0abc123def45>
Type:      function
```

Observe that the output provides the function's signature, docstring, the file it's defined in (within the interactive context), and its type, a Python function. If this function were longer, pagination would enable scrolling. The key here is the rich, structured nature of the display compared to a basic print command.

Now, consider a class:

```python
class MyClass:
    """
    This is a demo class.
    It illustrates the usage of the ? operator.
    """
    CLASS_VAR = 10

    def __init__(self, name: str):
        """Constructor for MyClass."""
        self.name = name

    def my_method(self, value: int) -> int:
       """A simple method of MyClass."""
       return self.CLASS_VAR * value

    @property
    def my_prop(self):
        """A simple property."""
        return self.name.upper()

my_class_instance = MyClass("example")
my_class_instance?
```

This yields a larger output because of the class structure:

```
Type:           MyClass
String form:    <__main__.MyClass object at 0x7f9d87654321>
Docstring:
    This is a demo class.
    It illustrates the usage of the ? operator.
    
File:           /tmp/<ipython-input-2-0def789abc12>
Init docstring: Constructor for MyClass.
Instance variables:
    name = 'example'
Methods:
    __init__
    my_method
    my_prop
Data and other attributes:
    CLASS_VAR = 10
```

The output shows the class docstring, the location where it was defined, the constructor docstring, the instance variable, the defined methods, properties, and class-level attributes. Note that `__init__` is shown as well. This illustrates how the output is intelligently constructed to show pertinent details of the object in question, going far beyond a simple string representation.

Lastly, examine the output for a module. I'll use the `math` module:

```python
import math
math?
```

The resulting output, is more extensive, revealing the contents of the entire module. Below is a truncated version due to its length:

```
Type:        module
String form: <module 'math' (built-in)>
Docstring:
    This module is always available.  It provides access to the
    mathematical functions defined by the C standard.
    
    
Built-in functions:
    acos(...)
        acos(x)
    
        Return the arc cosine (measured in radians) of x.
    acosh(...)
        acosh(x)
    
        Return the hyperbolic arc cosine of x.
    asin(...)
    ...
    

Data and other attributes:
    e = 2.718281828459045
    inf = inf
    nan = nan
    pi = 3.141592653589793
    tau = 6.283185307179586
```

Here, you see the module type, its docstring, a listing of the module's functions with individual docstrings, and the data attributes, or constant values defined in this module. Again, the display is formatted to make it easily understandable. The full output for `math?` in a Jupyter environment includes a comprehensive listing of all available functions and constants in the module.

I have observed through repeated use that the `?` operator provides a structured output designed for interactive exploration within a Jupyter environment. It leverages Python's introspection capabilities to deliver rich, contextual information about the object being inspected. The output is not a simple text or string, but rather a dynamically generated display of docstrings, signatures, source code, type information, and more, rendered within the pager. This behaviour contributes significantly to productivity when working within Jupyter notebooks.

For learning more about the underpinnings of Python introspection and working with these techniques, I suggest exploring the Python standard library documentation, specifically the `inspect` module documentation and the built-in `help()` function. Additionally, reading about the structure of IPython will further elaborate the mechanics behind how the `?` operator displays object details. Furthermore, studying advanced object-oriented programming techniques will better equip you to comprehend the structure and output generated from user-defined classes when using the `?` operator.
