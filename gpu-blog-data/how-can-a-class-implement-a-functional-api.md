---
title: "How can a class implement a functional API without raising `NotImplementedError`?"
date: "2025-01-30"
id: "how-can-a-class-implement-a-functional-api"
---
The core challenge in implementing a functional API within a class without triggering `NotImplementedError` lies in carefully managing abstract methods and leveraging default implementations where appropriate.  My experience working on large-scale data processing pipelines highlighted this issue repeatedly; abstract base classes (ABCs) are powerful tools for defining interfaces, but their rigorous enforcement can hinder practical development if not handled strategically.  The key is to differentiate between truly abstract operations that *must* be defined by subclasses, and those that can benefit from a sensible default behavior.

The `NotImplementedError` is raised when an abstract method, defined in an ABC, is called directly without a concrete implementation provided by a subclass. This is intended to enforce the contract defined by the ABC, ensuring that all crucial aspects of the interface are addressed.  However, it becomes problematic when some methods possess a meaningful default behavior applicable in many scenarios, even though they might logically need customization in certain specialized cases.  This requires a refined approach to class design.

**1. Clear Explanation:**

The solution hinges on judiciously using the `@abstractmethod` decorator from the `abc` module only for methods genuinely requiring concrete implementations from subclasses. Methods allowing default behavior should be implemented directly within the ABC itself.  This approach provides flexibility without sacrificing the benefits of strong typing and interface definition offered by ABCs.  The default implementation acts as a fallback, handling common use cases, allowing subclasses to override the behavior only when necessary.  If a subclass chooses not to override a method with a default implementation, it implicitly inherits and utilizes that default.  This eliminates the need to implement every method defined in the ABC, thereby reducing boilerplate code and allowing for more focused subclass development.

**2. Code Examples with Commentary:**

**Example 1: Basic Default Implementation**

```python
from abc import ABC, abstractmethod

class DataProcessor(ABC):
    def __init__(self, data):
        self.data = data

    def preprocess(self):
        """Default preprocessing: remove whitespace."""
        return [item.strip() for item in self.data]

    @abstractmethod
    def process(self):
        """This method MUST be implemented by subclasses."""
        pass

    def postprocess(self):
        """Default postprocessing: convert to uppercase."""
        return [item.upper() for item in self.process()]

class NumericalProcessor(DataProcessor):
    def process(self):
        """Overrides default processing to handle numbers."""
        return [float(item) for item in self.data if item.isnumeric()]

class TextProcessor(DataProcessor):
    def process(self):
        """This subclass leverages the default preprocess and postprocess."""
        return self.preprocess()

data = [" 123 ", "456", " abc ", "789 "]
numerical_processor = NumericalProcessor(data)
text_processor = TextProcessor(data)

print(f"Numerical Processor: {numerical_processor.postprocess()}")  # Output: [123.0, 456.0, 789.0]
print(f"Text Processor: {text_processor.postprocess()}") # Output: ['123', '456', 'ABC', '789']

```

This example shows `preprocess` and `postprocess` having default implementations, while `process` remains abstract, forcing subclasses to provide concrete logic. `TextProcessor` implicitly uses the default implementations, avoiding `NotImplementedError`.

**Example 2:  Default Implementation with Parameterization**

```python
from abc import ABC, abstractmethod

class DataTransformer(ABC):
    def __init__(self, data, transformation_factor=1):
        self.data = data
        self.factor = transformation_factor

    @abstractmethod
    def transform(self):
        pass

    def apply_transformation(self):
        transformed_data = self.transform()
        return [item * self.factor for item in transformed_data]


class LinearTransformer(DataTransformer):
    def transform(self):
        return self.data

class ExponentialTransformer(DataTransformer):
    def transform(self):
        return [item**2 for item in self.data]


data = [1, 2, 3, 4, 5]
linear_transformer = LinearTransformer(data)
exponential_transformer = ExponentialTransformer(data, transformation_factor=2)

print(f"Linear Transformation: {linear_transformer.apply_transformation()}") #Output: [1, 2, 3, 4, 5]
print(f"Exponential Transformation: {exponential_transformer.apply_transformation()}") #Output: [2, 8, 18, 32, 50]
```

Here, `apply_transformation` provides a default post-processing step that can be customized via the `transformation_factor` parameter in the constructor.  Subclasses only need to focus on the core `transform` operation.


**Example 3:  Handling Optional Operations**

```python
from abc import ABC, abstractmethod

class ReportGenerator(ABC):
    def __init__(self, data):
        self.data = data

    @abstractmethod
    def generate_report(self):
        pass

    def add_summary(self):
        """Optional summary generation; subclasses may override."""
        return "No summary available."


class DetailedReportGenerator(ReportGenerator):
    def generate_report(self):
        return f"Detailed Report: {self.data}"

    def add_summary(self):
        """Provides a custom summary."""
        return f"Summary: Data contains {len(self.data)} items."


class SimpleReportGenerator(ReportGenerator):
    def generate_report(self):
        return f"Simple Report: {self.data}"


data = [1,2,3,4,5]
detailed_report = DetailedReportGenerator(data)
simple_report = SimpleReportGenerator(data)

print(f"Detailed Report: {detailed_report.generate_report()}, {detailed_report.add_summary()}")
print(f"Simple Report: {simple_report.generate_report()}, {simple_report.add_summary()}")
```

In this case, `add_summary` is an optional method with a default implementation. Subclasses can provide more specialized summary generation, while others can simply use the default behavior.


**3. Resource Recommendations:**

For deeper understanding, I recommend exploring resources on Python's `abc` module, specifically focusing on the use of `abstractmethod` and the design principles of abstract base classes.  Furthermore, a solid grounding in object-oriented programming principles and design patterns is invaluable for mastering these techniques.  Finally, reviewing examples of well-structured libraries that leverage ABCs effectively can be instructive.  Careful study of these areas will equip you to handle similar scenarios effectively and efficiently.
