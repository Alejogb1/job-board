---
title: "Why is Python 3.6 unsupported by Cloud ML Engine trainers?"
date: "2025-01-30"
id: "why-is-python-36-unsupported-by-cloud-ml"
---
Python 3.6 reached its end-of-life on December 23, 2021, a critical date influencing the support policy of managed services like Cloud ML Engine trainers. This retirement directly impacts Google’s commitment to providing secure and stable environments; maintaining support for deprecated language versions becomes an increasing security and resource burden, hence its absence from officially supported configurations. My experience, building and deploying machine learning models on Cloud ML Engine over several years, consistently revealed the platform’s adherence to current and secure technology stacks. Legacy versions become a challenge to maintain.

The core reason behind the lack of support revolves around the operational overhead and security risks associated with maintaining outdated runtimes. Google Cloud services, including Cloud ML Engine trainers, are constructed upon a foundation of regularly updated infrastructure. This involves not only the underlying operating systems but also the Python interpreters themselves, together with their dependency ecosystems. When a version of Python reaches its end-of-life, it no longer receives security patches, critical bug fixes, or feature updates from the Python core team. This immediately introduces potential vulnerabilities that could be exploited in a cloud environment.

Furthermore, continued support for Python 3.6 would require Google to expend development resources to maintain the necessary compatibility layers and security protocols. These efforts are better spent focusing on newer versions of Python, versions that actively contribute to the performance and security of the Cloud ML Engine. The goal is not just to provide a functional environment, but also a secure, up-to-date, and scalable one. Cloud ML Engine’s training infrastructure needs to be hardened against potential exploits, and continuing to support outdated runtimes would actively work against that objective. In my own project migrations, the move to Python 3.8 and then 3.9 brought immediate performance benefits, particularly when combined with newer versions of dependencies like TensorFlow and scikit-learn; this highlighted the importance of the platform aligning with contemporary tools.

The move also simplifies the support structure. By focusing on a reduced set of currently maintained Python versions, Google can provide more effective and consistent customer service. When support requests arrive, issues arising from outdated language versions can be immediately ruled out, streamlining troubleshooting and incident resolution. This ensures a more reliable and predictable user experience, which is essential for large-scale machine learning projects. It also enables Google to optimize its infrastructure resources more effectively; less time is spent troubleshooting legacy version specific issues, and more can be invested in enhancements and innovation. From experience, debugging issues across multiple Python versions and their compatibility with different versions of TensorFlow and other ML libraries can consume substantial time. Aligning with supported runtimes avoids these unnecessary challenges.

Here are a few code examples, illustrative of the kinds of changes one would encounter when migrating away from Python 3.6 for use on Cloud ML Engine, and showcasing why updates are essential.

**Example 1: Type Hinting**

```python
# Python 3.6 (and earlier) - lacks postponed evaluation of annotations.
# from typing import List  #No longer necessary for 3.7+

def process_data(data_list: List[int]) -> List[float]:
    processed = [float(x) for x in data_list]
    return processed

# Python 3.7+ (postponed evaluation allows for forward references and eliminates the need for a lot of imports at the top of the module)
def process_data(data_list: list[int]) -> list[float]:
    processed = [float(x) for x in data_list]
    return processed
```

*Commentary*: Type hinting was vastly improved starting in Python 3.7. The `typing` module is now optional for many type hints due to postponed evaluation. Python 3.6 required explicit imports from `typing` for collection-based type hinting and also could not directly reference types that are defined later in the file. These improvements significantly improved code clarity and maintainability. My own experience moving from 3.6 to higher versions greatly reduced the boilerplate code needed for complex typing, leading to a more readable codebase. Cloud ML Engine favors these readability improvements due to the collaborative nature of many ML projects.

**Example 2: f-strings**

```python
# Python 3.6
name = "Alice"
age = 30
print("User: %s, Age: %d" % (name, age))

# Python 3.7+ (f-strings)
name = "Alice"
age = 30
print(f"User: {name}, Age: {age}")
```

*Commentary*: F-strings, introduced in Python 3.6, became a standard way of formatting strings in Python 3.7 and later. While both forms function, f-strings provide a more readable and concise way of string interpolation. Before f-strings, developers used cumbersome methods like `%` formatting or `.format()`, both of which are prone to errors and less readable. The direct variable interpolation available with f-strings enhances developer productivity; this, in turn, is a driving factor in adopting new language features. During development cycles, even small improvements like this add up to better developer velocity, which translates to faster releases of ML models in the cloud environment.

**Example 3: Data Class**

```python
# Python 3.6 (traditional class definition)
class UserData:
    def __init__(self, name, age, email):
        self.name = name
        self.age = age
        self.email = email

# Python 3.7+ (data classes)
from dataclasses import dataclass

@dataclass
class UserData:
    name: str
    age: int
    email: str
```

*Commentary:* Data classes, introduced in Python 3.7, provided a more concise way to define classes that primarily hold data. They handle a large amount of boilerplate associated with initialization and representation. While using them does require importing `dataclass`, this is preferable to needing to write repetitive `__init__` methods. Data classes significantly simplify the creation and maintenance of data-centric classes, common across many aspects of ML pipelines – from loading data to defining feature structures. This reduction in boilerplate is highly desirable for development productivity and maintainability, which directly benefits larger and collaborative cloud ML projects. Data classes also automatically provide some common methods such as `__repr__` which further enhances developer productivity.

To summarize, the reason for the absence of Python 3.6 support in Cloud ML Engine is not arbitrary; it is a necessary measure to ensure the security, reliability, and performance of the platform. Maintaining backward compatibility with outdated runtimes introduces significant overhead, exposes security vulnerabilities, and diverts resources from the development of newer, more secure, and efficient technology. The benefits of adopting newer versions of Python, such as improved performance, enhanced features, and increased security, outweigh the costs of migration, making this a worthwhile investment in the long-term health of the service and users’ projects.

For developers seeking resources to navigate the migration to newer versions of Python, I recommend focusing on several excellent, and freely available, sources. The official Python documentation offers comprehensive guides for upgrading to each new version of Python. Additionally, books and online courses from reputable sources, and particularly those focused on Python 3.7 and higher, are a good starting point. Reviewing changelogs for new Python releases, found readily online, will provide specific details of features and capabilities. Lastly, consulting style guides that incorporate best practices for modern Python development will be very useful for long term maintainability. These strategies have served me well in my own projects and I have found them invaluable for the migration of legacy systems to modern setups.
