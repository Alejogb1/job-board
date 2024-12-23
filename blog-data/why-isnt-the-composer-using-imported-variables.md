---
title: "Why isn't the composer using imported variables?"
date: "2024-12-23"
id: "why-isnt-the-composer-using-imported-variables"
---

Alright, let's tackle this. It's a question I’ve seen crop up more than once in my career, particularly when teams are transitioning from simpler scripting to more complex component-based architectures. It's not a straightforward "yes" or "no" answer; it hinges on fundamental concepts of encapsulation, modularity, and dependency management. So, let's delve into why, in many cases, a composer (whether it's a dependency injection container in a language like PHP, an Angular component, or a similar construct in another framework) shouldn't directly access variables that are imported, or more broadly, variables defined outside of its immediate scope.

The crux of the issue revolves around the principles of **separation of concerns** and **explicit dependencies**. Imagine a scenario I encountered a few years back: we had a team developing a large e-commerce platform. Initially, everything was fine. We used globally defined configuration variables, accessible anywhere. This seemed convenient initially. However, as we added features and more developers joined, the system turned into a spaghetti bowl. Modifying a configuration setting in one place had unpredictable ripple effects elsewhere. This was primarily because components were implicitly relying on globally accessible variables, violating the core principles of modular software design. The composer, in its idealized role, is meant to orchestrate the instantiation and configuration of dependencies, not to act as another consumer of potentially volatile global state.

Think of it this way: a well-designed component, instantiated and configured by the composer, shouldn't care *where* a piece of data comes from. It should only care that it receives what it *expects*, through well-defined interfaces or constructor parameters. When a composer starts directly accessing imported variables, it's essentially creating **implicit dependencies**. These dependencies are harder to trace, harder to reason about, and make the entire application more brittle.

Why is this so? Because if a composer references, say, a `global_settings` object directly, it means that anywhere this composer is used *must also* have access to that `global_settings` object in precisely the same way. If the structure or availability of this object changes, it can break all the components the composer is managing. Furthermore, it becomes harder to test the individual components because their behaviour now is tangled with the presence (or absence) of some outside entity.

Let me illustrate this with some code snippets, showcasing how *not* to do it and what the preferred alternative looks like.

**Snippet 1: The Incorrect Approach (Illustrative, may not be runnable in all environments)**

```python
# config.py
GLOBAL_SETTINGS = {
    "database_url": "some_url",
    "api_key": "some_key"
}

# my_component.py
class MyComponent:
    def __init__(self, data_access_object):
        self.data_access = data_access_object

    def process(self, input):
        print(f"Processing: {input}, using database at {GLOBAL_SETTINGS['database_url']}")
        #do something with data_access


# composer.py
import config
from my_component import MyComponent

def create_my_component(data_access_factory): #pretend this is an actual dependency injection container
    data_access_object = data_access_factory(config.GLOBAL_SETTINGS)
    return MyComponent(data_access_object)

# application.py
from composer import create_my_component

def make_db_access(settings):
    #imagine actual database instantiation and object
    return settings['database_url']

component_instance = create_my_component(make_db_access)

component_instance.process("some_data")
```

In this extremely simplified Python example, the `create_my_component` function in `composer.py` is directly accessing the `config.GLOBAL_SETTINGS`. This immediately creates a hidden dependency between our composer (which, in this case, is really just a single function) and the `config` module. If the `config` module changes structure, or is not available, the `create_my_component` function (and by extension, the entire application) would break. This is despite our intention for `MyComponent` only to depend on a data access object; it now implicitly depends on the format of `GLOBAL_SETTINGS`.

**Snippet 2: The Correct Approach (Explicit Dependencies)**

```python
# config.py
GLOBAL_SETTINGS = {
    "database_url": "some_url",
    "api_key": "some_key"
}


# my_component.py
class MyComponent:
    def __init__(self, data_access_object, db_url):
        self.data_access = data_access_object
        self.db_url = db_url #now the database url is passed in explicitly


    def process(self, input):
        print(f"Processing: {input}, using database at {self.db_url}")
        #do something with data_access

# composer.py
import config
from my_component import MyComponent

def create_my_component(data_access_factory, settings): #notice that settings is an explicit parameter
    data_access_object = data_access_factory(settings) #data access object factory gets full settings
    return MyComponent(data_access_object, settings['database_url']) #the settings database_url is passed explicitly to my component


# application.py
from composer import create_my_component

def make_db_access(settings):
    #imagine actual database instantiation and object
    return "data_access_object" #simplified placeholder

component_instance = create_my_component(make_db_access, config.GLOBAL_SETTINGS)

component_instance.process("some_data")

```

In this revised example, the composer now accepts `settings` as a parameter, making it clear that it depends on configuration. The `db_url` is passed into MyComponent in its constructor. This makes it a clearly defined, explicit dependency, making it much easier to debug, refactor, and reuse components. The dependency is now injected, rather than relying on a global import. The composer is no longer coupled to specific global variables.

**Snippet 3: Refactor with an abstract data source**

```python
# config.py
GLOBAL_SETTINGS = {
    "database": {
      "url": "some_url",
       "user": "some_user",
       "pass": "some_pass"
     },
     "api_key": "some_key"
}

# interfaces.py
from abc import ABC, abstractmethod

class DataAccess(ABC):
    @abstractmethod
    def query(self, query):
        pass


# database.py
from interfaces import DataAccess

class DatabaseAccess(DataAccess):
    def __init__(self, db_url):
        self.db_url = db_url

    def query(self, query):
        print(f"Querying {self.db_url} with {query}")
        return f"results for {query}"



# my_component.py
from interfaces import DataAccess
class MyComponent:
    def __init__(self, data_access_object: DataAccess):
        self.data_access = data_access_object

    def process(self, input):
        results = self.data_access.query(f"SELECT * FROM {input}")
        print(f"Processing: {input}, results: {results}")

# composer.py
import config
from my_component import MyComponent
from database import DatabaseAccess

def create_my_component(settings): #notice that the factory now creates a data source based on settings
    database_access = DatabaseAccess(settings['database']['url'])
    return MyComponent(database_access)


# application.py
from composer import create_my_component

component_instance = create_my_component(config.GLOBAL_SETTINGS)

component_instance.process("some_data")
```

This third example takes it further by using an abstract `DataAccess` interface, and a concrete `DatabaseAccess` implementation. The composer now instantiates the `DatabaseAccess` object from the configuration settings, making it completely flexible to use different implementations by supplying different factories, while `MyComponent` only depends on the interface. This pattern promotes loose coupling, and makes unit testing significantly easier. We now have a fully dependency-injected component.

The practical takeaway here is that while a composer *could* technically access global variables, doing so leads to brittle and hard-to-maintain code. The composer's primary responsibility is to construct objects and inject dependencies, not to directly consume global configuration data. It should receive the necessary parameters (or factories) through its own constructor or function parameters and pass those through to the objects it creates.

To delve deeper into these concepts, I recommend studying Martin Fowler’s work on dependency injection and inversion of control, specifically his articles on the topic, and the classic “Design Patterns: Elements of Reusable Object-Oriented Software” by the Gang of Four, which provides a strong foundation on design principles. For a practical exploration, I’d recommend the *Dependency Injection in .NET* book by Mark Seemann, even if you don’t use .NET, the core principles are universal.
