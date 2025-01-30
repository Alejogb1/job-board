---
title: "Why is Keras RL throwing a 'not implemented' error after overriding a class method?"
date: "2025-01-30"
id: "why-is-keras-rl-throwing-a-not-implemented"
---
The root cause of a "NotImplementedError" in Keras-RL after overriding a class method, specifically when interacting with its Agent classes, typically stems from a misunderstanding of the framework's internal dispatch mechanisms and the roles of abstract methods. Keras-RL Agents, such as DQNAgent or PolicyGradientAgent, are designed around a structured inheritance model with several key methods, many of which are intended to be overridden by the user when customizing the agent's behavior. The error surfaces when these core abstract methods, particularly those involved in core agent steps, are not completely implemented or fail to retain their intended signature, even with seemingly appropriate overrides.

The key is that Keras-RL is not just looking for a method of a given name on your overridden class. It often calls these methods via an internal, dynamic dispatch process. This process relies on the framework's internal understanding of the method's *type*, not just its name. Simply creating a method with the correct name and argument structure might not be enough if it doesn’t seamlessly integrate with the expected behavior of the parent class’s abstract method that you’ve implicitly replaced. The error therefore often doesn’t manifest during compilation but instead during a runtime call to the method. The dispatch mechanism attempts to find the appropriate implementation for the method via the class inheritance tree. If any component along this chain isn’t properly set up or doesn't meet the type expectations of Keras-RL's dispatch, a `NotImplementedError` is raised since it fails to find a concrete implementation that matches its needs. The situation is especially acute when methods meant to be abstract are not implemented with methods that behave as the expected return and parameter types by the base classes.

The fundamental design of Keras-RL requires that agents perform several sequential actions to learn from experience and to act in the environment. This sequence is defined by the structure of the Agent classes and the methods used to drive the agent's interaction within the reinforcement learning framework. The typical methods of note include `compute_action`, `forward`, `backward`, and `fit` or `step`. The exact method throwing the "NotImplementedError" depends on the particular agent class, but the underlying issue lies in Keras-RL's attempt to call a concrete method where the hierarchy doesn’t present a suitable replacement.

Consider, for instance, an agent needing to compute an action, and we choose to override the `compute_action` method of an agent. Below are three common scenarios where this can go wrong.

**Code Example 1: Incorrect Method Signature**

```python
from keras_rl.core import Agent
import numpy as np


class MyCustomAgent(Agent):
    def __init__(self, *args, **kwargs):
      super(MyCustomAgent, self).__init__(*args, **kwargs)

    # Incorrect override - missing state parameter
    def compute_action(self, *args):
        # Simplified - should incorporate actual logic
        return np.random.randint(0, 2)


# Intended use would usually require specifying further details for env.
# However, for demonstration purposes this works fine
agent = MyCustomAgent()
state = np.array([1, 2, 3])
try:
    action = agent.compute_action(state)
    print(f"Action: {action}")
except NotImplementedError as e:
    print(f"Error: {e}")
```

**Commentary:** In this example, the `compute_action` method is overridden in `MyCustomAgent`, but it does not adhere to the expected signature defined in the parent `Agent` class, which expects at least the `state` argument and often others depending on the agent subclass. In this instance we are missing any state argument, but the method itself still executes without error since the state parameter is optional via python's `*args`. However, when called internally by another method in the class, this *lack* of specific adherence to the expected method signature causes Keras-RL to raise the "NotImplementedError" because it cannot find a suitable `compute_action` method, despite the name being correct. The `Agent` class expects a method with a minimum parameter for state and a return value.

**Code Example 2: Missing Implementation (or Partial Implementation)**

```python
from keras_rl.core import Agent
import numpy as np

class MyCustomAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(MyCustomAgent, self).__init__(*args, **kwargs)

    # Missing crucial functionality
    def forward(self, state):
        pass # Deliberately left empty

    def compute_action(self, state):
       # This method is implemented and can be called successfully
       return np.random.randint(0, 2)

    def step(self, reward, observation):
       # This method would be called indirectly by fit later.
       # Since forward method is empty, this fails later
       pass

# Example initialization without an environment.
agent = MyCustomAgent()
state = np.array([1, 2, 3])

try:
    action = agent.compute_action(state)
    print(f"Action: {action}")
    # Trigger the problem - the call to fit will internally call forward.
    # the missing implementation results in an error
    agent.fit(state)
except NotImplementedError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Other error: {e}")
```

**Commentary:** Here, `compute_action` is correctly implemented, and we can call it successfully at first. However, the `forward` method, which is implicitly called within the Keras-RL pipeline, is intentionally left unimplemented. The problem here is not immediately apparent since `compute_action` and `fit` will execute when called directly. But the internal call to `forward` from `fit` causes the framework to throw the "NotImplementedError" during the fit method call. This illustrates that the overridden method isn't directly throwing the error itself but that the missing internal implementation leads to failure of an expected call to the forward method.

**Code Example 3: Incorrect Return Type**

```python
from keras_rl.core import Agent
import numpy as np

class MyCustomAgent(Agent):
    def __init__(self, *args, **kwargs):
      super(MyCustomAgent, self).__init__(*args, **kwargs)

    # Incorrect return type, should be np.array or int
    def compute_action(self, state):
        return [np.random.randint(0, 2)]

    def step(self, reward, observation):
       pass

# Example initialization.
agent = MyCustomAgent()
state = np.array([1, 2, 3])

try:
    action = agent.compute_action(state)
    print(f"Action: {action}")
    agent.fit(state)
except NotImplementedError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Other error: {e}")
```

**Commentary:** In this scenario, the method signatures appear to be correct in the method override. However, `compute_action` returns a Python list, not an integer or a NumPy array as expected by the Keras-RL internal dispatcher. The method executes without error, but the type mismatch is not noticed until later, during the internal execution, causing a "NotImplementedError" when Keras-RL attempts to further process the returned action using an operation incompatible with a list. Thus, the error isn't caused directly by the method itself, but by the internal call to the return value of that method during further processing.

To address these issues, several steps should be taken. First, review the documentation and source code of the specific Keras-RL Agent class that you're inheriting from. Identify the required parameters, return types, and the core functionality of the methods you intend to override. Ensure that your overridden methods adhere to these requirements with rigorous parameter checking, internal type checking, and a good debugging strategy. Second, whenever you implement or override an abstract method, ensure you have a good understanding of the intended purpose of the method and what happens to the returned value. Third, use a structured approach to debugging. When you see `NotImplementedError` during runtime, use print statements to isolate where the error is coming from. You can also use a debugging tool (such as the pdb python module) to step through the execution of the program and observe when the error is thrown.

For more detailed guidance, consult resources providing in-depth explanations of reinforcement learning principles and Keras-RL itself. The official Keras documentation is also helpful for understanding the underlying layer implementations. Additionally, numerous books and online resources discuss practical implementations of reinforcement learning that may prove useful in troubleshooting Keras-RL issues. Finally, the Keras-RL Github repository's issue tracker may offer solutions to common error cases.
