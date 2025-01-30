---
title: "How do `sess.graph` and `tf.get_default_graph()` differ in TensorFlow?"
date: "2025-01-30"
id: "how-do-sessgraph-and-tfgetdefaultgraph-differ-in-tensorflow"
---
The fundamental distinction between `sess.graph` and `tf.get_default_graph()` in TensorFlow arises from their scope and lifecycle relative to a TensorFlow session. `tf.get_default_graph()` retrieves the *default* graph, a globally accessible structure managed by the TensorFlow runtime, whereas `sess.graph` accesses the specific graph *associated with a particular session object*. These are often the same graph, particularly in simple scripts, but it’s critical to understand their separation to handle more complex scenarios.

The default graph is essentially the workspace where TensorFlow operations and tensors are created when no explicit graph is specified. When you execute statements like `x = tf.constant(2)`, TensorFlow implicitly adds this operation to the default graph. This provides a convenient, implicit execution environment suitable for most introductory work and standalone scripts. However, as projects become more advanced and require multiple, isolated computational graphs or specialized session configuration, the single default graph becomes a limiting factor.

A TensorFlow `Session`, instantiated using `sess = tf.compat.v1.Session()`, provides an environment to execute the operations defined within a graph. When a `Session` object is created without an explicit graph passed to its constructor, it defaults to executing operations in the default graph. Critically, however, the `sess` object holds a reference (`sess.graph`) to the specific graph it's operating on. While this might initially be the default graph, it need not remain so.

This separation allows for more complex workflows, like training a model with one graph and simultaneously evaluating with another, or reusing parts of a model while training different branches. It also enables finer-grained control over resource allocation and memory management since each graph can potentially have a unique configuration. The `tf.Graph` object is the core structure; the methods `tf.get_default_graph()` and `sess.graph` are simply different access mechanisms for these graph objects.

Here are three code examples to illustrate these differences and practical scenarios where they become relevant:

**Example 1: Implicit Use of Default Graph**

```python
import tensorflow as tf

# Implicit graph creation using default graph
x = tf.constant(2)
y = tf.constant(3)
z = tf.add(x, y)

# Create a session; it implicitly uses default graph
sess = tf.compat.v1.Session()

# Verify sess.graph and tf.get_default_graph point to the same graph
print(sess.graph is tf.get_default_graph()) # Output: True

result = sess.run(z)
print(result)  # Output: 5

sess.close()
```

In this first example, I create operations without explicitly constructing a graph, relying on the default. When the session is created (`sess = tf.compat.v1.Session()`), it defaults to executing within the same default graph that the operations `x`, `y`, and `z` belong to. The comparison `sess.graph is tf.get_default_graph()` demonstrates that initially they point to the same graph object. The session executes the `z` operation, resulting in the output 5. This is the most common scenario in introductory code.

**Example 2: Explicit Graph Creation and Session Assignment**

```python
import tensorflow as tf

# Create a new explicit graph
graph1 = tf.Graph()

# Create operations within graph1
with graph1.as_default():
    x1 = tf.constant(2)
    y1 = tf.constant(3)
    z1 = tf.add(x1, y1)

# Create another new explicit graph
graph2 = tf.Graph()

# Create operations within graph2
with graph2.as_default():
    x2 = tf.constant(5)
    y2 = tf.constant(10)
    z2 = tf.multiply(x2, y2)

# Create sessions associated with the explicit graphs
sess1 = tf.compat.v1.Session(graph=graph1)
sess2 = tf.compat.v1.Session(graph=graph2)

# Verify each session accesses its corresponding graph
print(sess1.graph is tf.get_default_graph()) # Output: False - Sess1 does not use default graph.
print(sess2.graph is tf.get_default_graph()) # Output: False - Sess2 does not use default graph.

print(sess1.graph is graph1) # Output: True
print(sess2.graph is graph2) # Output: True

result1 = sess1.run(z1)
result2 = sess2.run(z2)

print(result1) # Output: 5
print(result2) # Output: 50

sess1.close()
sess2.close()

# Verify default graph still exists separately.
with tf.get_default_graph().as_default():
  x3 = tf.constant(15)
  y3 = tf.constant(5)
  z3 = tf.add(x3,y3)
  sess3 = tf.compat.v1.Session()
  result3 = sess3.run(z3)
  print(result3) # Output: 20
  sess3.close()
```

This example demonstrates how distinct `tf.Graph` objects can be created and assigned to separate sessions. This approach is useful when wanting to maintain a completely isolated context for your operations.  I create two graphs, `graph1` and `graph2`, and explicitly add operations within each of them. I then create two sessions, `sess1` and `sess2`, and explicitly associate them with their respective graphs, using the `graph` parameter of the `Session` constructor. This clearly shows that neither session operates in the default graph. After performing operations using both sessions, the code demonstrates that the default graph is independent. It creates `x3`, `y3`, `z3` in the default graph and utilizes a session to retrieve the result. This isolates the graphs further and provides a more complex demonstration of the utility of working outside the default graph. This methodology provides maximum flexibility and control of tensor operations and resource use.

**Example 3: Sharing Operations Across Sessions, Using The Default Graph As a Common Graph**

```python
import tensorflow as tf

# Create shared ops inside default graph
with tf.compat.v1.variable_scope("shared_scope"):
    var1 = tf.compat.v1.get_variable("shared_var", initializer = 10.0)
    add_op = tf.compat.v1.assign_add(var1,5.0)

# Create two sessions and execute an operation that references ops defined in the default graph.
sess1 = tf.compat.v1.Session()
sess2 = tf.compat.v1.Session()

# Initialize the variables for both sessions.
sess1.run(tf.compat.v1.global_variables_initializer())
sess2.run(tf.compat.v1.global_variables_initializer())

# Output initial value
print("Initial value:", sess1.run(var1))  # Output: Initial value: 10.0

# Execute add op in both sessions separately.
sess1.run(add_op)
print("Session1 Value:",sess1.run(var1)) # Output: Session1 Value: 15.0
sess2.run(add_op)
print("Session2 Value:", sess2.run(var1))  # Output: Session2 Value: 15.0

# Verify both sessions are accessing the same variable.
print(sess1.run(var1) == sess2.run(var1)) # Output: True

sess1.close()
sess2.close()
```
Here, I show how operations created in the default graph can be shared and manipulated between multiple sessions, particularly when variables are involved. I utilize a variable scope which is a way to group and manage variables. A variable `var1` is created using `tf.compat.v1.get_variable()` and an operation to add to it is defined using `tf.compat.v1.assign_add()`. Both sessions are initialized via the call `tf.compat.v1.global_variables_initializer()` and both sessions can access and manipulate `var1`. The code verifies that both sessions are indeed operating on the same shared variable, demonstrating that when sessions are created without specifying a graph, they default to operating within the same shared, default graph. This approach is common when building models where training and validation can occur within separate sessions, but should reference the same set of trained parameters.

These examples show that, though the default graph and session-specific graphs may point to the same graph initially, it is always safest to utilize `sess.graph` for operations within a session. This reduces ambiguity and makes it clear which graph is being operated on. The default graph is a convenient mechanism for simple scripts; in more complex situations, however, it's far more useful to control graph creation and session assignment directly.

For further study, consult the TensorFlow documentation under the following topics: "Graphs and Sessions," "Creating and Using Graphs," and "Managing Shared Variables." Exploring the source code for `tf.Graph`, `tf.compat.v1.Session` and `tf.get_default_graph` can also give deeper insights. Additionally, the 'TensorFlow Mechanics' and 'Effective TensorFlow' books can provide detailed explanations and practical examples.
