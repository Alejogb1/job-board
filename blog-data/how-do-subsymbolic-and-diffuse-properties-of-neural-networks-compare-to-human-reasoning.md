---
title: "How do subsymbolic and diffuse properties of neural networks compare to human reasoning?"
date: "2024-12-11"
id: "how-do-subsymbolic-and-diffuse-properties-of-neural-networks-compare-to-human-reasoning"
---

Okay so you wanna know how these AI brains – neural networks – stack up against our own squishy ones when it comes to thinking right  It's a HUGE question  like comparing apples and spaceships  Apples are kinda understandable spaceships less so

Neural networks are all about numbers right  Lots and lots of them  They're subsymbolic meaning they don't work with symbols like words or concepts directly  Think of it like this your brain probably uses words to think about a cat  a neural network uses a huge matrix of numbers  that somehow represents a cat  It's a diffuse representation  the "catness" isn't stored in one place its spread all across the network's connections  a bit like a hologram  smash a part of it and you still get a kinda cat but a blurry one

Humans are different  We seem to work with symbols  concepts  rules  We can reason deductively like  all men are mortal Socrates is a man therefore Socrates is mortal  See  symbols logic  A neural network would probably need to see millions of examples of men dying to even *begin* to approach that kind of generalisation  It's kinda brute force learning versus symbolic manipulation

Another thing  our reasoning has this amazing thing called introspection we can think about our thinking  we can explain our reasoning  a network can't really do that  it can tell you what its output is but why it picked that output  that's a black box  a mystery  It's like asking a toddler why they like blue  they might say "because it's pretty" but that's not a real explanation  it's the same with a network  its outputs are just emergent properties from its training  there's no deep internal understanding

Diffuse properties are another key difference  human knowledge isn't just all jumbled together  we have categories  hierarchies  We know a cat is a mammal which is an animal  etc  Neural networks  they kinda learn these things but the way they represent that hierarchy is completely different from ours  It's all distributed  intertwined  no neat boxes

Let's look at some code snippets to illustrate this point


```python
# A simple feedforward neural network
import numpy as np

# Sample data (imagine this is a vast dataset of cat images)
X = np.random.rand(1000, 784) # 1000 images, each 784 pixels
y = np.random.randint(0, 2, 1000) # 0 or 1 (cat or not cat)


# Simple network architecture (fully connected)
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)


    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = 1 / (1 + np.exp(-z2))
        return a2


# Train the network (highly simplified for brevity)
nn = SimpleNN(784, 128, 1)
# ... training loop with backpropagation ...
```

This shows the subsymbolic nature  there are weights biases  but no explicit representation of "cat"  It's just numbers adjusting to match input and output


```python
# Symbolic reasoning example (very simple)
def is_mortal(x):
  if x == "Socrates":
    return True
  else:
    return False

print(is_mortal("Socrates")) # True
```

This is symbolic  clear rules  "Socrates" is a symbol that directly triggers a truth value


```python
# A more complex network example (still very basic)
import tensorflow as tf

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
  tf.keras.layers.Dense(10, activation='softmax') # 10 output classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ... training loop with large dataset ...
```


This one is a bit more sophisticated but still subsymbolic its representing categories (the 10 output classes) in a diffuse way  across weights and activations  No explicit definition of what constitutes each category


So what are the implications of this difference  well  it means that while neural networks are amazing at pattern recognition  they struggle with things like abstract reasoning  common sense  explanation  They are great at mimicking human-like behavior but the underlying mechanisms are very different


To delve deeper into this you could check out some papers on  cognitive science  connectionism  and symbolic AI  There's no one definitive book or paper  but exploring work by  David Marr on vision  John Haugeland on AI  and more recent work on hybrid symbolic-connectionist models will give you a broader picture

This is a complex field and there's tons more to unpack about  generalization  explainability  the role of embodiment in cognition etc  but hopefully this gives you a starting point for your own explorations  It's a wild ride comparing human brains and AI brains  lots of open questions  many unresolved debates but its fun  isn't it
