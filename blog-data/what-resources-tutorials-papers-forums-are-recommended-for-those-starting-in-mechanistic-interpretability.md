---
title: "What resources (tutorials, papers, forums) are recommended for those starting in mechanistic interpretability?"
date: "2024-12-11"
id: "what-resources-tutorials-papers-forums-are-recommended-for-those-starting-in-mechanistic-interpretability"
---

Hey so you wanna get into mechanistic interpretability huh that's awesome its a super cool field  I've been messing around with it for a while now and its seriously mind bending  Its like reverse engineering the brain but instead of brains its neural networks  Pretty wild right

Okay so where to start  Forget tutorials for now  Tutorials are great for learning syntax but they wont give you the intuition you need for this stuff  You need the *why* not just the *how*  Think of it like learning to play the guitar you can learn chords and scales but until you understand music theory you wont really *get* it

For the foundations you really gotta dig into some papers  There are some absolute bangers out there that lay the groundwork   Honestly the best place to start is probably looking at papers coming out of groups like DeepMind  They're usually pretty good about explaining their work clearly  And I don't just mean the results  I mean the *methodology*  How they actually got their results is way more important than the results themselves in this field

One paper that's really helped me is  "Understanding deep learning through deep learning" by some folks at Google Brain  Its a bit dense but it’s worth it  It covers concepts like probing classifiers and network dissection which are really fundamental techniques  Think of those techniques as your basic toolkit  You'll be using them constantly

Another one you should check out is anything related to circuit analysis  This is all about figuring out how specific parts of the network interact to produce certain behaviors its like tracing wires in a circuit board but for neurons  This approach is becoming increasingly popular as we want to understand not just what a network does but *how* it does it and its all about understanding the interactions of small groups of neurons

Then theres the whole interpretability zoo  There's tons of methods out there  like attention mechanisms  activation maximization  saliency maps  gradient based methods  and so many more  You'll eventually need to know these but start with the foundations first  Dont get bogged down in the details too early

Its like learning a programming language you need to know the basics before you jump into building a complex application  You dont want to spend hours trying to debug code that you dont even understand  So focus on understanding the core concepts first

For books I'd recommend something on linear algebra and a solid introduction to machine learning  Linear algebra is especially important because its the mathematical foundation for understanding how these networks work  I know its dry but stick with it its worth it trust me  There's a ton of free online resources for this as well

After the foundational stuff you can start looking at more advanced topics  like  neuron dissection  activation atlases  and probing classifiers  These techniques help us understand how individual neurons or small groups of neurons contribute to the overall function of the network

Lets look at some example code snippets  these are very simplified illustrations to get the general idea  Remember these are extremely stripped down just to give a flavor

**Snippet 1: Probing Classifier**

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

# Assume 'activations' is a numpy array of neuron activations
# and 'labels' is a numpy array of corresponding labels

# Train a logistic regression model to predict labels from activations
model = LogisticRegression()
model.fit(activations, labels)

# Evaluate the model's accuracy
accuracy = model.score(activations, labels)
print(f"Accuracy of probing classifier: {accuracy}")
```

This just gives you the idea of training a simple model on activations to see what it learns  It's a super basic example  but its the core idea behind probing classifiers  Its basically using a simple model like a linear regression or a logistic regression to see if you can predict the label from neuron activations

**Snippet 2: Activation Maximization**

```python
import tensorflow as tf

# Assume 'model' is your trained neural network
# and 'image' is your input image

# Create a loss function that maximizes the activation of a specific neuron
with tf.GradientTape() as tape:
    tape.watch(image)
    activations = model(image)
    loss = activations[:, specific_neuron_index] # Maximise activation of this neuron

# Calculate gradients and update the image
gradients = tape.gradient(loss, image)
image = image + learning_rate * gradients # Gradient ascent to maximise neuron activation


```

This shows a very very basic approach to activation maximization  We're trying to find an input image that makes a specific neuron fire maximally  Its basically a gradient ascent process  You iteratively modify your input image to increase the activation of a target neuron

**Snippet 3: Saliency Maps**

```python
import tensorflow as tf

# Assume 'model' is your trained neural network
# and 'image' is your input image

# Calculate gradients of the output with respect to the input image
with tf.GradientTape() as tape:
    tape.watch(image)
    output = model(image)

gradients = tape.gradient(output, image)

# Take the absolute value of the gradients to get saliency map
saliency_map = tf.abs(gradients)

```


This is a super basic example of how to generate saliency maps  We're calculating the gradients of the output with respect to the input image  The magnitude of the gradients tells us how much each pixel contributes to the output  Higher magnitude means higher contribution  Its a simple way to visualize what parts of an image the network is focusing on

These are seriously simplified examples but hopefully they give you a basic idea of the kind of stuff you’ll be dealing with  Dont worry if you dont fully grasp it now  Just keep going  read those papers  play around with the code  and most importantly  be curious  

Mechanistic interpretability is a new and exciting field and there is a lot to discover  So keep exploring and have fun with it  You got this  Let me know if you have any questions  seriously dont hesitate to ask  I am happy to help
