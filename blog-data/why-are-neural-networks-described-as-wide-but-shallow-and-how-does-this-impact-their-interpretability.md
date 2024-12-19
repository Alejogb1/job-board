---
title: "Why are neural networks described as 'wide but shallow,' and how does this impact their interpretability?"
date: "2024-12-11"
id: "why-are-neural-networks-described-as-wide-but-shallow-and-how-does-this-impact-their-interpretability"
---

Okay so you wanna know about wide but shallow neural nets right  Interesting question  It's all about the architecture right  Think of a neural network like a layered cake each layer is a bunch of neurons doing their thing  Deep networks are like a really tall cake lots and lots of layers  Wide networks are like a really big cake lots of neurons in each layer but not many layers  Shallow networks are the opposite few layers  So wide but shallow means lots of neurons per layer but only a few layers total

Why are some described this way  Well it's often a trade-off  Deep networks are great for learning complex features  Think of it like this a deep network can learn  "this is a cat" then "this is a tabby cat" then "this is a fluffy grey tabby cat"  It goes really deep into the details  But training them takes ages and they're like black boxes you can't really see what's going on inside super hard to interpret

Wide networks on the other hand can learn simpler features quickly  They're faster to train  Think of it like building a really simple decision tree for a cat identification problem  It's a bit less accurate maybe but much faster to build and easier to understand  Wide but shallow networks kinda sit in the middle  They aren't super deep so they are relatively faster to train than very deep networks and they are often wider allowing to learn more complex features than simple networks  However  they won't learn the super intricate relationships that a deep network would  It's like a compromise speed versus complexity

This trade-off directly impacts interpretability  Interpretability basically means how easy is it to understand what the network is doing  Deep networks are notoriously uninterpretable  It's hard to say why a deep net classified an image as a cat  It's making millions of tiny decisions across tons of layers  Wide but shallow networks are much more interpretable  You can kinda follow the flow of information a bit easier  There are fewer layers and the interactions between neurons are simpler   It's not perfectly clear but you can get a better idea of what's happening

Here's a code example of a simple wide but shallow network using TensorFlow  This is a super basic example for illustration purposes  You'd want more layers and neurons in a real-world application

```python
import tensorflow as tf

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(1024, activation='relu', input_shape=(784,)), # Wide layer
  tf.keras.layers.Dense(10, activation='softmax') # Output layer
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ... (training code here) ...
```

See that first Dense layer  1024 neurons that's wide  But only two layers total that's shallow  This is a multi-class classification problem  the output layer has 10 neurons representing 10 classes  The input layer assumes data of 784 features for this example   You could adjust those numbers  You could make it wider or add a layer or two but the general idea is there

Another example showing a wider network with slightly more layers


```python
import torch.nn as nn
import torch

class WideShallowNet(nn.Module):
    def __init__(self):
        super(WideShallowNet, self).__init__()
        self.fc1 = nn.Linear(784, 2048) # Very wide layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2048, 1024) # Another wide layer
        self.fc3 = nn.Linear(1024, 10) # Output layer

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = WideShallowNet()
# ... (training code in pytorch here) ...

```

This uses PyTorch  It's a bit more complex but still relatively shallow  It has three layers  but each layer is wide  Again this is just a simple example  for illustration

And finally here’s a slightly more complex example of a wide and relatively shallow network that uses different activation functions


```python
import numpy as np
from sklearn.neural_network import MLPClassifier

X = np.random.rand(100, 784) #example data
y = np.random.randint(0, 10, 100) # Example labels

clf = MLPClassifier(hidden_layer_sizes=(512, 256), activation='tanh', solver='adam', max_iter=1000)
clf.fit(X, y)
```

This example uses scikit-learn  It's not as directly customizable as TensorFlow or PyTorch but it's convenient for quick prototyping  The `hidden_layer_sizes` parameter defines the width of the hidden layers  Two hidden layers with 512 and 256 neurons is still considered relatively shallow compared to very deep networks

So yeah wide but shallow networks are a thing  They offer a balance between training speed and complexity  and that affects how easy they are to understand  For more detailed info check out  "Deep Learning" by Goodfellow Bengio and Courville  It's a pretty comprehensive book on the subject  And there's also a lot of papers on network architectures and interpretability  Search for papers on "explainable AI" or "XAI"  those should give you more specific resources  Good luck  Let me know if you have more questions
