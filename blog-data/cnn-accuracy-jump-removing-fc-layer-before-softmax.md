---
title: "CNN Accuracy Jump: Removing FC Layer Before Softmax? 🤯"
date: '2024-11-08'
id: 'cnn-accuracy-jump-removing-fc-layer-before-softmax'
---

```python
# Assuming you're using Keras
model = tf.keras.models.Sequential([
    # ... your convolutional layers ...
    tf.keras.layers.Flatten(),
    tf.keras.layers.Softmax() 
])

# Compile and train your model as usual
model.compile(...)
model.fit(...)
``` 

