---
title: "Do Keras callbacks like EarlyStopping utilize training history prior to a crash?"
date: "2024-12-23"
id: "do-keras-callbacks-like-earlystopping-utilize-training-history-prior-to-a-crash"
---

Let's tackle this one – I've certainly seen my fair share of training runs end unexpectedly, and it's always a question of how much of that work is salvageable, particularly when using callbacks like `EarlyStopping`. The short answer is: it depends, but primarily, no, not in the way you might initially think. `EarlyStopping` doesn't intrinsically "remember" history across program executions, so if your script crashes, that training history is essentially lost unless you've explicitly saved it. Let me elaborate based on some experiences I had with a rather large convolutional neural network for image classification a few years back, where I learned this lesson firsthand.

Initially, I had this beast of a model training for days, and while I had `EarlyStopping` in place, we had a power outage – not fun. When I restarted, `EarlyStopping` had no memory of the previous run's validation losses, so the whole training process effectively started from scratch. That's when it became quite apparent that callbacks within Keras, while very helpful within a single training session, operate solely within that session's context.

The core principle behind `EarlyStopping` is to monitor a specified metric (typically validation loss) and halt training when that metric ceases to improve over a predefined number of epochs (`patience`). This monitoring is done during the current training run – it relies on the model's `fit` method to generate metrics during the training process. These metrics are stored in the `history` object that `fit` returns. When your training script crashes, the execution context is lost along with the history object.

It's helpful to understand how Keras handles callbacks. The callbacks are actually functions called at specific points in the training process, like at the end of each epoch or each batch, allowing you to interact with the training loop and model. However, they do not inherently store state across different training runs or program executions. That state, such as the `history` object, is generally created and maintained within the specific session.

This lack of persistence isn't necessarily a limitation; it’s by design. It helps keep the callbacks lightweight, avoiding persistent memory management. However, it means we, as developers, need to implement a mechanism for persistent history if we need to resume training or leverage information across runs.

Now, let’s look at some code examples to illustrate.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Example 1: Basic usage of EarlyStopping
def create_model():
  model = keras.Sequential([
      layers.Dense(64, activation='relu', input_shape=(10,)),
      layers.Dense(10, activation='softmax')
  ])
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  return model

def train_model_with_early_stopping():
    model = create_model()
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    x_train = np.random.rand(100,10)
    y_train = np.eye(10)[np.random.randint(0, 10, 100)]
    x_val = np.random.rand(50,10)
    y_val = np.eye(10)[np.random.randint(0, 10, 50)]
    history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_val, y_val), callbacks=[early_stopping])

    print("Training history:", history.history)

if __name__ == "__main__":
    train_model_with_early_stopping()

```

In this first example, `EarlyStopping` works precisely as intended, but if the training were interrupted and the script restarted, the callback would have no prior context – it starts monitoring from the initial epoch after the restart.

The solution, of course, is to explicitly persist the relevant information, specifically the `history` object or the model weights, or both. Here is how that looks in practice.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import json

# Example 2: Saving training history and resuming training
def create_model():
  model = keras.Sequential([
      layers.Dense(64, activation='relu', input_shape=(10,)),
      layers.Dense(10, activation='softmax')
  ])
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  return model

def train_model_with_history_persistence():
    model = create_model()
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    x_train = np.random.rand(100,10)
    y_train = np.eye(10)[np.random.randint(0, 10, 100)]
    x_val = np.random.rand(50,10)
    y_val = np.eye(10)[np.random.randint(0, 10, 50)]
    
    history_file = "training_history.json"
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
           history_data = json.load(f)
        #Load saved model (assuming we saved the weights, which would be another step)
        print("Continuing from previous run...")
    else:
        history_data = {}


    
    history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_val, y_val), callbacks=[early_stopping])
    
    # Merge current history with loaded history
    for key, value in history.history.items():
       if key in history_data:
           history_data[key].extend(value)
       else:
           history_data[key] = value

    with open(history_file, 'w') as f:
        json.dump(history_data, f, indent=4)
    print("Training history saved and updated in training_history.json")
    

if __name__ == "__main__":
    train_model_with_history_persistence()
```

In this second example, I introduced a very basic saving of the training `history`.  We serialize the dictionary that comes back from the `fit` method to a json file.  This enables a persistent record that can be retrieved between runs, and any future runs can update the persistent history. A similar process should be used for model weights as well, using Keras functions `model.save_weights` and `model.load_weights`, typically using .h5 or .keras format.

Finally, I'll show a third example that's a bit more sophisticated, encapsulating the saving and loading of both model weights and training history in an abstract class, a pattern I often use.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import json

class ResumableModel(keras.Model):

    def __init__(self, model, model_dir, history_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.model_dir = model_dir
        self.history_file = history_file
        self.history = {}
        
        if os.path.exists(model_dir):
            self.model.load_weights(model_dir)
            print(f"Loaded model weights from {model_dir}")
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                self.history = json.load(f)
                print(f"Loaded training history from {history_file}")
            
    def save(self):
        self.model.save_weights(self.model_dir)
        with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=4)
        print("Model and History Saved.")
    
    def fit(self, *args, **kwargs):
       
        history = super().fit(*args, **kwargs)

        # Merge current history with loaded history
        for key, value in history.history.items():
          if key in self.history:
            self.history[key].extend(value)
          else:
              self.history[key] = value
        
        return history

# Example 3: Resumable model wrapper
def create_model():
  model = keras.Sequential([
      layers.Dense(64, activation='relu', input_shape=(10,)),
      layers.Dense(10, activation='softmax')
  ])
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  return model
  
def train_with_resumable_model():
    model = create_model()
    resumable_model = ResumableModel(model, "saved_weights/model.h5", "training_history.json")
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    x_train = np.random.rand(100,10)
    y_train = np.eye(10)[np.random.randint(0, 10, 100)]
    x_val = np.random.rand(50,10)
    y_val = np.eye(10)[np.random.randint(0, 10, 50)]
  
    history = resumable_model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_val, y_val), callbacks=[early_stopping])
    resumable_model.save()

if __name__ == "__main__":
    train_with_resumable_model()
```

Here, we've created an abstraction of sorts, wrapping the model with a class that provides additional functionality to save and load the model's weights and history. This allows more complex logic to be created on how that history is used, while remaining separated from the core modeling functionality. It also ensures we can easily resume training.

As for more in-depth reading, I’d strongly recommend the official Keras documentation on callbacks. Additionally, for a broader understanding of model training and management, the book "Deep Learning with Python" by François Chollet is invaluable. The TensorFlow documentation itself is another indispensable resource for specific details on callbacks, saving models, and managing training states.

To summarize, `EarlyStopping` doesn't remember training history across crashes; its state is lost when the Python process terminates. The responsibility for preserving this information lies with you. By implementing proper saving and loading mechanisms, like the simple examples I've shown here, we can effectively resume training after an interruption and preserve all that hard-earned training data.
