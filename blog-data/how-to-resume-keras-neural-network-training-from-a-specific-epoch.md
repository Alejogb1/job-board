---
title: "How to resume Keras neural network training from a specific epoch?"
date: "2024-12-23"
id: "how-to-resume-keras-neural-network-training-from-a-specific-epoch"
---

Okay, let's tackle this. I’ve been down this road countless times – interrupted training runs are an unfortunate reality, especially when dealing with complex models and lengthy datasets. The good news is, resuming a Keras training session from a specific epoch is straightforward, provided you've set things up correctly from the outset. It hinges primarily on leveraging the power of model checkpoints and potentially some customized callbacks.

My past experiences on projects involving large-scale image recognition have underscored how critical it is to save model progress. I recall one project where we were training a convolutional neural network for a medical imaging task; a sudden power outage at hour 20 of a 30-hour run was enough to make anyone swear under their breath. Thankfully, we had checkpointing enabled, which saved us from having to restart from scratch.

The core idea revolves around saving the model's weights and, ideally, the optimizer's state at regular intervals during training. Keras makes this easy via the `ModelCheckpoint` callback, which, when properly configured, writes both the model's architecture and current weights, often along with the optimizer state (depending on the file format and configuration) to disk.

Let me break down the process and provide you with a couple of code examples to illustrate the workflow:

**Setting Up Checkpointing**

First, during the *initial* training run, it’s crucial to configure a `ModelCheckpoint` callback. You need to specify the filepath where the model and its state should be saved, the frequency of the saves (e.g., after each epoch or only when a specific metric improves), and whether to save only the best weights or all weights across every checkpoint.

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np # for dummy data

# let's generate some random data for this example
num_classes = 10
input_shape = (28, 28, 1)

x_train = np.random.rand(1000, *input_shape).astype(np.float32)
y_train = np.random.randint(0, num_classes, 1000).astype(np.int64)
y_train = keras.utils.to_categorical(y_train, num_classes)


def create_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model



def initial_training():
    model = create_model()
    checkpoint_filepath = 'path_to_checkpoint/model_{epoch:02d}.h5'
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,  # save the entire model not just weights
        monitor='loss', # monitor the validation loss to save the best model
        save_best_only=True, # save only the weights with the lowest loss
        save_freq='epoch'  # saves after every epoch.
    )

    model.fit(x_train, y_train, epochs=5, callbacks=[model_checkpoint_callback])
    return model

# initiate the initial training
model = initial_training()

```

In this example, the model is trained for five epochs and, at the end of each epoch (since I set save_freq='epoch'), the model's weights, optimizer state, and overall architecture are saved in `.h5` format under the specified filepath, making it restartable. The `save_best_only=True` parameter makes sure to overwrite the checkpoint if the validation loss (specified via `monitor='loss'`) improves. If no validation data is available, set the monitor to the training loss which might not be ideal but is better than nothing. Be aware that the name of the file is constructed as a `f-string`, hence why `{epoch:02d}` is used, so ensure to correctly format any variables you may want to use inside the filename.

**Resuming the Training**

Now, let’s consider we had to stop the training mid-run for some reason. Resuming is also quite simple, but you need to load from the correct checkpoint file, which is determined by the epoch at which you stopped the previous training session.

```python
def resume_training(start_epoch=2):
    # load the previously trained model from a checkpoint
    # construct the checkpoint filepath based on the desired start epoch
    checkpoint_filepath = f'path_to_checkpoint/model_{start_epoch:02d}.h5'
    
    model = keras.models.load_model(checkpoint_filepath)

    checkpoint_filepath_resume = 'path_to_checkpoint_resume/model_resume_{epoch:02d}.h5'

    model_checkpoint_callback_resume = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath_resume,
        save_weights_only=False,
        monitor='loss',
        save_best_only=True,
        save_freq='epoch'
    )

    model.fit(x_train, y_train, initial_epoch=start_epoch, epochs=10, callbacks=[model_checkpoint_callback_resume])
    return model
# this will load the weights of the model trained till epoch 2, and continue from that point on.
resumed_model = resume_training(2)
```

In this example, the function `resume_training` loads the model weights and architecture from the saved file corresponding to the 2nd epoch. We are specifying `initial_epoch=2` to Keras so that the internal counter is correctly set. Further training is then executed for 8 more epochs. Importantly, I've changed the checkpoint filepath and added a resume label to the name to ensure that the original saved weights are not overwritten and are kept for reference.

**Customized Callbacks for Enhanced Control**

While `ModelCheckpoint` covers most scenarios, you may need more control over what gets saved or when. For this, you can create custom callbacks. For example, I encountered a situation where the optimizer state was not correctly being reloaded due to some incompatibility issue, so I created a custom callback to save the optimizer state specifically.

```python
class SaveOptimizerCallback(keras.callbacks.Callback):
    def __init__(self, filepath):
        super(SaveOptimizerCallback, self).__init__()
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        optimizer = self.model.optimizer
        weights = optimizer.get_weights()
        np.save(f'{self.filepath}_optimizer_epoch_{epoch}.npy', weights)

    def load_optimizer_weights(self, epoch):
        optimizer = self.model.optimizer
        try:
           weights = np.load(f'{self.filepath}_optimizer_epoch_{epoch}.npy', allow_pickle=True)
           optimizer.set_weights(weights)
        except:
          print(f"Could not load the saved optimizer weights from {self.filepath}_optimizer_epoch_{epoch}.npy.")

def resume_training_with_optimizer(start_epoch=2):
    # load the previously trained model from a checkpoint
    # construct the checkpoint filepath based on the desired start epoch
    checkpoint_filepath = f'path_to_checkpoint/model_{start_epoch:02d}.h5'
    
    model = keras.models.load_model(checkpoint_filepath)
    
    optimizer_filepath = 'path_to_optimizer_weights/optimizer_weights'
    
    custom_optimizer_callback = SaveOptimizerCallback(optimizer_filepath)
    
    # load optimizer weights
    custom_optimizer_callback.load_optimizer_weights(start_epoch)

    checkpoint_filepath_resume = 'path_to_checkpoint_resume/model_resume_{epoch:02d}.h5'

    model_checkpoint_callback_resume = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath_resume,
        save_weights_only=False,
        monitor='loss',
        save_best_only=True,
        save_freq='epoch'
    )

    model.fit(x_train, y_train, initial_epoch=start_epoch, epochs=10, callbacks=[model_checkpoint_callback_resume, custom_optimizer_callback])
    return model
# this will load the weights of the model trained till epoch 2, and continue from that point on.
resumed_model = resume_training_with_optimizer(2)
```

In this specific example, we are building a custom callback, `SaveOptimizerCallback` to specifically save the optimizer weights at the end of every epoch (using `np.save` to save the weights in `.npy` format and the `get_weights` method) and a method (`load_optimizer_weights`) that loads these weights using `np.load` and the `set_weights` method to set it back to the optimizer. It's not as robust as a fully built solution, but it provides a good starting point for handling edge cases.

**Recommended Resources**

For further in-depth study, I strongly recommend exploring the following resources:

1.  **"Deep Learning with Python" by François Chollet:** This book is an excellent resource for a clear and detailed explanation of Keras and its functionalities, including callbacks and model saving.
2.  **The Official TensorFlow/Keras Documentation:** Refer directly to the official documentation for the most up-to-date information on Keras callbacks, specifically `ModelCheckpoint`, model loading (`keras.models.load_model`), and working with the `fit` method, including the `initial_epoch` parameter.
3.  **Research papers on model checkpointing strategies:** Look for articles in computer vision or natural language processing conferences (e.g., CVPR, NeurIPS, ACL) which cover techniques and strategies for robust checkpointing in deep learning.
4.  **Keras examples repository:** Explore the official Keras example repository on GitHub. There are numerous examples showcasing different callback configurations, and these real-world use-cases can provide a good perspective on different ways to tackle particular issues.

In closing, resuming Keras training from a particular epoch is not a difficult task, but it demands an understanding of how callbacks work and the various parameters involved in the model checkpointing process. Having a clear checkpointing strategy, as well as potentially custom callbacks for edge cases or special needs, will save you a significant amount of time and headache in the long run. Remember, prevention is always better than cure; in deep learning, this translates to careful planning for training interruptions.
