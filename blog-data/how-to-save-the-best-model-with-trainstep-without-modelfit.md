---
title: "How to Save the best model with train_step without model.fit()?"
date: "2024-12-15"
id: "how-to-save-the-best-model-with-trainstep-without-modelfit"
---

alright, so you're looking to save the best model during training, but you're doing a custom training loop with `train_step` instead of the usual `model.fit()`. i get it, you want that granular control, and sometimes the built-in stuff isn't flexible enough. i've been there, trust me. i spent a good chunk of a summer back in '17 trying to optimize some reinforcement learning agent, and the `fit()` method was just too much of a black box.

the thing is, `model.fit()` does a bunch of behind-the-scenes magic, including keeping track of validation losses and saving the best model based on some metric. when you're rolling your own loop, you need to handle that logic yourself. it’s not that complicated though, just a couple of extra pieces you need to put together.

here's the general idea. we'll keep track of the best validation loss (or whatever metric you care about) seen so far, and whenever we see an improvement, we'll save the model weights. think of it like a running champion leaderboard, we only save the current champ model.

let's start with the foundational steps. first we need a way to actually save the model. we can do this with `tf.keras.models.save_model()`. it's a very versatile tool. it can save the entire model, including its architecture and weights, or just the weights themselves. for our purpose, we'll stick to saving only the weights, since you're probably fine with the model structure. you can even save in different formats, like the `SavedModel` format that is more for deployment. But for now let’s keep it simple.

here is an example on how to do it, you need to have a model first:

```python
import tensorflow as tf
import numpy as np

# Assume you have a model defined
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1)
])

# sample data for demonstration
x_train = np.random.rand(100, 10)
y_train = np.random.rand(100, 1)
x_val = np.random.rand(30, 10)
y_val = np.random.rand(30, 1)

# create an optimizer, loss and metric function
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.MeanSquaredError()
metric = tf.keras.metrics.MeanSquaredError()

# training loop using train_step
def train_step(x,y):
  with tf.GradientTape() as tape:
    y_pred = model(x,training=True)
    loss = loss_fn(y,y_pred)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients,model.trainable_variables))
  metric.update_state(y,y_pred)
  return loss

# validation loop
def validation_step(x,y):
  y_pred = model(x,training=False)
  val_loss = loss_fn(y,y_pred)
  metric.update_state(y,y_pred)
  return val_loss
```

now, that's our usual training stuff for doing a custom training loop. now let's get to the saving part.

for the saving part, we need a variable to keep track of the best loss, and we’ll initialize it to a very large number, that way the first validation loss will be better. we also will need to provide a path to save the model in, and it's good to have it as a variable. here is how we are going to implement it:

```python
best_val_loss = float('inf')
save_path = 'best_model_weights.h5'

num_epochs = 10 #number of training epochs
for epoch in range(num_epochs):
    #training
    for i in range(len(x_train)):
        loss = train_step(x_train[i:i+1],y_train[i:i+1])
    print(f'epoch {epoch + 1} training mean square error: {metric.result()}')
    metric.reset_state()
    # validation
    for i in range(len(x_val)):
        val_loss = validation_step(x_val[i:i+1],y_val[i:i+1])
    current_val_loss = metric.result()
    print(f'epoch {epoch + 1} validation mean square error: {current_val_loss}')
    metric.reset_state()
    #save if better
    if current_val_loss < best_val_loss:
      best_val_loss = current_val_loss
      model.save_weights(save_path)
      print('saved model weights')
```

that's basically it. each epoch, after you calculate the validation metric, you just compare it to the best one you had, and save if is better. in this case, it is the mean squared error. but it could be accuracy, or any metric you're using. i do this all the time with classification problems, it's pretty standard and works like a charm.

now, a couple of points to note. first, make sure the `save_path` is where you want to store your weights. and also, if you need, instead of saving only the weights with `model.save_weights()`, you could save the entire model with `tf.keras.models.save_model()`. and second, remember to load the weights with `model.load_weights(save_path)` later on, if you are saving only the weights.

this approach will save you a lot of time, specially if the training is very long. because, i don't know if you have been there, but i remember once training a neural net on some astrophysics data, and after 3 days, my computer crashed, i lost the entire progress, because i was not saving the model weights. that was really frustrating, so now i make sure to save regularly. and if you want to be more robust, you can add callbacks to be saved by some number of steps, that can be very useful if you are training for a lot of epochs, this is something tensorflow already gives you, is good to learn it, check the official documentation.

i personally prefer the code to the documentation most of the time. but the documentation has all the available parameters, and you can adjust the saving behaviour according to you needs. it’s a great resource if you want more details, the official tensorflow documentation page for saving models can be very helpful, there is a lot of examples there, that you can check. but the approach above works just fine, i use it in most of my projects.

and about books, i remember that for learning more about reinforcement learning that summer, i read "reinforcement learning: an introduction" by sutton and barto. even if it's not totally related, it has some good insights into making you think about how to train a model in general. and as for general machine learning, "deep learning" by goodfellow et al. it is a great reference book and has all that you need, if you really want to go deep in machine learning, it’s a great pick.

so there you have it. no more black boxes, you're in charge of saving your best model. you get the control of your training, and you will never lose your progress, the way that i did back in 2017, that was terrible. that was when i learned about saving the best model. and believe me, there is no better feeling that seeing your model saving with a good loss value. and to be honest, i know it is not much of a joke, but it has saved me more times than i can count: why did the developer quit their job? because they didn't get arrays!

happy coding, and let me know if anything else comes up. i am here to help.
