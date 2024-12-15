---
title: "Why am I getting a Checkpointing error with apply_gradients()?"
date: "2024-12-15"
id: "why-am-i-getting-a-checkpointing-error-with-applygradients"
---

ah, checkpointing errors with `apply_gradients()`, that's a classic. i've been there, trust me. it’s one of those things that can make you scratch your head for hours, especially when everything else seems to be working fine. let's unpack this.

first off, when you see checkpointing errors related to `apply_gradients()`, it usually boils down to one of a few common scenarios. it's not always immediately obvious, but the root cause generally lies in how tensorflow is tracking the tensors and operations within your model and gradient updates.

the basic idea behind checkpointing is to save the state of your model and optimizer so you can later resume training or use the trained model. when you are using `tf.train.Checkpoint` (which i assume you are), and you apply gradients via `apply_gradients()`, tensorflow needs to know exactly which variables are being updated. if there is any disconnect between what the checkpoint thinks it should be saving, and what's actually going on during `apply_gradients()`, you end up with an error.

one of the biggest culprits is using variables that are not directly part of the model you are checkpointing. it is important to keep in mind that these issues can be a silent killer of experiments, i can tell you that much. let me tell you a personal anecdote; back in my early days, i was building a segmentation model using a custom loss function and everything seemed peachy on the surface. i was getting some decent training loss and the architecture was solid. i remember being very happy with the architecture. however, as soon as i started checkpointing i hit the wall. i could not resume any of the experiments. i got all kinds of checkpointing errors. it took me almost a week to realize that a helper function that was doing some intermediate calculation was using `tf.Variable` for some temporary values but they were not being included in the checkpoint. i had added the variable out of laziness for efficiency and had not thought of the checkpointing, the `tf.Variable` that was not part of the model. i had to move it to a pure tensor operation and after that, the training could resume and i could checkpoint the whole thing without any issues. i learned a big lesson that day. since then, i have never overlooked the checkpointing of any of my models again.

let me give you some examples in code:

first scenario: **untracked variables**.
```python
import tensorflow as tf

# pretend model
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.w = tf.Variable(tf.random.normal((1,)), name="w")
        self.b = tf.Variable(tf.zeros((1,)), name="b")

    def call(self, x):
        return x * self.w + self.b


# let's create the model
model = MyModel()

# optimizer
optimizer = tf.keras.optimizers.Adam()

# checkpoint
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

# this untracked variable causes the problem when applying gradients
untracked_var = tf.Variable(1.0, name="untracked")


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = tf.reduce_mean(tf.square(logits - y))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # this is a good way to update the untracked variable with something that will not affect the gradients
    untracked_var.assign_add(1.0)
    return loss


# pretend data
x_data = tf.random.normal((10, 1))
y_data = tf.random.normal((10, 1))

for step in range(5):
    loss = train_step(x_data, y_data)
    print(f'step: {step}, loss: {loss}')

# save the checkpoint
checkpoint_path = "./my_checkpoint"
checkpoint.save(checkpoint_path)
print("checkpoint created successfully")
```

if you run this code, it will run and the checkpoint will be created. but try to load the model and you might see an error if your code is not carefully coded. notice that the `untracked_var` is not part of the checkpoint it will be reinitialized when you load the model if you have code that tries to use it. when applying gradients and if the untracked variable is part of the gradients you will encounter the error. it's crucial that *all* variables involved in gradient computations are explicitly tracked by the `tf.train.Checkpoint`. this includes the trainable variables of your model and variables directly used by the optimizer. the key thing to notice is that here there is no direct application of the untracked variables to the gradients this is ok and this example won't cause an error, but if you apply the gradient it will.

another very common scenario is that you are changing your model's architecture dynamically, that is, you are not using a keras model class or something similar. this is something that i see in many research experiments that i review. tensorflow's checkpointing mechanism heavily relies on a static graph. if you are adding or removing layers or changing the number of variables between training sessions, the checkpoint might not be able to load back the proper state.

here is a more subtle example that can occur if you are building models that change its weights and structure in an imperceptible way.
```python
import tensorflow as tf

# this function can generate an issue
def create_model(input_shape):
    # the model is randomly being built differently each time
    if tf.random.uniform((1,)).numpy() < 0.5:
        return tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=input_shape),
            tf.keras.layers.Dense(1)
        ])
    else:
        return tf.keras.Sequential([
           tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
           tf.keras.layers.Dense(1)
        ])


# initialize the model
input_shape = (10,)
model = create_model(input_shape)

# optimizer
optimizer = tf.keras.optimizers.Adam()

# checkpoint
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = tf.reduce_mean(tf.square(logits - y))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


# generate data
x_data = tf.random.normal((100, 10))
y_data = tf.random.normal((100, 1))

for step in range(5):
    loss = train_step(x_data, y_data)
    print(f'step: {step}, loss: {loss}')

#save checkpoint
checkpoint_path = "./my_checkpoint_2"
checkpoint.save(checkpoint_path)

print("checkpoint created successfully")

# reload the model for next train
input_shape = (10,)
model = create_model(input_shape) #this can trigger the issue since it has different number of layers.

checkpoint.restore(checkpoint_path)
print("checkpoint loaded successfully")
```

in this example, the model is not explicitly changed during training but during reload. this can trigger an issue and if the number of layers change you might have an error. this is a bit more complex, but it will also show you the power of checkpointing and the issues you can get into. if your code is like this it's very difficult to debug and it's the most common reason for an issue with checkpointing.

another case i've seen is where some tensor is not properly being traced within the graph and you are not using the `tf.function` decorator, and the checkpoint does not really know what to save or how to save it. using `tf.function` is key to getting all the proper tracing of the operations. here is another example of a not very common but possible scenario with `tf.function`
```python
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.w = tf.Variable(tf.random.normal((1,)), name="w")
        self.b = tf.Variable(tf.zeros((1,)), name="b")

    def call(self, x):
        return x * self.w + self.b

# instantiate model
model = MyModel()

# optimizer
optimizer = tf.keras.optimizers.Adam()

# checkpoint
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)


def my_loss_calculation(logits, y):
    #some complex loss function that does not use tf ops.
    l = tf.reduce_mean(tf.square(logits - y))
    return l


@tf.function
def train_step_no_tf(x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        # this loss function breaks checkpointing
        loss = my_loss_calculation(logits, y)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# generate data
x_data = tf.random.normal((100, 1))
y_data = tf.random.normal((100, 1))

for step in range(5):
    loss = train_step_no_tf(x_data, y_data)
    print(f'step: {step}, loss: {loss}')

#save checkpoint
checkpoint_path = "./my_checkpoint_3"
checkpoint.save(checkpoint_path)
print("checkpoint created successfully")

checkpoint.restore(checkpoint_path) # this can break with a checkpoint error
print("checkpoint loaded successfully")
```
in the previous code, if the my_loss_calculation is not using a `tf.*` operations and you are working with numpy arrays, this can also break the graph and cause an error since not all the operations can be traced. i do not recommend working with pure python objects inside tf.functions, they are not tensor friendly.

so, to sum things up, here's a quick checklist to debug checkpointing issues with `apply_gradients()`:

1.  **ensure that all trainable variables are properly part of the `tf.train.Checkpoint`**: avoid creating un-tracked variables that can be used in the gradients or the training loop.
2.  **use a static graph**: avoid dynamic changes to the model architecture between train and load cycles, specially if it affects the gradients or the trainable variables.
3.  **use `tf.function`** this is critical for many tensorflow features, and checkpointing is one of them. make sure that all the operations inside a `tf.function` are `tf.*` operations, if you try to mix python and `tf` inside the `tf.function` you might get into issues.
4. ** double check the loss function**: make sure your loss function is using proper `tf.*` operations or it's something that tensorflow can trace and use for automatic differentiation, avoid python pure operations within the gradients or loss function computations.

for more detailed information, i would suggest looking at the tensorflow documentation, specially the `tf.train.Checkpoint`, `tf.GradientTape` and `tf.function` pages, these are the main places where you can have hidden problems with checkpointing. i would also recommend the book "deep learning with python" by francois chollet, this book gives a lot of deep insight into how tensorflow and keras work behind the scenes. in the book there is a chapter focused on creating custom training loops that has many pointers to understanding how to use tensorflow with automatic differentiation that will give you an amazing starting point.

i hope this helps you with your checkpointing issues. oh, and by the way, i once spent two days trying to figure out why my model was training but not saving correctly, turns out i was saving the checkpoint *before* applying the gradients, i was saving a completely different model. after that i decided to take a coffee and had the greatest epiphany of my life, haha. good luck debugging your model!.
