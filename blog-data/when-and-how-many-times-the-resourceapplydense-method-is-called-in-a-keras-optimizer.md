---
title: "When and how many times the '_resource_apply_dense' method is called in a keras Optimizer?"
date: "2024-12-15"
id: "when-and-how-many-times-the-resourceapplydense-method-is-called-in-a-keras-optimizer"
---

well, let's talk about `_resource_apply_dense` in keras optimizers. it’s a pretty deep cut, and i've spent more time than i'd like to say staring at this stuff.

first off, to nail down when and how often it’s called, we've got to understand what `_resource_apply_dense` actually *does*. it's the core method that applies gradients to the model's weights using a dense representation. in keras, many optimizers don't directly manipulate weight values; they instead use resources and variables, hence the resource prefix. think of it like handling specific bits of memory associated with your weights.

now, the 'when' is directly tied to the model training process. more precisely, it is called within the `apply_gradients` function of the optimizer, which is usually called in the `train_step` method of the model. this usually happens during a `fit` call. so, each time your model processes a batch of data during training, your model compute the gradient, then this gradient is passed to the optimiser, finally the optimiser calls this method `_resource_apply_dense` to update the weights of the model using these gradients. this means the method is invoked once for each trainable weight in your model, *per batch*, *per training step*. so, it's a pretty busy method if you have a lot of parameters in the network.

to see this in action, let’s look at a little bit of how keras structures its optimizers, it would be useful, i think. so when you create an instance of your optimizer, and then call `apply_gradients` this triggers the update of your variables.

here’s what i've noticed: if we use a plain `Adam` optimizer, it first computes the gradients of the losses with respect to the variables. then it updates the accumulators (like moving average of the gradients and square of gradients) and finally the variables using those.

```python
import tensorflow as tf

# toy model example
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# dummy data and target
dummy_input = tf.random.normal((32, 784))
dummy_target = tf.random.uniform((32, 10), minval=0, maxval=10, dtype=tf.int32)
dummy_target = tf.one_hot(dummy_target, depth=10)

with tf.GradientTape() as tape:
    output = model(dummy_input)
    loss = tf.keras.losses.categorical_crossentropy(dummy_target, output)

# compute the gradients
gradients = tape.gradient(loss, model.trainable_variables)

# apply gradients - this triggers _resource_apply_dense calls
optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# check the number of calls of apply gradients
print("num of trainable params:", len(model.trainable_variables))

```

this shows how the `apply_gradients` would be called at a normal training step.

now, regarding the 'how many times', it's directly related to how many trainable parameters the model has. each trainable parameter, like a single weight in a dense layer, gets its own gradient update. this is where the 'dense' part in `_resource_apply_dense` comes from, since those weights are typically stored in a dense matrix format. for convolutional or other kind of layers it's a little different it's handled by `_resource_apply` (without the 'dense' suffix)

so, let's make it clear, during the call to `apply_gradients`, the function `_resource_apply_dense` is called for *every* trainable parameter of the model. and this happens *for every batch* during your training phase, when the loss is backpropagated.

i remember this one time i was working on a particularly huge transformer model, and the training was just crawling along. i was tracking every tensor and op, trying to figure out why it was so slow. i was profiling everything when i realized how frequently `_resource_apply_dense` was getting invoked. just this method alone was taking a chunk of training time. in my case that led me to some memory optimizations related to how i was preparing the batches. it's funny how sometimes something so fundamental can become a performance bottleneck, right?

here is another code example showing how to go deeper inside an optimizer `apply_gradients` method:

```python
import tensorflow as tf

# toy model example
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# dummy data and target
dummy_input = tf.random.normal((32, 784))
dummy_target = tf.random.uniform((32, 10), minval=0, maxval=10, dtype=tf.int32)
dummy_target = tf.one_hot(dummy_target, depth=10)

with tf.GradientTape() as tape:
    output = model(dummy_input)
    loss = tf.keras.losses.categorical_crossentropy(dummy_target, output)

# compute the gradients
gradients = tape.gradient(loss, model.trainable_variables)
grads_vars = zip(gradients, model.trainable_variables)

# apply gradients - let's go deeper
_ = optimizer.apply_gradients(grads_vars)
for grad, var in grads_vars:
    print(f"name: {var.name} ; shape: {var.shape}")

    # access directly the update method
    # optimizer._resource_apply_dense(grad, var)
    # see the source code of the optimiser for this method
    # in tf v2.15 is here (example) :
    # tf.keras.optimizers.Adam._resource_apply_dense

```

in this code, if you uncomment the `optimizer._resource_apply_dense(grad, var)`, you will see that it does the same as calling apply_gradients, this will apply the gradient to the variable. this way we can see how the optimization method, at its base, updates each variable (weight) using the gradient information. it is essentially applying gradient descent to all the variables.

now, for resources if you really want to dive deeper into the internals of keras optimizers, the best way is always going to be the source code itself. go to the tensorflow github repository, and explore keras's optimizer implementations. it's actually quite readable. start with the base `Optimizer` class in `tf.keras.optimizers` and then the specific optimizer you're interested in (like `Adam`). a really good resource, although not specific to keras but generally on optimisation techniques, is the "deep learning" book from goodfellow, bengio, courville. specifically, the chapter on optimisation algorithms. it gives you a very complete and mathematical way of understanding this particular step of the optimization.

another tip: when debugging, use tf's eager execution to your advantage. this lets you step through the code and observe what happens inside the `_resource_apply_dense` with debuggers like pdb. this would let you see the gradients as they are and what's the update rule that is being applied.

let me show you one last example, where we print the updated variable value, after it was updated with a gradient:

```python
import tensorflow as tf

# toy model example
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='relu', input_shape=(1,)),
])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

# dummy data and target
dummy_input = tf.constant([[1.0]])
dummy_target = tf.constant([[0.0]])

with tf.GradientTape() as tape:
    output = model(dummy_input)
    loss = tf.keras.losses.mean_squared_error(dummy_target, output)

# compute the gradients
gradients = tape.gradient(loss, model.trainable_variables)
grads_vars = zip(gradients, model.trainable_variables)

# Before update
initial_weight = model.trainable_variables[0].numpy()
print(f"Initial weight value: {initial_weight}")

# apply gradients
optimizer.apply_gradients(grads_vars)

# After update
final_weight = model.trainable_variables[0].numpy()
print(f"Updated weight value: {final_weight}")
```

if you run this, you'll notice the value of the weight changed, that is because `apply_gradients` will execute internally the `_resource_apply_dense` and make this update using stochastic gradient descent. this shows in a very explicit way what this method does, it changes the value of a variable based on a calculated gradient.

so, in a nutshell, `_resource_apply_dense` is called once for each trainable parameter, for every batch of data during the model's training. it's the method where the gradients are finally used to update the model's weights. if you have more detailed questions about optimisers and gradients, let me know.
