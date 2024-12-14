---
title: "What is the unexpected shape of output from raw_rnn and how to inspect weights in raw_rnn?"
date: "2024-12-14"
id: "what-is-the-unexpected-shape-of-output-from-rawrnn-and-how-to-inspect-weights-in-rawrnn"
---

ah, raw_rnn. that takes me back. i remember the first time i messed with it, thinking i understood the underlying mechanics, only to be completely bamboozled by the output shape. it's a rite of passage for any serious tensorflow user, i guess.

so, the thing with `raw_rnn` is that it's intentionally low-level. it gives you a lot of flexibility, which also means a lot of responsibility, especially when it comes to managing state and understanding what's going on with the output. other rnn wrappers kind of hold your hand with some assumptions, `raw_rnn` throws all of those out the window. 

the unexpected shape arises because, unlike its higher-level counterparts like `dynamic_rnn`, `raw_rnn` doesn't automatically gather and stack the outputs of the recurrent cells across timesteps. what happens is that in each cell iteration, *you* decide what to return and in what shape. and, that shape is what gets thrown back at you. it's not a nice, neat `[batch_size, max_time, hidden_size]` matrix unless *you* make it so.

let me illustrate with a basic example, before i dive into inspecting weights and all that. suppose you define a simple lstm cell, like this:

```python
import tensorflow as tf

cell = tf.keras.layers.LSTMCell(units=128)
```

and you have some input, say a sequence of shape `[batch_size, max_time, input_dim]`. for simplicity, i'll use random data:

```python
batch_size = 32
max_time = 20
input_dim = 64

inputs = tf.random.normal((batch_size, max_time, input_dim))
```

now, when using `raw_rnn`, you define the `loop_fn`. this function controls what happens at each timestep. most importantly, it defines what to return. the `loop_fn` will take the time as argument. and previous cell state, and input.

here's a basic example of how to set up the loop\_fn to accumulate all the outputs over time:

```python
def loop_fn(time, cell_output, cell_state, loop_state):
    elements_finished = (time >= max_time)
    
    if cell_output is None:
        next_cell_state = cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
        emit_output = tf.zeros([batch_size, cell.units], dtype=tf.float32)
        next_input = inputs[:, 0, :]
        next_loop_state = tf.TensorArray(dtype=tf.float32, size=max_time)

    else:
        next_cell_state = cell_state
        next_input = tf.cond(
            tf.reduce_any(elements_finished),
            lambda: tf.zeros([batch_size, input_dim], dtype=tf.float32),
            lambda: inputs[:, time, :],
        )
        emit_output = cell_output
        next_loop_state = loop_state.write(time - 1, cell_output)


    return elements_finished, next_input, next_cell_state, emit_output, next_loop_state
```

you can also define an initial state as well, and it's important to note that `raw_rnn` expects that, as opposed to `dynamic_rnn` that sets it for you under the hood:

```python
initial_state = cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)

_, final_state, loop_states = tf.raw_rnn(
        cell=cell,
        loop_fn=loop_fn,
        initial_state=initial_state,
        dtype=tf.float32
    )
outputs = loop_states.stack()
outputs = tf.transpose(outputs, [1, 0, 2])
```

if you look carefully at the `loop_fn`, we are using a `TensorArray` to accumulate outputs. if we don't do that, we will not have the output over time. and by stacking the outputs and transposing, we are effectively getting something similar to what `dynamic_rnn` spits out: `[batch_size, max_time, cell.units]`. without that explicit accumulation, `raw_rnn` will only give you the output of the very last timestep in the `cell_output` of the loop\_fn function

the important thing to remember here is that, in the loop function, we are returning `emit_output` in each step. that's what we get after all of the recurrency.

now, on to inspecting weights. the beauty of using keras cells is that they store their weights in a consistent way. you can access them using the `.trainable_weights` attribute.

let's say you want to check the weights of our lstm cell:

```python
print(cell.trainable_weights)
```

this will output a list of `tf.Variable` objects, each representing a weight matrix or bias vector in the lstm cell. the exact order and shape of these weights will depend on the specific cell you're using (lstm in our case), but you'll typically find weights for the input gate, forget gate, cell gate, output gate, and recurrent connections.

here is a more concrete example, let's take a look at the lstm's kernel:

```python
for var in cell.trainable_weights:
    if "kernel" in var.name:
      print(var.name)
      print(var.shape)
      kernel = var
```
now you can inspect the `kernel`. if we are talking about an lstm, the kernel contains input and recurrent weights for the 4 gates. it's shape will be `[input_dim + cell.units, cell.units * 4]`

now, you can directly inspect and work with the weights as tensors. it's your playground. this is what makes `raw_rnn` awesome for debugging or for really crazy setups. however it's not for the faint of heart.

a final example, is how to inspect the bias, it would be almost the same:
```python
for var in cell.trainable_weights:
    if "bias" in var.name:
        print(var.name)
        print(var.shape)
        bias = var
```
in this case, it will output bias tensor of shape `[cell.units*4]`

remember to use a tensorboard to inspect all the weights during training. it's quite useful, and if you do some advanced analysis, you can get the weight norms of your rnn layers over time.

as a rule of thumb, if you start with the cell layers from keras, it's a safe way to go, as keras cells have a standard implementation of the way weights are organized.

a final word of caution though. sometimes i forget, when working with rnn's and raw\_rnn is, that my inputs might be of `float64` and the internal state of the cell might be `float32`. if that is the case, you will get some casting issues. so always remember to check your dtypes.

regarding resources, i'd recommend going beyond the tensorflow documentation. while it's a great starting point, for deeper understanding, you should look into the original papers on rnn's, and lstms. "long short-term memory" by hochreiter and schmidhuber is an absolute must-read. also, for a broader perspective on recurrent networks, anything by elman is usually good. you can look up books such as "deep learning" by goodfellow, bengio and courville. it contains a good amount of information about rnns.

also it's important to note that there is no magic involved. these weights are just numbers, stored as tensors. that's basically it. and they are updated as part of the training process, using backpropagation.

anyway, i hope that clarifies the somewhat mysterious nature of `raw_rnn` outputs and gives you a starting point for weight inspection. it can feel daunting at first, but, trust me, the flexibility it gives you is worth the initial learning curve. once you've stared at enough raw_rnn outputs, you start to see the matrix... or at least a bunch of tensors stacked together.
