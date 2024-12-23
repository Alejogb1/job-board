---
title: "How are Haiku and Jax weights initialized?"
date: "2024-12-23"
id: "how-are-haiku-and-jax-weights-initialized"
---

Alright, let's talk about how Haiku and Jax handle weight initialization. It's a topic I’ve spent quite a bit of time navigating, especially back during a project where we were pushing the limits of model complexity with very specific convergence goals. The nuanced decisions around initialization can, and often do, drastically impact training performance. You can easily spend hours chasing down vanishing gradients or plateauing losses if your initialization strategy is not well-considered. So, let’s break this down.

First, it's important to note that neither Haiku nor Jax *intrinsically* define initialization strategies. Rather, Jax provides the fundamental building blocks for defining them via its `jax.random` module and transformation capabilities, while Haiku leverages these tools to offer convenient abstractions. You’re not relying on some magic function; it's explicit, deterministic, and flexible. This means that weight initialization is primarily controlled by function transforms and explicit definition. Unlike some libraries which bake in a set of default options, this approach provides immense control.

To start, let’s delve into the mechanics of generating random numbers. Jax uses a functional approach to randomness. This means that every time you want a random number, you have to pass a `jax.random.PRNGKey`. These keys are essentially seeds, but they are advanced differently than traditional random number generators. Instead of in-place modifications, each `jax.random` function returns a *new* key alongside the random values. This ensures referential transparency and makes your models purely functional. It is quite a subtle shift from standard programming paradigms, and it took me a few iterations to wrap my head around it, particularly when debugging multi-device setups.

Now, let’s translate that to Haiku. Haiku relies on a `hk.Module` architecture where you'll typically define your layers. When you create a layer, such as a `hk.Linear`, `hk.Conv2D`, etc., these modules don't hold their weights. Weights are created and stored in a "params" dictionary that is returned by the model's initialization function. Inside the `__init__` method of a `hk.Module`, we use `self.get_parameter` to tell Haiku that a certain parameter must exist. These parameters are not *immediately* initialized; rather, they are *declared*. This might feel a bit delayed compared to other systems, but this approach grants Haiku its unique modularity. Initialization happens only when you call `hk.transform` and then the `init` function of the resulting transformation. The result is a tuple consisting of parameters and an optional state for trainable and non-trainable variables.

The actual initialization process is user-defined and it typically comes down to providing an initialization function to `hk.get_parameter` or similar methods which under the hood use `jax.random` functions. For example, when creating a linear layer, I might specify an orthogonal initialization strategy for the kernel and a zeros initialization for the bias.

Here is a first snippet, illustrating a basic layer and how initialization is done manually:

```python
import jax
import jax.numpy as jnp
import haiku as hk

def my_linear(input_dim, output_dim, init_scale=1.0):
    def init_func(shape, dtype=jnp.float32):
        return jax.random.normal(hk.next_rng_key(), shape, dtype=dtype) * init_scale

    w = hk.get_parameter("w", [input_dim, output_dim], init=init_func)
    b = hk.get_parameter("b", [output_dim], init=jnp.zeros)
    
    def forward(x):
        return jnp.dot(x, w) + b
    
    return forward

def net_fn(x):
    linear_layer = my_linear(5, 10)
    return linear_layer(x)


key = jax.random.PRNGKey(42)
transformed_net = hk.transform(net_fn)
dummy_input = jnp.ones(5)

params = transformed_net.init(key, dummy_input)
output = transformed_net.apply(params, key, dummy_input)
print(params)
print(output)

```

In this example, `my_linear` explicitly defines its initialization using `jax.random.normal` and also shows how to inject different initialization functions. We use a simple scaling factor (`init_scale`) to provide flexibility in controlling variance. Notice that each call to `hk.next_rng_key()` consumes a piece of the PRNG key. Each subsequent call to the same layer during the initialisation will produce independent random numbers thanks to that key management.

Now, you might be wondering about popular, often-used initialization methods like Xavier or Kaiming. These are generally implemented in terms of the basic random functions, but they have a more intelligent scaling strategy based on input dimensions of the layer to mitigate the issues of vanishing or exploding gradients which were common in early models. In Haiku and Jax, you'd typically create a custom function to perform this scaling logic.

Here's a snippet demonstrating a Xavier initialization implementation:

```python
import jax
import jax.numpy as jnp
import haiku as hk

def xavier_init(fan_in, fan_out, dtype=jnp.float32):
    limit = jnp.sqrt(6 / (fan_in + fan_out))
    def init_func(shape, dtype=jnp.float32):
        return jax.random.uniform(hk.next_rng_key(), shape, dtype=dtype, minval=-limit, maxval=limit)
    return init_func


def my_linear_xavier(input_dim, output_dim):
    init = xavier_init(input_dim, output_dim)
    w = hk.get_parameter("w", [input_dim, output_dim], init=init)
    b = hk.get_parameter("b", [output_dim], init=jnp.zeros)

    def forward(x):
        return jnp.dot(x, w) + b

    return forward
def net_fn_xavier(x):
    linear_layer = my_linear_xavier(5, 10)
    return linear_layer(x)
key = jax.random.PRNGKey(42)
transformed_net = hk.transform(net_fn_xavier)
dummy_input = jnp.ones(5)

params = transformed_net.init(key, dummy_input)
output = transformed_net.apply(params, key, dummy_input)
print(params)
print(output)
```

In this Xavier implementation, the `xavier_init` function calculates the appropriate range for the uniform distribution based on input (`fan_in`) and output (`fan_out`) dimensions of the weight matrix. You may notice, this is still using the basic building block `jax.random.uniform` but with parameters that are mathematically derived to avoid issues of gradients.

Finally, let's look at a simple Kaiming initialization:

```python
import jax
import jax.numpy as jnp
import haiku as hk
from functools import partial


def kaiming_uniform(gain, fan_in, dtype=jnp.float32):

    bound = jnp.sqrt(6 / fan_in) * gain
    def init_func(shape, dtype=jnp.float32):
        return jax.random.uniform(hk.next_rng_key(), shape, dtype=dtype, minval=-bound, maxval=bound)

    return init_func

def my_conv_kaiming(input_channels, output_channels, kernel_shape, stride):

    init_func = kaiming_uniform(gain=jnp.sqrt(2),fan_in=input_channels*jnp.prod(jnp.asarray(kernel_shape)) )

    w = hk.get_parameter("w", [ *kernel_shape,input_channels,output_channels], init=init_func)
    b = hk.get_parameter("b", [output_channels], init=jnp.zeros)

    def forward(x):
        return jax.nn.conv_general_dilated(x, w, window_strides=stride, padding="SAME") + b
    return forward

def net_fn_conv(x):
    conv_layer = my_conv_kaiming(3,16,(3,3), (1,1))
    return conv_layer(x)

key = jax.random.PRNGKey(42)
transformed_net = hk.transform(net_fn_conv)

dummy_input = jnp.ones((1,32,32,3))

params = transformed_net.init(key, dummy_input)
output = transformed_net.apply(params, key, dummy_input)
print(params)
print(output.shape)

```

This example uses a Kaiming initialization, with the gain term being passed into the init function. The key difference here, as opposed to Xavier, is that it scales the variance by *only* the input connections, not both input and output connections.

In summary, the beauty of Haiku and Jax lies in their explicit approach to initialization. You have complete control using basic random generators along with the knowledge of what you would like in the initialization scheme. This encourages deeper understanding, and avoids any hidden behavior. You can implement virtually any initialization scheme by understanding the mechanics behind it and applying them using the fundamental functions of `jax.random`.

For a deeper dive, I’d strongly recommend checking out the original papers on Xavier and Kaiming initializations (specifically "Understanding the difficulty of training deep feedforward neural networks" by Glorot & Bengio, and "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification" by He et al. ) . Also, spend some time with the Jax and Haiku documentation itself as the best source of truth. Finally, reading up on Functional programming paradigms can give you a better understanding of why Jax and Haiku are designed the way they are. It really helped me avoid errors in the early days of my use.
