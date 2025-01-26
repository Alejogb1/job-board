---
title: "What causes the 'Keyword argument not understood: 'query_shape'' TypeError when implementing a Vision Transformer (ViT) with Keras and TensorFlow 2.6.0?"
date: "2025-01-26"
id: "what-causes-the-keyword-argument-not-understood-queryshape-typeerror-when-implementing-a-vision-transformer-vit-with-keras-and-tensorflow-260"
---

TensorFlow 2.6.0 exhibits strict adherence to API parameter names within its Keras implementation, specifically when interfacing with model architectures that utilize layers built outside of the core Keras library. This rigidity, particularly evident in Vision Transformer (ViT) implementations derived from external sources or earlier TensorFlow versions, often manifests as a `TypeError: Keyword argument not understood: 'query_shape'` during model creation or subsequent fit/evaluation phases.

The root cause of this error lies in the incompatibility of parameter names defined in an externally created ViT model and the parameter names expected by TensorFlow's Keras layers, notably the `MultiHeadAttention` layer. While the core logic of attention mechanisms remains consistent, the naming conventions for parameters controlling input shape, specifically the query input, can differ. A ViT implementation might, for instance, use `query_shape` to specify the dimensions of the query tensor passed to the attention layer. In contrast, the `tf.keras.layers.MultiHeadAttention` layer, especially as implemented in TensorFlow 2.6.0, does not recognize this parameter name. It uses internally deduced or explicitly provided parameters that control the input shapes and projection transformations.

This disparity arises because custom ViT implementations frequently rely on an older, or less standard, approach to define input shapes directly as arguments to an attention layer. Prior to a more refined API, developers often used custom layers that accepted shape parameters directly, which is now considered an anti-pattern in the standard Keras/TensorFlow ecosystem. Furthermore, ViT implementations are often backported or repurposed from other libraries or frameworks where such a `query_shape` parameter might be a valid, standard part of the layer signature. When moving such a model into the environment of TensorFlow 2.6.0 and its official Keras modules, this conflict is exposed. Specifically, if the attention layer within the ViT's architecture is attempting to pass ‘query_shape’ as a keyword argument, it triggers a `TypeError`, because the `MultiHeadAttention` module does not process or accept such an argument during layer instantiation.

The `MultiHeadAttention` layer within TensorFlow 2.6.0 utilizes the shapes of the incoming tensors directly or relies on explicitly specified dimensions of the key, query, and value projection matrices, all managed internally without accepting an explicit 'query_shape' keyword argument. The resolution to this problem involves either adapting the ViT implementation to comply with TensorFlow's `MultiHeadAttention` layer’s expected inputs or to replace the custom `MultiHeadAttention` implementation with the Keras version, provided a correct parameterization is applied. This often means modifying the ViT code to use standard methods to deduce tensor shape and to avoid passing the incompatible argument.

Here are three code examples, demonstrating scenarios and solutions:

**Example 1: Incorrect ViT Implementation:**

```python
import tensorflow as tf

class CustomMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, query_shape, **kwargs):
        super(CustomMultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.query_shape = query_shape # Using query_shape directly

    def call(self, query, key, value):
        # Simplified implementation for demonstration
        return query

class ViTBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, query_shape, **kwargs):
        super(ViTBlock, self).__init__(**kwargs)
        self.mha = CustomMultiHeadAttention(num_heads=num_heads, key_dim=key_dim, query_shape=query_shape)

    def call(self, x):
        return self.mha(x, x, x)


# Example of usage that triggers the error
try:
  vit_block = ViTBlock(num_heads=8, key_dim=64, query_shape=(197, 768))
  input_tensor = tf.random.normal(shape=(1, 197, 768))
  output = vit_block(input_tensor)
except Exception as e:
  print(f"Error Encountered: {e}")
```

This code demonstrates a custom implementation of `MultiHeadAttention` within a ViT block that directly accepts a `query_shape` argument in its constructor, which the custom `call` method does not use, yet it remains part of the constructor's parameters. When this code is executed, even though the `call` function doesn't utilize `query_shape`, the error will be raised during the instantiation of the `CustomMultiHeadAttention` in the `ViTBlock` initializer by the `super` call of `CustomMultiHeadAttention`. The attempt to initialize it with a positional parameter that is not explicitly understood by the parent class is the cause.

**Example 2: Corrected ViT Implementation using `tf.keras.layers.MultiHeadAttention`:**

```python
import tensorflow as tf

class ViTBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, **kwargs): # Removed query_shape
        super(ViTBlock, self).__init__(**kwargs)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

    def call(self, x):
        return self.mha(x, x, x)

# Example of usage
vit_block = ViTBlock(num_heads=8, key_dim=64)
input_tensor = tf.random.normal(shape=(1, 197, 768))
output = vit_block(input_tensor)
print("Successfully passed through the ViT block")
```

This revised code replaces the custom attention layer with the standard `tf.keras.layers.MultiHeadAttention`. Importantly, it removes the erroneous `query_shape` from the `ViTBlock` constructor. The `MultiHeadAttention` layer infers input shapes from the tensors passed during the `call` method execution which eliminates the need to explicitly specify `query_shape`. Now, this code executes without raising the `TypeError`. It emphasizes how using the standardized Keras attention layer resolves the underlying issue, provided the parameterization aligns with expected behavior.

**Example 3: Corrected ViT Implementation using custom Attention with correct parameterization:**

```python
import tensorflow as tf

class CustomMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, **kwargs):
        super(CustomMultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.query_projection = tf.keras.layers.Dense(units=key_dim)
        self.key_projection = tf.keras.layers.Dense(units=key_dim)
        self.value_projection = tf.keras.layers.Dense(units=key_dim)

    def call(self, query, key, value):
        query_proj = self.query_projection(query)
        key_proj = self.key_projection(key)
        value_proj = self.value_projection(value)

        # Simplified implementation for demonstration, actual attention logic would be here
        return query_proj  # Return projected Query

class ViTBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, **kwargs):
        super(ViTBlock, self).__init__(**kwargs)
        self.mha = CustomMultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

    def call(self, x):
        return self.mha(x, x, x)


# Example of usage
vit_block = ViTBlock(num_heads=8, key_dim=64)
input_tensor = tf.random.normal(shape=(1, 197, 768))
output = vit_block(input_tensor)
print("Successfully passed through the ViT block")
```
Here, I illustrate a corrected custom attention implementation by removing the direct `query_shape` specification. Instead, the code now instantiates Dense layers (`query_projection`, `key_projection`, `value_projection`) responsible for projecting the inputs to the correct dimension before the actual attention calculation, which is abstracted in this example. The key point is that the input tensor shapes are now implied by the usage of the `Dense` layers, without relying on the `query_shape` argument which triggers the `TypeError`. The explicit input shapes are handled through the `Dense` layers, which directly project input tensors to the intended embedding dimension, thus removing the need to rely on passing shapes explicitly.

When encountering this issue, I recommend a three-pronged approach. First, verify if the ViT implementation uses `tf.keras.layers.MultiHeadAttention` or a custom layer. If custom, thoroughly examine its parameterization. Next, compare the layer signatures against those provided in TensorFlow 2.6.0 documentation for the `MultiHeadAttention` layer, or subsequent versions of the same. Finally, modify the code to either use the Keras version or ensure that parameterization adheres to the expected API by using projection layers to manage shape transformations. I have found that the TensorFlow API documentation and the official Keras tutorials are valuable resources, and often provide adequate guidance for modifying these specific errors. Additionally, examining other, well-maintained open-source ViT implementations can offer practical examples and code snippets that help resolve such incompatibilities. A clear understanding of Keras layers, and how input shapes are handled within `tf.keras.layers.MultiHeadAttention`, are crucial to resolving this specific issue with ViT implementations.
