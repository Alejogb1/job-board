---
title: "How can batch normalization be shuffled for MoCo in TensorFlow 2?"
date: "2025-01-30"
id: "how-can-batch-normalization-be-shuffled-for-moco"
---
Batch normalization's inherent sensitivity to batch statistics poses a unique challenge when applying it within the Momentum Contrast (MoCo) framework, specifically when attempting to maintain consistent batch statistics across the query and key branches. Shuffling batch statistics in MoCo, though seemingly counterintuitive given normalization's goal of consistent means and variances, is precisely what's necessary to avoid information leakage and maintain effective contrastive learning. Without strategic shuffling, the key branch's network, a moving average of the query network, would operate on batch statistics strongly correlated with the current query batch, undermining the premise of contrasting against different, negative examples.

The core issue stems from how MoCo constructs its contrastive pairs. During training, a query image is passed through the query encoder, while its corresponding key image is encoded by the key encoder – which is typically an exponentially smoothed version of the query encoder. The key images are not typically in the same batch as the query images; they are often selected from a queue of previous key embeddings. If batch normalization is applied directly within the key encoder, it would compute its statistics based on these key examples, which, while technically different, have already influenced the query encoder. This direct association undermines the contrastive nature of MoCo, effectively making the task too simple by correlating positive and negative samples through the shared batch norm statistics.

To circumvent this, we need to maintain separate batch normalization statistics for each key. The fundamental principle is to decouple the batch normalization calculations for the query and key branches, even when they are operating on similar or related inputs. This involves managing an inventory of batch normalization statistics— specifically means and variances— for every element stored in the MoCo key queue. When a key is retrieved from the queue, its corresponding stored batch normalization statistics must be applied instead of those computed from the current batch of keys.

Here's how I've approached this challenge in TensorFlow 2. First, we need to maintain a queue that stores not just the key embeddings but also their associated batch norm statistics. We cannot simply calculate the stats when keys are queued; the stored stats must be those observed when the keys were *originally encoded*. This means that when we enqueue a key embedding, we must also enqueue the output of batch normalization prior to any further encoding. I have experimented with maintaining these paired statistics within a TensorFlow variable to allow quick access during dequeue operations. This approach avoids the overhead of recomputing these statistics every time we fetch a key.

```python
import tensorflow as tf

class ShuffledBatchNorm(tf.keras.layers.Layer):
    def __init__(self, axis=-1, momentum=0.99, epsilon=1e-3, **kwargs):
        super(ShuffledBatchNorm, self).__init__(**kwargs)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.moving_mean = None
        self.moving_variance = None

    def build(self, input_shape):
        param_shape = [input_shape[self.axis]]
        self.moving_mean = self.add_weight(
            name='moving_mean',
            shape=param_shape,
            initializer=tf.zeros_initializer(),
            trainable=False
        )
        self.moving_variance = self.add_weight(
            name='moving_variance',
            shape=param_shape,
            initializer=tf.ones_initializer(),
            trainable=False
        )
        super().build(input_shape)


    def call(self, inputs, training=None, bn_stats=None):
        if training or bn_stats is None: # Apply batch norm as usual if training or no precomputed stats provided
            mean, variance = tf.nn.moments(inputs, axes=[i for i in range(len(inputs.shape)) if i != self.axis], keepdims=True)
            self.moving_mean.assign(self.moving_mean * self.momentum + mean * (1 - self.momentum))
            self.moving_variance.assign(self.moving_variance * self.momentum + variance * (1 - self.momentum))

            output = tf.nn.batch_normalization(inputs, mean, variance, None, None, self.epsilon) # Scale and shift are managed elsewhere for this context

            if training: # Return actual statistics for storage
                return output, mean, variance
            else:
                return output
        else: # Apply stored batch norm stats from the queue
            mean, variance = bn_stats
            return tf.nn.batch_normalization(inputs, mean, variance, None, None, self.epsilon)

```

This custom `ShuffledBatchNorm` layer encapsulates the logic of applying either batch statistics computed on the fly (during the query processing) or applying previously calculated, enqueued statistics. The layer's behavior hinges on the `bn_stats` argument, which, when populated, triggers the substitution of the current batch statistics with the provided precomputed statistics.

The second crucial piece is the queue management. Upon encoding a key, the output of this `ShuffledBatchNorm` layer *along with its computed mean and variance* are enqueued into a FIFO queue along with the key embedding itself. When a key is dequeued, the associated mean and variance are retrieved and provided to the ShuffledBatchNorm layer *during the processing of that key*. This assures the batch norm layer uses the statistics stored with the key rather than the ones for the current mini-batch.

```python
import collections

class KeyQueue:
    def __init__(self, queue_size, embedding_dim):
        self.queue_size = queue_size
        self.embedding_dim = embedding_dim
        self.queue = collections.deque(maxlen=queue_size)

    def enqueue(self, key_embedding, bn_mean, bn_variance):
        self.queue.append((key_embedding, bn_mean, bn_variance))

    def dequeue(self):
        if not self.queue:
            return None, None, None
        key_embedding, bn_mean, bn_variance = self.queue.popleft()
        return key_embedding, bn_mean, bn_variance

    def is_full(self):
        return len(self.queue) == self.queue_size

    def is_empty(self):
        return len(self.queue) == 0
```

This `KeyQueue` manages the key embeddings and their associated batch norm statistics. When new keys are created after a query, they are enqueued alongside the mean and variance observed during their encoding.  When keys are fetched, they are dequeued with their stats.

Finally, we assemble the components into a MoCo encoder. The key encoder, which uses the smoothed parameters from the query encoder, utilizes the `ShuffledBatchNorm` layer. During key encoding, the batch norm statistics retrieved from the `KeyQueue` are fed into this layer. This guarantees that each key utilizes the proper batch statistics. The query encoder uses the same layer but with the conventional batch norm behaviour.

```python
class MoCoEncoder(tf.keras.Model):
    def __init__(self, embedding_dim, queue_size, base_encoder, **kwargs):
        super(MoCoEncoder, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.queue_size = queue_size
        self.base_encoder = base_encoder # This could be any model that produces a latent representation
        self.key_queue = KeyQueue(queue_size, embedding_dim)
        self.momentum = 0.999
        self.key_encoder = tf.keras.models.clone_model(self.base_encoder)
        self._update_weights(source=self.base_encoder, target=self.key_encoder, tau=0.0) # Initialize the key encoder weights with query encoder weights


    def call(self, query_images, training=True):
         # Process Query Images
        query_latent, query_bn_mean, query_bn_variance = self.base_encoder(query_images, training=training)
        query_latent = tf.nn.l2_normalize(query_latent, axis=-1) #Normalize query embeddings

        if training:
            # Enqueue Key Images
            with tf.GradientTape() as tape:
               key_latent, key_bn_mean, key_bn_variance = self.key_encoder(query_images, training=training)
            key_latent = tf.nn.l2_normalize(key_latent, axis=-1)
            self.key_queue.enqueue(key_latent, key_bn_mean, key_bn_variance)
            
            # Dequeue and Process old Keys
            key_latent_dequeued = []
            key_bn_mean_dequeued = []
            key_bn_variance_dequeued = []

            while not self.key_queue.is_empty():
               key_embedding, bn_mean, bn_variance = self.key_queue.dequeue()
               key_latent_dequeued.append(key_embedding)
               key_bn_mean_dequeued.append(bn_mean)
               key_bn_variance_dequeued.append(bn_variance)

            if key_latent_dequeued:
              key_latent_dequeued = tf.stack(key_latent_dequeued)
              key_bn_mean_dequeued = tf.stack(key_bn_mean_dequeued)
              key_bn_variance_dequeued = tf.stack(key_bn_variance_dequeued)


              key_latent_processed = self.key_encoder(key_latent_dequeued, training=False, bn_stats=[key_bn_mean_dequeued, key_bn_variance_dequeued] )
              return query_latent, key_latent_processed
            else:
              return query_latent, tf.zeros([query_images.shape[0],self.embedding_dim], dtype=tf.float32)
        else: # In evaluation do not queue or dequeue
          return query_latent, tf.zeros([query_images.shape[0],self.embedding_dim], dtype=tf.float32)


    def _update_weights(self, source, target, tau):
            for source_weight, target_weight in zip(source.trainable_variables, target.trainable_variables):
                  target_weight.assign(target_weight * tau + source_weight*(1-tau))


    def train_step(self, data):
        query_images, _ = data
        with tf.GradientTape() as tape:
           query_latent, key_latent = self(query_images, training=True)

           #Contrastive loss
           similarity = tf.matmul(query_latent, key_latent, transpose_b=True)
           labels = tf.range(tf.shape(query_latent)[0])
           loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=similarity)
           loss = tf.reduce_mean(loss)


        trainable_variables = self.base_encoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        self._update_weights(source=self.base_encoder, target=self.key_encoder, tau=self.momentum)

        return {"loss": loss}
```

This final class `MoCoEncoder` ties it all together. This simplified example shows a minimal MoCo loop. During each step of training, query images are passed through the query encoder, which outputs embeddings and batch norm statistics. Then, those are then enqueued. When enough keys are available, they are dequeued with their associated batch norm statistics, and then the embeddings of those keys are created via the key encoder. Critically, during the key's encoding, we apply the associated statistics, ensuring that no information is shared with the current batch. This prevents information leakage. Finally, the model outputs the query embeddings and key embeddings which are then passed to a contrastive loss.

For further exploration into this topic, I would suggest a review of the original MoCo paper, which provides the foundational concept. Additionally, resources focusing on batch normalization's behaviour in various contexts could be of further value in gaining a broader understanding. Examining other contrastive learning techniques will further emphasize this unique requirement of batch statistics management in MoCo. The best way to truly understand this issue is through careful implementation and observation of how shuffling these statistics impacts the loss and the final performance.
