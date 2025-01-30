---
title: "How can BERT models be saved at each epoch for later training?"
date: "2025-01-30"
id: "how-can-bert-models-be-saved-at-each"
---
Transfer learning with BERT and similar large transformer models often benefits from iterative training; a model pre-trained on a vast corpus may require further specialization on a smaller, task-specific dataset. Saving model checkpoints at the end of each training epoch allows for experimentation with different training lengths and enables rollback to an earlier point if the current training begins to overfit or yields poor results. Implementing this checkpointing strategy involves careful configuration of model saving and loading mechanisms within a chosen deep learning framework.

The fundamental approach involves leveraging the built-in model saving and loading capabilities provided by frameworks like PyTorch or TensorFlow. In essence, after each complete pass through the training dataset (an epoch), the current state of the model, including its learned parameters (weights and biases), optimizer states, and any other relevant training information, is serialized and stored on disk. This stored snapshot can then be loaded later to either resume training from that specific point or to evaluate the model’s performance at that stage of training. I have employed this strategy extensively while fine-tuning BERT for various text classification tasks, observing significant benefits in terms of adaptability and model stability during training.

The process typically requires specifying a directory where the saved checkpoints will be stored, a method for creating unique filenames for each epoch, and the actual saving and loading code using API functions provided by the deep learning library. Consistent checkpoint naming facilitates later analysis and simplifies the selection of the best-performing model based on validation metrics.

Consider using a systematic naming convention that includes the epoch number; this will prevent unintended overwriting of previous checkpoint files. Furthermore, saving optimizer state information along with model weights allows training to resume from a precise point without loss of learning progress. For instance, if using Adam optimizer, its moment values should also be persisted during checkpointing, and this is automatically done by using standard saving APIs.

Here’s how one would implement this in PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer
import os

# Assume `model` is a pre-trained BERT model and `optimizer` is configured.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
num_classes = 2 # For binary classification
model = nn.Sequential(model, nn.Linear(768, num_classes)) # Custom classifier
optimizer = optim.Adam(model.parameters(), lr=1e-5)
epochs = 5
checkpoint_dir = 'checkpoints' # Directory to store saved checkpoints

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


for epoch in range(epochs):
  # Training loop (omitted for brevity)
    for i in range(10): # Example training loop
      input_ids = torch.randint(0, tokenizer.vocab_size, (1,512))
      attention_mask = torch.ones(input_ids.shape)
      optimizer.zero_grad()
      output = model(input_ids, attention_mask=attention_mask)
      loss = torch.nn.CrossEntropyLoss()(output[0], torch.randint(0, num_classes, (1,)))
      loss.backward()
      optimizer.step()

    # Save model and optimizer states at the end of each epoch.
    checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
    print(f"Epoch {epoch+1} checkpoint saved to {checkpoint_path}")

# Example on how to load:
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch_number = checkpoint['epoch']
print(f"Model loaded at epoch: {epoch_number}")


```

This code snippet illustrates a basic implementation of epoch-based checkpointing. The `torch.save` function serializes a dictionary containing the model’s state, the optimizer’s state, and the epoch number, while `torch.load` restores this information. Note that I have created a custom classifier head on top of the Bert base model and will save that along with the base model. This allows complete recovery of training state, including the custom head which is necessary when using this approach.

Next, consider a scenario where one would want to limit the number of checkpoints stored and always keep the best models. This would be valuable when storage capacity is a constraint, or the training process has many epochs. This requires keeping track of validation metrics and deleting older, less performant models while keeping only the best performing one (or a limited number of best ones).

```python
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer
import os
import numpy as np


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
num_classes = 2 # For binary classification
model = nn.Sequential(model, nn.Linear(768, num_classes)) # Custom classifier
optimizer = optim.Adam(model.parameters(), lr=1e-5)
epochs = 5
checkpoint_dir = 'checkpoints'
best_val_loss = np.inf # Initialize best validation loss as infinity
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

def validate(model):
  # Placeholder validation logic, replace with actual validation dataset and metrics
  return torch.rand(1,1).item() # Example random validation loss

for epoch in range(epochs):
    # Training loop (omitted for brevity)
    for i in range(10): # Example training loop
      input_ids = torch.randint(0, tokenizer.vocab_size, (1,512))
      attention_mask = torch.ones(input_ids.shape)
      optimizer.zero_grad()
      output = model(input_ids, attention_mask=attention_mask)
      loss = torch.nn.CrossEntropyLoss()(output[0], torch.randint(0, num_classes, (1,)))
      loss.backward()
      optimizer.step()


    #Validate the model and decide whether to save or not
    val_loss = validate(model)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint_path = os.path.join(checkpoint_dir, f'best_model.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            }, checkpoint_path)
        print(f"Epoch {epoch+1} checkpoint saved, val_loss: {val_loss} to {checkpoint_path}")
    else:
        print(f"Epoch {epoch+1} not saved, val_loss:{val_loss}")

#Example on how to load:
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch_number = checkpoint['epoch']
val_loss = checkpoint['val_loss']

print(f"Model loaded at epoch: {epoch_number}, Validation loss: {val_loss}")
```

This updated snippet saves only the model that produces the best validation loss so far.  The validation step must be implemented according to the needs of your specific use case. I have used a random loss, as the actual validation process is problem-dependent and would need to be implemented by the user. A similar logic can be employed to keep track of several best models instead of only the single one. This approach ensures that only the most effective model checkpoints are retained, minimizing storage overhead. In my own practice, I often utilize a sliding window of the top 3-5 models based on the validation metrics.

Lastly, in TensorFlow, the process is similar but uses TensorFlow’s checkpointing system:

```python
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import os

# Assume `model` is a pre-trained TFBertModel and `optimizer` is configured
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained('bert-base-uncased')
num_classes = 2
model = tf.keras.Sequential([model, tf.keras.layers.Dense(num_classes)]) # Custom classifier
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

epochs = 5
checkpoint_dir = 'checkpoints'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

checkpoint_path = os.path.join(checkpoint_dir,'model_epoch_{epoch}')

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only = False # Set true if you only want to save best models
)

def validate(model):
  # Placeholder validation logic, replace with actual validation dataset and metrics
  return tf.random.normal([1]).numpy().item() # Example validation loss

# Example trainig loop (omitted)
for epoch in range(epochs):
  # Training loop, same logic as in the previous examples, not important in this context
    for i in range(10): # Example training loop
      input_ids = tf.random.uniform(shape=(1,512), maxval=tokenizer.vocab_size, dtype=tf.int32)
      attention_mask = tf.ones(input_ids.shape)

      with tf.GradientTape() as tape:
          output = model(input_ids, attention_mask=attention_mask)[0]
          loss = tf.reduce_mean(tf.keras.losses.CategoricalCrossentropy(from_logits=True)(tf.one_hot(tf.random.uniform(shape=(1,), minval=0, maxval=num_classes, dtype=tf.int32), num_classes),output))
      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
    val_loss = validate(model)
    print(f"Epoch {epoch+1}, validation loss: {val_loss}")

    cp_callback.on_epoch_end(epoch, logs = {'val_loss': val_loss}) # Manually call the checkpoint callback to control the saving process

latest = tf.train.latest_checkpoint(checkpoint_dir) # Gets the latest saved checkpoint name
model.load_weights(latest) # Loads the latest checkpointed model

```

This TensorFlow example uses `tf.keras.callbacks.ModelCheckpoint` to handle the saving of weights at each epoch. Again, a simplified training loop is shown as the focus here is on the checkpointing mechanism. TensorFlow provides mechanisms for saving more than only weights; for example the complete model structure can be saved, and this can be controlled via the same callback API. Similarly to PyTorch, optimizer state is also handled automatically by the framework when saving/loading entire models or weights. I tend to favour using `save_weights_only=True` as in most situations saving the structure itself is not needed. The logic here is more verbose and requires explicit usage of `on_epoch_end` in the training loop in order to properly save the weights. In my experience this is more cumbersome and prone to errors than in the PyTorch case.

For further investigation into this topic, I recommend consulting the official documentation of both PyTorch and TensorFlow related to model saving and loading, checkpointing, and callbacks; this will lead to understanding of advanced techniques as well as better ways to control saving frequency and parameters. Additionally, deep learning tutorials related to fine-tuning transformer models commonly have example usages of checkpointing techniques. Furthermore, the Hugging Face Transformers library provides its own `Trainer` class which includes built-in checkpointing, and investigating its source code can be informative as well.
