---
title: "How do I load a pretrained model and use it to train on a small dataset?"
date: "2024-12-23"
id: "how-do-i-load-a-pretrained-model-and-use-it-to-train-on-a-small-dataset"
---

Alright,  Having spent a good chunk of my career navigating the nuances of machine learning pipelines, this is a scenario I've encountered more times than I can count. Loading a pretrained model and adapting it to a smaller, specialized dataset is a crucial technique, particularly when computational resources or data availability are constraints. It’s a balancing act: leveraging existing knowledge while fine-tuning to capture specifics of your new data. Here's how I approach it, focusing on common pitfalls and practical strategies.

The fundamental idea is transfer learning. Instead of training a model from scratch, which requires vast amounts of data and computational power, you’re using a model that has already learned a wealth of features from a large, generalized dataset. This usually comes in the form of pre-trained weights. The process generally involves a few key steps: selecting a suitable pretrained model, adapting the architecture if necessary, loading the pretrained weights, and then fine-tuning it on your smaller dataset.

Selecting the "right" model depends largely on the nature of your task and dataset. For instance, if you're dealing with image data, a model pre-trained on ImageNet, such as a ResNet or VGG architecture, is often a good starting point. For natural language tasks, models pre-trained on large text corpora, like BERT or GPT models, are the usual contenders. The key consideration is the similarity between the pre-training task and your target task. If the representations learned by the pretrained model are broadly relevant to your data, you'll see better initial performance and require less fine-tuning.

Now, let’s look at the actual code. I’ll use Python with TensorFlow/Keras for demonstration, as it’s a common and versatile framework. Let's say we're working with a small image classification task. I have had a similar case years ago where I was trying to classify different types of microscopic algae, a project that did not have a vast amount of training data.

**Example 1: Basic Image Classification with TensorFlow/Keras**

```python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define input image size and number of classes
img_height, img_width = 224, 224
num_classes = 3 # Example: 3 types of algae

# Load a pretrained ResNet50 model (excluding the top fully connected layer)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the layers of the base model (prevent training these during fine-tuning)
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Load and augment data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Replace with your actual paths
train_generator = train_datagen.flow_from_directory(
    'path/to/your/training/data',  # e.g., 'data/train'
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='categorical'
)

# Train the model
model.fit(train_generator, epochs=10)
```

In this first example, we load a ResNet50, remove its classification head, and add new layers tailored to our `num_classes`. We freeze the convolutional layers of the ResNet50 to leverage its pre-trained weights without altering them during the initial fine-tuning phase. Data augmentation is employed to mitigate the effect of having a smaller dataset. This method is incredibly effective in quickly adapting a pre-trained model to your domain.

Let's consider a second scenario: dealing with sequence data like text. Suppose I'm adapting a pre-trained BERT model for sentiment analysis on product reviews, again, something I worked on a few years back for a niche e-commerce company.

**Example 2: Text Classification with Transformers (Hugging Face)**

```python
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2) # binary sentiment: positive/negative

# Load and preprocess your text data (replace with your data loading)
data = pd.read_csv('path/to/your/reviews.csv')  # Assuming your data has 'text' and 'sentiment' columns
X = data['text'].tolist()
y = data['sentiment'].tolist() # 0 for negative, 1 for positive

# Split into training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize the text data
def tokenize(sentences, tokenizer):
    input_ids = []
    attention_masks = []
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=128,  # Adjust max length as needed
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='tf'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    return tf.concat(input_ids, axis=0), tf.concat(attention_masks, axis=0)


train_input_ids, train_attention_masks = tokenize(X_train, tokenizer)
val_input_ids, val_attention_masks = tokenize(X_val, tokenizer)
train_labels = tf.convert_to_tensor(y_train)
val_labels = tf.convert_to_tensor(y_val)

# Define optimizer and loss
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

# Train step function
@tf.function
def train_step(input_ids, attention_masks, labels):
    with tf.GradientTape() as tape:
      outputs = model(input_ids, attention_masks=attention_masks)
      loss = loss_fn(labels, outputs.logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    metric.update_state(labels, outputs.logits)
    return loss

# Training loop
epochs = 3  # You might need more
batch_size = 8

for epoch in range(epochs):
    for batch in range(0, len(train_input_ids), batch_size):
        batch_input_ids = train_input_ids[batch:batch + batch_size]
        batch_attention_masks = train_attention_masks[batch:batch + batch_size]
        batch_labels = train_labels[batch:batch + batch_size]
        loss = train_step(batch_input_ids, batch_attention_masks, batch_labels)

    val_outputs = model(val_input_ids, attention_masks=val_attention_masks)
    val_metric.update_state(val_labels, val_outputs.logits)
    print(f"Epoch: {epoch + 1}, Loss: {loss.numpy():.4f}, Train Accuracy: {metric.result().numpy():.4f}, Val Accuracy: {val_metric.result().numpy():.4f}")
    metric.reset_state()
    val_metric.reset_state()
```
This code uses the `transformers` library to load BERT, and you can see how I preprocess the input text data by tokenizing them. The key is setting the right `num_labels` argument when loading `TFBertForSequenceClassification`. Fine-tuning the whole BERT model, including its transformer layers is typically performed for such scenarios. This example further showcases how we manage text data and incorporate a more complex transformer based pre-trained model.

Let’s look at a final example, which considers a case that goes beyond vanilla training. Let's suppose that in a past project I needed to transfer learning for a model that needs a modified output layer, rather than simply swapping out a fully connected layer like in Example 1. For this example, consider using a pre-trained encoder-decoder model like a transformer model for a sequence-to-sequence translation, and adapting it for code generation, something I have had to do in the past.

**Example 3:  Encoder-Decoder Model Adaptation**

```python
import tensorflow as tf
from transformers import T5Tokenizer, TFT5ForConditionalGeneration
import pandas as pd
from sklearn.model_selection import train_test_split


#Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = TFT5ForConditionalGeneration.from_pretrained("t5-small")

# Example Data loading and Preprocessing (replace with your own data)
df = pd.DataFrame({
    'input_code': ['python:print("hello")', 'javascript:console.log("world")', 'java:System.out.println("foo");'],
    'output_code': ['print("hello")', 'console.log("world")', 'System.out.println("foo");']
})
X = df['input_code'].tolist()
y = df['output_code'].tolist()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def tokenize_data(input_texts, target_texts, tokenizer, max_len=128):
  input_ids = []
  attention_masks = []
  target_ids = []

  for input_text, target_text in zip(input_texts, target_texts):
    input_encoded = tokenizer.encode_plus(input_text, max_length=max_len, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='tf')
    target_encoded = tokenizer.encode_plus(target_text, max_length=max_len, padding='max_length', truncation=True, return_tensors='tf')

    input_ids.append(input_encoded['input_ids'])
    attention_masks.append(input_encoded['attention_mask'])
    target_ids.append(target_encoded['input_ids'])

  return tf.concat(input_ids, axis=0), tf.concat(attention_masks, axis=0), tf.concat(target_ids, axis=0)


train_input_ids, train_attention_masks, train_labels = tokenize_data(X_train, y_train, tokenizer)
val_input_ids, val_attention_masks, val_labels = tokenize_data(X_val, y_val, tokenizer)

# Define optimizer and loss
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

@tf.function
def train_step(input_ids, attention_masks, labels):
    with tf.GradientTape() as tape:
        outputs = model(input_ids, attention_masks=attention_masks, labels=labels)
        loss = outputs.loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    metric.update_state(labels, outputs.logits)
    return loss


epochs = 3
batch_size = 2

for epoch in range(epochs):
    for batch in range(0, len(train_input_ids), batch_size):
        batch_input_ids = train_input_ids[batch:batch + batch_size]
        batch_attention_masks = train_attention_masks[batch:batch + batch_size]
        batch_labels = train_labels[batch:batch + batch_size]
        loss = train_step(batch_input_ids, batch_attention_masks, batch_labels)

    val_outputs = model(val_input_ids, attention_masks=val_attention_masks, labels=val_labels)
    val_loss = val_outputs.loss
    val_metric.update_state(val_labels, val_outputs.logits)
    print(f"Epoch: {epoch + 1}, Loss: {loss.numpy():.4f}, Train Accuracy: {metric.result().numpy():.4f}, Validation Loss: {val_loss.numpy():.4f} Val Accuracy: {val_metric.result().numpy():.4f}")
    metric.reset_state()
    val_metric.reset_state()
```

In the above example, we use the T5 model from the transformers library. This example shows how you can transfer learn an encoder-decoder model with minimal adaptation to the model's layers. Note, that we’re using this example to adapt a pre-trained transformer model for a sequence-to-sequence task using a dataset with source and target code examples. I’ve had to do similar projects for specific coding languages, and fine-tuning is almost always necessary for even a pre-trained model. The model directly minimizes the cross entropy loss of the target outputs conditioned on the inputs, and so we do not need to add additional output layers.

Key resources I would highly recommend for diving deeper into these techniques include "Deep Learning" by Goodfellow, Bengio, and Courville for a comprehensive theoretical background; and for specific model architectures and applications, research papers from the respective authors of the models you intend to use. For practical implementations, the documentation of libraries like TensorFlow and Hugging Face Transformers is indispensable. They have excellent tutorials and guides that often mirror the process described here.

In summary, loading pre-trained models and fine-tuning them on smaller datasets is a powerful method in modern machine learning. Careful model selection, proper data preprocessing, and a solid training loop is necessary for this to work successfully. It’s a skill that, with practice, can significantly reduce development time and improve model performance, especially when data or computational resources are limited.
