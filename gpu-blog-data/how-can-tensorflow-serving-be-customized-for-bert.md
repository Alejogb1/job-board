---
title: "How can TensorFlow Serving be customized for BERT or Transformer models?"
date: "2025-01-30"
id: "how-can-tensorflow-serving-be-customized-for-bert"
---
Deploying BERT or Transformer models using TensorFlow Serving presents unique challenges due to their architectural complexity and resource demands. Standard TensorFlow Serving setups, while functional, often fall short of optimal performance for these models. Customization is frequently required to address specific needs related to input preprocessing, output handling, and serving optimization. My experience deploying multiple BERT-based NLP systems in production environments has highlighted several critical areas for such customization, which I will elaborate on below.

A core aspect of tailoring TensorFlow Serving for BERT or Transformers involves modifying the standard input and output logic. BERT models, unlike simpler architectures, require tokenization and input formatting before being fed into the model. Conversely, their raw output tensors often need post-processing, such as applying softmax for classification tasks or extracting relevant features. These pre- and post-processing steps are not inherently handled by standard TensorFlow Serving and thus, require custom implementation within the model's serving graph. I've found it particularly useful to utilize TensorFlow's `tf.data` API within the serving function to perform these operations efficiently, rather than relying on external processes. This allows these operations to take advantage of optimized TensorFlow operations, resulting in improved throughput and reduced latency.

The initial strategy I employed, and often recommend, involves embedding pre- and post-processing steps directly within the saved model. Instead of solely exporting the raw BERT model, a complete TensorFlow graph is created encompassing tokenization, model inference, and the final output conversion. In practice, this is accomplished through creating a custom layer that combines the tokenizer and the BERT model into a single callable unit.

Here's an example illustrating how to integrate a tokenizer directly into the serving graph (simplified for brevity, using Hugging Face's tokenizer API):

```python
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

class BertServingLayer(tf.keras.layers.Layer):
    def __init__(self, pretrained_model_name, **kwargs):
        super(BertServingLayer, self).__init__(**kwargs)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.bert_model = TFBertModel.from_pretrained(pretrained_model_name)

    def call(self, inputs):
        encoded_inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors='tf')
        outputs = self.bert_model(encoded_inputs)
        return outputs.last_hidden_state[:, 0, :] # Extract CLS token output

# Example usage with text inputs
def serving_function():
    input_tensor = tf.keras.Input(shape=(1,), dtype=tf.string, name='text_input')
    serving_layer = BertServingLayer(pretrained_model_name='bert-base-uncased')
    output_tensor = serving_layer(input_tensor)
    serving_model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    return serving_model

if __name__ == '__main__':
    model = serving_function()
    tf.saved_model.save(model, 'saved_bert_model')
    print("Model saved successfully")
```

In this code, the `BertServingLayer` encapsulates the tokenizer and the BERT model. The `call` method automatically transforms raw text input into the required input format, passes it through the BERT model, and returns the embedding output. The `serving_function` creates a Keras model around this layer, accepting a string tensor as input, thereby directly accepting text input at inference. The output of this layer is directly compatible with serving. By saving this model using `tf.saved_model.save`, we encapsulate all processing logic within the SavedModel, significantly simplifying deployment with TensorFlow Serving. Note that the output extraction method (`outputs.last_hidden_state[:, 0, :]`) may differ based on specific model use cases, and may be replaced by any desired post processing logic.

Another customization avenue involves adapting the model's signature. The standard SavedModel signature is generally generic, accepting named inputs and producing named outputs. For complex models like BERT, explicit specification of input tensor shapes and types, as well as the expected output structure, becomes critical for clients. Furthermore, it allows for efficient batching and avoids unnecessary type checking. Custom signatures help TensorFlow Serving understand the intended model interface, enabling smoother integration with client-side code and optimal inference scheduling.

Here's an example illustrating how to define a custom signature:

```python
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import numpy as np

def serving_function_with_signature():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.string, name="text_input")])
    def predict_fn(text_input):
        encoded_inputs = tokenizer(list(text_input), padding=True, truncation=True, return_tensors='tf')
        outputs = model(encoded_inputs)
        probabilities = tf.nn.softmax(outputs.logits, axis=-1)
        return {"probabilities": probabilities}

    return predict_fn

if __name__ == '__main__':
    serving_fn = serving_function_with_signature()

    class ServeModel(tf.train.Checkpoint):
      def __init__(self, model_func):
        super(ServeModel, self).__init__()
        self.predict = model_func
    serve_model = ServeModel(serving_fn)

    tf.saved_model.save(
        serve_model,
        'saved_bert_model_signature',
        signatures={'serving_default': serve_model.predict.get_concrete_function()}
    )
    print("Model saved with signature successfully")
```

Here, a custom `predict_fn` is defined using `tf.function`, which accepts a single `text_input` tensor of type `string` with a flexible batch size. This is a significant improvement over relying on TensorFlow Serving to infer the input type and shape. Furthermore, a custom Checkpoint model class is created which takes the function and saves it. By calling `get_concrete_function()`, a specific signature is registered, allowing the model to expose the intended interface with explicit input and output names. This improves clarity and makes integration easier.

Finally, advanced serving optimization might be necessary for high throughput or low latency scenarios. TensorRT integration, specifically, can yield substantial improvements in inference speed. However, applying this often requires some manual work. I typically generate the TensorRT graph on the GPU and use a custom TensorFlow op within the served model to utilize the TensorRT execution engine. This requires building the engine separately and embedding its execution logic into the serving graph. The main logic of building and using a TensorRT graph must be created using appropriate API functions from the TF-TensorRT framework.

The example below demonstrates a simplified conceptual flow of wrapping a TensorRT execution within a custom TensorFlow operation (TensorRT integration is not included to maintain brevity; the emphasis here is on the interface):

```python
import tensorflow as tf
import numpy as np

class TensorRTWrapper(tf.keras.layers.Layer):
    def __init__(self, trt_engine_path, **kwargs):
        super(TensorRTWrapper, self).__init__(**kwargs)
        self.trt_engine_path = trt_engine_path

    def call(self, inputs):
        # This would normally load and execute a TensorRT engine
        # For this example, we use a dummy output as if TensorRT is running.
        output_shape = tf.shape(inputs)[0], 768 # Example embedding size 768
        dummy_output = tf.random.normal(output_shape)

        return dummy_output

def serving_function_trt(trt_engine_path):
    input_tensor = tf.keras.Input(shape=(None,), dtype=tf.string, name='text_input')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def process_text(text):
      encoded = tokenizer(list(text), padding=True, truncation=True, return_tensors='tf')
      return encoded['input_ids'], encoded['attention_mask']

    input_ids, attention_mask = tf.keras.layers.Lambda(process_text)(input_tensor)
    trt_layer = TensorRTWrapper(trt_engine_path)
    output_tensor = trt_layer((input_ids, attention_mask))
    serving_model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    return serving_model

if __name__ == '__main__':
    model_trt = serving_function_trt("dummy_trt_path")
    tf.saved_model.save(model_trt, 'saved_bert_model_trt')
    print("Model saved with TensorRT wrapper successfully")
```

In this case, the `TensorRTWrapper` class serves as a conceptual placeholder for loading and running a previously created TensorRT execution engine. The `call` method performs a placeholder random tensor generation but would normally load and infer with the provided input using a TensorRT engine. This can require extensive use of TensorRT's API to integrate the functionality.

To summarize, customizing TensorFlow Serving for BERT and Transformer models requires integrating tokenizer and post-processing directly into the serving graph, defining clear input and output signatures, and employing performance optimizations such as TensorRT. Documentation provided by the TensorFlow team on SavedModel format, serving signatures, and optimizing inference with TensorRT are invaluable resources. I recommend studying these and experimenting with the described strategies to determine optimal settings for any given deployment. Additionally, resources focusing on NLP model deployment pipelines and performance optimization, such as advanced TensorFlow model optimization guides and research papers on efficient inference, can greatly improve the process.
