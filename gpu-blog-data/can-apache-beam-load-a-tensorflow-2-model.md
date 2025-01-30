---
title: "Can Apache Beam load a TensorFlow 2 model once?"
date: "2025-01-30"
id: "can-apache-beam-load-a-tensorflow-2-model"
---
Apache Beam, when used in conjunction with TensorFlow, does indeed present an efficient mechanism for loading a TensorFlow 2 model once within a distributed processing pipeline, leveraging its inherent capabilities to manage state across workers. The critical technique centers around utilizing `tf.saved_model.load` within a Beam `DoFn`’s `setup` method. This ensures model loading occurs only once per worker instance, rather than per processed element, thereby preventing redundant and computationally expensive reload operations.

My experience developing large-scale machine learning pipelines on Google Cloud Platform using Apache Beam and TensorFlow has repeatedly demonstrated the practicality of this approach. In scenarios involving billions of data points, the performance gains by avoiding per-element model reloads are substantial, shifting the bottleneck from model instantiation to data processing throughput. The key here is Beam's execution model: data is partitioned and processed by workers across a cluster. When a `DoFn` is instantiated on a given worker, its `setup` method is executed one time only. This provides the opportune moment to load a TensorFlow model. Following that setup phase, the `process` method handles each element within the worker’s partition, using the pre-loaded model, maximizing efficiency and reducing the overhead of model loading per item.

Let's delve into a conceptual example. Imagine a text classification task where you have a pre-trained TensorFlow model. Instead of loading this model with each document being processed, we want each worker to load it once at start-up.

Here is the first code example, illustrating the basic mechanism:

```python
import apache_beam as beam
import tensorflow as tf
from apache_beam.options.pipeline_options import PipelineOptions

class TextClassificationDoFn(beam.DoFn):
  def __init__(self, model_path):
    self.model_path = model_path
    self.model = None

  def setup(self):
    self.model = tf.saved_model.load(self.model_path)

  def process(self, element):
    # Assume element is a string representing text to classify.
    prediction = self.model(tf.constant([element]))
    return prediction

def run_pipeline(input_data, model_path):
    options = PipelineOptions()
    with beam.Pipeline(options=options) as pipeline:
        _ = (
            pipeline
            | 'CreateInput' >> beam.Create(input_data)
            | 'ClassifyText' >> beam.ParDo(TextClassificationDoFn(model_path))
            | 'PrintResults' >> beam.Map(print)
        )

if __name__ == '__main__':
  # Assume model is saved in ./my_saved_model directory
  # For simplicity, an example text corpus
  input_text = ['This is a positive review.',
                'This is a negative review.',
                'This is a neutral sentence.']
  model_path = './my_saved_model' #Placeholder path, needs an actual saved model
  run_pipeline(input_text, model_path)
```
In this first example, the `TextClassificationDoFn` class encapsulates the model loading logic. The `setup` method, called once per worker, loads the model from `self.model_path` using `tf.saved_model.load`. The `process` method receives text data elements and utilizes the pre-loaded `self.model` for classification. The pipeline is initiated using a set of sample input texts and the specified model location. However, there is an important assumption: that a model is pre-existing in the `model_path`. This highlights a critical step in a real-world application where the model needs to be saved in a format suitable for `tf.saved_model.load`.

Consider this example with a modified `process` function to provide additional detail about the prediction, which could include preprocessing:
```python
import apache_beam as beam
import tensorflow as tf
from apache_beam.options.pipeline_options import PipelineOptions

class TextClassificationDoFnEnhanced(beam.DoFn):
  def __init__(self, model_path):
    self.model_path = model_path
    self.model = None

  def setup(self):
    self.model = tf.saved_model.load(self.model_path)

  def process(self, element):
      # Assumes the input is a string representing raw text.
      # Some pre-processing steps could go here: tokenization, padding, etc.
      # For the sake of example, let's just make the input tensor
      processed_input = tf.constant([element])

      predictions = self.model(processed_input)
      # Assuming the model returns a score, or a tensor of scores.
      # Here, just for demonstration, let's just return the raw tensor and the input
      return (element, predictions)


def run_pipeline_enhanced(input_data, model_path):
    options = PipelineOptions()
    with beam.Pipeline(options=options) as pipeline:
        _ = (
            pipeline
            | 'CreateInput' >> beam.Create(input_data)
            | 'ClassifyText' >> beam.ParDo(TextClassificationDoFnEnhanced(model_path))
            | 'PrintResults' >> beam.Map(print)
        )

if __name__ == '__main__':
  input_text = ['This is a great day.',
                'The situation is concerning.',
                'The weather is fine']
  model_path = './my_saved_model'
  run_pipeline_enhanced(input_text, model_path)
```
Here, the `TextClassificationDoFnEnhanced` class demonstrates a process that returns the input and the prediction tensor. This demonstrates the idea that more complex logic around preprocessing or post-processing can also be done within the process step, using the already loaded model, once again avoiding redundant loads.

Finally, consider a scenario where more than one model might need loading, based on some condition. This can be done by adding more model loading to the `setup` method:

```python
import apache_beam as beam
import tensorflow as tf
from apache_beam.options.pipeline_options import PipelineOptions

class MultiModelDoFn(beam.DoFn):
  def __init__(self, model_path_1, model_path_2):
      self.model_path_1 = model_path_1
      self.model_path_2 = model_path_2
      self.model_1 = None
      self.model_2 = None


  def setup(self):
    self.model_1 = tf.saved_model.load(self.model_path_1)
    self.model_2 = tf.saved_model.load(self.model_path_2)

  def process(self, element):
      # Assumes element is a tuple of (text_input, model_choice).
      text_input, model_choice = element

      if model_choice == 1:
        prediction = self.model_1(tf.constant([text_input]))
      elif model_choice == 2:
        prediction = self.model_2(tf.constant([text_input]))
      else:
        prediction = None # Handle unknown choice

      return (text_input, model_choice, prediction)

def run_pipeline_multi_model(input_data, model_path_1, model_path_2):
    options = PipelineOptions()
    with beam.Pipeline(options=options) as pipeline:
        _ = (
            pipeline
            | 'CreateInput' >> beam.Create(input_data)
            | 'ProcessWithModels' >> beam.ParDo(MultiModelDoFn(model_path_1, model_path_2))
            | 'PrintResults' >> beam.Map(print)
        )

if __name__ == '__main__':
    input_data = [('This is model 1 data', 1),
                    ('This is model 2 data', 2),
                    ('This is another model 1 data', 1),
                    ('This is model 3, but its handled as default', 3)]
    model_path_1 = './my_model_1'
    model_path_2 = './my_model_2'
    run_pipeline_multi_model(input_data, model_path_1, model_path_2)

```
The `MultiModelDoFn` demonstrates the loading of two separate models in the `setup` function and subsequently selects the appropriate model based on a `model_choice` value present in the input element. This showcases the expandability of the single-load strategy, supporting scenarios with multiple models or model versions.

In summary, Apache Beam, through the use of `DoFn` and its `setup` method, provides an elegant method to load TensorFlow 2 models once per worker within a distributed processing environment, a crucial pattern for performance when dealing with machine learning models.

Regarding further learning, I recommend exploring the official Apache Beam documentation, specifically its sections on the `DoFn` class, its lifecycle, and the usage of pipelines. Furthering knowledge of TensorFlow's SavedModel format will enable proper model exports for deployment. Detailed knowledge on pipeline orchestration tools like Apache Airflow or Cloud Composer will allow for scheduling and monitoring these pipelines. Finally, understanding concepts in distributed processing would allow for more informed resource management.
