---
title: "How can I read a model.pb file from an S3 bucket in Python?"
date: "2025-01-30"
id: "how-can-i-read-a-modelpb-file-from"
---
TensorFlow's protobuf-based model files, specifically `model.pb`, require careful handling when retrieved from cloud storage like Amazon S3. They are not simple text files and must be parsed correctly to reconstruct the computational graph. I've encountered numerous issues with direct downloads and attempted deserializations, revealing that a multi-step process involving appropriate libraries is crucial.

The process necessitates using the `boto3` library to interact with S3, alongside TensorFlow's own functionality for graph loading. Crucially, the model file needs to be downloaded as a byte stream first, rather than treated as a file path directly accessible within the S3 bucket. Direct file system paths don't apply to cloud storage, and attempting to provide a bucket location directly to TensorFlow's model loading will result in errors. Furthermore, relying solely on `tf.io.gfile` for S3 interaction often proves less flexible than a full `boto3` implementation. Here's how I’ve successfully approached this:

Firstly, establishing an S3 connection is fundamental. I’ve opted for `boto3`, as it provides fine-grained control over bucket access and allows more precise management of resource allocation. Authentication should be handled outside the scope of this function, either via environment variables, instance roles, or direct key provision in the client setup.

```python
import boto3
import tensorflow as tf
import io

def load_model_from_s3(bucket_name, s3_key):
    """
    Loads a TensorFlow model from an S3 bucket, returning a Graph object.
    Args:
        bucket_name (str): The name of the S3 bucket.
        s3_key (str): The S3 key (path within the bucket) to the model.pb file.

    Returns:
        tf.compat.v1.Graph: The loaded TensorFlow graph object.

    Raises:
        Exception: If any errors occur during S3 retrieval or graph loading.
    """
    try:
        s3 = boto3.client('s3')
        response = s3.get_object(Bucket=bucket_name, Key=s3_key)
        model_bytes = response['Body'].read()

        # Load the graph from the byte stream
        graph = tf.compat.v1.Graph()
        with graph.as_default():
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(model_bytes)
            tf.import_graph_def(graph_def, name='')
        return graph

    except Exception as e:
        raise Exception(f"Error loading model from S3: {e}")


# Example Usage
if __name__ == "__main__":
    try:
       graph = load_model_from_s3("your-s3-bucket-name", "path/to/your/model.pb")
       print("Model loaded successfully.")
       # You can proceed to access graph elements using:
       # operations = graph.get_operations()
       # tensors = graph.get_all_tensors()
    except Exception as e:
        print(f"Error: {e}")
```

In this example, the `load_model_from_s3` function retrieves the `model.pb` from S3, reads it into a byte stream, and uses `tf.compat.v1.GraphDef.ParseFromString` to load it into a `tf.compat.v1.Graph` object. The `graph` object then provides the necessary hooks to access individual operations and tensors defined within the model’s graph. The `if __name__ == '__main__':` block serves to illustrate usage with placeholder bucket and key values. Real implementation should replace these with actual S3 details. The crucial aspect here is the handling of the object body from `boto3` as a byte stream. Directly trying to treat the response as a file path will not work. The try-except blocks provide essential error handling to catch potential S3 retrieval or TensorFlow parsing issues.

The second example explores working with specific graph operations or tensors after loading the graph and executing an inference using the graph. After loading, you will often need to extract the operations required for feeding input and for extracting output data from the graph. This involves identifying these operation by name as specified within the `model.pb` file. You may need to inspect the model details with Tensorboard or other tools to identify these names before running the code.

```python
import boto3
import tensorflow as tf
import io
import numpy as np

def load_and_run_inference(bucket_name, s3_key, input_tensor_name, output_tensor_name, input_data):
    """Loads a TensorFlow model, performs inference and returns output tensor.

        Args:
            bucket_name (str): S3 Bucket Name.
            s3_key (str): S3 key to the model.pb.
            input_tensor_name (str): The name of the input tensor in the graph.
            output_tensor_name (str): The name of the output tensor in the graph.
            input_data (np.ndarray): Input data for the model.

        Returns:
            np.ndarray: The output tensor result of inference.

        Raises:
            Exception: If any issues with S3, graph loading or inference.
    """
    try:
        graph = load_model_from_s3(bucket_name, s3_key)

        # Input placeholder and output tensor
        input_tensor = graph.get_tensor_by_name(input_tensor_name)
        output_tensor = graph.get_tensor_by_name(output_tensor_name)

        with tf.compat.v1.Session(graph=graph) as sess:
            output_result = sess.run(output_tensor, feed_dict={input_tensor: input_data})
        return output_result

    except Exception as e:
        raise Exception(f"Error in inference: {e}")


# Example Usage
if __name__ == "__main__":
    try:
        input_shape = (1, 28, 28, 1)
        input_example = np.random.rand(*input_shape).astype(np.float32)
        output = load_and_run_inference(
        "your-s3-bucket-name",
        "path/to/your/model.pb",
        "input_1:0", # example placeholder input node
        "dense_1/BiasAdd:0", # example output node
        input_example)
        print(f"Inference output shape: {output.shape}")
        print(f"Inference output: {output}")


    except Exception as e:
        print(f"Error: {e}")
```

This example extends the prior one by specifying input and output tensor names. It then sets up a TensorFlow session to execute inference, passing the `input_data` into the model and returning the corresponding `output_result`. Placeholder tensor names `"input_1:0"` and `"dense_1/BiasAdd:0"` are used for demonstration, real world applications will require correct names from your specific `model.pb` file. Note the use of numpy to create an example input of the correct shape for the graph and proper data type.

Lastly, it’s worth considering how to load models using the newer SavedModel format, which is more common in modern TensorFlow applications and is easier to manage using native TensorFlow methods, including those supporting S3. Using the SavedModel format, you may bypass the need to load a `model.pb` from S3 directly. Below is a more straightforward approach to loading a SavedModel from S3. Here, TensorFlow's internal mechanisms can be leveraged. In this particular example, the model is saved in a compressed format.

```python
import boto3
import tensorflow as tf
import tarfile
import io
import os

def load_saved_model_from_s3(bucket_name, s3_key, extract_dir="temp_model"):
    """Loads a TensorFlow SavedModel from an S3 bucket, returning a SavedModel object.

        Args:
            bucket_name (str): The S3 bucket name.
            s3_key (str): The S3 key (path within the bucket) to the saved model (tar.gz).
            extract_dir (str): Local temporary directory to extract contents to.

        Returns:
            tf.compat.v1.saved_model.load: The loaded TensorFlow SavedModel object.

        Raises:
            Exception: If any errors occur during S3 retrieval or model loading.
    """
    try:
         s3 = boto3.client('s3')
         response = s3.get_object(Bucket=bucket_name, Key=s3_key)
         compressed_bytes = response['Body'].read()

        # Extract to local temp directory:
         if not os.path.exists(extract_dir):
                os.makedirs(extract_dir)
         with tarfile.open(fileobj=io.BytesIO(compressed_bytes), mode='r:gz') as tar:
              tar.extractall(path=extract_dir)

         # Loading model
         model = tf.compat.v1.saved_model.load_v2(export_dir=extract_dir)
         return model

    except Exception as e:
        raise Exception(f"Error loading SavedModel from S3: {e}")

#Example Usage
if __name__ == "__main__":
    try:
        model = load_saved_model_from_s3("your-s3-bucket-name", "path/to/saved_model.tar.gz")
        print("SavedModel loaded successfully.")
        # You can proceed with predictions, for example:
        # infer_func = model.signatures["serving_default"]
        # output = infer_func(tf.constant(input_data))
    except Exception as e:
        print(f"Error: {e}")

```
This example demonstrates downloading the compressed SavedModel, extracting the tar file to a temporary folder and then using the TensorFlow API to load the model from the file path. Note that a temporary folder is required for this operation. The example also highlights how to obtain a function for model inference after loading. This method provides a more modern approach, leveraging the structured metadata and versioning offered by SavedModel format.

For more in-depth understanding of these topics, I recommend exploring the official documentation for `boto3`, which includes comprehensive details on S3 interactions and various authentication methods. The TensorFlow documentation itself offers detailed insights into graph representation, loading methods, and the SavedModel format. Additionally, research into best practices for cloud model deployment, and security considerations for model loading is crucial for real-world application and production environments. Consider resources that cover topics like access control lists, bucket policies, encryption, and secure coding practices with cloud storage. Consulting the relevant API references and official tutorials for each library will enhance comprehension and problem-solving abilities for this complex subject.
