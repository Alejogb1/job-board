---
title: "How to structure data for TensorFlow Serving classification requests via gRPC?"
date: "2025-01-30"
id: "how-to-structure-data-for-tensorflow-serving-classification"
---
TensorFlow Serving's gRPC interface expects a very specific structure for classification requests.  The crucial detail often overlooked is the necessity for strict adherence to the `PredictRequest` protocol buffer message definition.  Over the years, I've debugged countless integration issues stemming from minor discrepancies in this structure, leading to cryptic error messages.  This response details the correct protocol buffer message construction for TensorFlow Serving classification, along with illustrative code examples in Python, C++, and Java.

**1.  Explanation of the `PredictRequest` Structure:**

The core of a TensorFlow Serving gRPC classification request lies within the `PredictRequest` message. This message contains several fields, but for classification, the essential components are:

* **`model_spec`:** This field identifies the model to be used for prediction.  It requires at least the `name` field specifying the model's name as registered with TensorFlow Serving.  The `signature_name` field, while optional, is highly recommended.  It allows specifying a particular signature within the model, particularly useful when a model has multiple input and output signatures.  For classification, the signature name is typically `serving_default`.

* **`inputs`:** This is a map containing input tensors.  The key is the name of the input tensor (as defined in your TensorFlow model), and the value is a `TensorProto` message representing the input data.  The `TensorProto` message needs to be populated correctly, specifying the data type (`dtype`) and the shape of the input tensor.  This is where many errors occur, often due to shape mismatches or incorrect data type specifications. The input tensor should represent the feature vector for the classification task.  Its shape should align with the expected input shape of your TensorFlow model.

The response from TensorFlow Serving is a `PredictResponse` message containing the output tensors. For classification, this typically includes the predicted class labels and, potentially, associated probabilities or scores.  The structure of the output tensors mirrors the structure you defined in your TensorFlow model's signature.

**2. Code Examples:**

**a) Python:**

```python
import tensorflow as tf
import tensorflow_serving_api as tf_serving

# Create a PredictRequest message
request = tf_serving.PredictRequest()
request.model_spec.name = "my_classification_model"
request.model_spec.signature_name = "serving_default"

# Prepare input data (example: a single image represented as a numpy array)
input_data = np.array([[[128, 128, 128]], [[64, 64, 64]]], dtype=np.uint8) # Example shape (2,1,1,3)

# Convert numpy array to TensorProto
tensor_proto = tf.make_tensor_proto(input_data, shape=[2,1,1,3])

# Add the input tensor to the request
request.inputs["input_image"].CopyFrom(tensor_proto) # Assumes input tensor name is "input_image"


# Send the request and retrieve the response (error handling omitted for brevity)
with grpc.insecure_channel('localhost:8500') as channel:
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  response = stub.Predict(request, 10.0) # timeout set to 10 seconds

# Process the response (access predicted classes and probabilities)
predictions = tf.make_ndarray(response.outputs["output_classes"]) # Assumes output tensor name is "output_classes"
print(predictions)
```

This Python example leverages the `tensorflow_serving_api` library to create and send the `PredictRequest`. Note the explicit specification of the model name, signature name, and the conversion of NumPy array to `TensorProto`.  Remember to replace `"my_classification_model"`, `"input_image"`, `"output_classes"`, and the localhost address and port with your model and server details.

**b) C++:**

```cpp
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "grpcpp/grpcpp.h"

// ... (Grpc channel and stub creation) ...

tensorflow::serving::PredictRequest request;
request.mutable_model_spec()->set_name("my_classification_model");
request.mutable_model_spec()->set_signature_name("serving_default");

// Prepare input data (example: a single feature vector)
std::vector<float> input_data = {0.1f, 0.5f, 0.2f};

// Create a TensorProto
tensorflow::TensorProto tensor_proto;
tensorflow::TensorShape shape({3}); // Shape of the input vector
tensorflow::Tensor tensor(tensorflow::DT_FLOAT, shape);
auto tensor_mapped = tensor.flat<float>();
for(int i = 0; i < input_data.size(); ++i){
    tensor_mapped(i) = input_data[i];
}
tensor.AsProtoField(&tensor_proto);


(*request.mutable_inputs())["input_features"].PackFrom(tensor_proto);


// ... (Send request and process response) ...
```

This C++ example demonstrates creating the `PredictRequest` manually, including the creation and population of the `TensorProto` using the TensorFlow C++ API.  Thorough error handling and resource management are crucial for production-ready C++ code.  Again, remember to replace placeholders with your specific model and server information.

**c) Java:**

```java
import org.tensorflow.framework.TensorProto;
import org.tensorflow.framework.TensorShapeProto;
import org.tensorflow.serving.PredictionServiceGrpc;
import org.tensorflow.serving.PredictRequest;
import org.tensorflow.serving.PredictResponse;
import io.grpc.*;

// ... (Grpc channel and stub creation) ...

PredictRequest request = PredictRequest.newBuilder()
    .setModelSpec(ModelSpec.newBuilder().setName("my_classification_model").setSignatureName("serving_default").build())
    .build();

// Prepare input data (example: a single feature vector)
float[] inputData = {0.1f, 0.5f, 0.2f};

// Create a TensorProto (simplified for brevity,  consider using TensorFlow Java API for robust creation)
TensorProto.Builder tensorProtoBuilder = TensorProto.newBuilder();
TensorShapeProto.Builder shapeBuilder = TensorShapeProto.newBuilder().addDim(TensorShapeProto.Dim.newBuilder().setSize(3).build());
tensorProtoBuilder.setDtype(DataType.DT_FLOAT);
tensorProtoBuilder.setTensorShape(shapeBuilder);

// Add the floats to TensorProto (simplified, handle potential errors in production code)
for (float value : inputData) {
    tensorProtoBuilder.addFloatVal(value);
}
TensorProto tensorProto = tensorProtoBuilder.build();

// Add the input tensor to the request
PredictRequest.Builder reqBuilder = request.toBuilder();
reqBuilder.putInputs("input_features", tensorProto);
request = reqBuilder.build();


// ... (Send request and process response) ...
```

The Java example uses the TensorFlow Serving Java gRPC client to construct the request.  Similar to the C++ example, manual `TensorProto` construction is shown for illustrative purposes.   In a real-world scenario, the TensorFlow Java API should be utilized for more robust tensor creation and management.


**3. Resource Recommendations:**

The TensorFlow Serving documentation is indispensable.  Consult the API reference for precise details on the `PredictRequest` and `PredictResponse` messages.   Furthermore, thoroughly understand the TensorFlow protocol buffer definitions. Familiarize yourself with the gRPC concepts and best practices for efficient communication.  Finally, invest time learning the specifics of your chosen client library (Python, C++, Java, etc.) to handle gRPC communication and protocol buffer manipulation effectively.
