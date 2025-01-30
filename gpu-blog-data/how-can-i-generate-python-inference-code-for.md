---
title: "How can I generate Python inference code for a saved TensorFlow model without relying on TensorFlow libraries?"
date: "2025-01-30"
id: "how-can-i-generate-python-inference-code-for"
---
A saved TensorFlow model, specifically its SavedModel format, encapsulates not just the trained weights, but also the computational graph defining the operations. Extracting the inference logic and implementing it without TensorFlow libraries demands understanding this graph's structure, protobuf serialization, and potentially low-level numerical operations. This is feasible, though it requires substantial effort and careful implementation.

The SavedModel, at its core, is a directory containing multiple files. Notably, the `saved_model.pb` file (or `saved_model.pbtxt` for human-readable versions) holds a serialized `MetaGraphDef` protobuf message. This message describes the computational graph, including node definitions, input/output tensors, and associated metadata. To implement inference outside TensorFlow, I must: 1) parse this protobuf, 2) interpret the graph, and 3) perform the necessary numerical computations.

The `MetaGraphDef` contains a collection of `GraphDef` messages, each defining a computational graph. Within a `GraphDef`, nodes (operations) are specified as `NodeDef` messages. Each `NodeDef` contains an `op` field indicating the operation type (e.g., `MatMul`, `Add`, `Conv2D`), input tensor names, and operation-specific attributes. Input tensors are referenced by name, often linking back to other node outputs. To perform inference, I need to traverse this graph in a topological order (i.e., ensuring that the inputs to an operation have been computed before that operation) and evaluate each operation.

Decoding and interpreting the protobuf structure forms the first hurdle. Fortunately, Google provides the `protobuf` library which allows parsing these messages. However, I am responsible for the interpretation and execution of each defined operation. This differs drastically from the convenience of using TensorFlow's optimized kernels. I must implement custom logic for every operation encountered in the model, such as matrix multiplication, convolution, activation functions, and batch normalization. Efficient and accurate implementations of these operations are crucial for achieving results comparable to TensorFlow's native inference.

Let’s consider three illustrative scenarios of how I might approach this, starting from relatively simple, and then moving to more complex examples.

**Example 1: A Simple Linear Model (MatMul and Add)**

Let’s imagine a model represented in `saved_model.pb` that performs a simple linear transformation: `output = matmul(input, weights) + bias`. The corresponding `NodeDef` messages in the `GraphDef` would likely include operations for matrix multiplication (`MatMul`) and addition (`Add`). The weights and biases are constant tensors, initialized with trained values which are themselves stored as `Const` operations.

```python
import protobuf
from google.protobuf import text_format

# Assuming the saved_model.pb is decoded to a MetaGraphDef object 'meta_graph'
# Below is an example parsing of the graph nodes

# Replace this with actual reading and parsing of 'saved_model.pb'
with open("saved_model.pbtxt", "r") as f:
    meta_graph = text_format.Parse(f.read(), protobuf.tensorflow.MetaGraphDef())

# Access the graph
graph_def = meta_graph.graph_def

node_map = {node.name: node for node in graph_def.node}

def linear_model_inference(input_values):
    tensor_values = {}

    for node_name, node in node_map.items():
       if node.op == "Const":
            tensor_values[node_name] = protobuf.tensor_util.MakeNdarray(node.attr['value'].tensor)
       elif node.op == "Placeholder":
           if node.name == 'input_tensor': # Assumption based on graph inspection
               tensor_values[node.name] = input_values
       elif node.op == "MatMul":
           input1_name = node.input[0]
           input2_name = node.input[1]
           tensor_values[node_name] = numpy.matmul(tensor_values[input1_name], tensor_values[input2_name])
       elif node.op == "Add":
           input1_name = node.input[0]
           input2_name = node.input[1]
           tensor_values[node_name] = tensor_values[input1_name] + tensor_values[input2_name]

    # Assuming final output is named 'output' according to GraphDef inspection
    return tensor_values['output']


# Example usage
input_data = numpy.array([[1, 2, 3]], dtype=numpy.float32) # Example, shape would vary
output = linear_model_inference(input_data)
print(output)

```

In this example, I manually traverse the `GraphDef`, identify `Const`, `Placeholder`, `MatMul`, and `Add` operations and, relying on the tensor values and inputs of each operation, calculate the output. I made assumptions regarding input node name (`'input_tensor'`) and output node name (`'output'`). In a true implementation, the graph needs to be inspected to make these inferences. The `protobuf.tensor_util.MakeNdarray` is key, as it converts the protobuf representation to an efficient numpy array for calculations. This code would require `protobuf` and `numpy` to run correctly.

**Example 2: Convolutional Layer (Conv2D and BiasAdd)**

Let's extend to include a convolutional layer, which would involve the `Conv2D` operation, followed by a `BiasAdd`. This scenario introduces more parameters to manage within the custom implementation, such as strides, padding, and data format.

```python
# Assume access to graph_def as in example 1

def conv2d_inference(input_image):
    tensor_values = {}

    for node_name, node in node_map.items():
        if node.op == "Const":
            tensor_values[node_name] = protobuf.tensor_util.MakeNdarray(node.attr['value'].tensor)
        elif node.op == "Placeholder":
           if node.name == 'input_tensor':
               tensor_values[node.name] = input_image
        elif node.op == "Conv2D":
            input_tensor_name = node.input[0]
            filter_tensor_name = node.input[1]

            strides = [int(s) for s in node.attr["strides"].list.i]
            padding = node.attr["padding"].s.decode('utf-8')
            data_format = node.attr['data_format'].s.decode('utf-8')

            input_data = tensor_values[input_tensor_name]
            filter_data = tensor_values[filter_tensor_name]

            # Implement the 2D Convolution, taking into account strides and padding
            # Note: The implementation here would require careful coding to match Tensorflow's Conv2D kernel
            tensor_values[node_name] = custom_conv2d(input_data, filter_data, strides, padding, data_format)
        elif node.op == "BiasAdd":
            input_tensor_name = node.input[0]
            bias_tensor_name = node.input[1]
            tensor_values[node_name] = tensor_values[input_tensor_name] + tensor_values[bias_tensor_name]

    # Assumption made on naming convention of output tensor after inspecting graph
    return tensor_values['output_conv_layer'] # For the output after conv and bias

def custom_conv2d(input_data, filter_data, strides, padding, data_format):
      # This is a simplified implementation.
      # A full, efficient implementation would be required
      input_height, input_width, input_channels = input_data.shape
      filter_height, filter_width, filter_channels, output_channels = filter_data.shape
      if data_format == "NHWC":
         input_height, input_width, input_channels = input_data.shape
         filter_height, filter_width, filter_channels, output_channels = filter_data.shape
      elif data_format == "NCHW":
        input_channels, input_height, input_width = input_data.shape
        output_channels, filter_channels, filter_height, filter_width = filter_data.shape


      stride_h, stride_w = strides[1], strides[2]

      if padding == 'VALID':
        out_h = (input_height - filter_height)// stride_h + 1
        out_w = (input_width - filter_width) // stride_w + 1
      elif padding == "SAME":
        out_h = (input_height + stride_h -1) // stride_h
        out_w = (input_width + stride_w -1) // stride_w

      output_data = numpy.zeros((out_h, out_w, output_channels))

      for h_out in range(out_h):
        for w_out in range(out_w):
             for out_channel in range(output_channels):
                for h_filter in range(filter_height):
                    for w_filter in range(filter_width):
                        for in_channel in range(filter_channels):
                          h_in = h_out * stride_h + h_filter
                          w_in = w_out * stride_w + w_filter
                          if 0 <= h_in < input_height and 0<= w_in < input_width:
                             if data_format == "NHWC":
                                output_data[h_out, w_out, out_channel] += (input_data[h_in, w_in, in_channel]
                                                        * filter_data[h_filter, w_filter, in_channel, out_channel])
                             elif data_format == "NCHW":
                                output_data[h_out, w_out, out_channel] += (input_data[in_channel, h_in, w_in]
                                                        * filter_data[out_channel, in_channel, h_filter, w_filter])


      return output_data

# Example usage with an image
input_image = numpy.random.rand(32, 32, 3).astype(numpy.float32) #example image, shape and dtype vary
output_conv = conv2d_inference(input_image)
print(output_conv.shape)
```
The `custom_conv2d` is a simplified example. A full implementation needs to handle all padding and stride variations, as well as any potential optimizations to match TensorFlow's performance, which is typically highly optimized for hardware, utilizing efficient algorithms for convolution such as the fast Fourier transform, or Winograd algorithms. The `data_format` argument (`'NHWC'` or `'NCHW'`) is also extremely important, and should be handled appropriately.

**Example 3: Batch Normalization**

Introducing batch normalization involves implementing operations like `FusedBatchNormV3` or a series of individual operations to calculate mean and variance and the scale and shift transformations. This is computationally more complex due to the need to use the moving average and variance tensors, which are also stored as constant tensors.

```python
# Assume access to node_map as in previous examples

def batchnorm_inference(input_values):
  tensor_values = {}
  for node_name, node in node_map.items():
        if node.op == "Const":
            tensor_values[node_name] = protobuf.tensor_util.MakeNdarray(node.attr['value'].tensor)
        elif node.op == "Placeholder":
            if node.name == 'input_tensor':
               tensor_values[node.name] = input_values
        elif node.op == 'FusedBatchNormV3': # Example. Could also be separate ops.
           input_tensor_name = node.input[0]
           scale_tensor_name = node.input[1]
           offset_tensor_name = node.input[2]
           mean_tensor_name = node.input[3]
           variance_tensor_name = node.input[4]

           input_data = tensor_values[input_tensor_name]
           scale = tensor_values[scale_tensor_name]
           offset = tensor_values[offset_tensor_name]
           mean = tensor_values[mean_tensor_name]
           variance = tensor_values[variance_tensor_name]
           epsilon = node.attr['epsilon'].f


           output_data = (input_data - mean) / numpy.sqrt(variance + epsilon)
           output_data = output_data * scale + offset

           tensor_values[node_name] = output_data
        # ... other operations

  return tensor_values['output']

input_data = numpy.random.rand(10, 256).astype(numpy.float32)
output_bn = batchnorm_inference(input_data)
print(output_bn.shape)
```
The `FusedBatchNormV3` operation is a common representation of Batch Normalization in TensorFlow. I've simplified here the use of its parameters, but careful management of the `scale`, `offset`, `mean`, and `variance` constant tensors is key for correct inference. Different variants exist, for example, when batch norm is not fused. In this case, I would have to compute the mean, variance and scale and shift transformations separately.

These examples illustrate the core process involved in performing inference without TensorFlow. The code needs to handle a wide range of operations including all types of mathematical computations and activations as well as different data layouts and padding, among other variables. This is very time intensive and error-prone. This requires careful, thorough understanding of the TensorFlow graph structure, protobufs and mathematical implementations and is not a trivial exercise.

For further study, I recommend studying the TensorFlow source code itself, specifically the definitions for the `NodeDef` message in the protobuf specifications. Reading through the C++ implementations of TensorFlow's kernels for commonly used operations will provide valuable insights. Books on deep learning and numerical computation are helpful in implementing correct and optimized mathematical operations. Academic papers discussing optimization of inference can also be very valuable in building fast, performant implementations. Specifically, papers on graph compilation and optimization could be useful.  These resources, taken together, provide the foundation for understanding the internal workings of TensorFlow and for producing independent inference code.
