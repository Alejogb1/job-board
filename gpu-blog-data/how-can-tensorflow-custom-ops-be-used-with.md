---
title: "How can TensorFlow custom ops be used with ONNX in Java?"
date: "2025-01-30"
id: "how-can-tensorflow-custom-ops-be-used-with"
---
The challenge of integrating TensorFlow custom operations (ops) with ONNX models within a Java environment arises from the inherent platform specificity of TensorFlow’s C++ backend and ONNX's standardized, platform-agnostic representation. Specifically, TensorFlow custom ops are compiled into shared libraries (.so on Linux, .dll on Windows, .dylib on macOS) with dependencies on the TensorFlow C++ API. ONNX, on the other hand, operates on an intermediate graph representation, devoid of concrete runtime implementations for individual operations. Effectively bridging this gap requires a structured approach, centering around the creation of compatible ONNX export and import mechanisms, and the provision of Java-accessible implementations of the custom operations.

I’ve encountered this problem firsthand working on a distributed machine learning inference engine where portability across diverse environments was paramount. We were heavily invested in TensorFlow for model training, including custom ops for specific signal processing algorithms. However, deploying inference on heterogeneous hardware, especially embedded devices, demanded a shift to ONNX. Initially, the lack of direct integration proved a significant hurdle. The solution we developed involved a multi-stage process: first, modifying the TensorFlow graph to correctly represent the custom op during ONNX export; second, implementing the custom op in a way compatible with the chosen ONNX runtime in Java; and finally, ensuring a seamless handoff between the exported ONNX graph and our custom implementation in the Java ecosystem.

The primary issue stems from the fact that the TensorFlow-to-ONNX conversion process, often achieved using the `tf2onnx` tool, generally doesn't recognize custom TensorFlow ops natively. `tf2onnx` relies on a predefined mapping between TensorFlow operations and their ONNX equivalents. If a custom operation lacks a corresponding ONNX definition, it will not be properly translated, leading either to failed exports or malformed ONNX graphs. One must therefore either map the custom op to an existing ONNX operation or create a custom ONNX operator definition, thereby requiring custom export logic.

Here's how to manage the custom operation representation within the TensorFlow graph during export. Assume our custom op, named `MyCustomOp`, accepts two float tensors and returns their element-wise sum after a non-linear transformation defined within the op. Within the TensorFlow Python environment, the custom op would be instantiated using a `tf.load_op_library` call, loading a dynamically linked shared library.

```python
# Example 1: Python code demonstrating custom op usage in TensorFlow.
import tensorflow as tf
import numpy as np

# Assume 'my_custom_op.so' contains the definition of MyCustomOp
my_custom_op = tf.load_op_library('./my_custom_op.so')

# Create tensors for input.
tensor_a = tf.constant(np.random.rand(3,3), dtype=tf.float32)
tensor_b = tf.constant(np.random.rand(3,3), dtype=tf.float32)

# Usage of the custom operation.
output = my_custom_op.my_custom_op(tensor_a, tensor_b)

# Perform a standard calculation after using our custom op.
final_output = tf.nn.relu(output)

#  Export the model to ONNX including the custom operation. This would likely fail without the appropriate tf2onnx extensions.
# We’d need to register our custom operation within tf2onnx (which is not shown here),
#  before proceeding with the export operation.
# Placeholder for export using tf2onnx.
# onnx_model, external_tensor_storage = tf2onnx.convert.from_function(
#     lambda x, y: tf.nn.relu(my_custom_op.my_custom_op(x, y)),
#       input_signature=[tf.TensorSpec((3, 3), tf.float32), tf.TensorSpec((3, 3), tf.float32)],
#       output_path="./custom_op.onnx"
# )

print(output) # output is a TensorFlow EagerTensor object.

```

This snippet illustrates a simple TensorFlow graph using `MyCustomOp`.  The key is that `tf2onnx` needs to be aware of `my_custom_op.my_custom_op`.  This often involves creating a custom exporter plugin for `tf2onnx` or leveraging the `custom_ops` mechanism, where the exporter translates the TensorFlow op to a custom ONNX operator by defining a custom ONNX definition using the `opset` within the `tf2onnx.convert.from_function` or similar export function. Without this crucial bridging, the ONNX export will fail, as `tf2onnx` doesn't know how to translate `MyCustomOp` to a corresponding ONNX node type.

Now, consider the Java side, where we will use an ONNX runtime such as ONNX Runtime (ORT). We need to provide the implementation for our custom ONNX operator that mimics the `MyCustomOp` behaviour, which we defined in the shared library of TensorFlow. The implementation needs to be in Java. Here's an outline:

```java
// Example 2:  Java class for implementing the ONNX custom op using ORT's custom op API.
import ai.onnxruntime.*;
import java.nio.FloatBuffer;
import java.util.Arrays;

public class MyCustomOpImpl implements OrtCustomOp {
    @Override
    public String getOpType() {
        return "MyCustomOp"; // This must match the custom ONNX node type.
    }

    @Override
    public OrtStatus createKernel(OrtApi api, OrtKernelInfo info, OrtKernelContext context, OrtKernel kernel) {
        return new OrtStatus(OrtErrorCode.ORT_OK, "");
    }

    @Override
    public void compute(OrtApi api, OrtKernelContext context, OrtKernel kernel) throws OrtException {

            // Retrieve inputs
            OrtValue inputA = api.kernelContext_getInput(context,0);
            OrtValue inputB = api.kernelContext_getInput(context, 1);
            
            // Error handling for input retrieval
            if(inputA == null || inputB == null){
                 throw new OrtException("Missing inputs for custom op");
             }

            // Extract data from OrtValue objects
            float[] arrA = OrtUtil.getFloatArray(inputA, api);
            float[] arrB = OrtUtil.getFloatArray(inputB, api);

            if(arrA.length != arrB.length) {
                throw new OrtException("Inputs must have the same number of elements");
            }


            // Perform element-wise sum and the non-linear transform.
            float[] result = new float[arrA.length];
             for(int i = 0; i < arrA.length; ++i){
                result[i] = customTransform(arrA[i] + arrB[i]);
            }

           // Create an output tensor from result.

           long[] shape = inputA.getInfo().getShape(); // Reuse the shape of the first input
           OrtValue outputValue = OrtUtil.createOrtValue(result, shape, api);


           // Assign the value to the output.
           api.kernelContext_setOutput(context, 0, outputValue);
    }
    //  Private custom transform
    private float customTransform(float value){
       return (float)Math.pow(value,3);
    }


    @Override
    public void destroyKernel(OrtApi api, OrtKernel kernel) {
         // no resources to free.
    }
}
```

This Java code provides an implementation conforming to ORT’s custom operation interface. The `compute` method retrieves the input tensors, applies our custom element-wise sum and transformation, and outputs the result as a new ONNX tensor.  The `getOpType()` function specifies the string that will be associated with our custom operator node in the ONNX model graph. This operator type string must match that defined when defining the custom ONNX operator in the `tf2onnx` export phase (not shown in Example 1). The `createKernel` and `destroyKernel` methods are callbacks for ORT kernel management and are left minimal here for simplicity.

To integrate this within our Java ONNX runtime, the implementation needs to be registered within an ORT session. A typical initialization pattern might look like this:

```java
// Example 3: Java code setting up the ONNX Runtime with the custom op.
import ai.onnxruntime.*;
import java.io.IOException;

public class OnnxInference {

   public static void main(String[] args) throws OrtException, IOException {
        // Create an ONNX environment.
        OrtEnvironment env = OrtEnvironment.getEnvironment();

        // Create session options for loading model with custom ops.
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.registerCustomOpLibrary("path/to/your/java/implementation/directory");

        // Add custom operation to SessionOptions.
        options.registerCustomOp(new MyCustomOpImpl());

        // Load the ONNX model, using session options.
        try (OrtSession session = env.createSession("path/to/custom_op.onnx", options)) {

             // Create input tensors to be used with session.run.
             float[] inputAArray = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
             float[] inputBArray = {9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

             long[] inputShape = {3, 3};

            OrtValue inputA = OrtUtil.createOrtValue(inputAArray, inputShape, env);
             OrtValue inputB = OrtUtil.createOrtValue(inputBArray, inputShape, env);

             // Run inference
             OrtValue[] output = session.run(null,  // This can contain inputs names for multi-input models.
                        new OrtValue[]{inputA, inputB},  // Inputs for the ONNX model.
                          null);    // Names of output tensors, null will return all outputs.
            
             // Verify we received one output in the inference.
             if (output.length > 0) {

                //  We know our output is float array based on the ONNX graph output type.
                float[] results = OrtUtil.getFloatArray(output[0], env);
                System.out.println(Arrays.toString(results));
            }else{
                  System.out.println("No output from custom op model");
            }
        }
   }

}

```

This example shows how to load the ONNX model, register our `MyCustomOpImpl` instance, and then run inference. The `registerCustomOp` function of `SessionOptions` is crucial; it tells ORT where to locate the custom operator implementations during model execution. Without proper registration, ORT will fail to execute the ONNX graph if it contains nodes referencing our custom operator. The code loads the model exported using the mechanism described before (Example 1) and then feeds sample inputs into the model and displays the final output.

In summary, incorporating custom TensorFlow ops into ONNX and utilizing them in Java necessitates a coordinated approach. Firstly, extensions to the TensorFlow ONNX export process are required to handle non-standard operations. Secondly, the custom operator must be implemented in Java using an interface provided by the specific ONNX runtime being used. Finally, these implementations must be registered with the runtime prior to executing the ONNX model.

For further exploration, the ONNX documentation provides insights into custom operators, the ONNX Runtime provides detailed guides and examples for Java integration, and the TensorFlow documentation covers the mechanisms for constructing and exporting models with custom ops.  Furthermore, research into the specific custom export mechanisms for `tf2onnx` is recommended as they are often nuanced and context-specific. Examining existing open-source projects implementing custom operators within similar ONNX runtime frameworks can also offer valuable guidance.
