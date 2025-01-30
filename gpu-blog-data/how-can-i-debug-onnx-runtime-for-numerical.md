---
title: "How can I debug ONNX Runtime for numerical inference errors?"
date: "2025-01-30"
id: "how-can-i-debug-onnx-runtime-for-numerical"
---
The source of numerical inference errors in ONNX Runtime frequently stems from discrepancies between the model’s intended numerical behavior and the realities of floating-point computation, particularly within hardware acceleration contexts. These errors often manifest as unexpected output values, NaNs, or divergences from expected results when compared to, for instance, PyTorch or TensorFlow outputs. Debugging these issues requires a systematic approach that examines both the model's mathematical properties and the execution environment. I've encountered this frequently, particularly when deploying models to edge devices with different float representations.

Debugging numerical inference errors in ONNX Runtime demands an understanding of several key areas: model conversion, operator implementations, precision, and execution configurations. Errors can arise from the conversion of the original model to ONNX, which may involve subtle transformations that alter numerical characteristics. Further, each operator within ONNX Runtime has its own implementation, which might exhibit numerical behavior different from its corresponding implementation in the original framework. Differences in float precision (e.g., float32 vs. float16) between the training environment and the inference environment will introduce deviations. Finally, settings such as the execution provider and thread pool configurations impact how operations are processed and can influence numerical stability.

When I debug such issues, my first step is always to rigorously validate the ONNX model itself. This begins by visualizing the graph using tools like Netron to ensure that the conversion process did not introduce unexpected changes or remove crucial steps. Next, I utilize the ONNX Runtime's Python API to perform a comparison between the expected outputs and those produced by the runtime. I typically start with a small set of hand-crafted inputs that I understand the expected outputs for, and then gradually expand the input set. If divergences appear during this process, I begin to narrow down the source of the error by isolating the problematic node within the graph. I typically achieve this by adding a series of output nodes throughout the model and comparing each node's output with what I'd get with the original framework.

A valuable technique is to leverage ONNX Runtime's logging features to expose detailed information about the execution of each operator. This logging reveals the type of hardware or software provider being utilized, the shapes and types of tensor inputs/outputs, and runtime statistics. Enabling verbose logging sometimes points directly to an issue by highlighting nodes with unusual runtime characteristics, such as extreme or NaN outputs.

Here are three code examples demonstrating debugging approaches:

**Example 1: Comparing ONNX Runtime Output with Reference Output**

This Python code snippet demonstrates the core process of comparing outputs from ONNX Runtime to reference results typically obtained from the originating framework.

```python
import onnxruntime as ort
import numpy as np

def compare_outputs(onnx_path, input_data, reference_output):
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    ort_output = ort_session.run(None, {input_name: input_data})[0]

    print(f"ONNX Runtime Output:\n{ort_output}")
    print(f"Reference Output:\n{reference_output}")

    diff = np.abs(ort_output - reference_output)
    max_diff = np.max(diff)
    print(f"Max Absolute Difference: {max_diff}")
    
    if max_diff > 1e-5:
      print("Significant difference detected.")
    else:
       print("Outputs are numerically close.")

# Example usage
onnx_model_path = "path/to/your/model.onnx"
test_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
reference_out = np.random.rand(1, 1000).astype(np.float32) # Replace with actual expected output

compare_outputs(onnx_model_path, test_input, reference_out)
```

This code loads an ONNX model, performs inference with a test input, and compares the output with a known reference output. It prints both outputs and the maximum absolute difference. A difference exceeding a defined threshold (here, 1e-5) suggests numerical issues, triggering a deeper investigation.

**Example 2: Isolating a Problematic Node by Inserting Output Nodes**

This example illustrates how to modify an ONNX model using the onnx library to add output nodes for inspecting intermediate computations:

```python
import onnx
import numpy as np

def add_intermediate_outputs(model_path, node_names):
  model = onnx.load(model_path)
  output_names = [out.name for out in model.graph.output]
  
  for node_name in node_names:
    node = next((n for n in model.graph.node if n.name == node_name), None)
    if node:
        for out_tensor in node.output:
           if out_tensor not in output_names:
                model.graph.output.extend([onnx.helper.make_tensor_value_info(
                out_tensor, onnx.TensorProto.FLOAT, None)])
                print(f"Adding output: {out_tensor}")

  onnx.save(model, model_path.replace(".onnx", "_debug.onnx"))

# Example Usage
onnx_model_path = "path/to/your/model.onnx"
node_names_to_check = ["Conv_0", "Relu_1"] # Names of nodes to inspect
add_intermediate_outputs(onnx_model_path, node_names_to_check)

#Run the debug model with onnxruntime to get intermediate outputs using the same comparison
#from example one
debug_onnx_model_path = onnx_model_path.replace(".onnx", "_debug.onnx")
test_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
#Assume we've collected reference_outputs for "Conv_0" and "Relu_1" nodes
reference_outputs_debug = {"Conv_0": np.random.rand(1, 64, 112, 112).astype(np.float32),
                          "Relu_1": np.random.rand(1, 64, 112, 112).astype(np.float32)}
ort_session_debug = ort.InferenceSession(debug_onnx_model_path)
input_name = ort_session_debug.get_inputs()[0].name
run_output_debug = ort_session_debug.run(None, {input_name: test_input})
output_names = [output.name for output in ort_session_debug.get_outputs()]
#Run output_debug contains all outputs from the model, iterate through and find matches and compare
for output in output_names:
  if output in reference_outputs_debug.keys():
    print(f"Comparing output for {output}")
    index = output_names.index(output)
    compare_outputs("", run_output_debug[index], reference_outputs_debug[output])
```

This code loads an existing ONNX model, adds output nodes for specified intermediate layers (e.g., the output of "Conv_0"), and saves a new debuggable model. Executing the debug model, it is possible to compare the output of these nodes to corresponding outputs from the reference implementation to pinpoint numerical discrepancies.

**Example 3: Inspecting Execution Provider and Precision**

This example demonstrates how to check the execution provider and configure precision:

```python
import onnxruntime as ort
import numpy as np

def check_execution_provider(onnx_path):
    providers = ort.get_available_providers()
    print(f"Available Providers: {providers}")
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider']) #or 'CUDAExecutionProvider'
    print(f"Execution Provider Used: {ort_session.get_providers()}")

def run_with_precision(onnx_path, input_data, precision="float32"):
    ort_session_options = ort.SessionOptions()
    if precision=="float16":
        ort_session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        ort_session = ort.InferenceSession(onnx_path, sess_options=ort_session_options, providers=["CPUExecutionProvider"]) #May require other providers
        input_name = ort_session.get_inputs()[0].name
        input_data = input_data.astype(np.float16)
        output = ort_session.run(None, {input_name: input_data})
    elif precision=="float32":
        ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        input_name = ort_session.get_inputs()[0].name
        output = ort_session.run(None, {input_name: input_data})
    return output
# Example usage
onnx_model_path = "path/to/your/model.onnx"
test_input = np.random.rand(1, 3, 224, 224).astype(np.float32)

check_execution_provider(onnx_model_path)
float16_output = run_with_precision(onnx_model_path, test_input, "float16")
float32_output = run_with_precision(onnx_model_path, test_input, "float32")
print(f"Float 32 output shape {float32_output[0].shape}")
print(f"Float 16 output shape {float16_output[0].shape}")

```

This code first identifies the available execution providers. Then, it attempts to run the model using a specified execution provider and, if selected, executes the model with float16 precision. By changing the selected execution provider or precision, it is possible to determine whether these factors are causing the numerical issues. I have seen edge devices without the necessary hardware acceleration providers leading to precision issues, so always checking this is key.

For further investigation, I frequently consult documentation on ONNX operator definitions to understand their behavior, particularly regarding numerical stability. Additionally, studying resources on numerical methods and floating-point arithmetic helps contextualize the kinds of errors that are frequently seen. I’ve found several online textbooks on computer arithmetic helpful, and I’d recommend exploring literature that describes the nature of floating point operations. Finally, reviewing the ONNX runtime GitHub repository is often insightful as it includes information about specific implementation choices and any reported numerical bugs.
