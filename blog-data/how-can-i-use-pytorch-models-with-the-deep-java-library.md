---
title: "How can I use PyTorch models with the Deep Java Library?"
date: "2024-12-16"
id: "how-can-i-use-pytorch-models-with-the-deep-java-library"
---

Alright, let's talk about integrating PyTorch models with the Deep Java Library (DJL). I've spent a good chunk of my career navigating the intricacies of different machine learning frameworks, and getting these two to play nice is a common challenge. The core issue arises from the inherent differences in how models are defined and executed: PyTorch is Python-centric, leveraging its dynamic computational graphs, while DJL is a Java-based interface intended to be framework-agnostic. This requires a bridging mechanism, and thankfully, DJL is equipped to handle this reasonably well.

First, let's address the fundamental approach. DJL doesn't directly *import* a PyTorch `.pt` or `.pth` file as-is. Instead, it leverages the concept of a *model zoo* and model *engines*. Essentially, a PyTorch model needs to be converted into an intermediate representation (often referred to as a 'trace') that DJL can understand. This generally involves exporting the model from PyTorch into either a `TorchScript` format or an `ONNX` format, both of which DJL can load. `TorchScript` is generally preferred if the model was originally trained using PyTorch since DJL can leverage the full capabilities of the PyTorch engine. `ONNX` serves as a good alternative if interoperability across different frameworks is desired and some capabilities are less crucial.

Let's break this down with examples. For the purpose of these code examples, let’s assume we have a simple PyTorch model, `SimpleNet`, defined as follows (python code):

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    input_size = 10
    hidden_size = 20
    output_size = 5
    model = SimpleNet(input_size, hidden_size, output_size)
    dummy_input = torch.randn(1, input_size)
    # Trace using TorchScript
    traced_model = torch.jit.trace(model, dummy_input)
    torch.jit.save(traced_model, "simplenet_torchscript.pt")
    # Alternative: Export to ONNX
    torch.onnx.export(model, dummy_input, "simplenet_onnx.onnx",
        opset_version=10,
        input_names = ['input'],   # Optional, but helps with clarity
        output_names = ['output'])
```

This code snippet demonstrates how to prepare the model for DJL by saving two different formats: one using torchscript and another one using ONNX. The `torch.jit.trace` method creates a *scripted* version of the model based on a dummy input which makes it easy for DJL to interpret. The `torch.onnx.export` method does the same, but to ONNX format and includes a few extra arguments to ensure consistent naming of input and output tensors.

Now, let’s switch to the Java side and use DJL to load and run it. Here’s a Java example for loading the *TorchScript* version:

```java
import ai.djl.*;
import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;

public class TorchScriptInference {

  public static void main(String[] args) throws Exception {
     Criteria<Input, Output> criteria = Criteria.builder()
            .setTypes(Input.class, Output.class)
            .optModelPath(Paths.get("."))
            .optModelName("simplenet_torchscript")
            .optEngine("PyTorch")
            .build();

    try (ZooModel<Input, Output> model = criteria.loadModel()) {
         Predictor<Input, Output> predictor = model.newPredictor();
         NDManager manager = model.getNDManager();

         NDArray inputTensor = manager.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
         inputTensor = inputTensor.reshape(new Shape(1, 10)); // Reshape to match the expected input

         Input input = new Input();
         input.add("input", inputTensor);

         Output output = predictor.predict(input);
         System.out.println("Prediction: " + output.getData());

         inputTensor.close();
    }

  }
}
```

This code creates a `Criteria` object specifying the path to the model (which in this case is the current directory) and the engine. It then loads the traced `torchscript` model from the file `simplenet_torchscript.pt` (note, DJL knows that if the filename is without the extension, it should search for that extension), creates an input tensor, and runs inference using the `predictor`. The final output is printed to the console. It’s crucial to use the same input shape as the model expects which is achieved by reshaping the NDArray. Notice that we’re specifying PyTorch as the engine in the `Criteria`, this is fundamental for DJL to pick the appropriate engine for your model.

If you have exported the model to `ONNX`, then the following code will load and run it.

```java
import ai.djl.*;
import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;

public class OnnxInference {
    public static void main(String[] args) throws Exception {
        Criteria<Input, Output> criteria = Criteria.builder()
                .setTypes(Input.class, Output.class)
                .optModelPath(Paths.get("."))
                .optModelName("simplenet_onnx")
                .optEngine("OnnxRuntime")
                .build();

        try (ZooModel<Input, Output> model = criteria.loadModel()) {
            Predictor<Input, Output> predictor = model.newPredictor();
            NDManager manager = model.getNDManager();

           NDArray inputTensor = manager.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
            inputTensor = inputTensor.reshape(new Shape(1, 10));

            Input input = new Input();
            input.add("input", inputTensor);

            Output output = predictor.predict(input);
             System.out.println("Prediction: " + output.getData());

           inputTensor.close();

        }
    }
}

```
As you can see, the only difference is in the `criteria` part where we set the engine to be `"OnnxRuntime"` this is the only part that distinguishes that this code will load the ONNX model instead of the `torchscript` model.

Several points are worth highlighting. First, ensuring your environment has the appropriate DJL dependencies for `PyTorch` or `OnnxRuntime` is key. You typically achieve this through maven or gradle by including relevant DJL artifacts. Second, the `optModelName` parameter in `Criteria` specifies the name of the model file without an extension. DJL searches the given path for files with known extensions for the specified engine, such as `.pt` or `.onnx`. Third, DJL expects input as an `NDArray` (n-dimensional array), so the input data needs to be converted accordingly, and reshaped if required. Finally, the output from the predictor is of type `Output`, which contains various properties of interest. It is generally required to extract the result by using the method `getData()`.

For deeper dives, I strongly recommend consulting the official DJL documentation; it’s the most comprehensive and authoritative source of information for this framework. Additionally, for a better understanding of `torchscript`, refer to the official PyTorch documentation for tracing and scripting, and for `onnx`, the ONNX specifications are very valuable. Specifically, pay close attention to versioning issues – if the ONNX version used for export is not supported by your `OnnxRuntime` build, you might encounter issues. Further, some operators can be tricky when using `ONNX`, so it's good to check the export log if you encounter problems. Also, if you’re dealing with more complex models that have custom layers, you might face challenges with `TorchScript` as it has limited support for custom operations. In those situations, `ONNX` might be a better alternative. Working with these frameworks and addressing issues through troubleshooting provides an extremely valuable practical understanding. The key, as always, is careful documentation, methodical debugging, and a solid understanding of the underlying principles.
