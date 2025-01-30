---
title: "How can a PyTorch model be loaded in Java using DJL?"
date: "2025-01-30"
id: "how-can-a-pytorch-model-be-loaded-in"
---
The crux of deploying a PyTorch model within a Java application lies in bridging the gap between Python's deep learning ecosystem and Java's more established enterprise environment. Deep Java Library (DJL) serves as this bridge, providing a high-level, platform-agnostic API to load and execute models trained in frameworks such as PyTorch. I've encountered this scenario numerous times during the transition of ML prototypes to production-ready services, and the process, while not overly complex, warrants a detailed breakdown.

The first step requires preparing your PyTorch model for Java consumption. This usually involves saving the model in a format that DJL can readily interpret. While several options exist, I've consistently found using TorchScript to be the most straightforward and reliable. TorchScript provides a way to serialize a PyTorch model into an intermediary representation which is then easily loaded by DJL without requiring a Python interpreter on the target machine. This serialization process transforms your dynamic Python model into a static graph suitable for efficient execution within the JVM.

The primary reason TorchScript is preferred is because it decouples the model from the Python runtime. DJL's engine abstraction allows you to utilize different backends (TensorFlow, PyTorch, ONNX Runtime, etc.). However, when dealing with PyTorch, relying on a Python installation adds significant overhead and potential dependency conflicts to your Java deployments. TorchScript bypasses this by directly providing a representation that the PyTorch JNI library used within DJL can directly consume.

Here’s a practical example of how I've saved a simple PyTorch model to a TorchScript file.

```python
import torch
import torch.nn as nn

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

# Instantiate the model
model = SimpleModel()

# Dummy input
example_input = torch.randn(1, 10)

# Trace the model (convert to TorchScript)
traced_model = torch.jit.trace(model, example_input)

# Save the traced model
torch.jit.save(traced_model, "simple_model.pt")

print("Model saved as simple_model.pt")
```

This Python code snippet first defines a rudimentary linear model, a `SimpleModel` with a single linear layer. It then generates a dummy input tensor that mimics the expected input to the model. The crucial step is the call to `torch.jit.trace`, which creates a TorchScript representation by tracing the model's execution with the provided example input. Finally, this traced model is saved to a file named `simple_model.pt`. This is the file we will later load using DJL.

The tracing mechanism works by observing the operations performed on the input tensor, effectively "compiling" the model into a graph. It captures the sequence of operations irrespective of how the Python code was structured. If, instead, we had the model do complex decision branches based on dynamic properties of the input, tracing might not be an appropriate tool. In such cases, scripting is a more flexible alternative.

Now that we have the saved model, the next step is to use DJL within a Java application. First, ensure that your Java project includes the appropriate DJL dependencies. You’ll need at least the core DJL artifact along with the PyTorch engine artifact. These will typically be added using a dependency management tool like Maven or Gradle. The precise dependency statements will vary based on your tool, however. Here's an example of the core DJL dependency in Gradle, for instance: `implementation 'ai.djl:api:0.25.0'` along with `implementation 'ai.djl.pytorch:pytorch-engine:0.25.0'`. Be sure to use the version of DJL applicable to your project requirements. With these dependencies in place, the following Java code will load and execute our `simple_model.pt`.

```java
import ai.djl.*;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;

import java.nio.file.Path;
import java.nio.file.Paths;

public class ModelLoader {
    public static void main(String[] args) throws ModelException, TranslateException {
        Path modelPath = Paths.get("simple_model.pt");

        Criteria<NDArray, NDArray> criteria =
            Criteria.builder()
                .setTypes(NDArray.class, NDArray.class)
                .optModelPath(modelPath)
                .build();

        try(ZooModel<NDArray, NDArray> model = criteria.loadModel()){
            // input tensor
            NDManager manager = model.getNDManager();
            NDArray input = manager.randomNormal(new Shape(1, 10));

            // Predictor allows you to perform inference
            Predictor<NDArray, NDArray> predictor = model.newPredictor(new SimpleTranslator());

            // Prediction step
            NDArray output = predictor.predict(input);

            System.out.println("Output: " + output);

            output.close();
            input.close();
        }
    }
    private static class SimpleTranslator implements Translator<NDArray, NDArray>{
        @Override
        public NDArray processOutput(TranslatorContext ctx, NDArray list) {
           return list;
        }
        @Override
        public NDArray processInput(TranslatorContext ctx, NDArray input) {
           return input;
        }

    }
}
```

This Java code first specifies the path to the saved TorchScript model, `simple_model.pt`.  It then constructs a `Criteria` object, which defines how the model should be loaded.  Importantly, `setTypes` specifies the input and output data type which are, in our case, NDArrays. The `optModelPath` option points to our `simple_model.pt` file.  A `ZooModel` object is then created and `loadModel()` attempts to load our model. If the model loads successfully, a new `NDManager` object is created for managing tensors in DJL and subsequently a sample input tensor `input` with a shape of [1,10] using `randomNormal`. A `Predictor` is obtained from the model via `model.newPredictor`, which will use a custom translator. Finally, the prediction is executed using the `predict` method and the result is printed to the console. The SimpleTranslator passes the input and output tensors directly, which is why we don't need custom translation for this model.

A key aspect of this process is how DJL manages tensors and operations.  DJL utilizes NDArrays, a multi-dimensional array structure which provides an abstraction over different backend implementations.  As such, the Java code does not directly interact with PyTorch tensors. The model's input and output data, while represented as PyTorch tensors within the model itself, are handled through the DJL NDArray abstraction.

Finally, the code includes resource cleanup using the try-with-resources block and explicitly closes the allocated tensors using the close method on NDArrays. Resource management is critical for avoiding memory leaks during long-running applications that process many prediction tasks.

To extend this further, here's an example demonstrating a more complex scenario where custom pre-processing and post-processing would be required, using a different model. This model is a simple image classification model, trained in PyTorch, and now needs to be used in Java.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# Define a simple convolutional model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Instantiate the model
model = SimpleCNN()

# Dummy input (batch size of 1)
dummy_input = torch.randn(1, 3, 28, 28)

# Trace the model (convert to TorchScript)
traced_model = torch.jit.trace(model, dummy_input)

# Save the traced model
torch.jit.save(traced_model, "simple_cnn.pt")
print("CNN Model saved as simple_cnn.pt")
```

This Python code will create `simple_cnn.pt`, which will be consumed by DJL. Notice that in this model, the inputs are images, with shape `(1, 3, 28, 28)` for channels-first input with a batch size of one. Here's how this model would be loaded in Java:

```java
import ai.djl.*;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.*;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class ImageModelLoader {
    public static void main(String[] args) throws ModelException, TranslateException, IOException {
        Path modelPath = Paths.get("simple_cnn.pt");
        Path imagePath = Paths.get("test.png"); // Dummy image

         // Create dummy image file for testing (replace with actual image processing)
        byte[] dummyImageData = new byte[28 * 28 * 3];
        Files.write(imagePath, dummyImageData);



        Criteria<Image, Classifications> criteria =
                Criteria.builder()
                        .setTypes(Image.class, Classifications.class)
                        .optModelPath(modelPath)
                        .build();

        try(ZooModel<Image, Classifications> model = criteria.loadModel()) {
           NDManager manager = model.getNDManager();

           // Load dummy image for inference
           Image image = ImageFactory.getInstance().fromFile(imagePath);
           Predictor<Image, Classifications> predictor = model.newPredictor(new ImageTranslator());

           // Perform prediction
           Classifications classifications = predictor.predict(image);
           System.out.println("Output: " + classifications);
        }
    }
    private static class ImageTranslator implements Translator<Image, Classifications>{
        @Override
        public NDArray processInput(TranslatorContext ctx, Image image) {
            NDManager manager = ctx.getNDManager();
            NDArray array = image.toNDArray(manager);
            array = array.transpose(2, 0, 1); // Adjust to channels-first
            array = array.reshape(new long[]{1, 3, 28, 28}); // batching dimension
            return array;
        }

        @Override
        public Classifications processOutput(TranslatorContext ctx, NDArray array) throws TranslateException {
            return new Classifications(array);
        }
    }
}
```

This code performs a similar model loading procedure to the prior example. However, the `Criteria` now specifies the input type as `Image` and output type as `Classifications`. A dummy image file is generated, which DJL then loads as `Image`. The `Predictor` uses a custom `ImageTranslator`. In `processInput`, the image is converted to an NDArray, the dimension order changed to be consistent with the model requirements (from (height, width, channels) to (channels, height, width)) using `transpose`, and then a batch dimension is prepended using `reshape`. The output from the model is passed directly to `Classifications` which is used to print the output. This example highlights the importance of pre- and post-processing for model input and output.

For deeper understanding, I would recommend reading the official DJL documentation, specifically the sections concerning PyTorch model deployment. Additionally, exploring resources detailing the inner workings of TorchScript would be advantageous. A good foundation in deep learning, along with familiarity with Java, will prove invaluable as well. Further, familiarize yourself with the fundamentals of tensors, especially how different backends manage them.
