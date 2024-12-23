---
title: "How do I use PyTorch models in Deep Java Library (DJL)?"
date: "2024-12-23"
id: "how-do-i-use-pytorch-models-in-deep-java-library-djl"
---

Alright,  Integrating PyTorch models into a Java environment using Deep Java Library (DJL) is something I’ve frequently navigated over the years, often in contexts where server-side processing demanded the robust nature of Java coupled with the power of PyTorch-trained models. It's a fairly common requirement, and thankfully, DJL streamlines the process quite effectively, although a few nuances need attention to ensure smooth sailing.

Essentially, what we’re doing is moving a model trained within the Python-centric ecosystem of PyTorch into the Java realm. DJL acts as an abstraction layer, providing a common api that shields us from the direct intricacies of both the backend deep learning engine and the differences between languages. The process, at a high level, involves saving your PyTorch model in a format DJL understands, and then loading and running that model through DJL in your Java application.

The first crucial step is model export. PyTorch provides several mechanisms for saving models, but DJL generally favors TorchScript. This format serializes your model’s architecture and weights in a way that's both efficient and compatible across different environments. I’ve found this particularly useful when dealing with complex model structures, where simply saving weights might not be sufficient for preserving the full operational graph. To save your PyTorch model in TorchScript format, you’d use code that looks something like this:

```python
import torch
import torch.nn as nn

# Example simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

# Create an instance of the model and trace it with a dummy input
model = SimpleModel()
dummy_input = torch.randn(1, 10) # Batch size 1, 10 input features

traced_model = torch.jit.trace(model, dummy_input)

# Save the traced model to a file
torch.jit.save(traced_model, "simple_model.pt")
```
Here, the `torch.jit.trace` function is key. It executes the model with the provided `dummy_input` and captures the sequence of operations performed. This captured execution, now a TorchScript module, is what DJL can then directly consume. The file `simple_model.pt` is what you'll work with in Java.

Now, let’s move over to the Java side. I’m going to assume a gradle or maven setup with DJL as a dependency (which involves the core DJL engine, and also the pytorch specific provider if needed, depending on your runtime environment). Loading and executing the model would generally look something akin to the example below, along with the necessary setup:

```java
import ai.djl.*;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;

public class LoadAndPredict {

    public static void main(String[] args) throws ModelException, MalformedModelException, TranslateException {
        // Define the model's location and criteria
        Path modelPath = Paths.get("path/to/your/model/simple_model.pt"); //Change this path to your actual location
        Criteria<Input, Output> criteria = Criteria.builder()
                .setTypes(Input.class, Output.class)
                .optModelPath(modelPath)
                .optEngine("PyTorch")
                .build();


        try(ZooModel<Input, Output> model = criteria.loadModel()){
           NDManager manager = model.getNDManager();
           Predictor<Input, Output> predictor = model.newPredictor();

           // Example input
           NDArray input = manager.create(new float[]{1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f}).reshape(1, 10);
           Input djlInput = new Input();
           djlInput.add(input);

           // Run the prediction
           Output djlOutput = predictor.predict(djlInput);

            //Process output
           NDArray outputTensor = djlOutput.getData().getAsNDArray();
           float[] output = outputTensor.toFloatArray();
           System.out.println("Output: " + java.util.Arrays.toString(output));
        }
    }
}
```
In this Java snippet, we first set up the `Criteria` which tells DJL where to find the TorchScript model, what types of input/output to expect, and that we are using PyTorch. Then, we load the model via the `ZooModel` object. We create an `NDManager` which handles memory allocation for the NDArrays used in DJL and finally, a `Predictor` that allows running inferences on the loaded model. Critically, note that you will need to create input data in a DJL-compatible `NDArray`, and encapsulate it in an `Input` object. Finally we extract the data from the `Output`, again as an `NDArray`, and can transform it to a Java array to print the results.

One aspect that’s often underestimated when working with DL models is preprocessing and postprocessing. Real-world data seldom fits the model's input requirements directly. DJL allows for custom `Translator` classes to handle these tasks, which become vital for end-to-end solutions. This often involves data normalization, reshaping, or mapping output results to interpretable predictions, depending on the use case. For simple cases like the above example, you can manipulate the data as necessary without needing a custom translator, but in real world scenarios with more complex input and outputs this is usually a necessity.

Let’s consider a scenario where we need to preprocess input images and interpret the output of a classification model. Assume we have a pre-trained image classification model in PyTorch, saved as `image_classifier.pt`. In Java, instead of having direct numerical data, our input is an image, represented as a byte array. Here is how we may adjust our `Criteria` and handle the image prediction:

```java
import ai.djl.*;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class ImagePrediction {

    public static void main(String[] args) throws ModelException, MalformedModelException, TranslateException, IOException {
        // Define the model's location and criteria
         Path modelPath = Paths.get("path/to/your/image_classifier.pt"); // Change this path to your actual model location

        Criteria<Image, Classifications> criteria = Criteria.builder()
                .setTypes(Image.class, Classifications.class)
                .optModelPath(modelPath)
                .optEngine("PyTorch")
                .build();

        try(ZooModel<Image, Classifications> model = criteria.loadModel()){
          Predictor<Image, Classifications> predictor = model.newPredictor();

          Path imagePath = Paths.get("path/to/your/image.jpg"); // Change this to the path of an image file
          byte[] imageBytes = Files.readAllBytes(imagePath);
          Image image = ImageFactory.getInstance().fromBytes(imageBytes);

            Classifications predictions = predictor.predict(image);
            System.out.println(predictions);
           }
        }
    }
```
The critical change here is that the `Criteria` is defined with `Image.class` as the input type and `Classifications.class` as the output type. This means that DJL automatically infers the need for the relevant input conversion logic to pass the image as an `NDArray` and that we can get the model output as a `Classifications` object, which we can easily read. This is a large step, as normally, you would need to manually preprocess the image to the appropriate numerical representation needed by the model.

For resources, I'd highly recommend the official DJL documentation itself, which provides detailed explanations and examples. Also, 'Deep Learning with Python' by François Chollet is a fantastic resource for solidifying your understanding of deep learning concepts that directly transfer to DJL. Specifically for TorchScript, the official PyTorch documentation is the best source; look for sections detailing `torch.jit`. The book 'Programming PyTorch for Deep Learning' by Ian Pointer provides a good overview of how PyTorch models work, which is beneficial when debugging issues. You can often encounter unexpected behaviors, but a solid understanding of what’s occurring on the PyTorch side allows you to diagnose problems effectively.

In my experience, the most common difficulties often arise from mismatches in input shapes or incorrect data type conversions. Carefully examine the input shape that your PyTorch model expects, and make sure that data is passed correctly to the DJL model. Additionally, checking the model's output directly in PyTorch can greatly assist in ensuring that you understand what the outputs should look like, allowing you to be sure that it is a problem with the Java DJL side of the setup. Getting a model from training to production, especially across different languages, has its challenges. By systematically working through the steps and keeping the underlying principles in mind, you can bridge the gap between a PyTorch-trained model and a robust Java deployment with DJL.
