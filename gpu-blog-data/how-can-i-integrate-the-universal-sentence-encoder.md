---
title: "How can I integrate the Universal Sentence Encoder TensorFlow model into ML.NET?"
date: "2025-01-30"
id: "how-can-i-integrate-the-universal-sentence-encoder"
---
The challenge of integrating pre-trained TensorFlow models, specifically the Universal Sentence Encoder (USE), within the .NET ecosystem using ML.NET requires careful handling of model serialization and inference pipeline construction. ML.NET primarily operates on its own data structures and execution environment, demanding an intermediary layer to bridge the TensorFlow graph representation.

My experience with migrating a sentiment analysis pipeline from Python-centric TensorFlow to a .NET service illuminated several core requirements. The most crucial being that direct model loading from a SavedModel or similar TensorFlow format is not natively supported by ML.NET. Instead, we must leverage the ONNX (Open Neural Network Exchange) format, a standardized model representation designed for interoperability. Thus, the initial, pivotal step is converting the Universal Sentence Encoder from its TensorFlow representation into an equivalent ONNX model.

Here's a structured breakdown of the required steps, along with demonstrative code examples:

**1. TensorFlow to ONNX Conversion:**

The conversion process generally occurs within a Python environment, where TensorFlow and ONNX libraries are readily accessible. You'd typically use the `tf2onnx` package. This conversion also requires the specification of input and output tensor names of the USE model, which can be found in its documentation or by inspecting the TensorFlow graph using tools like `TensorBoard`.

```python
import tensorflow as tf
import tf2onnx

def convert_use_to_onnx(saved_model_path, output_path, input_name='inputs', output_name='outputs'):
    """Converts a SavedModel Universal Sentence Encoder to ONNX.

    Args:
        saved_model_path (str): Path to the SavedModel directory.
        output_path (str): Path to save the ONNX model.
        input_name (str): Name of the input tensor.
        output_name (str): Name of the output tensor.
    """

    try:
      tf_module = tf.saved_model.load(saved_model_path)
      signature = tf_module.signatures['serving_default']

      input_specs = [tf.TensorSpec(shape=(None,), dtype=tf.string, name=input_name)]
      output_specs = [tf.TensorSpec(shape=(None, 512), dtype=tf.float32, name=output_name)]

      onnx_model, _ = tf2onnx.convert.from_signature(signature, input_signature=input_specs,
                                                      output_signature=output_specs,
                                                      output_path=output_path)

      print(f"Successfully converted SavedModel to ONNX: {output_path}")
    except Exception as e:
      print(f"Error during conversion: {e}")


# Example Usage (assuming saved model is at './use_saved_model' and output is './use.onnx'):
saved_model_location = "./use_saved_model"
output_onnx_path = "./use.onnx"
convert_use_to_onnx(saved_model_location, output_onnx_path)
```

*Commentary:* This Python code snippet defines a function, `convert_use_to_onnx`, encapsulating the conversion logic. It uses `tf.saved_model.load` to load the USE saved model, identifies input and output specifications (`input_specs` and `output_specs`), then leverages `tf2onnx.convert.from_signature` to perform the conversion. The critical part here is accurately determining the input and output tensor specifications, notably the string input for sentences and the 512-dimensional float output vector. Incorrect specifications can result in failed conversion or unusable ONNX models. The example usage provides concrete paths to a hypothetical saved model and the resulting ONNX file location.

**2. Loading the ONNX Model in ML.NET:**

With the ONNX model prepared, we shift our focus to the .NET environment and ML.NET. ML.NET provides an `OnnxScoringEstimator`, which facilitates the loading and execution of ONNX models. This involves defining the input and output data schemas within ML.NET, mirroring the specifications within the ONNX model.

```csharp
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;


public class SentenceData
{
    [LoadColumn(0)]
    public string Sentence { get; set; }
}

public class SentenceEmbedding
{
    [VectorType(512)]
    [ColumnName("outputs")]
    public float[] Embedding { get; set; }
}

public class OnnxIntegration
{

    public void EmbedSentences(string onnxModelPath, List<string> sentences)
    {
        var mlContext = new MLContext();

        // Define input schema
        var data = mlContext.Data.LoadFromEnumerable(sentences.Select(s => new SentenceData { Sentence = s }));

        // Define ONNX scoring pipeline
        var pipeline = mlContext.Transforms.ApplyOnnxModel(
            modelFile: onnxModelPath,
            outputColumnNames: new[] { "outputs" },
            inputColumnNames: new[] { "inputs" }
        );

        // Fit and transform
        var transformer = pipeline.Fit(data);
        var output = transformer.Transform(data);

        // Extract embeddings
        var embeddings = mlContext.Data.CreateEnumerable<SentenceEmbedding>(output, reuseRowObject: false).ToList();

        //Process embeddings
        for(int i = 0; i < embeddings.Count; i++) {
             Console.WriteLine($"Embedding for sentence '{sentences[i]}': {string.Join(", ", embeddings[i].Embedding.Take(5))} ...");
        }
    }
}


// Example Usage:
// Assuming the converted ONNX model is at './use.onnx' and sentences
// to be embedded are in a list.
string onnxPath = "./use.onnx";
List<string> inputSentences = new List<string>() { "This is a test sentence.", "Another test." };
new OnnxIntegration().EmbedSentences(onnxPath, inputSentences);
```

*Commentary:* This C# code defines two simple data structures, `SentenceData` to represent the raw input and `SentenceEmbedding` for the resulting vector embeddings. The core logic resides within the `EmbedSentences` method, which uses `ApplyOnnxModel` to create an inference pipeline referencing the provided ONNX model path. `outputColumnNames` and `inputColumnNames` specify the ONNX tensor names for output and input columns, respectively, matching the Python conversion process. The method then transforms the input data using the trained pipeline and extracts the output embeddings using `CreateEnumerable<SentenceEmbedding>`.  The example illustrates a basic usage by providing a sample set of sentences and outputting the first five elements of the resultant vectors.

**3. Handling Text Preprocessing:**

While the ONNX model handles the core embedding computation, the input data for the USE model, as originally designed in TensorFlow, is a string tensor.  ML.NET has no specific built-in transformer for preprocessing raw text directly into a string tensor.  You can't just map strings as inputs; you have to convert them in a way that mimics the underlying tensorflow graph's input structure. This may involve custom steps to mimic tokenization and encoding that the original model expected.  Depending on the complexity of the use case, a very simple "do-nothing" approach that simply passes the raw string can sometimes work, but it's often not ideal and might require extra preprocessing in a different stage of the pipeline if required.

```csharp
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;

public class SentenceData
{
    [LoadColumn(0)]
    public string Sentence { get; set; }
}

public class SentenceEmbedding
{
    [VectorType(512)]
    [ColumnName("outputs")]
    public float[] Embedding { get; set; }
}

public class OnnxIntegrationAdvanced
{
    public void EmbedSentences(string onnxModelPath, List<string> sentences)
    {
          var mlContext = new MLContext();


         // Define input schema
        var data = mlContext.Data.LoadFromEnumerable(sentences.Select(s => new SentenceData { Sentence = s }));

        // Define ONNX scoring pipeline
        var pipeline = mlContext.Transforms.CopyColumns("Sentence", "inputs")
            .Append(mlContext.Transforms.ApplyOnnxModel(
            modelFile: onnxModelPath,
            outputColumnNames: new[] { "outputs" },
            inputColumnNames: new[] { "inputs" }
         ));

        // Fit and transform
        var transformer = pipeline.Fit(data);
        var output = transformer.Transform(data);


         var embeddings = mlContext.Data.CreateEnumerable<SentenceEmbedding>(output, reuseRowObject: false).ToList();

        for(int i = 0; i < embeddings.Count; i++) {
             Console.WriteLine($"Embedding for sentence '{sentences[i]}': {string.Join(", ", embeddings[i].Embedding.Take(5))} ...");
        }
    }
}


// Example Usage:
// Assuming the converted ONNX model is at './use.onnx' and sentences
// to be embedded are in a list.
string onnxPathAdv = "./use.onnx";
List<string> inputSentencesAdv = new List<string>() { "This is a test sentence.", "Another test." };
new OnnxIntegrationAdvanced().EmbedSentences(onnxPathAdv, inputSentencesAdv);
```

*Commentary:* This third example enhances the previous one by explicitly adding a `CopyColumns` step before invoking the ONNX scoring. This makes the pipeline more explicit by mapping the `Sentence` column to the `inputs` tensor of the model.  While, in this specific scenario the CopyColumns provides simple and works, this might not always be enough and could need more custom processing to perfectly align the data format to the ONNX model. The rest of the inference flow remains the same, demonstrating a more robust method of binding string inputs to the ONNX model's input layer.

**Resource Recommendations:**

*   **TensorFlow Documentation:** For detailed information on the Universal Sentence Encoder, particularly its input/output specifications and SavedModel format.
*   **ONNX Documentation:** For understanding the ONNX format, its specifications, and related tooling.
*   **ML.NET Documentation:** For detailed usage of `OnnxScoringEstimator` and other relevant ML.NET components. The API documentation and samples are invaluable for practical implementations.
*   **tf2onnx Library Repository:** The Github repository provides detailed guidance and example code for `tf2onnx` tool and understanding common issues.
*   **Machine Learning Blogs and Articles:** Search specific articles on integrating TensorFlow models with .NET for deeper practical insights and troubleshooting tips.

Integrating a TensorFlow model like the Universal Sentence Encoder with ML.NET requires several distinct steps, primarily converting to ONNX format, defining appropriate input and output schemas in ML.NET, and ensuring the input data matches expected specifications. While this detailed breakdown showcases a functional approach, certain use cases may necessitate further customized text preprocessing steps before model ingestion.
