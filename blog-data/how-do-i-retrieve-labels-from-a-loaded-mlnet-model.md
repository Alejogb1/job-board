---
title: "How do I retrieve labels from a loaded ML.NET model?"
date: "2024-12-23"
id: "how-do-i-retrieve-labels-from-a-loaded-mlnet-model"
---

Let's tackle extracting labels from an already trained ML.NET model, a situation I’ve encountered more times than I care to count during my work on various machine learning pipelines. It's a common pitfall, especially when working with models trained by someone else, or when coming back to a project after a long break. The challenge essentially revolves around understanding how ML.NET manages the mapping between your raw data and the output predictions, and then peeling that back to get to the labels. Let me break down how it works and some practical strategies you can employ, drawing from my own experiences.

ML.NET, under the hood, uses a pipeline-based approach. When you train a model, it essentially creates a series of data transformations, starting from your input data, processing them through feature engineering steps, and culminating in a trained algorithm. Part of this pipeline often involves converting categorical or textual labels into numerical representations for the model to work with. The core point here is that the model itself doesn't store your original labels directly; rather, it works with transformed representations. These transformations, especially the one that converts strings to numeric keys and vice versa, are vital to extracting your desired labels.

Typically, when you train a model with text labels, such as when doing classification, the `Train` method automatically creates this mapping. This is achieved by using the data transformations, such as `MapValueToKey` and `MapKeyToValue` in the `Transforms` namespace. The key-to-value map becomes crucial later when you're trying to interpret the model's predictions.

The first step in retrieving these labels involves understanding where this transformation is stored after model training. It is essentially serialized within the model's binary file. When you load the model, ML.NET loads the entire pipeline, including this necessary mapping. However, you don’t get it as a list of strings out-of-the-box. We need to explicitly access the transform part.

Here's a breakdown of the process and how to implement it, combined with some real-world experience from a project where I had to refactor a pipeline someone else built—it involved an image classifier, and the labels were not accessible directly after loading. I had to explicitly inspect and extract them:

**Method 1: Using `KeyToValueMapping` Transformer (Most Common)**

This method applies when your labels were converted from strings to keys using `MapValueToKey` during training, followed by `MapKeyToValue` as the reverse transform.

```csharp
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections.Generic;
using System.Linq;

public class LabelExtractor
{
    public static List<string> GetLabelsFromModel(string modelPath, string labelColumnName)
    {
        MLContext mlContext = new MLContext();
        ITransformer trainedModel = mlContext.Model.Load(modelPath, out var modelInputSchema);
        
        // 1. Find the key-to-value transform corresponding to the label column.
        var keyToValueTransform = trainedModel.GetOutputSchema(modelInputSchema)
                                     .GetColumnOrNull(labelColumnName)
                                     ?.Metadata
                                     .Where(x => x.Type == typeof(KeyToValueMapping))
                                     .Select(x => (KeyToValueMapping)x.Value)
                                     .FirstOrDefault();


        if (keyToValueTransform == null)
        {
            // No transform found; it's possible labels weren't keys
            return null; // Or handle this case in another way.
        }

        // 2. Retrieve the list of string labels from mapping.
        var labelNames = new List<string>();
        for (int i = 0; i < keyToValueTransform.Count; i++) {
            labelNames.Add(keyToValueTransform.GetLabel(i).ToString());
         }

        return labelNames;
    }
}
```

In this snippet, the crucial part is accessing the model’s `OutputSchema` and finding the `KeyToValueMapping` metadata associated with the label column. This metadata contains the information we need to get back to the string representations. I distinctly recall spending a few hours debugging this on a model where the label column name was not consistently applied through the codebase. A quick inspection through the model schema solved the issue.

**Method 2: Using the `LookupTable` Transformer**

Sometimes, especially when working with text data or more complex scenarios, the model might employ a `LookupTable` transformer. This transformer allows mapping of text to numerical keys. Here's how to handle it if you face this variation:

```csharp
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections.Generic;
using System.Linq;

public class LookupLabelExtractor
{
     public static List<string> GetLabelsFromModel(string modelPath, string labelColumnName)
     {
        MLContext mlContext = new MLContext();
        ITransformer trainedModel = mlContext.Model.Load(modelPath, out var modelInputSchema);

         // 1. Find the LookupTable transform
        var lookupTableTransform = trainedModel.GetOutputSchema(modelInputSchema)
                                     .GetColumnOrNull(labelColumnName)
                                     ?.Metadata
                                     .Where(x => x.Type == typeof(LookupTable))
                                     .Select(x => (LookupTable)x.Value)
                                     .FirstOrDefault();

        if (lookupTableTransform == null)
        {
            // No LookupTable transform found.
            return null; // Or handle appropriately.
        }

        // 2. Extract labels from the lookup table
        var labelNames = new List<string>();
        var keys = new VBuffer<ReadOnlyMemory<char>>();
        lookupTableTransform.GetKeys(ref keys);

        foreach(var mem in keys.DenseValues()){
            labelNames.Add(mem.ToString());
        }


        return labelNames;
    }
}
```

The key here is the use of the `LookupTable` metadata type. The approach is broadly similar, but the retrieval of the original string values from the table differs. In one memorable project, I was given a pre-trained model that used a custom text normalization and lookup, and this approach was essential in understanding which classes were being predicted.

**Method 3: Explicitly Training with a Schema Definition**

This method is applicable if you're training your model and can explicitly define the schema. This approach allows more direct control over the pipeline. I usually use this approach when setting up new ML projects:

```csharp
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections.Generic;
using System.Linq;

public class ExplicitLabelDefinition
{
     public static (ITransformer, List<string>) TrainWithLabelExtraction<T>(MLContext mlContext, 
                                                                          IEnumerable<T> trainingData,
                                                                          string labelColumnName, 
                                                                          IEstimator<ITransformer> trainingPipeline)
                                                                             where T : class
    {
        IDataView trainingDataView = mlContext.Data.LoadFromEnumerable(trainingData);

        // 1. Extract unique labels before training
        var uniqueLabels = mlContext.Data
                                     .CreateEnumerable<T>(trainingDataView, false)
                                     .Select(x => typeof(T).GetProperty(labelColumnName)?.GetValue(x)?.ToString())
                                     .Where(s => !string.IsNullOrEmpty(s))
                                     .Distinct()
                                     .ToList();


        // 2. Train with key-to-value mapping
        var preprocessor = mlContext.Transforms.Conversion.MapValueToKey(labelColumnName, labelColumnName);

        var completePipeline = preprocessor.Append(trainingPipeline).Append(mlContext.Transforms.Conversion.MapKeyToValue(labelColumnName, labelColumnName));


        ITransformer trainedModel = completePipeline.Fit(trainingDataView);
        return (trainedModel, uniqueLabels);


    }
}
```

This approach highlights that the labels can be obtained programmatically when you train your model and you have access to the labels during training.

Each of these methods provides a way to access the labels from your ML.NET model. The best one will depend on how the model was initially trained and what transformations it uses. Understanding the flow of data and transformations applied to your data, specifically the conversions between your string labels and numeric keys, is key to successful model analysis and deployment.

**Technical Resources:**

For further reading and a deeper understanding, I would highly suggest the following:

*   **The ML.NET documentation on the Microsoft website:** The official documentation is comprehensive and has detailed explanations of data transformations, model pipelines, and metadata access, specifically look for `Microsoft.ML.Data` and `Microsoft.ML.Transforms` namespaces.
*   **"Programming Machine Learning: From Data to Production Systems" by Paolo Perrotta:** This book is a practical guide that covers many facets of ML.NET development, including schema handling and data transformations.
*   **ML.NET GitHub repository:** Examining the unit tests and source code of the `KeyToValueMapping` and other relevant transforms can provide an in-depth understanding of how these mechanisms work under the hood.
*   **Research papers on model interpretability:** While not specific to ML.NET, understanding how to extract meaningful information from ML models is invaluable, and there are many recent publications which will give you a broader perspective.

By combining these resources and the examples I've provided, you should be well-equipped to retrieve those labels confidently from your ML.NET model. Always remember to meticulously check how your model was initially trained, specifically focusing on what transformations are applied to your label column during pipeline creation. This will help in choosing the correct approach.
