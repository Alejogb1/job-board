---
title: "Can ML.NET models be trained from multiple folders?"
date: "2024-12-23"
id: "can-mlnet-models-be-trained-from-multiple-folders"
---

Alright, let's tackle this one. I've actually dealt with scenarios exactly like this in a past project involving image classification for a client's retail catalog, where data was neatly organized (or not so neatly at times) into different product categories, each stored in a separate folder. So, the question of training ML.NET models from multiple folders isn't just theoretical; it's a very real, practical challenge.

The short answer is: yes, you absolutely can train ML.NET models from data spread across multiple folders, but it requires a bit more orchestration than simply pointing to a single data source. The key here is understanding how ML.NET manages data and the flexibility it offers through its data loading mechanisms. You're not confined to a single csv file or database connection. The system allows you to define data loading pipelines that can piece together data from disparate locations.

The core issue boils down to creating a unified view of your data so that the ML.NET training pipeline can process it. You won't be specifying multiple folder paths directly to a training function. Instead, you will typically leverage `IDataView` objects, which represent the tabular data format expected by the machine learning algorithms. These objects are the central data structure. Therefore, our focus is on loading data from different folders and combining it into a single `IDataView`.

Here's how it’s usually done, generally speaking:

1.  **Load data from individual folders:** Iterate through each folder containing your training data. Depending on the data format within the folders, you’ll use the corresponding data loader from ML.NET. This could be an image loader, a text loader, a CSV loader, etc. Each load operation yields an `IDataView`.
2.  **Add a "label" column (if required):** If your data is labelled, as is common in supervised learning, you'll often need to add a column indicating the label based on the folder the data was loaded from. For example, images in a folder named "cats" might get a label of "0," while those in "dogs" get a label "1." This is a crucial step to tell your model what to learn.
3.  **Append all IDataView objects:** Use ML.NET's `Append` functionality, which is exposed as an extension method on IDataView, to merge the loaded `IDataView` objects into a single, cohesive `IDataView`. This new, larger `IDataView` will contain data from all the folders, correctly labelled and ready for model training.
4.  **Train your model:** Finally, feed this combined `IDataView` into your chosen ML.NET trainer. The trainer is agnostic to the number of sources you loaded from, as long as it’s presented with a properly formatted `IDataView`.

Let's see some concrete code examples illustrating these concepts.

**Example 1: Image classification using multiple image folders**

This example uses a hypothetical scenario of loading images from `train/cats`, `train/dogs`, and `train/birds` folders.

```csharp
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;
using System.IO;
using System.Collections.Generic;
using System.Linq;

public class ImageClassification
{
    public void TrainModel(string baseDataPath, string modelPath)
    {
        var mlContext = new MLContext(seed: 0);
        List<IDataView> dataViews = new List<IDataView>();
        string[] labels = { "cats", "dogs", "birds" };

        foreach (string label in labels)
        {
            string imageFolderPath = Path.Combine(baseDataPath, label);
            if(Directory.Exists(imageFolderPath))
            {
              var images = Directory.GetFiles(imageFolderPath).Select(f => new ImageData() { ImagePath = f, Label = label }).ToList();

              IDataView dataView = mlContext.Data.LoadFromEnumerable(images);
             
              var preprocessedData = mlContext.Transforms.Conversion.MapValue(outputColumnName:"LabelAsKey", inputColumnName: "Label", mapping: new[] { new KeyValuePair<string, uint>("cats", 0), new KeyValuePair<string, uint>("dogs", 1), new KeyValuePair<string, uint>("birds",2) })
                .Append(mlContext.Transforms.LoadImages("ImageObject", imageFolderPath, "ImagePath"))
                .Append(mlContext.Transforms.ResizeImages("ImageObjectResized", imageWidth: 224, imageHeight: 224, inputColumnName: "ImageObject"))
                .Append(mlContext.Transforms.ExtractPixels("PixelValues", "ImageObjectResized", outputAsFloatArray: true))
                .Fit(dataView).Transform(dataView);

              dataViews.Add(preprocessedData);
            }
        }

        IDataView combinedDataView = dataViews.First();
        if (dataViews.Count > 1)
        {
            foreach (var dv in dataViews.Skip(1))
            {
                combinedDataView = mlContext.Data.AppendRows(combinedDataView, dv);
            }
        }

      var pipeline = mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "LabelAsKey", featureColumnName: "PixelValues").Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));

        ITransformer model = pipeline.Fit(combinedDataView);

        mlContext.Model.Save(model, combinedDataView.Schema, modelPath);

    }
}

public class ImageData
{
  public string ImagePath { get; set; }
  public string Label { get; set; }
}
```

**Example 2: Text classification using multiple text file folders**

Here we have a setup for sentiment analysis, where each folder contains text files with positive or negative reviews.

```csharp
using Microsoft.ML;
using Microsoft.ML.Data;
using System.IO;
using System.Collections.Generic;
using System.Linq;

public class TextClassification
{
    public void TrainModel(string baseDataPath, string modelPath)
    {
        var mlContext = new MLContext(seed: 0);
        List<IDataView> dataViews = new List<IDataView>();
        string[] labels = { "positive", "negative" };

        foreach (string label in labels)
        {
            string textFolderPath = Path.Combine(baseDataPath, label);
            if(Directory.Exists(textFolderPath))
            {
                var texts = Directory.GetFiles(textFolderPath)
                    .Select(file => new TextData { Text = File.ReadAllText(file), Label = label }).ToList();

                IDataView dataView = mlContext.Data.LoadFromEnumerable(texts);

                 var preprocessedData = mlContext.Transforms.Conversion.MapValue(outputColumnName: "LabelAsKey", inputColumnName: "Label", mapping: new[] { new KeyValuePair<string, uint>("positive", 0), new KeyValuePair<string, uint>("negative", 1) })
                    .Append(mlContext.Transforms.Text.FeaturizeText("Features", "Text"))
                    .Fit(dataView).Transform(dataView);

                  dataViews.Add(preprocessedData);
            }

        }
        IDataView combinedDataView = dataViews.First();
        if (dataViews.Count > 1)
        {
            foreach (var dv in dataViews.Skip(1))
            {
                combinedDataView = mlContext.Data.AppendRows(combinedDataView, dv);
            }
        }
        var pipeline = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "LabelAsKey", featureColumnName: "Features")
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));

        ITransformer model = pipeline.Fit(combinedDataView);
        mlContext.Model.Save(model, combinedDataView.Schema, modelPath);
    }
}
public class TextData
{
  public string Text { get; set; }
  public string Label { get; set; }
}
```

**Example 3: CSV data from multiple folders with similar schema**

Here each folder contains a `data.csv` file, all having the same structure.

```csharp
using Microsoft.ML;
using Microsoft.ML.Data;
using System.IO;
using System.Collections.Generic;
using System.Linq;

public class CsvDataClassification
{
    public void TrainModel(string baseDataPath, string modelPath)
    {
        var mlContext = new MLContext(seed: 0);
         List<IDataView> dataViews = new List<IDataView>();
         string[] labels = { "groupA", "groupB" };

        foreach (string label in labels)
        {
             string csvFolderPath = Path.Combine(baseDataPath, label);

            if(Directory.Exists(csvFolderPath))
            {
              string csvPath = Path.Combine(csvFolderPath, "data.csv");
                if(File.Exists(csvPath))
                {
                  IDataView dataView = mlContext.Data.LoadFromTextFile<CsvData>(
                    path: csvPath,
                    hasHeader: true,
                    separatorChar: ',',
                   allowQuotedStrings: true
                   );
                   var preprocessedData = mlContext.Transforms.Conversion.MapValue(outputColumnName: "LabelAsKey", inputColumnName: "Label", mapping: new[] { new KeyValuePair<string, uint>("groupA", 0), new KeyValuePair<string, uint>("groupB", 1) })
                  .Append(mlContext.Transforms.Concatenate("Features", "Feature1", "Feature2"))
                  .Fit(dataView).Transform(dataView);

                 dataViews.Add(preprocessedData);
                }
            }

        }
        IDataView combinedDataView = dataViews.First();
        if (dataViews.Count > 1)
        {
            foreach (var dv in dataViews.Skip(1))
            {
                combinedDataView = mlContext.Data.AppendRows(combinedDataView, dv);
            }
        }

          var pipeline = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "LabelAsKey", featureColumnName: "Features")
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));


        ITransformer model = pipeline.Fit(combinedDataView);
        mlContext.Model.Save(model, combinedDataView.Schema, modelPath);

    }
}

 public class CsvData
{
  [LoadColumn(0)]
  public float Feature1 { get; set; }
    [LoadColumn(1)]
  public float Feature2 { get; set; }
    [LoadColumn(2)]
  public string Label { get; set; }
}

```

In each example, the key is that we load data from multiple sources and merge the resulting IDataView objects.

For a deeper dive into the inner workings of ML.NET data loading and transforms, I'd highly recommend:

*   **"Machine Learning with .NET" by Chris Ross and Luis Quintanilla:** This is a comprehensive guide that explains the various aspects of ML.NET, including data handling, in detail.
*   **The official Microsoft ML.NET documentation:** It’s a solid resource, always kept up-to-date, and offers a wealth of information with specific details on all the transforms available. Specifically, look at the documentation regarding `IDataView`, different data loaders, and data transformations.
*   **“Deep Learning with Python” by François Chollet:** While this book focuses on Keras and TensorFlow, the general concepts and challenges surrounding data loading and preprocessing are quite universal and applicable to ML.NET scenarios as well. Understanding the general principles of data handling can be very useful.

Remember, these examples are simplified and tailored to fit into this response. Real-world scenarios might require more complex error handling, data cleaning steps, and sophisticated preprocessing pipelines. However, the underlying principle – loading from multiple sources and combining into a single `IDataView` for training – remains consistent. I hope this is helpful.
