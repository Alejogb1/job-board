---
title: "Why is tfjs throwing a TypeError: Cannot read properties of undefined (reading 'outputs')?"
date: "2025-01-30"
id: "why-is-tfjs-throwing-a-typeerror-cannot-read"
---
TensorFlow.js (tfjs), when integrated into a web environment, often encounters a `TypeError: Cannot read properties of undefined (reading 'outputs')` during model execution, frequently stemming from misconfigurations in how the model output is handled or expected, particularly after a model is loaded or converted. This error typically signifies that the framework is attempting to access an 'outputs' property on an object that does not exist or is undefined at the time of access. My experiences with web-based machine learning projects reveal this issue arises from several distinct causes, all directly impacting the pipeline of data flow within tfjs.

The root problem lies in the expectation of a model's output structure. When a tfjs model is loaded—whether a model saved from Python via `tf.saved_model.save`, a converted Keras model, or a pre-trained model from TensorFlow Hub—tfjs internally defines how it expects the model's output to be structured. This structure includes the names and types of tensors generated after the input data is processed. If the code that uses the loaded model does not align with these expected outputs, the dreaded `TypeError` emerges because tfjs cannot find the 'outputs' property it anticipates. Common scenarios include an incorrect prediction call, model input data that doesn't match the model's expected input shape, or an incomplete or corrupted model loading process.

Let’s consider a typical workflow, often involving an asynchronous model loading:

```javascript
async function loadAndPredict(inputTensor) {
  const modelUrl = 'path/to/my/model/model.json';
  try {
    const model = await tf.loadGraphModel(modelUrl);
    const output = model.execute(inputTensor);

    console.log(output.outputs); // Potential error location
    
    // Further processing of outputs.
  } catch (error) {
    console.error("Error loading or executing model:", error);
  }
}
```

In this example, the `tf.loadGraphModel` is used to load a SavedModel that was converted to a JSON compatible format for web deployment. Crucially, the code makes the implicit assumption that the output of `model.execute()` has a property called "outputs".  If `model.execute()` returns a tensor or a single output (rather than a keyed structure) that is not wrapped in an object with an "outputs" property, then accessing `output.outputs` will generate the aforementioned error. The immediate fix is to examine how the model returns outputs. In many cases, the `model.execute()` returns the tensor directly, not as an object. The corrected code would thus be:

```javascript
async function loadAndPredict(inputTensor) {
  const modelUrl = 'path/to/my/model/model.json';
    try {
      const model = await tf.loadGraphModel(modelUrl);
      const output = model.execute(inputTensor);

      console.log(output); // Correctly access tensor directly
      // Further processing of output.
    } catch (error) {
        console.error("Error loading or executing model:", error);
    }
}
```

This correction illustrates the first point: understanding the structure of what `model.execute` is returning. It isn't always, nor should it be assumed to be, an object with `outputs`. 

A second variation of this problem arises when working with models with multiple output tensors, a typical architecture in object detection or multi-task learning.  Suppose the model expects two separate output tensors, ‘boxes’ and ‘classes’.  Let’s illustrate the error using a modified snippet.

```javascript
async function predictWithMultiOutputs(inputTensor) {
   const modelUrl = 'path/to/multi/output/model.json';
   try {
      const model = await tf.loadGraphModel(modelUrl);
      const predictions = model.execute(inputTensor);

      console.log(predictions.outputs[0]); // This is incorrect
      console.log(predictions.outputs[1]);
     
      // Process output tensors
   } catch (error) {
     console.error("Error loading model or during execution:", error);
   }
}
```

In this context, a model with multiple outputs does not produce an object with a ‘outputs’ property that is an array containing the output tensors. The model will, by default, return an array of tensors directly.  The problem here isn't that `outputs` is undefined, but rather that `predictions` is already the array of tensors (or in some other cases a keyed object), not an object containing the actual output tensors within a `outputs` property. Accessing `predictions[0]` directly, instead of `predictions.outputs[0]`, resolves this type of error. Here is the correction:

```javascript
async function predictWithMultiOutputs(inputTensor) {
    const modelUrl = 'path/to/multi/output/model.json';
    try {
        const model = await tf.loadGraphModel(modelUrl);
        const predictions = model.execute(inputTensor);

        console.log(predictions[0]); //Correct access of tensors
        console.log(predictions[1]);

        // Process output tensors
    } catch (error) {
        console.error("Error loading model or during execution:", error);
    }
}
```

In the event that the model is more complex, and its `model.execute` returns a keyed structure that has a key named 'output' containing the tensor, or keys named 'boxes' and 'classes', then one must retrieve the tensor using their corresponding key rather than an array index.

Finally, another subtle origin of this error is model loading itself. If the model isn’t fully loaded or if the `model.json` file is corrupted, the tfjs object may not initialize with the necessary metadata needed for processing.  This often leads to a state where `model.execute` is either not a function or returns an undefined object, thus preventing the access of an 'outputs' property.  Here's an illustrative example:

```javascript
async function predictWithIncompleteLoad(inputTensor) {
  const modelUrl = 'path/to/potentially/corrupted/model.json';
    try {
      const model = await tf.loadGraphModel(modelUrl);

       if (!model) {
         console.error("Model did not load correctly.")
         return;
       }
      
      const output = model.execute(inputTensor);

      console.log(output.outputs); // This may fail due to loading error
    } catch (error) {
      console.error("Error during loading or execution:", error);
    }
}
```

The fix, in this scenario, is to explicitly test whether the model object was successfully loaded by checking for its truthiness, or more robustly, inspect its attributes such as `model.inputNodes`, and re-attempt the loading if necessary. Additionally, ensuring the JSON file is properly saved when exporting models from Python and that all required related files (e.g. `weights.bin`) are correctly available is crucial.  Adding logging within the `try`/`catch` statement is often the quickest diagnostic tool to isolate these issues.

In summary, troubleshooting `TypeError: Cannot read properties of undefined (reading 'outputs')` when using tfjs requires a systematic approach focusing on: validating that the output structure returned from `model.execute` aligns with the code attempting to process it, checking the correctness of multi-output processing and verifying that model loading is successful and complete. For additional information, reviewing the official TensorFlow.js documentation regarding model loading, specifically graph models and their execution lifecycle, would be highly beneficial. Resources dedicated to debugging and tensor manipulation using tfjs are also useful. Finally, consulting the guides for loading and deploying models from other frameworks such as Python (Keras or SavedModel formats) for tfjs is often invaluable.
