---
title: "How can custom metrics be collected in Katib (AutoML) on Kubeflow?"
date: "2025-01-30"
id: "how-can-custom-metrics-be-collected-in-katib"
---
Katib, a crucial component of Kubeflow for Automated Machine Learning (AutoML), relies heavily on metrics to guide its hyperparameter tuning process. Standard metrics like loss, accuracy, and precision are usually straightforward to collect, but scenarios often demand the use of custom metrics that are specific to the application. These custom metrics require explicit configuration within the Katib Experiment definition and a modification of the training job to report them in a way that Katib understands.

The fundamental challenge arises because Katib's metric collection mechanism expects to parse specific output patterns from the training job's logs. Therefore, successfully implementing custom metric collection involves two key steps: 1) ensuring the training job emits the custom metric in a structured, parsable format within its logs, and 2) configuring the Katib Experiment definition to understand and extract this metric.

Specifically, Katib uses regular expressions to locate and parse metric values within the logs. This means we are bound by the log emission format our training application employs. I've encountered situations where initial log formats were incompatible with Katib's regex engine, necessitating modification to training job code to conform.

Let me elaborate on the typical process with examples based on my experience developing an image classification experiment.

**Code Example 1: Modifying the Training Application**

Here, I'll illustrate a situation where my initial training script, using TensorFlow, needed modification to emit a custom F1-score metric that Katib could identify. Initially, only loss and validation accuracy were being logged directly to the console, and this is how the script outputted its values:

```python
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score

# ... (TensorFlow model definition and training loop) ...

def train_and_evaluate(model, train_dataset, val_dataset, epochs):
    for epoch in range(epochs):
        # Training loop code...
        # Calculate metrics as before
        val_loss, val_acc = model.evaluate(val_dataset)

        # Calculate F1-score (assuming `y_true` and `y_pred` are available in the validation loop)
        y_true = np.concatenate([y.numpy() for x,y in val_dataset])
        y_pred = model.predict(val_dataset)
        y_pred_classes = np.argmax(y_pred, axis=1)
        f1 = f1_score(y_true, y_pred_classes, average='weighted')

        print(f"Epoch: {epoch+1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}, F1 Score: {f1:.4f}")
        # Additional model saving and other operations.
```

This output format, while informative for monitoring, doesn't lend itself directly to Katib's regex-based parsing. I needed to format the F1-score output specifically so that Katib could extract the values. The output in its original form mixes multiple metrics on one line, making the regex selection ambiguous. To solve this, I re-wrote my training loop to output each metric on its own line, with a consistent structure that could be matched by a regular expression.

```python
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score

def train_and_evaluate(model, train_dataset, val_dataset, epochs):
    for epoch in range(epochs):
        # Training loop code...
        val_loss, val_acc = model.evaluate(val_dataset)

        y_true = np.concatenate([y.numpy() for x,y in val_dataset])
        y_pred = model.predict(val_dataset)
        y_pred_classes = np.argmax(y_pred, axis=1)
        f1 = f1_score(y_true, y_pred_classes, average='weighted')


        print(f"validation_loss={val_loss:.4f}")
        print(f"validation_accuracy={val_acc:.4f}")
        print(f"f1_score={f1:.4f}")
        # Additional model saving and other operations.
```

By emitting the metric with the specific `metric_name=metric_value` format, I simplified the extraction process for Katib using a more targeted regex pattern. I also ensured that each metric is reported on a new line, preventing ambiguity in the parsing process.

**Code Example 2: Configuring the Katib Experiment**

Now, with my training job correctly logging the custom F1-score, I needed to update the Katib Experiment definition. Specifically, the `metricsCollectorSpec` field in the experiment YAML needs to include this new metric, its associated regex, and optimization direction. Here is an excerpt of the relevant part of my original Katib experiment YAML:

```yaml
      metricsCollectorSpec:
        source:
          fileSystemPath:
            path: /var/log/katib
        metrics:
          - name: validation_loss
            regex: validation_loss=([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)
          - name: validation_accuracy
            regex: validation_accuracy=([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)
```

This original definition specified that Katib should monitor for `validation_loss` and `validation_accuracy` by matching the appropriate lines in the container output logs with regular expressions. To incorporate the `f1_score` metric, I added a new metric definition:

```yaml
      metricsCollectorSpec:
        source:
          fileSystemPath:
            path: /var/log/katib
        metrics:
          - name: validation_loss
            regex: validation_loss=([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)
          - name: validation_accuracy
            regex: validation_accuracy=([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)
          - name: f1_score
            regex: f1_score=([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)
```

The regex `f1_score=([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)` will match the line in the container output logs, extracts the float value after the `=` sign, and assign it to the `f1_score` metric in Katib. This addition to the `metrics` list makes the custom metric accessible to Katib's hyperparameter tuning algorithms. Note that no `goal` was specified in my experiment specification, so Katib would implicitly be trying to minimize `validation_loss` by default. I could set a goal of `maximize` for the `f1_score` metric instead, and Katib would then attempt to find hyperparameter values that maximized the F1-score.

**Code Example 3: Setting the Optimization Goal**

The original experiment specification did not specify which of the defined metrics should be the target of optimization. To make F1-score the target, I can specify a goal in the `objective` block of the `Experiment` definition. Note: this step is not directly related to the metric *collection*, but it *utilizes* the collected metric.

```yaml
  objective:
    type: maximize
    goal: f1_score
```

This simple change, placed in the appropriate location of the Experiment specification, signals that Katib’s optimization should aim at maximizing F1-score. This, combined with my modifications to the training script and `metricsCollectorSpec`, enables effective experimentation focused on this specific custom metric.

The process wasn’t always this linear. I've debugged multiple instances where regex patterns were too broad, capturing unintended values, or were too narrow, missing the intended metrics. Regularly validating the emitted log patterns with the regex expressions is critical for accurate metric capture. I've also found that using a more specific metric name, such as `validation_f1_score`, helps to avoid any potential clashes if there are similar-sounding metrics in a more complex training script or log.

For those working with Katib, I recommend these resource types for further guidance:

1.  **Kubeflow Documentation:** The official documentation for Katib and Kubeflow provides comprehensive information on experiment definitions, metric collection, and optimization algorithms. Pay particular attention to the sections related to experiment specifications, `metricsCollectorSpec`, and the structure of the `objective` field.

2. **Example Experiments:** Examining sample Katib experiment definitions from the Kubeflow repository provides practical insights into the configuration required for different use cases, including those with custom metrics. These provide a hands-on view on how the components connect in practice.

3. **Community Forums:** Active participation in Kubeflow community forums is invaluable for troubleshooting specific issues and understanding the nuances of the Katib platform. The Katib development team and community often provide solutions for corner cases and are aware of potential challenges in configurations.

By paying close attention to the output format of the training job and correctly configuring the Katib Experiment definition, collection of custom metrics in Katib becomes manageable. The key takeaway is that the entire process requires clear communication between your training code and Katib's metric collection mechanism, facilitated by well-formed logs and matching regular expressions.
