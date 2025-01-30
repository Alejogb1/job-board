---
title: "How can TF model.summary() be displayed on a plot?"
date: "2025-01-30"
id: "how-can-tf-modelsummary-be-displayed-on-a"
---
TensorFlow's `model.summary()` method provides a text-based overview of a neural network's architecture, which, while informative, lacks the visual clarity often desired for complex models. The challenge is, this method generates formatted strings, not numerical data readily plottable. Bridging this gap requires an indirect approach that involves parsing the output of `model.summary()` and representing the extracted information graphically. I've encountered this need frequently during network architecture exploration, especially when assessing layer parameter counts or visualizing hierarchical relationships.

Fundamentally, `model.summary()` prints a string to standard output. The string contains information organized into a table-like structure: layer name, output shape, and number of parameters. To visualize this, we need to convert this tabular representation into a data structure that a plotting library can interpret. This involves:

1.  **Capturing the Output:** Redirect the standard output from `model.summary()` to a string variable.
2.  **Parsing the String:** Extract the relevant information (layer name, output shape, parameter count) from this string.
3.  **Data Structuring:** Organize the extracted data into a suitable format (e.g., lists, dictionaries, Pandas DataFrames).
4.  **Visualization:** Use a plotting library (like Matplotlib or Seaborn) to create a graph representing the model summary.

Let's illustrate this with a few examples. First, consider the common task of visualizing the number of trainable parameters per layer. Here, we're primarily interested in layer names and their corresponding parameter counts.

```python
import tensorflow as tf
import io
import matplotlib.pyplot as plt
import re

def capture_summary(model):
    buffer = io.StringIO()
    model.summary(print_fn=lambda x: buffer.write(x + "\n"))
    summary_text = buffer.getvalue()
    return summary_text

def parse_parameter_counts(summary_text):
    lines = summary_text.splitlines()
    data = []
    for line in lines:
        if "Trainable params" in line:
             total_params = int(re.findall(r'\d+', line.split(':')[1].replace(',', ''))[0])
        parts = line.split()
        if len(parts) >= 5 and parts[1] != "layer": #skip the headers
            try:
                layer_name = ' '.join(parts[0:-3])
                param_count = int(parts[-1].replace(',', ''))
                data.append((layer_name, param_count))
            except ValueError:
                continue #skip non-param lines

    return data, total_params

def plot_parameter_counts(model):
    summary = capture_summary(model)
    data, total_params = parse_parameter_counts(summary)
    layer_names, param_counts = zip(*data)

    plt.figure(figsize=(10, 6))
    plt.bar(layer_names, param_counts)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Number of Trainable Parameters")
    plt.title(f"Trainable Parameters per Layer (Total Params: {total_params})")
    plt.tight_layout()
    plt.show()

# Example usage:
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
plot_parameter_counts(model)

```

In the above example, the `capture_summary` function diverts the output of `model.summary()` into a string buffer. The function `parse_parameter_counts` extracts layer names and their trainable parameter counts, converting the counts from string representation to integers. The `plot_parameter_counts` function then generates a bar chart visualizing these extracted parameter counts, additionally displaying total parameters count in the title. I've included error handling to skip header rows and any lines that cannot be properly parsed into the required format.

Now, consider a more detailed visualization incorporating both trainable and non-trainable parameter counts.

```python
def parse_all_parameter_counts(summary_text):
    lines = summary_text.splitlines()
    data = []
    total_params_train = 0
    total_params_non_train = 0
    for line in lines:
        if "Trainable params" in line:
            total_params_train = int(re.findall(r'\d+', line.split(':')[1].replace(',', ''))[0])
        if "Non-trainable params" in line:
            total_params_non_train = int(re.findall(r'\d+', line.split(':')[1].replace(',', ''))[0])
        parts = line.split()
        if len(parts) >= 5 and parts[1] != "layer": #skip the headers
            try:
                 layer_name = ' '.join(parts[0:-3])
                 param_count_total = int(parts[-1].replace(',', ''))
                 trainable_str = parts[-2]
                 if trainable_str == '-':
                     param_count_trainable = 0
                 else:
                    param_count_trainable = int(trainable_str.replace(',', ''))
                 param_count_non_trainable = param_count_total - param_count_trainable
                 data.append((layer_name,param_count_trainable,param_count_non_trainable))
            except ValueError:
                continue

    return data,total_params_train, total_params_non_train


def plot_all_parameter_counts(model):
    summary = capture_summary(model)
    data,total_params_train, total_params_non_train  = parse_all_parameter_counts(summary)

    layer_names, param_counts_trainable, param_counts_non_trainable = zip(*data)

    plt.figure(figsize=(12, 6))
    width = 0.35
    x = range(len(layer_names))
    plt.bar(x, param_counts_trainable, width, label='Trainable Parameters')
    plt.bar([pos + width for pos in x], param_counts_non_trainable, width, label='Non-trainable Parameters')

    plt.xticks([pos + width/2 for pos in x], layer_names, rotation=45, ha="right")
    plt.ylabel("Number of Parameters")
    plt.title(f"Parameters per Layer (Total Trainable: {total_params_train}, Non-Trainable: {total_params_non_train})")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Example usage:
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(10,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
plot_all_parameter_counts(model)
```

This enhanced version, `parse_all_parameter_counts`, extracts both trainable and non-trainable parameters using regular expressions and string parsing, if they are present in the output. The `plot_all_parameter_counts` function produces a grouped bar chart, visually differentiating between these two types. This kind of visualization can be helpful for spotting layers with a large number of fixed parameters, such as batch normalization. The code also displays both trainable and non-trainable parameter counts in the title, providing overall statistics.

Finally, for a more complex network, we can potentially visualize hierarchical structures by considering layer types. While `model.summary()` does not directly provide this, we can infer it from the layer names.

```python
def parse_layer_types(summary_text):
    lines = summary_text.splitlines()
    data = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 5 and parts[1] != "layer": #skip the headers
            try:
                layer_name = ' '.join(parts[0:-3])
                if "dense" in layer_name.lower():
                    layer_type = "Dense"
                elif "conv" in layer_name.lower():
                     layer_type = "Convolutional"
                elif "batch" in layer_name.lower():
                      layer_type = "BatchNormalization"
                elif "dropout" in layer_name.lower():
                      layer_type = "Dropout"
                else:
                      layer_type = "Other"
                data.append((layer_name, layer_type))
            except ValueError:
                continue
    return data

def plot_layer_types(model):
    summary = capture_summary(model)
    data = parse_layer_types(summary)

    layer_names, layer_types = zip(*data)
    layer_type_counts = {}
    for type_ in layer_types:
       if type_ in layer_type_counts:
           layer_type_counts[type_] += 1
       else:
            layer_type_counts[type_] = 1
    labels = list(layer_type_counts.keys())
    sizes = list(layer_type_counts.values())


    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Layer Type Distribution')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.tight_layout()
    plt.show()



# Example Usage
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
     tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation='softmax')
])
plot_layer_types(model)
```

This example, `parse_layer_types`, categorizes layers based on keywords in their names. The `plot_layer_types` then presents this as a pie chart, showing the distribution of layer types within the model. This particular implementation provides a basic layer type detection logic; enhancements may be needed for more complex cases or if model's custom layer names vary.

These examples provide methods for capturing `model.summary()`'s output, parsing relevant information, and generating different types of visualizations using Matplotlib. They show the flexibility needed for such transformation as the output is not a simple, easily usable object.

For further exploration and enhancement of these techniques, consult the official TensorFlow documentation for `model.summary()`, Matplotlib's comprehensive tutorials, and texts on advanced Python string manipulation and regular expressions. There are also several resources online covering data visualization best practices that can help improve the quality and informativeness of the charts. These resources, when taken together, create a better understanding of visualizing model architectures in ways beyond text.
