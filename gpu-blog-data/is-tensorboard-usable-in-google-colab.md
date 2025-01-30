---
title: "Is TensorBoard usable in Google Colab?"
date: "2025-01-30"
id: "is-tensorboard-usable-in-google-colab"
---
TensorBoard's integration within Google Colab environments, while not directly supported through a built-in interface button like some other tools, is indeed achievable and quite practical. The key lies in leveraging Colab’s shell access and tunneling capabilities to display TensorBoard in the notebook environment. I’ve frequently employed this technique during deep learning model development on Colab, allowing for real-time visualization of training metrics and graph structures.

Colab provides a virtual machine instance, which means that TensorBoard, which is fundamentally a web server, needs a specific mechanism to become accessible through the browser that's rendering the Colab notebook. The standard `tensorboard --logdir` command will start the server on the virtual machine, but without further action, the notebook's browser will not see it. This requires setting up a tunnel to redirect network traffic from the server running on the VM to a local port accessible from the user's browser. The Colab environment allows users to execute arbitrary shell commands via the `!` prefix in code cells; we leverage this to launch the TensorBoard server and establish the necessary tunneling.

The process essentially consists of three main steps: first, ensuring that TensorBoard is installed; second, generating the required log directories; and third, launching the TensorBoard server and creating a public-facing tunnel via `ngrok`. The third step is crucial because Colab instances don't have public IPs. Ngrok provides a convenient way to expose the locally running TensorBoard on a temporary public URL. This URL enables the browser to communicate with the TensorBoard server residing on the Colab virtual machine.

Here are three specific code examples illustrating how to integrate TensorBoard effectively:

**Example 1: Basic TensorBoard Usage with Dummy Data**

This example demonstrates the fundamental steps of using TensorBoard. We begin by installing `tensorboard` if it's not already present and creating sample data. Then, we launch TensorBoard and establish the tunnel using `ngrok`.

```python
# Install TensorBoard if not already present
!pip install -q tensorboard

# Create dummy log data
import tensorflow as tf
import datetime
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
with file_writer.as_default():
    for i in range(100):
        tf.summary.scalar('my_metric', i*0.01, step=i)

# Start TensorBoard and ngrok tunnel
get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(log_dir)
)

!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip ngrok-stable-linux-amd64.zip

get_ipython().system_raw('./ngrok http 6006 &')

import time
time.sleep(5) # Wait for ngrok to start

! curl -s http://localhost:4040/api/tunnels | python3 -c \
    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
```

In this example:

*   `!pip install -q tensorboard` ensures that TensorBoard is installed silently if not already present.
*   TensorFlow's `tf.summary` API is used to generate sample data and save it to a logs directory. This mirrors how you might create logging during model training.
*   The key to making TensorBoard accessible is the `get_ipython().system_raw(...)` command. This starts the TensorBoard server on port 6006 and makes it available to all network interfaces (`0.0.0.0`).
*   We download the `ngrok` binary and establish a tunnel that forwards traffic from the randomly assigned public `ngrok` URL to the port 6006 running within the Colab VM.
*   The final command queries the `ngrok` API to retrieve the generated public URL which you can then copy into your browser to view TensorBoard.

**Example 2: TensorBoard with Keras Model Training**

This example demonstrates how to integrate TensorBoard during the training of a Keras neural network. This is perhaps a more realistic use case than simple, dummy data logging.

```python
import tensorflow as tf
from tensorflow import keras
import datetime

# Define a simple Keras model
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test  = x_test.reshape(10000, 784).astype("float32") / 255

# Define TensorBoard callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model
model.fit(x_train, y_train,
          epochs=5,
          callbacks=[tensorboard_callback],
          verbose=0)

# Start TensorBoard and ngrok tunnel
get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(log_dir)
)

!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip ngrok-stable-linux-amd64.zip

get_ipython().system_raw('./ngrok http 6006 &')

import time
time.sleep(5) # Wait for ngrok to start

! curl -s http://localhost:4040/api/tunnels | python3 -c \
    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
```

Key aspects in this example:

*   We employ the `keras.callbacks.TensorBoard` callback, which automatically logs relevant metrics and graph data during model training.
*   The model itself is a simple multi-layer perceptron trained on the MNIST dataset.
*   The rest of the process (starting TensorBoard and ngrok) is identical to the first example, but with the new log directory generated by the Keras callback. This allows for real-time metric analysis of a training run.

**Example 3: Using a Specific Port and Avoiding Re-downloading ngrok**

This third example builds upon the previous, adding small improvements often desired in repeated Colab sessions. This example illustrates how to reuse the same port for ngrok and avoids repeatedly downloading it.

```python
import tensorflow as tf
from tensorflow import keras
import datetime

# Define a simple Keras model (same as example 2)
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Load dataset (same as example 2)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test  = x_test.reshape(10000, 784).astype("float32") / 255

# Define TensorBoard callback (same as example 2)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model (same as example 2)
model.fit(x_train, y_train,
          epochs=5,
          callbacks=[tensorboard_callback],
          verbose=0)

# Check if ngrok is already downloaded, and if not, download it
import os
if not os.path.exists("ngrok"):
    !wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
    !unzip ngrok-stable-linux-amd64.zip

# Start TensorBoard and ngrok tunnel, forcing port 6006.
get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(log_dir)
)

get_ipython().system_raw('./ngrok http --log=stdout 6006 &')

import time
time.sleep(5) # Wait for ngrok to start

! curl -s http://localhost:4040/api/tunnels | python3 -c \
    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
```

The main improvements in this example:

*   We added an initial check for the `ngrok` executable. If it does not exist, the zip archive is downloaded and extracted. This avoids unnecessary downloads in subsequent code cell executions within the same Colab notebook.
*   We are more explicit about using port 6006, for both `tensorboard` and `ngrok`, which keeps consistency between different runs and helps with troubleshooting.

For anyone seeking further depth into the workings of TensorBoard and its interaction with TensorFlow, the official TensorFlow documentation provides detailed explanations of the API, including summaries and callbacks. Exploring the TensorBoard sections there offers a comprehensive understanding of underlying mechanisms. Additionally, documentation on Google Colab's shell usage and execution environments is valuable for understanding how the Python code interacts with the underlying virtual machine. Ngrok's documentation provides more detail on tunneling and other features useful in similar situations. These resources, when used together, offer an in-depth comprehension of how to integrate TensorBoard effectively in a Google Colab setting.
