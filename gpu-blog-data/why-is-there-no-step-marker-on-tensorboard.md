---
title: "Why is there no step marker on TensorBoard?"
date: "2025-01-30"
id: "why-is-there-no-step-marker-on-tensorboard"
---
The absence of a universally displayed, explicit step marker on TensorBoard's primary scalar plots stems from its inherent design principles centered around visualizing *time-series data*, where a global step count is not always the most relevant or accurate representation of progress. My experience working with varied distributed training pipelines has reinforced this perspective; the notion of a single, synchronized ‘step’ can break down significantly when operating across multiple GPUs or machines, especially when employing asynchronous operations.

TensorBoard's core functionality revolves around visualizing how metrics change *over time*. It receives data points, often scalar values, paired with a time-based identifier—the 'wall time,' or epoch time, recorded at the moment the data point is logged. This time-based system offers several crucial advantages, particularly in complex training scenarios:

Firstly, it naturally accommodates asynchronous logging. Different parts of a training graph or different parallel processes may log data at different speeds. Using wall time allows TensorBoard to consistently stitch together disparate data streams and correctly sequence the progression of logged values. Introducing an explicit 'step' counter would necessitate a synchronization mechanism across all logging sources which is often impractical or, worse, introduces artificial bottlenecks.

Secondly, many metrics aren't intrinsically tied to a specific training 'step.' Validation metrics, for instance, are often computed periodically after a batch or several batches of training, often at irregular intervals. Representing validation scores directly against the number of gradient steps wouldn’t reflect the real progression of the training cycle. Similarly, for metrics gathered at the end of each epoch, which may contain thousands of steps, plotting against individual steps would render the plot largely unreadable.

Thirdly, wall time provides a measure of real-world training duration. This is often more pertinent than the number of steps when comparing training runs across different hardware or network configurations. A training run might proceed much faster on a more powerful system; using only step numbers for comparison could potentially lead to misleading interpretations of the experiment's overall performance.

While TensorBoard does not natively display a global step counter on the scalar charts, it does allow for a 'step' to be used as part of a tag name or as a separate scalar within a logged summary. The user can then tailor visualization using custom tag names, if needed. This allows for a great degree of flexibility, enabling users to display step-related information, when desired, but not at the cost of flexibility and generality.

Let's examine some example scenarios with accompanying code and explanations:

**Example 1: Logging a simple loss curve with explicit step count.**

This is the most straightforward case where a global step is logged along with loss values, although TensorBoard still treats 'step' as a scalar value and not an absolute axis.

```python
import tensorflow as tf
import time

logdir = "logs/example1"
writer = tf.summary.create_file_writer(logdir)

steps = 100
loss = 1.0 # initial loss
with writer.as_default():
    for step in range(steps):
      loss = loss * 0.99 # simulate loss decrease
      tf.summary.scalar("loss", loss, step=step)
      time.sleep(0.01)
      tf.summary.scalar("training_step", step, step=step) # Log the step count itself

print("Run: tensorboard --logdir logs/example1")
```

*Explanation*:
This code creates a basic training loop where both a simulated `loss` and the current `step` number are logged. We pass the explicit `step` parameter into `tf.summary.scalar()` to indicate the associated step for each log. Although this data is present in TensorBoard, the plot will display it against wall time. To properly examine the loss vs. the training step on the same axis, one must navigate to the 'training_step' scalar plot. Furthermore, note that this implementation has artificially forced time to be proportional to steps with a `time.sleep()` call. In real-world training, this won’t be the case.

**Example 2: Logging both training loss and validation accuracy at different frequencies.**

This showcases the benefit of wall time-based plotting. Validation accuracy is often calculated much less frequently than the training loss, and the step number of validation does not directly align with the training steps.

```python
import tensorflow as tf
import time
import random

logdir = "logs/example2"
writer = tf.summary.create_file_writer(logdir)

steps = 100
loss = 1.0
with writer.as_default():
    for step in range(steps):
        loss = loss * 0.99 + random.random() * 0.01 # Simulate loss decrease, add noise
        tf.summary.scalar("training_loss", loss, step=step)
        time.sleep(0.01)

        if step % 10 == 0:  # Simulate validation every 10 steps
            accuracy = random.random() * 0.6 + 0.4 # simulate validation accuracy
            tf.summary.scalar("validation_accuracy", accuracy, step=step)
            print(f"Validation step:{step} Accuracy:{accuracy:.4f}")

print("Run: tensorboard --logdir logs/example2")
```
*Explanation*:
Here, training loss is logged every step, whereas validation accuracy is logged only every tenth step. If TensorBoard plotted using a global step axis, the validation accuracies would be sparse and visually misaligned with training. With wall-time based plots, the plot will show accurately the temporal relationship between the training loss and validation accuracy even though they are not logged at the same cadence.

**Example 3: Asynchronous Logging using Multiple Writers**

Demonstrates a more realistic scenario with parallel writers and highlights the crucial role of timestamps for correct plot ordering.

```python
import tensorflow as tf
import time
import threading

logdir = "logs/example3"
writers = [tf.summary.create_file_writer(f"{logdir}/writer_{i}") for i in range(2)] # Create two writers

steps = 100

def log_data(writer, writer_num, start_delay):
    time.sleep(start_delay) # Simulate delay in data logging
    with writer.as_default():
      for step in range(steps):
        loss = 1.0 - (step / steps) * 0.9 # simulate loss reduction
        tf.summary.scalar(f"loss_writer_{writer_num}", loss, step=step)
        time.sleep(random.random() * 0.02) # Simulate uneven time gaps
        print(f"Writer:{writer_num} step:{step}")


threads = [threading.Thread(target=log_data, args=(writers[i], i, i*0.5)) for i in range(2)] # Start logging threads with differing delays
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()

print("Run: tensorboard --logdir logs/example3")
```

*Explanation*:
In this example, two threads write to separate log files, with each simulating its training metrics at varying rates and with different starting delays.  If plots were based strictly on a step count, one would need a shared counter, which would complicate implementation. TensorBoard’s wall time plotting automatically aligns the data in the correct time order even with different starting times and logging cadences per thread.

**Recommendations:**

To thoroughly understand TensorBoard, I advise exploring the official TensorFlow documentation on `tf.summary`. Additionally, carefully examine different types of summaries and how they interact. Experiment with logging data at various frequencies with differing delays, then observe how TensorBoard handles the time-series visualization. Finally, delve into practical guides focusing on interpreting training progress using TensorBoard. This hands-on approach will provide far more insights than relying solely on step counters. Analyzing the underlying data structures written to the log files will further deepen one's understanding. This allows for the direct observation of wall time and how TensorBoard uses it to construct visual representations.
