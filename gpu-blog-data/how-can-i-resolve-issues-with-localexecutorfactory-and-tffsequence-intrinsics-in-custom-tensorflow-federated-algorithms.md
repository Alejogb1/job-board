---
title: "How can I resolve issues with `local_executor_factory` and `tff.sequence_*` intrinsics in custom TensorFlow Federated algorithms?"
date: "2025-01-26"
id: "how-can-i-resolve-issues-with-localexecutorfactory-and-tffsequence-intrinsics-in-custom-tensorflow-federated-algorithms"
---

In my experience developing federated learning systems, a recurring source of frustration involves the subtle interplay between TensorFlow Federated’s (TFF) `local_executor_factory` and the `tff.sequence_*` intrinsics. The core problem arises because `local_executor_factory` by default uses a graph-based execution strategy, and sequence intrinsics (like `tff.sequence_map`, `tff.sequence_sum`, etc.) operate on data represented as sequences. These sequences often are not trivially compatible with a purely graph-centric execution, which leads to errors regarding tensor shapes and data types. Let's examine the reasons behind this and how to address them effectively.

The heart of the issue lies in TFF's execution model.  `local_executor_factory` creates execution contexts that primarily expect single, concrete values, not potentially variable-length sequences.  When you invoke a computation that uses `tff.sequence_*` with this executor, the computation attempts to operate on a sequence that isn't directly representable within the static TensorFlow graph that `local_executor_factory` relies on. You see errors, such as mismatches in data types when the graph expects a single tensor but receives a list, or shape errors arising from the inability to unroll a sequence in a statically defined graph. This often manifests as cryptic stack traces including "ValueError: Cannot convert a symbolic Tensor to a numpy array" or shape mismatches during graph construction.

To clarify, TFF represents sequences as a nested structure of tensors. Internally, these sequences are processed by TFF's core framework, which must efficiently handle different sequence lengths without graph recompilation. A locally constructed graph, however, is not optimized for these dynamic changes. The `local_executor_factory` doesn't intrinsically understand how to translate the operations of `tff.sequence_*` into purely graph operations, even if the data is available within the executor context.

The solution involves shifting towards a more sequence-aware execution environment, or, as a fallback, carefully preparing data for graph-based computations.  The `tff.simulation.build_inprocess_environment` is a good choice because it directly executes operations rather than converting them into graphs. This environment is tailored for handling sequences effectively, thus bypassing the inherent limitations of graph-centric executors for sequence processing. It’s also crucial to make sure that if using a local executor, sequence objects are converted to tensors suitable for the computation graph. I will outline different ways to resolve those issues and provide examples below.

**Code Example 1: Transitioning to an In-Process Environment**

This example highlights a scenario where a `tff.sequence_sum` intrinsic leads to a failure under `local_executor_factory` and demonstrates a successful switch to an in-process environment.

```python
import tensorflow as tf
import tensorflow_federated as tff

# Define a function that uses sequence operations
@tff.tf_computation(tff.SequenceType(tf.float32))
def sum_sequence_elements(sequence):
  return tff.sequence_sum(sequence)


# Define a computation that uses the function above.
@tff.federated_computation(tff.type_at_clients(tff.SequenceType(tf.float32)))
def compute_federated_sum(federated_sequence):
    return tff.federated_map(sum_sequence_elements, federated_sequence)


# Simulate some input data.
example_sequence = [[1.0, 2.0], [3.0, 4.0, 5.0]]
example_federated_data = [example_sequence, example_sequence]


# Fails with local_executor_factory
try:
  local_executor = tff.simulation.local_executor_factory()
  result = local_executor.create_computation(compute_federated_sum)(example_federated_data)
  print("Result with local_executor_factory:", result)
except Exception as e:
    print("Error with local_executor_factory:", e)


# Works with in-process environment
inprocess_environment = tff.simulation.build_inprocess_environment()
result_inprocess = inprocess_environment.create_computation(compute_federated_sum)(example_federated_data)
print("Result with inprocess environment:", result_inprocess)

```

The first part of this script shows how `local_executor_factory` will fail, with the underlying cause being the handling of sequences. This error arises because the `tff.sequence_sum` operation, under a graph-based execution, has issues interpreting variable-length sequence data as it constructs the TensorFlow graph. Switching to `tff.simulation.build_inprocess_environment` resolves the error since sequences can be directly processed. This difference illustrates the core point: the execution environment is critical when sequences are involved.

**Code Example 2: Explicit Tensor Conversion within `tf.function`**

In certain cases where you must use `local_executor_factory`, it is imperative to work with the constraints imposed by graph execution. If `tff.sequence_*` are nested within a larger function that can be converted to a graph, you should convert the sequence into a tensor. The example below demonstrates one way to accomplish this.

```python
import tensorflow as tf
import tensorflow_federated as tff

# Define a function to convert a sequence to a tensor with a specific padded length.
@tf.function
def sequence_to_tensor(sequence, padded_length):
    # Pad the sequence to ensure a uniform length
    tensor_representation = tf.keras.preprocessing.sequence.pad_sequences(
        sequence, padding='post', dtype=tf.float32, maxlen = padded_length
    )
    return tensor_representation

# Define a function that uses sequence operations
@tff.tf_computation(tff.SequenceType(tf.float32), tf.int32)
def process_sequence_with_padding(sequence, max_len):
    tensor_sequence = sequence_to_tensor(sequence, max_len)
    return tf.reduce_sum(tensor_sequence)


@tff.federated_computation(tff.type_at_clients(tff.SequenceType(tf.float32)))
def compute_federated_sum_padded(federated_sequence):
    max_length_local = tff.federated_map(lambda x : tf.shape(x)[0], tff.federated_map(lambda y: sequence_to_tensor(y,10), federated_sequence))
    max_length_global = tff.federated_max(max_length_local)
    return tff.federated_map(process_sequence_with_padding, (federated_sequence, max_length_global))

# Simulate some input data.
example_sequence = [[1.0, 2.0], [3.0, 4.0, 5.0]]
example_federated_data = [example_sequence, example_sequence]

# Works with local_executor_factory, with pre-padding.
local_executor = tff.simulation.local_executor_factory()
result_local = local_executor.create_computation(compute_federated_sum_padded)(example_federated_data)
print("Result with local_executor_factory:", result_local)
```

Here, we convert the sequence to a tensor using TensorFlow's `pad_sequences`. The key idea is that we are now working with tensors of known shape inside the graph. The padding here may introduce extra computation, but can be mitigated by using appropriate padding strategies. This ensures that the graph is constructed in a way that's compatible with the `local_executor_factory`. In a full federated learning setup, you would typically process the data into tensors of the same size before passing the data to a TFF computation. The key lesson here is: if using `local_executor_factory`, be explicit about transforming the sequence into something that can be efficiently worked on in graph execution contexts.

**Code Example 3: Careful Nesting of Functions**

This final example demonstrates how the combination of graph-based operations with sequence processing needs care in order to work with `local_executor_factory`. You need to make sure that the sequence operations are not within a function that's graph-compiled (e.g. via `@tf.function`), otherwise you will introduce similar issues as shown in the first example.

```python
import tensorflow as tf
import tensorflow_federated as tff

# Function defined using a tf.function, which causes sequence operations to raise errors within the TFF graph.
@tf.function
def graph_func_with_sequence(sequence):
  return tff.sequence_sum(sequence)


# This function will error since it has graph operations as part of its construction.
@tff.tf_computation(tff.SequenceType(tf.float32))
def incorrect_sequence_processing(sequence):
    return graph_func_with_sequence(sequence)


# Proper use of sequence in a tf function,
@tff.tf_computation(tff.SequenceType(tf.float32))
def correct_sequence_processing(sequence):
  return tff.sequence_sum(sequence)



# Compute using the correct sequence handling.
@tff.federated_computation(tff.type_at_clients(tff.SequenceType(tf.float32)))
def compute_federated_sum_correct(federated_sequence):
  return tff.federated_map(correct_sequence_processing, federated_sequence)

# Compute using incorrect sequence handling.
@tff.federated_computation(tff.type_at_clients(tff.SequenceType(tf.float32)))
def compute_federated_sum_incorrect(federated_sequence):
  return tff.federated_map(incorrect_sequence_processing, federated_sequence)

# Simulate some input data.
example_sequence = [[1.0, 2.0], [3.0, 4.0, 5.0]]
example_federated_data = [example_sequence, example_sequence]

# Fails with local_executor_factory because of graph function
try:
  local_executor = tff.simulation.local_executor_factory()
  result_incorrect = local_executor.create_computation(compute_federated_sum_incorrect)(example_federated_data)
  print("Incorrect result with local_executor_factory:", result_incorrect)
except Exception as e:
    print("Error with local_executor_factory, incorrect:", e)

# Works with local_executor_factory given that it's not nested in graph function.
local_executor = tff.simulation.local_executor_factory()
result_correct = local_executor.create_computation(compute_federated_sum_correct)(example_federated_data)
print("Correct result with local_executor_factory:", result_correct)
```
In this case, `graph_func_with_sequence` is decorated with `@tf.function`. Therefore, any sequence ops are translated to graph operations. This means that `tff.sequence_sum` needs a tensor and not a `tff.SequenceType`. Therefore, the computation will fail under the default graph context of `local_executor_factory`. The `correct_sequence_processing` does not contain a graph-function, and therefore, `tff.sequence_sum` is correctly executed. This case shows that you need to be careful with when and where the graph functions and sequence functions are defined.

**Resource Recommendations**

For a deeper understanding of TFF execution, I would suggest a thorough review of the official TFF documentation, particularly the sections related to execution models, computation types, and the different types of executors available. Pay close attention to the distinction between graph-based and non-graph-based execution as this has the most impact on how the sequence data is handled. The API documentation on `tff.sequence_*` intrinsics, including their type signatures, is also very beneficial. Furthermore, explore tutorials and examples that demonstrate handling sequences in TFF computations, and use the debugging capabilities provided by the framework.

In summary, these issues are not a flaw in TFF, but rather consequences of its flexibility and the varied execution contexts available. It's crucial to understand how your chosen executor interprets data, especially sequence data, and to adjust your computational logic to meet the specific execution environment. When in doubt, transition to `tff.simulation.build_inprocess_environment`, or be very clear with how to map your sequences to tensors before using the local executor.
