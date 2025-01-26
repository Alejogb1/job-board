---
title: "How can TensorFlow model files be accessed in C?"
date: "2025-01-26"
id: "how-can-tensorflow-model-files-be-accessed-in-c"
---

TensorFlow, being primarily a Python-centric library, does not natively offer a straightforward API for accessing model files (.pb or SavedModel directories) directly within C applications. Bridging this gap necessitates using the TensorFlow C API, a significantly lower-level interface compared to its Python counterpart. My experience building custom inference engines for embedded systems required deep familiarity with this approach, revealing both the power and the associated complexities. The core challenge lies in translating the high-level operations performed by Python's TensorFlow into equivalent C code, specifically related to loading and executing graph definitions.

Fundamentally, accessing a TensorFlow model in C requires the following steps: First, you load the serialized graph definition from the model file using the appropriate TensorFlow C API functions. This involves parsing the `.pb` file or retrieving the `MetaGraphDef` from a SavedModel directory. Next, a `TF_Session` needs to be created to provide a context for executing the graph. Input tensors must be populated, and the inference operation must be invoked using `TF_SessionRun`. Finally, output tensors are extracted. These operations are inherently manual compared to Python and necessitate careful memory management to prevent leaks.

One crucial aspect is distinguishing between the frozen graph format (.pb) and the SavedModel format. A frozen graph encapsulates the graph definition and the variable values into a single file, suitable for simpler inference tasks. SavedModel, on the other hand, is a more structured directory format, suitable for models that require separate checkpoints, assets and potentially several graphs. The loading procedure differs slightly between these formats, but the core concepts remain the same. For frozen graphs, you use `TF_NewGraph`, `TF_ImportGraphDefOptions`, and `TF_GraphImportGraphDef` to load the graph from a `.pb` file. For SavedModel, `TF_LoadSessionOptions`, `TF_LoadSessionFromSavedModel`, and `TF_GetSessionGraph` accomplish this, also retrieving the graph and session in a single operation.

The C API mandates explicit tensor manipulation. Creating input tensors often involves preparing a `TF_Tensor` struct, specifying its datatype, dimensions, and allocating memory for the actual data. Similarly, extracting output tensors requires deallocating memory after use. Unlike Python's automatic memory management, care must be taken to deallocate memory for graphs, sessions, and tensors to avoid resource exhaustion, particularly in resource-constrained environments like embedded devices. This manual process constitutes the primary learning curve when transitioning from Python to the C API.

Now, let's delve into concrete examples. I will assume we're using a compiled TensorFlow C library, appropriately configured in the include and link paths of your development environment. The first example focuses on loading a `.pb` frozen graph, setting up an input, and running inference:

```c
#include "tensorflow/c/c_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    // 1. Load the frozen graph
    TF_Graph* graph = TF_NewGraph();
    TF_Status* status = TF_NewStatus();
    const char* model_path = "path/to/my_frozen_graph.pb";
    FILE* file = fopen(model_path, "rb");
    if (!file) {
        fprintf(stderr, "Error opening model file.\n");
        return 1;
    }
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    void* model_data = malloc(file_size);
    if (fread(model_data, 1, file_size, file) != file_size) {
        fprintf(stderr, "Error reading model file.\n");
        fclose(file);
        free(model_data);
        return 1;
    }
    fclose(file);

    TF_Buffer* graph_def = TF_NewBuffer();
    graph_def->data = model_data;
    graph_def->length = file_size;

    TF_ImportGraphDefOptions* import_opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(graph, graph_def, import_opts, status);
    TF_DeleteImportGraphDefOptions(import_opts);
    TF_DeleteBuffer(graph_def);
    free(model_data); // Free data after importing
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Error importing graph: %s\n", TF_Message(status));
        TF_DeleteStatus(status);
        TF_DeleteGraph(graph);
        return 1;
    }

    // 2. Create a Session
    TF_SessionOptions* sess_opts = TF_NewSessionOptions();
    TF_Session* session = TF_NewSession(graph, sess_opts, status);
    TF_DeleteSessionOptions(sess_opts);
    if(TF_GetCode(status) != TF_OK){
      fprintf(stderr, "Error creating session: %s\n", TF_Message(status));
        TF_DeleteStatus(status);
        TF_DeleteGraph(graph);
        return 1;
    }

    // 3. Prepare input tensor (example: assuming single float input)
    float input_data = 0.5f;
    int64_t dims[] = {1};
    TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT, dims, 1, &input_data, sizeof(float));


    // 4. Run inference
    TF_Output input_op = {TF_GraphOperationByName(graph, "input_node_name"), 0}; // Replace with actual input name
    TF_Output output_op = {TF_GraphOperationByName(graph, "output_node_name"), 0}; // Replace with actual output name
    TF_Tensor* output_tensor = NULL;
    TF_SessionRun(session, NULL, &input_op, &input_tensor, 1, &output_op, &output_tensor, 1, NULL, 0, NULL, status);

    if (TF_GetCode(status) != TF_OK) {
      fprintf(stderr, "Error running session: %s\n", TF_Message(status));
      TF_DeleteStatus(status);
      TF_DeleteSession(session, status);
      TF_DeleteGraph(graph);
      TF_DeleteTensor(input_tensor);
      return 1;
     }


    // 5. Extract output and cleanup (assuming single float output)
    float output_value = *((float*)TF_TensorData(output_tensor));
    printf("Output: %f\n", output_value);
    TF_DeleteTensor(output_tensor);
    TF_DeleteTensor(input_tensor);
    TF_DeleteSession(session, status);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);
    return 0;
}
```

This example demonstrates the basic process for a `.pb` model. Note that the input and output node names ("input\_node\_name" and "output\_node\_name") need to be replaced with the actual names present in the graph definition. The manual file reading, memory allocation, and error handling are crucial.

My second example focuses on loading a SavedModel:

```c
#include "tensorflow/c/c_api.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    // 1. Load the SavedModel
    TF_Status* status = TF_NewStatus();
    const char* saved_model_dir = "path/to/my_savedmodel_dir";
    TF_SessionOptions* session_options = TF_NewSessionOptions();
    TF_Buffer* tags = TF_NewBufferFromString("serve"); // or relevant tag
    TF_Session* session = TF_LoadSessionFromSavedModel(session_options, NULL, saved_model_dir, tags, NULL, status);
    TF_DeleteSessionOptions(session_options);
    TF_DeleteBuffer(tags);

      if(TF_GetCode(status) != TF_OK){
        fprintf(stderr, "Error loading SavedModel: %s\n", TF_Message(status));
        TF_DeleteStatus(status);
          return 1;
    }

    TF_Graph* graph = TF_GetSessionGraph(session);
     if(graph == NULL){
         fprintf(stderr, "Error getting graph from session.\n");
        TF_DeleteSession(session, status);
        TF_DeleteStatus(status);
         return 1;

     }
    // 2. Prepare input tensor (example: assuming single float input)
    float input_data = 0.75f;
    int64_t dims[] = {1};
    TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT, dims, 1, &input_data, sizeof(float));


    // 3. Run inference (same as frozen graph example)
    TF_Output input_op = {TF_GraphOperationByName(graph, "serving_default_input_node"), 0};  // Replace with actual input name
    TF_Output output_op = {TF_GraphOperationByName(graph, "serving_default_output_node"), 0}; // Replace with actual output name
    TF_Tensor* output_tensor = NULL;
    TF_SessionRun(session, NULL, &input_op, &input_tensor, 1, &output_op, &output_tensor, 1, NULL, 0, NULL, status);


    if (TF_GetCode(status) != TF_OK) {
      fprintf(stderr, "Error running session: %s\n", TF_Message(status));
      TF_DeleteStatus(status);
      TF_DeleteSession(session, status);
        TF_DeleteTensor(input_tensor);
        return 1;
    }

    // 4. Extract output and cleanup
     float output_value = *((float*)TF_TensorData(output_tensor));
     printf("Output: %f\n", output_value);
    TF_DeleteTensor(output_tensor);
    TF_DeleteTensor(input_tensor);
    TF_DeleteSession(session, status);
    TF_DeleteStatus(status);
    return 0;
}
```

This example utilizes `TF_LoadSessionFromSavedModel` to load a SavedModel. The `tags` parameter is particularly important, specifying which model variant (if present). Again, remember to use the correct input and output node names, often found under the `serving_default` signatures.

For the third example, imagine a model with multiple inputs and outputs:

```c
#include "tensorflow/c/c_api.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
 // (Load graph and session: either .pb or SavedModel, as in previous examples - not shown for brevity)
    TF_Status* status = TF_NewStatus();

    // Assuming session and graph loaded previously.
    TF_Graph* graph = NULL;
    TF_Session* session = NULL;

   // ... Load the graph and session here as in previous examples ...
    // 1. Prepare multiple input tensors
     float input1_data[2] = {1.0f, 2.0f};
     int64_t input1_dims[] = {1, 2};
     TF_Tensor* input1_tensor = TF_NewTensor(TF_FLOAT, input1_dims, 2, input1_data, 2 * sizeof(float));

     int32_t input2_data = 10;
     int64_t input2_dims[] = {1};
     TF_Tensor* input2_tensor = TF_NewTensor(TF_INT32, input2_dims, 1, &input2_data, sizeof(int32_t));


    TF_Output input_ops[] = {
       {TF_GraphOperationByName(graph, "input_1_node"), 0},  // Input 1
       {TF_GraphOperationByName(graph, "input_2_node"), 0}  // Input 2
       };

    TF_Tensor* input_tensors[] = {input1_tensor, input2_tensor};

    TF_Output output_ops[] = {
       {TF_GraphOperationByName(graph, "output_1_node"), 0}, // Output 1
       {TF_GraphOperationByName(graph, "output_2_node"), 0}  // Output 2
    };

    TF_Tensor* output_tensors[2] = {NULL, NULL};


    // 2. Run inference
    TF_SessionRun(session, NULL, input_ops, input_tensors, 2, output_ops, output_tensors, 2, NULL, 0, NULL, status);
    if (TF_GetCode(status) != TF_OK) {
      fprintf(stderr, "Error running session: %s\n", TF_Message(status));
        TF_DeleteStatus(status);
         TF_DeleteSession(session, status);
          TF_DeleteTensor(input1_tensor);
         TF_DeleteTensor(input2_tensor);
        return 1;
    }


    // 3. Extract and cleanup multiple outputs

     float output1_value = *((float*)TF_TensorData(output_tensors[0]));
     printf("Output 1: %f\n", output1_value);

    int32_t output2_value = *((int32_t*)TF_TensorData(output_tensors[1]));
      printf("Output 2: %d\n", output2_value);


    TF_DeleteTensor(output_tensors[0]);
    TF_DeleteTensor(output_tensors[1]);
    TF_DeleteTensor(input1_tensor);
     TF_DeleteTensor(input2_tensor);
    TF_DeleteSession(session, status);
    TF_DeleteStatus(status);
    return 0;
}
```
This illustrates how to manage multiple inputs and outputs. The key is setting up input/output arrays with the correct `TF_Output` and `TF_Tensor` structs and passing those arrays to `TF_SessionRun`.

For further study, the official TensorFlow C API documentation is essential. Additionally, exploring the TensorFlow source code, particularly the relevant C++ files, can provide deep insights. For practical examples, the TensorFlow repository includes numerous C API test cases that serve as excellent learning resources. Books covering TensorFlow internals are also advantageous, but focus on practical examples rather than the complexities of the C API. Finally, remember that debugging C code requires a solid foundation in C programming.
