---
title: "How can I extract variables from a SavedModelBundle in C++?"
date: "2025-01-30"
id: "how-can-i-extract-variables-from-a-savedmodelbundle"
---
A fundamental challenge when working with TensorFlow SavedModelBundles in C++ lies in accessing the underlying variable tensors after loading the model. Unlike the Python API which provides more direct access via name lookup and symbolic connections, the C++ API requires a more structured approach involving graph traversal and tensor handles. Through several projects involving TensorFlow integration within embedded systems, I've developed a dependable method for variable extraction, primarily using the `GetVariables` function and associated functionalities within `tensorflow::SavedModelBundle`.

The core issue stems from the fact that a SavedModelBundle, after being loaded, presents itself as an opaque collection of computation graphs rather than a readily accessible data store of tensors. When you load a model using `tensorflow::LoadSavedModelBundle`, the returned `tensorflow::SavedModelBundle` object contains a `MetaGraphDef`, holding various aspects of the saved model, including the computation graph, signatures, and variables. These variables, however, aren't exposed directly as named entities. Instead, you interact with them indirectly using specific TensorFlow APIs designed to operate within the graph's structure. The `GetVariables` method, found within `tensorflow::SavedModelBundle`, offers a controlled way to retrieve a collection of handles to these variables, which are represented as `tensorflow::Tensor` objects.

The workflow generally involves three key steps: loading the bundle, obtaining variable handles, and accessing the variable data. The `tensorflow::SavedModelBundle` class offers `GetVariables` method that, when provided with a valid session, returns a `std::vector<tensorflow::Tensor>`. This vector represents an aggregation of all the variables within the loaded model. Each `tensorflow::Tensor` object, in turn, provides access to its data via methods like `flat<T>`, which flattens the tensor data into a continuous block of memory.

I typically use the following methodology, often encapsulated in a helper function, to facilitate this process. This method avoids direct manipulation of raw memory and adheres to the TensorFlow API.

**Code Example 1: Basic Variable Extraction**

```cpp
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/tensor.h"
#include <iostream>
#include <vector>

std::vector<tensorflow::Tensor> extractVariables(const std::string& model_path) {
    tensorflow::SavedModelBundle bundle;
    tensorflow::SessionOptions session_options;
    tensorflow::RunOptions run_options;
    tensorflow::Status status = tensorflow::LoadSavedModelBundle(session_options, run_options, model_path, {tensorflow::kSavedModelTagServe}, &bundle);

    if (!status.ok()) {
      std::cerr << "Error loading model: " << status.ToString() << std::endl;
      return {};
    }
    
    std::vector<tensorflow::Tensor> variables;
    status = bundle.GetVariables(&variables);
    
    if (!status.ok())
    {
       std::cerr << "Error extracting variables: " << status.ToString() << std::endl;
       return {};
    }

    return variables;
}

int main() {
    std::string model_path = "./path/to/your/saved_model"; // Replace with the actual path

    std::vector<tensorflow::Tensor> variables = extractVariables(model_path);

    if (variables.empty()) {
       std::cerr << "No variables extracted." << std::endl;
       return 1;
    }

    for (const auto& variable : variables) {
      std::cout << "Variable shape: " << variable.shape().DebugString() << std::endl;
      // Further processing can be performed here (see examples 2 & 3)
    }
    return 0;
}
```

**Commentary:**

This example demonstrates the basic structure. The `extractVariables` function encapsulates the loading procedure of the SavedModelBundle and uses the `GetVariables` function to obtain all variables as a vector of `tensorflow::Tensor` objects. If the load fails or variable retrieval fails, it logs an error and returns an empty vector, avoiding access to null resources. The `main` function shows how to call this function and iterate through the obtained variables, printing their shapes, which is often a first step in debugging. This is a barebone implementation that will only give you the tensors; the next examples will focus on getting to the data.

The example above retrieves all variables without distinguishing between them by name. Often, it is necessary to identify specific variables by name. This requires traversing the protobuf structures returned by a few methods before using the `GetVariables` method.

**Code Example 2: Named Variable Extraction**

```cpp
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

tensorflow::Tensor extractNamedVariable(const std::string& model_path, const std::string& variable_name) {
    tensorflow::SavedModelBundle bundle;
    tensorflow::SessionOptions session_options;
    tensorflow::RunOptions run_options;
    tensorflow::Status status = tensorflow::LoadSavedModelBundle(session_options, run_options, model_path, {tensorflow::kSavedModelTagServe}, &bundle);

    if (!status.ok()) {
      std::cerr << "Error loading model: " << status.ToString() << std::endl;
       return tensorflow::Tensor(); // Return empty tensor
    }

    const tensorflow::MetaGraphDef &meta_graph_def = bundle.GetMetaGraphDef();
    std::vector<std::string> variable_names;

    // Extract variable names from the metagraph
    for (const auto& node : meta_graph_def.graph_def().node()) {
       if (node.op() == "VariableV2" || node.op() == "VarHandleOp") {
            variable_names.push_back(node.name());
       }
    }

    auto it = std::find(variable_names.begin(), variable_names.end(), variable_name);
    if (it == variable_names.end()) {
       std::cerr << "Variable '" << variable_name << "' not found." << std::endl;
       return tensorflow::Tensor(); // Return empty tensor
    }

     std::vector<tensorflow::Tensor> variables;
     status = bundle.GetVariables(&variables);

     if (!status.ok()) {
         std::cerr << "Error getting variables: " << status.ToString() << std::endl;
         return tensorflow::Tensor();
     }

     for(const auto& variable : variables){
       std::string full_name = variable.tensor_name();
       size_t lastSlash = full_name.find_last_of('/');
       if (lastSlash != std::string::npos){
           full_name = full_name.substr(lastSlash +1);
       }
       if (full_name == variable_name){
         return variable;
       }
     }

    std::cerr << "Variable '" << variable_name << "' not found in extracted tensors." << std::endl;
    return tensorflow::Tensor();
}


int main() {
    std::string model_path = "./path/to/your/saved_model";
    std::string target_variable_name = "dense/kernel"; // Replace with your target variable name

    tensorflow::Tensor variable = extractNamedVariable(model_path, target_variable_name);
    if (variable.NumElements() > 0) {
        std::cout << "Shape of variable '" << target_variable_name << "': " << variable.shape().DebugString() << std::endl;

        // Access variable data using flat<T> (see Example 3).
    } else {
        std::cerr << "Error retrieving variable or variable is empty." << std::endl;
    }
   return 0;
}
```

**Commentary:**

This example expands upon the previous one. Before calling `GetVariables`, it iterates through the `MetaGraphDef`'s `GraphDef` to explicitly identify nodes representing TensorFlow variables and extracts their names. It then searches for the target variable name within this collection. Once the name is found in the graph def it iterates through the tensors returned by `GetVariables`, checks their names, and returns the corresponding tensor. The `tensor_name()` method returns the entire path of the tensor, which needs to be parsed to extract the name that is searched for in the `GraphDef`. If the variable is found, it is returned; otherwise, an empty tensor is returned. This prevents attempting operations on an invalid tensor.

Finally, to access the tensor's data after obtaining the tensor handle, a specialized template method of the `Tensor` class can be used: `flat<T>`. This method provides a convenient way to access the tensor’s underlying data block as a typed array.

**Code Example 3: Accessing Variable Data**

```cpp
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

tensorflow::Tensor extractNamedVariable(const std::string& model_path, const std::string& variable_name) {
   // (Same function as in Example 2)
    tensorflow::SavedModelBundle bundle;
    tensorflow::SessionOptions session_options;
    tensorflow::RunOptions run_options;
    tensorflow::Status status = tensorflow::LoadSavedModelBundle(session_options, run_options, model_path, {tensorflow::kSavedModelTagServe}, &bundle);

    if (!status.ok()) {
      std::cerr << "Error loading model: " << status.ToString() << std::endl;
       return tensorflow::Tensor(); // Return empty tensor
    }

    const tensorflow::MetaGraphDef &meta_graph_def = bundle.GetMetaGraphDef();
    std::vector<std::string> variable_names;

    // Extract variable names from the metagraph
    for (const auto& node : meta_graph_def.graph_def().node()) {
       if (node.op() == "VariableV2" || node.op() == "VarHandleOp") {
            variable_names.push_back(node.name());
       }
    }

    auto it = std::find(variable_names.begin(), variable_names.end(), variable_name);
    if (it == variable_names.end()) {
       std::cerr << "Variable '" << variable_name << "' not found." << std::endl;
       return tensorflow::Tensor(); // Return empty tensor
    }

     std::vector<tensorflow::Tensor> variables;
     status = bundle.GetVariables(&variables);

     if (!status.ok()) {
         std::cerr << "Error getting variables: " << status.ToString() << std::endl;
         return tensorflow::Tensor();
     }

     for(const auto& variable : variables){
       std::string full_name = variable.tensor_name();
       size_t lastSlash = full_name.find_last_of('/');
       if (lastSlash != std::string::npos){
           full_name = full_name.substr(lastSlash +1);
       }
       if (full_name == variable_name){
         return variable;
       }
     }

    std::cerr << "Variable '" << variable_name << "' not found in extracted tensors." << std::endl;
    return tensorflow::Tensor();
}


int main() {
    std::string model_path = "./path/to/your/saved_model";
    std::string target_variable_name = "dense/kernel"; // Replace with your target variable name

    tensorflow::Tensor variable = extractNamedVariable(model_path, target_variable_name);
    if (variable.NumElements() > 0) {
        std::cout << "Shape of variable '" << target_variable_name << "': " << variable.shape().DebugString() << std::endl;
        
        // Access the data (assuming float, adjust as needed):
        auto flat_tensor = variable.flat<float>();
        int num_elements = variable.NumElements();
        std::cout << "First 10 values of '" << target_variable_name << "': ";
        for (int i = 0; i < std::min(num_elements, 10); ++i) {
            std::cout << flat_tensor(i) << " ";
        }
         std::cout << std::endl;
        
    } else {
         std::cerr << "Error retrieving variable or variable is empty." << std::endl;
    }
    return 0;
}
```

**Commentary:**

This final example shows how to read data from the extracted tensor using `flat<T>`. Before calling the `flat<T>`, it is important to ensure that the variable contains data, otherwise an error can occur when accessing the elements. The example above assumes that the tensor contains data of type `float`. The program iterates through the first 10 elements (or up to the total number of elements if less than 10). Type conversion and handling of data types are left to specific user requirements, as TensorFlow supports various numeric and string types. The example above uses a float type but other types can be used like int or double using `<int>` or `<double>`.

For further exploration of SavedModelBundles and related APIs, the official TensorFlow C++ API documentation is the primary resource. In particular, the documentation for the `tensorflow::SavedModelBundle`, `tensorflow::Session`, and `tensorflow::Tensor` classes is particularly useful. The TensorFlow source code itself is also valuable for understanding the lower-level implementation details. I found the detailed explanation of the `MetaGraphDef` structure useful as well. Finally, the TensorFlow C++ examples provided in the official source tree also offer practical insights, though direct examples of variable extraction can be difficult to locate.
