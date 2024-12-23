---
title: "How can I obtain weights for each layer using the OpenVINO API?"
date: "2024-12-23"
id: "how-can-i-obtain-weights-for-each-layer-using-the-openvino-api"
---

Alright, let's talk about extracting layer weights using OpenVINO. It’s a frequent need, especially when you’re debugging, optimizing, or transferring knowledge between frameworks. I’ve personally faced this challenge numerous times, typically when trying to understand how specific parts of a model are behaving after quantization or other post-training transformations. It isn't always as straightforward as it initially seems.

The OpenVINO api, while powerful for inference, doesn’t offer a dedicated, direct function to simply grab all the weights from every layer in a nicely organized manner. You’ll find that the weights aren’t stored in a single convenient location but are instead part of the *compiled* model representation. This means you’ll need to navigate the model's internal structure, identify layers that contain weights, and then access the underlying memory buffers holding those values. Think of it as peeling back the layers of an onion, where each layer reveals more about the data and structure. It requires a good grasp of how OpenVINO represents a model's intermediate representation or IR.

The first step is always to load your model into an `ie_core::ExecutableNetwork`. This allows you to actually use the model and examine its internal components. Once loaded, we need to iterate through the network’s operations to locate those that hold weights – these are typically convolution, fully connected (dense), and embedding layers, among others. Each operation is an `ie_core::Operation`, which has its input and output tensors, as well as *attributes*. It's these *attributes* that we need to explore to find our weights. Within an `ie_core::Operation`, weights are often (but not always!) stored as constant input tensors that hold the numerical values.

Now, let’s get to some code. We'll use C++ because it's the primary interface to OpenVINO and because it provides lower-level access, which helps when dealing with memory buffers directly. I’ve streamlined the error handling for clarity, but in a real production environment you would want to handle those more robustly.

```c++
#include <iostream>
#include <ie_core.hpp>
#include <string>
#include <unordered_map>

using namespace InferenceEngine;

std::unordered_map<std::string, std::vector<float>> extract_weights(const ExecutableNetwork& network) {
    std::unordered_map<std::string, std::vector<float>> weights_map;

    auto network_ops = network.GetExecGraphInfo().getFunction()->get_ops();

    for (const auto& op : network_ops) {
        std::string op_name = op->get_friendly_name();
        std::string op_type = op->get_type_name();

        if (op_type == "Convolution" || op_type == "FullyConnected") {

           for (size_t i = 0; i < op->inputs().size(); ++i) {
               auto input = op->inputs()[i];
               if (input.get_element_type() == element::f32 &&
                   input.get_shape().size() > 1) { // Check for data types, shapes to identify weights
                   
                   auto const_node = std::dynamic_pointer_cast<ngraph::op::Constant>(input.get_source_output().get_node_shared_ptr());

                   if (const_node)
                   {
                        auto tensor = const_node->get_vector<float>();
                        weights_map[op_name + "_weight_" + std::to_string(i)] = tensor;
                   }
               }
            }
        }
        else if (op_type == "Embedding") {
                for (size_t i = 0; i < op->inputs().size(); ++i) {
                    auto input = op->inputs()[i];
                    if (input.get_element_type() == element::f32 && input.get_shape().size() > 1) {
                        auto const_node = std::dynamic_pointer_cast<ngraph::op::Constant>(input.get_source_output().get_node_shared_ptr());
                        if (const_node)
                        {
                             auto tensor = const_node->get_vector<float>();
                             weights_map[op_name + "_weight_" + std::to_string(i)] = tensor;
                        }
                    }
                }
        }

    }
    return weights_map;
}

int main() {
    try {
        Core ie;
        std::string model_path = "your_model.xml"; // Replace with your model path
        auto network = ie.ReadNetwork(model_path);
        ExecutableNetwork executable_network = ie.LoadNetwork(network, "CPU");


        auto weights = extract_weights(executable_network);


        for (const auto& [name, values] : weights) {
            std::cout << "Layer: " << name << ", Weights Count: " << values.size() << std::endl;
            // You could print a subset of these values, or save them to a file.
            // for (int i = 0; i < std::min(10, static_cast<int>(values.size())); ++i){
            //     std::cout << values[i] << " ";
            // }
            // std::cout << std::endl;
        }


    } catch (const std::exception& ex) {
        std::cerr << "Error occurred: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}

```

In this snippet, we are essentially traversing the computational graph. We’re checking if an operation is a `Convolution` or `FullyConnected`, or `Embedding` layer. These are typically weight-bearing layers. Inside the operation, I'm iterating through the inputs, specifically looking for `float32` tensors that are not scalars. Then using the `ngraph` api, which OpenVINO is based on we attempt to cast the input tensor as a constant. If this cast is successful then we proceed to fetch its values. I’m using a `unordered_map` to store them, keyed by the layer’s name.

Now, the above will provide a basic way of getting the weights, but it might not always apply directly without modifications. Especially when dealing with models that have fused layers or custom ops. We need a solution that's a little more versatile. Let's dive into a Python implementation that utilizes some of the more high-level APIs for dealing with operations.

```python
from openvino.runtime import Core
import numpy as np


def extract_weights_python(executable_network):
    weights_dict = {}
    for node in executable_network.get_ops():
        if node.type_name in ["Convolution", "FullyConnected", "Embedding"]:
             for i, input_node in enumerate(node.inputs()):
                if input_node.get_element_type() == "f32" and len(input_node.shape) > 1:

                    try:
                       const_node = input_node.get_source_output().get_node()
                       if const_node.type_name == "Constant":

                         tensor_data = const_node.get_vector()
                         weights_dict[f"{node.friendly_name}_weight_{i}"] = np.array(tensor_data)


                    except Exception as e:
                        print(f"Error processing {node.friendly_name}: {e}")


    return weights_dict


if __name__ == "__main__":
    ie = Core()
    model_path = "your_model.xml" # Replace with your model path
    net = ie.read_model(model=model_path)
    exec_net = ie.compile_model(model=net, device_name="CPU")
    weights = extract_weights_python(exec_net)

    for name, values in weights.items():
        print(f"Layer: {name}, Weights Count: {values.size}")
        #print(values[:10]) # print a subset

```
This Python implementation is similar to our C++ version but uses `get_ops` for accessing all nodes. The important part is `get_source_output().get_node()` which we use to determine if the input node is constant. If it is constant and its a weight tensor, then it extracts the underlying data into a numpy array. This is a more convenient way when dealing with Python projects.

Lastly, let’s look at something more involved. Sometimes, a single float tensor is not the weight. Instead it could be broken into multiple tensors, or even be contained inside a *blob*. Consider a case where you’re dealing with custom ops or quantized models. You might need to go deeper into the network's *ir*. Let's demonstrate that:

```c++
#include <iostream>
#include <ie_core.hpp>
#include <string>
#include <unordered_map>
#include <ngraph/opsets/opset1.hpp>

using namespace InferenceEngine;

std::unordered_map<std::string, std::vector<float>> extract_weights_advanced(const ExecutableNetwork& network) {
    std::unordered_map<std::string, std::vector<float>> weights_map;
    auto network_ops = network.GetExecGraphInfo().getFunction()->get_ops();

    for (const auto& op : network_ops) {
        std::string op_name = op->get_friendly_name();
        std::string op_type = op->get_type_name();

       
        for (size_t i = 0; i < op->inputs().size(); ++i) {
            auto input = op->inputs()[i];
            if (input.get_element_type() == element::f32 && input.get_shape().size() > 1) {

                auto source_node = input.get_source_output().get_node_shared_ptr();


                if (auto const_op = std::dynamic_pointer_cast<ngraph::op::Constant>(source_node))
                    {
                        auto tensor = const_op->get_vector<float>();
                         weights_map[op_name + "_weight_" + std::to_string(i)] = tensor;
                    }
                else if(auto convert_op = std::dynamic_pointer_cast<ngraph::op::Convert>(source_node)) {

                        if (auto const_op = std::dynamic_pointer_cast<ngraph::op::Constant>(convert_op->input_value(0).get_node_shared_ptr()))
                        {
                            auto tensor = const_op->get_vector<float>();
                            weights_map[op_name + "_weight_" + std::to_string(i)] = tensor;
                        }
                    }
                 // add more custom handling logic here... 
            }
        }


    }
    return weights_map;
}

int main() {
    try {
        Core ie;
        std::string model_path = "your_model.xml"; // Replace with your model path
        auto network = ie.ReadNetwork(model_path);
        ExecutableNetwork executable_network = ie.LoadNetwork(network, "CPU");

        auto weights = extract_weights_advanced(executable_network);

        for (const auto& [name, values] : weights) {
            std::cout << "Layer: " << name << ", Weights Count: " << values.size() << std::endl;

        }

    } catch (const std::exception& ex) {
        std::cerr << "Error occurred: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
```
Here, we’re checking for the constant nodes as before but also checking if they are inside a convert node, meaning an explicit type conversion was performed. The idea is to add more conditional checks, based on the expected patterns in your specific model and the IR. For instance, it’s possible for weights to be stored as a series of operations rather than direct constant tensors. These checks might look for specific node types or patterns. The crucial part is recognizing the specific pattern of how OpenVINO represents the weights within your model.

For a deeper understanding, you should study the Intel OpenVINO documentation, paying special attention to the concepts of intermediate representation (IR), executable networks, and the graph execution APIs. The *Ngraph* documentation is also essential, as OpenVINO is built upon it. A deep dive into the source code examples in the OpenVINO toolkit can offer clarity. The OpenVINO Model Optimizer documentation will shed light on how weights are typically handled after the model is converted. Additionally, the research paper "OpenVINO: An Open-Source Toolkit for Deep Learning Inference" provides the architectural perspective that helps in understanding its internal workings. Furthermore, a deep look into the *ngraph* library, specifically its constant operations and shape manipulation functionalities, will enhance your understanding of how data is structured.

To conclude, extracting weights is not always straightforward, and it often demands a thorough understanding of OpenVINO's representation of your model. The code snippets above are a great starting point. As your use cases become more complex, you may need to adapt them to your specific model’s nuances.
