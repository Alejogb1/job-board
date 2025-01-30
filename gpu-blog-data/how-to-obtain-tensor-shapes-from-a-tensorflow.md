---
title: "How to obtain tensor shapes from a TensorFlow GraphDef in C++?"
date: "2025-01-30"
id: "how-to-obtain-tensor-shapes-from-a-tensorflow"
---
TensorFlow's GraphDef, a protocol buffer representing a computation graph, stores shape information for tensors within individual node definitions. Accessing this shape data in C++ requires careful parsing of the `NodeDef` messages, particularly the `AttrValue` associated with the output tensors. From my experience building custom model deployment tooling, this process often involves navigating nested proto structures and handling variable-rank tensors.

The `GraphDef` itself is a container for a sequence of `NodeDef` messages. Each `NodeDef` describes a specific operation (e.g., convolution, addition) and holds details about its input and output tensors. Crucially, the output tensors’ shapes are not directly encoded as a separate field; rather, they are part of the attributes of the node and are usually determined during graph construction or optimization. Therefore, you cannot simply access a single, global shape table. Instead, you need to examine the `AttrValue` associated with the output tensor's name.

The process involves the following steps:
1. **Load the GraphDef:** The `GraphDef` is typically loaded from a `.pb` file using TensorFlow's C++ API. This involves reading the file content into a buffer and then parsing it into a `GraphDef` proto message.
2. **Iterate Through NodeDefs:** You iterate over each `NodeDef` contained within the `GraphDef`.
3. **Identify Output Tensors:** Each `NodeDef` has an `output` field, a repeated list of strings that specify the names of the tensors produced by the operation.
4. **Locate the Shape AttrValue:** For each output tensor name, examine the `attr` field of the `NodeDef`. This field is a map of attribute names to `AttrValue` proto messages. Look for a key that matches the tensor's name followed by `:shape`. This naming convention is how shape information is typically stored, though it's not a strict requirement of the protocol buffer.
5. **Parse the AttrValue:** If the attribute is found, you must parse the `AttrValue`. If the shape is concrete (e.g., [1, 28, 28, 3]), then the `AttrValue` will contain a `shape` field, which is a `TensorShapeProto` message. This message contains a list of dimensions (`dim`). Each `dim` holds the size of that dimension and may be an integer or have an associated `string` name. If the size is -1, then the dimension is of unknown size. If the shape is not concrete (meaning some dimensions are unspecified at graph creation) then you may not be able to get a valid shape value and may see that no such attribute exists.
6. **Extract Dimension Sizes:** Iterate through the `dim` field within the `TensorShapeProto` to extract the size of each dimension. Note that each `dim` can be represented either as an integer, or a string, if it is not a fixed dimension. This string representation indicates the name of a variable within the graph which might specify the size at runtime.
7. **Handle Unknown Shapes:** Be prepared to handle cases where a shape might be partially or completely unknown. This will be reflected in missing or `None` dimensions or unspecified `dim` fields.
8. **Store the Shapes:** Store the retrieved shapes in an appropriate data structure (e.g., a map of tensor names to vectors of dimension sizes) for subsequent use.

Let's illustrate this process with some C++ code examples.

```cpp
// Example 1: Basic shape extraction for a single NodeDef.
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include <iostream>
#include <string>
#include <vector>
#include <map>

void extractNodeOutputShapes(const tensorflow::NodeDef& node,
                            std::map<std::string, std::vector<int64_t>>& shapes) {
  for (const auto& output_tensor_name : node.output()) {
    std::string shape_attr_name = output_tensor_name + ":shape";
    if (node.attr().count(shape_attr_name) > 0) {
      const auto& attr_value = node.attr().at(shape_attr_name);
      if (attr_value.has_shape()) {
        std::vector<int64_t> tensor_shape;
        for (const auto& dim : attr_value.shape().dim()) {
             if(dim.has_size())
              tensor_shape.push_back(dim.size());
             else
             {
                //Dimension may not be a concrete size, such as a batch size
                tensor_shape.push_back(-1);
                
             }
        }
        shapes[output_tensor_name] = tensor_shape;
      }
      else{
           //Shape attribute exists but has no concrete shape.
           shapes[output_tensor_name] = {-1};
      }
    } else {
      //Shape attribute does not exist.
      shapes[output_tensor_name] = {-1};
    }
  }
}
```

This first example illustrates the core logic to extract shapes from a single `NodeDef`. It iterates through each output tensor name, constructs the attribute name, and retrieves the `AttrValue` containing the `TensorShapeProto` information. It includes error checking to confirm that the shape exists. Importantly, this example shows how non-constant sizes should be handled and stored.

```cpp
// Example 2:  Extract shapes from all NodeDefs in a GraphDef
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <fstream>

bool loadGraphDefFromPath(const std::string& path, tensorflow::GraphDef& graph_def) {
    std::ifstream stream(path, std::ios::binary);
     if (!stream.is_open()){
        std::cerr << "Failed to open " << path << std::endl;
        return false;
    }
    
    std::string content(std::istreambuf_iterator<char>(stream), {});
    
    if(!graph_def.ParseFromString(content)){
         std::cerr << "Failed to parse GraphDef from string." << std::endl;
        return false;
    }
    return true;
}


int main(int argc, char** argv) {
  if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_graphdef.pb>" << std::endl;
        return 1;
    }
    std::string graph_path = argv[1];


  tensorflow::GraphDef graph_def;
  if (!loadGraphDefFromPath(graph_path, graph_def))
  {
      return 1;
  }

  std::map<std::string, std::vector<int64_t>> all_shapes;
  for (const auto& node : graph_def.node()) {
    extractNodeOutputShapes(node, all_shapes);
  }

  for (const auto& shape_pair : all_shapes) {
      std::cout << "Tensor: " << shape_pair.first << " Shape: [";
      for(size_t i=0; i<shape_pair.second.size(); ++i)
      {
          std::cout << shape_pair.second[i];
           if(i<shape_pair.second.size() -1)
                std::cout << ",";
      }
       std::cout << "]" << std::endl;
    }
  return 0;
}

```
The second code snippet presents a complete example that loads a `GraphDef` from disk, then iterates over all of the `NodeDef` messages and prints out the shapes. It utilizes the extraction logic in the previous example and demonstrates the typical approach for handling the full graph structure. Note that for some layers, the shapes may not be directly available via the shape attribute, depending on the graph construction.

```cpp
// Example 3: Shape extraction when TensorShapeProto has named dimensions.
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include <iostream>
#include <string>
#include <vector>
#include <map>

void extractNodeOutputShapesWithNames(const tensorflow::NodeDef& node,
                                       std::map<std::string, std::vector<std::variant<int64_t, std::string>>>& shapes) {
    for (const auto& output_tensor_name : node.output()) {
        std::string shape_attr_name = output_tensor_name + ":shape";
        if (node.attr().count(shape_attr_name) > 0) {
            const auto& attr_value = node.attr().at(shape_attr_name);
            if (attr_value.has_shape()) {
                std::vector<std::variant<int64_t, std::string>> tensor_shape;
                for (const auto& dim : attr_value.shape().dim()) {
                    if (dim.has_size())
                       tensor_shape.push_back(dim.size());
                    else if (dim.has_name())
                        tensor_shape.push_back(dim.name());
                     else
                        tensor_shape.push_back(-1); 
                }
                shapes[output_tensor_name] = tensor_shape;
            }
             else{
                 //Shape attribute exists but has no concrete shape
                shapes[output_tensor_name] = {-1};
            }
        }
       else {
           //Shape attribute does not exist
           shapes[output_tensor_name] = {-1};
        }
    }
}

int main() {

 tensorflow::GraphDef graph_def;
    //Populate the graph with a custom example
    tensorflow::NodeDef node;
     node.set_name("Placeholder");
      node.set_op("Placeholder");
       node.add_output("Placeholder");
      tensorflow::AttrValue shape_attr;
       tensorflow::TensorShapeProto* shape = shape_attr.mutable_shape();
        tensorflow::TensorShapeProto_Dim* dim = shape->add_dim();
        dim->set_name("batch");
        dim = shape->add_dim();
       dim->set_size(28);
        dim = shape->add_dim();
        dim->set_size(28);
         dim = shape->add_dim();
        dim->set_size(3);
      node.mutable_attr()->insert({"Placeholder:shape",shape_attr});
    
    graph_def.add_node()->CopyFrom(node);

    std::map<std::string, std::vector<std::variant<int64_t, std::string>>> all_shapes;

  for (const auto& node : graph_def.node()) {
      extractNodeOutputShapesWithNames(node, all_shapes);
  }


    for (const auto& shape_pair : all_shapes) {
        std::cout << "Tensor: " << shape_pair.first << " Shape: [";
        for(size_t i=0; i<shape_pair.second.size(); ++i)
        {
            if (std::holds_alternative<int64_t>(shape_pair.second[i])) {
                 std::cout << std::get<int64_t>(shape_pair.second[i]);
            }
             else if(std::holds_alternative<std::string>(shape_pair.second[i])){
                std::cout << std::get<std::string>(shape_pair.second[i]);
             }
             else{
                std::cout << "unknown";
             }
             if(i<shape_pair.second.size() -1)
                  std::cout << ",";
        }
        std::cout << "]" << std::endl;
      }

  return 0;
}
```
This final example illustrates a variation in shape extraction where dimension sizes may be named. In the example, a custom GraphDef is constructed where one of the dimensions is a string name. This can happen where certain sizes are placeholders within the graph, e.g., batch size. The code shows how to modify the extraction logic to account for both integer sizes and dimension names, storing them within a `std::variant`.

For further study, I would recommend delving deeper into the TensorFlow C++ API documentation, particularly sections relating to `GraphDef`, `NodeDef`, `AttrValue`, and `TensorShapeProto`.  Examining the TensorFlow source code itself can also be highly instructive, focusing on the `tensorflow/core/framework` and `tensorflow/core/protobuf` directories. Finally, I find hands-on experience by creating test graphs with varying shape complexity to be the most effective way to develop proficiency in shape analysis.
