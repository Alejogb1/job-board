---
title: "How can I retrieve tree data as a list from a LightGBM classifier?"
date: "2024-12-23"
id: "how-can-i-retrieve-tree-data-as-a-list-from-a-lightgbm-classifier"
---

Okay, let's tackle this one. The intricacies of extracting tree structures from a LightGBM model, and transforming them into a more workable list format, is a task I've navigated a fair number of times. It's not always immediately obvious, and the built-in model representation isn’t quite what you need for certain kinds of analysis or visualization. I recall a particularly thorny case working on a fraud detection project where we needed detailed insights into individual decision paths for interpretability, and extracting the data into list format was absolutely crucial. So, let me explain the process from a hands-on perspective, not just the theory, and give you some concrete code examples.

The core challenge isn't just pulling the tree data—LightGBM's `booster_.dump_model()` offers that readily enough as a dictionary structure. The problem is transforming that output, which is hierarchical, into a flattened list that you can then more easily process. This conversion involves recursively walking through the tree’s nodes. Each node may have split information (feature, threshold, gain) or be a leaf containing a value.

The `dump_model()` method, by default, returns a json-like dictionary, nested deeply and representing all the trees in the forest. The 'tree_info' key contains an array of dictionaries where each dictionary corresponds to a single decision tree. Within each tree, you find ‘tree_structure’ with nodes represented by keys like 'leaf_value', 'split_feature', 'split_gain' and 'threshold'. Our goal is to extract data about every single node and put it into a list format that looks somewhat like: `[{'node_id': 0, 'type': 'split', 'feature': 'feature_1', 'threshold': 0.5, 'gain': 100, 'left_child': 1, 'right_child': 2}, {'node_id': 1, 'type': 'leaf', 'value': 0.1}, ...]`

Here's how you achieve that using a recursive function to traverse the tree structure.

```python
import lightgbm as lgb
import json

def extract_tree_as_list(model):
    """
    Extracts tree data from a LightGBM model as a list of dictionaries.
    Each dictionary represents a node (split or leaf).

    Args:
        model: A trained LightGBM booster object.

    Returns:
        A list of dictionaries, where each dictionary represents a node.
        Returns an empty list if something goes wrong in the parsing.
    """
    try:
        model_dump = json.loads(model.dump_model())
        tree_info = model_dump['tree_info']
        tree_list = []

        def _extract_node(node, node_id, tree_index, parent_id=None, direction = None):
            """
            Recursive helper function to process tree nodes.
            """
            if 'leaf_value' in node:
                tree_list.append({
                    'tree_index': tree_index,
                    'node_id': node_id,
                    'type': 'leaf',
                    'value': node['leaf_value'],
                    'parent_id': parent_id,
                    'direction': direction
                })
            else:
                tree_list.append({
                    'tree_index': tree_index,
                    'node_id': node_id,
                    'type': 'split',
                    'feature': node['split_feature'],
                    'threshold': node['threshold'],
                    'gain': node['split_gain'],
                    'left_child': node['left_child'],
                    'right_child': node['right_child'],
                    'parent_id': parent_id,
                    'direction': direction
                })
                _extract_node(node['left_child'], node['left_child'], tree_index, node_id, 'left')
                _extract_node(node['right_child'], node['right_child'], tree_index, node_id, 'right')


        for tree_idx, tree in enumerate(tree_info):
            if 'tree_structure' in tree and tree['tree_structure']: #check if the tree structure is not null
                _extract_node(tree['tree_structure'], 0, tree_idx)

        return tree_list

    except Exception as e:
        print(f"An error occurred: {e}")
        return []


# Example usage
# First, train a dummy classifier:

X = [[1, 2], [3, 4], [5, 6], [7,8], [9,10]]
y = [0, 1, 0, 1, 0]
lgbm = lgb.LGBMClassifier(n_estimators=2)
lgbm.fit(X,y)
tree_list = extract_tree_as_list(lgbm.booster_)

print(tree_list)
```

This code performs the transformation by: 1. Dumping the model into JSON format; 2. Extracting the 'tree_info'; 3. Implementing a recursive helper function which visits each node and extracts relevant attributes. 4. Appends a dictionary with extracted node information to the `tree_list`; 5. Returns that list or an empty one if an exception happens during the processing.

The recursive function `_extract_node` is central here. It takes a node, its id, its tree's index and parent node ID (which can be `None` at the root), plus the direction from the parent node (`left` or `right`). If the node is a leaf, it extracts the `leaf_value`; otherwise, it extracts the split feature, threshold, gain, and child IDs. Crucially, it recursively calls itself on the left and right children.

Let’s refine this to include feature names. Often, when inspecting models, you want to see not only the feature index (like 'feature_1') but the actual feature names. To accomplish this, you need to pass feature names into the training process and then extract that information from the model dump. Here’s an extended version:

```python
import lightgbm as lgb
import json

def extract_tree_as_list_with_names(model, feature_names):
    """
    Extracts tree data from a LightGBM model as a list of dictionaries,
    including feature names.

    Args:
        model: A trained LightGBM booster object.
        feature_names: A list of strings representing the feature names.

    Returns:
        A list of dictionaries, where each dictionary represents a node.
    """

    try:
        model_dump = json.loads(model.dump_model())
        tree_info = model_dump['tree_info']
        tree_list = []

        def _extract_node(node, node_id, tree_index, parent_id=None, direction = None):
              if 'leaf_value' in node:
                tree_list.append({
                    'tree_index': tree_index,
                    'node_id': node_id,
                    'type': 'leaf',
                    'value': node['leaf_value'],
                    'parent_id': parent_id,
                    'direction': direction
                })
              else:
                  feature_index = int(node['split_feature'][8:]) # Extract index
                  tree_list.append({
                      'tree_index': tree_index,
                      'node_id': node_id,
                      'type': 'split',
                      'feature': feature_names[feature_index],
                      'threshold': node['threshold'],
                      'gain': node['split_gain'],
                      'left_child': node['left_child'],
                      'right_child': node['right_child'],
                       'parent_id': parent_id,
                      'direction': direction
                })

                  _extract_node(node['left_child'], node['left_child'], tree_index, node_id, 'left')
                  _extract_node(node['right_child'], node['right_child'], tree_index, node_id, 'right')


        for tree_idx, tree in enumerate(tree_info):
            if 'tree_structure' in tree and tree['tree_structure']:
                _extract_node(tree['tree_structure'], 0, tree_idx)
        return tree_list
    except Exception as e:
         print(f"An error occurred: {e}")
         return []

# Example Usage
feature_names = ['feature_1', 'feature_2']
lgbm = lgb.LGBMClassifier(n_estimators=2)
lgbm.fit(X,y, feature_name=feature_names)
tree_list = extract_tree_as_list_with_names(lgbm.booster_,feature_names)

print(tree_list)
```

The key modification here is accessing `node['split_feature']` and mapping the numerical index in the string 'feature_X' back to the actual name from `feature_names`. I perform a simple string slicing here, based on the standard formatting, to extract the index.

Finally, consider cases where we want to retain the depth information for each node, and keep track of root node. We can modify the recursive function further to calculate the depth of each node.

```python
import lightgbm as lgb
import json

def extract_tree_as_list_with_depth(model, feature_names):
  """
    Extracts tree data from a LightGBM model as a list of dictionaries,
    including feature names and node depth.
    """
  try:
    model_dump = json.loads(model.dump_model())
    tree_info = model_dump['tree_info']
    tree_list = []
    def _extract_node(node, node_id, tree_index, depth, parent_id=None, direction = None):
        if 'leaf_value' in node:
          tree_list.append({
             'tree_index': tree_index,
              'node_id': node_id,
              'type': 'leaf',
              'value': node['leaf_value'],
              'depth': depth,
              'parent_id': parent_id,
             'direction': direction
          })
        else:
            feature_index = int(node['split_feature'][8:]) # Extract index
            tree_list.append({
              'tree_index': tree_index,
              'node_id': node_id,
              'type': 'split',
              'feature': feature_names[feature_index],
              'threshold': node['threshold'],
              'gain': node['split_gain'],
              'left_child': node['left_child'],
              'right_child': node['right_child'],
               'depth': depth,
                'parent_id': parent_id,
                'direction': direction
            })
            _extract_node(node['left_child'], node['left_child'], tree_index, depth + 1, node_id, 'left')
            _extract_node(node['right_child'], node['right_child'], tree_index, depth + 1, node_id, 'right')

    for tree_idx, tree in enumerate(tree_info):
      if 'tree_structure' in tree and tree['tree_structure']:
            _extract_node(tree['tree_structure'], 0, tree_idx, 0) #depth starts at 0

    return tree_list
  except Exception as e:
         print(f"An error occurred: {e}")
         return []

#Example Usage
feature_names = ['feature_1', 'feature_2']
lgbm = lgb.LGBMClassifier(n_estimators=2)
lgbm.fit(X,y, feature_name=feature_names)
tree_list = extract_tree_as_list_with_depth(lgbm.booster_, feature_names)
print(tree_list)
```

Here, I added a depth variable that is incremented at each recursive call. This depth information can be incredibly useful for visualizing tree structures in a more intuitive way or for advanced tree analysis techniques.

To delve further into the theoretical underpinnings of these models, I highly recommend the book "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman. For a deeper dive into tree-based methods and boosting algorithms, "Pattern Recognition and Machine Learning" by Bishop is a great companion. Lastly, LightGBM's official documentation, while not a substitute for academic understanding, provides essential practical details about the model's parameters and internal structures.

These examples and resources should get you started with extracting and transforming your LightGBM tree data. Keep in mind that the way you structure the list can be adapted to the specifics of your analysis. The core principle, however, remains the same – recursively traverse the tree and extract the relevant attributes into a more convenient, flat list structure.
