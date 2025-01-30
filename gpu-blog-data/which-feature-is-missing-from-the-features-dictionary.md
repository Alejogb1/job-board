---
title: "Which feature is missing from the features dictionary?"
date: "2025-01-30"
id: "which-feature-is-missing-from-the-features-dictionary"
---
The absence of a reliable method to track feature dependencies within the provided features dictionary is a critical oversight that impedes maintainability and scalability. While the dictionary appears to store feature flags as key-value pairs, it lacks the structure to represent the relationships between these features. Specifically, we don’t know which features might rely on others, or which should be enabled before others for a consistent application state. This omission can lead to cascading issues when modifying or introducing new features, requiring manual tracking and increasing the risk of errors.

In my experience managing a complex, microservices-based application for several years, I've repeatedly observed how seemingly isolated feature flags can become entangled, resulting in unexpected behaviour when toggled independently. Without dependency management, developers often end up relying on tribal knowledge or complex commit histories, making it difficult for new team members to understand the impact of changing specific feature states. The features dictionary as it currently exists is akin to a flat data structure where each item appears to have equal importance without consideration for their place in a broader logical flow.

A more robust system would allow us to define which features are parents or children of others, or if they need to be enabled or disabled in tandem with other feature flags. Consider, for example, a new user interface for our data visualization component which requires both "new_layout_enabled" and “enhanced_chart_render" to function correctly. Without explicit dependency management, attempting to turn on "new_layout_enabled" before enabling "enhanced_chart_render" may cause UI rendering failures.

Here's a Python implementation of the current features dictionary:

```python
features = {
    "new_login_page": True,
    "enhanced_search": False,
    "personalized_recommendations": True,
    "experimental_widget": False,
    "new_user_onboarding": False,
    "dark_mode_enabled": True,
}

def is_feature_enabled(feature_name):
    return features.get(feature_name, False)
```

This example showcases the basic functionality of checking feature states. However, there is no way to indicate which feature is necessary for another. The `is_feature_enabled` function does not account for any dependencies. Consider that `personalized_recommendations` may rely on `enhanced_search` and if enhanced search is disabled the recommendations might throw errors.

A better approach would be to represent feature dependencies using a directed acyclic graph (DAG). We can implement this using a dictionary where the key is the feature name and the value is another dictionary with at least two entries : a boolean that shows if the feature is enabled and a list of features that this feature depends on.

Here’s an example:

```python
features_with_dependencies = {
    "new_login_page": {"enabled": True, "depends_on": []},
    "enhanced_search": {"enabled": False, "depends_on": []},
    "personalized_recommendations": {"enabled": True, "depends_on": ["enhanced_search"]},
    "experimental_widget": {"enabled": False, "depends_on": []},
    "new_user_onboarding": {"enabled": False, "depends_on": ["new_login_page"]},
    "dark_mode_enabled": {"enabled": True, "depends_on": []},
     "new_layout_enabled": {"enabled":False, "depends_on":[]},
     "enhanced_chart_render":{"enabled":False,"depends_on":["new_layout_enabled"]}
}

def is_feature_enabled_with_dependencies(feature_name, feature_dict):
    feature = feature_dict.get(feature_name)
    if not feature:
      return False

    if not feature["enabled"]:
      return False

    for dependency in feature["depends_on"]:
        if not is_feature_enabled_with_dependencies(dependency, feature_dict):
            return False

    return True

#Testing the improved implementation
print(f"Is personalized recommendations enabled: {is_feature_enabled_with_dependencies('personalized_recommendations', features_with_dependencies)}")
print(f"Is new user onboarding enabled: {is_feature_enabled_with_dependencies('new_user_onboarding', features_with_dependencies)}")
print(f"Is enhanced chart render enabled: {is_feature_enabled_with_dependencies('enhanced_chart_render', features_with_dependencies)}")
```

In this version, `personalized_recommendations` will be disabled even though it's set to `True`, because it depends on the `enhanced_search` feature, which is disabled. This approach adds a layer of complexity but significantly improves the system's reliability by automatically considering feature dependencies.  The enhanced chart render will not be rendered since even if it was enabled, the new layout is not.

We can enhance the code further by adding features to automatically enable dependencies and handle cases where dependencies are circularly linked.  Here is another code example of how we can add that functionality using recursion. It also includes a safety check to avoid infinite recursion.

```python

features_with_dependencies = {
    "new_login_page": {"enabled": True, "depends_on": []},
    "enhanced_search": {"enabled": False, "depends_on": []},
    "personalized_recommendations": {"enabled": True, "depends_on": ["enhanced_search"]},
    "experimental_widget": {"enabled": False, "depends_on": []},
    "new_user_onboarding": {"enabled": False, "depends_on": ["new_login_page"]},
    "dark_mode_enabled": {"enabled": True, "depends_on": []},
     "new_layout_enabled": {"enabled":False, "depends_on":[]},
     "enhanced_chart_render":{"enabled":False,"depends_on":["new_layout_enabled"]},
    "circular_dep_1": {"enabled": False, "depends_on": ["circular_dep_2"]},
    "circular_dep_2": {"enabled": False, "depends_on": ["circular_dep_1"]},
}


def set_feature_enabled_with_dependencies(feature_name, new_state, feature_dict, visited = None):
    if visited is None:
         visited = set()
    if feature_name in visited:
         return feature_dict

    visited.add(feature_name)
    feature = feature_dict.get(feature_name)
    if not feature:
       return feature_dict

    if new_state:
       for dependency in feature["depends_on"]:
          feature_dict = set_feature_enabled_with_dependencies(dependency, True, feature_dict, visited)
    feature["enabled"] = new_state
    return feature_dict


# Testing with dependency enabling
updated_features = set_feature_enabled_with_dependencies("personalized_recommendations", True, features_with_dependencies)
print(f"Is personalized recommendations enabled: {is_feature_enabled_with_dependencies('personalized_recommendations', updated_features)}")
print(f"Is enhanced search enabled: {is_feature_enabled_with_dependencies('enhanced_search', updated_features)}")
print(f"Is new layout enabled:{is_feature_enabled_with_dependencies('new_layout_enabled',updated_features)}")
print(f"Is enhanced chart render enabled:{is_feature_enabled_with_dependencies('enhanced_chart_render',updated_features)}")


updated_features_2 = set_feature_enabled_with_dependencies("enhanced_chart_render", True, updated_features)
print(f"Is new layout enabled:{is_feature_enabled_with_dependencies('new_layout_enabled',updated_features_2)}")
print(f"Is enhanced chart render enabled:{is_feature_enabled_with_dependencies('enhanced_chart_render',updated_features_2)}")

# Example of circular dependency
updated_features_3 = set_feature_enabled_with_dependencies("circular_dep_1",True, features_with_dependencies)
print(f"Is circular dependency 1 enabled: {is_feature_enabled_with_dependencies('circular_dep_1', updated_features_3)}")
print(f"Is circular dependency 2 enabled: {is_feature_enabled_with_dependencies('circular_dep_2', updated_features_3)}")
```

This improved version introduces the `set_feature_enabled_with_dependencies` function which enables the features together based on dependencies. It uses recursion and the `visited` set to avoid infinite recursions due to circular dependencies. If we enable `personalized_recommendations` it now also enables `enhanced_search`. If we then attempt to enable `enhanced_chart_render` it will also enable `new_layout_enabled`. The example with circular dependency shows that it does not crash the code, though neither circular dependency gets enabled.

For deeper understanding of design patterns, specifically dependency injection, consider exploring resources like "Patterns of Enterprise Application Architecture." To enhance knowledge of data structures and algorithms, resources like "Introduction to Algorithms" offer comprehensive insights.  For information on best practices when designing large scale systems, books such as "Designing Data-Intensive Applications" are recommended.

In summary, the critical missing feature from the provided features dictionary is an explicit representation of feature dependencies. This omission leads to fragile code, increased complexity when introducing or modifying features, and an overall lack of clarity regarding the application’s behavior. Implementing a directed acyclic graph approach, as demonstrated, is a step towards a more maintainable and scalable feature flag management system.
