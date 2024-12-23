---
title: "Can two connected Directed Acyclic Graphs (DAGs) be visualized in a single graph?"
date: "2024-12-23"
id: "can-two-connected-directed-acyclic-graphs-dags-be-visualized-in-a-single-graph"
---

 The short answer is yes, absolutely, though the method depends heavily on what you're trying to achieve with the visualization. You're essentially asking about representing two separate sets of directed relationships within a single visual space, maintaining the acyclic property of each set. I've faced this exact scenario a few times, particularly when dealing with complex data pipelines that had, let's say, independent preprocessing and postprocessing phases. Treating them as a single, overarching DAG often simplifies certain analytical tasks even if they operate in distinct logical domains.

The challenge lies in conveying that distinction visually while preserving the clarity of each DAG. Simply merging the nodes and edges, without careful planning, leads to an unreadable mess, possibly even introducing spurious connections that violate the acyclic nature of the original graphs. We need strategies to separate the visual representation of the two DAGs yet keep them together in one frame.

One of the simplest, but not always most effective, approaches is the use of color coding and spatial separation. If your two DAGs are relatively isolated in their functional roles, you could use different colors for nodes and edges belonging to each graph, effectively segregating them. Furthermore, consider arranging the graphs so that they occupy different regions of your canvas – perhaps one above the other, or side-by-side. This provides a clear visual cue to their independent nature, even within a combined visualization.

However, if the two DAGs share some data dependencies or interactions, or if their structures are complex and potentially intermingling, then visual separation alone might not be sufficient. We need to be a bit more strategic, potentially introducing 'virtual' nodes or edges to explicitly indicate relationships between the graphs if they exist. This might involve introducing special edge styles (dashed, dotted, etc.) for connections between DAGs and is especially beneficial when your DAGs aren't entirely independent entities. Another powerful method is using distinct layering or depth effects to allow them to visually reside within different planes and not become intertwined.

To illustrate, let's imagine we're dealing with a data processing pipeline. one DAG is `preprocess_dag` and another is `postprocess_dag`. They are mostly independent, but the `postprocess_dag` depends on the finalized output of the `preprocess_dag`. Let's model the two using Python’s `networkx` library to depict their respective structures, then visualize them together, demonstrating some of the techniques discussed.

First, let's establish a simple `preprocess_dag`.

```python
import networkx as nx
import matplotlib.pyplot as plt

preprocess_dag = nx.DiGraph()
preprocess_dag.add_nodes_from(['raw_data', 'clean_data', 'transformed_data'])
preprocess_dag.add_edges_from([('raw_data', 'clean_data'), ('clean_data', 'transformed_data')])


postprocess_dag = nx.DiGraph()
postprocess_dag.add_nodes_from(['transformed_data_p', 'model_output', 'report'])
postprocess_dag.add_edges_from([('transformed_data_p', 'model_output'), ('model_output', 'report')])
```
Here, I created two directed acyclic graphs, `preprocess_dag`, and `postprocess_dag`, representing distinct stages of a data processing pipeline. I have also included a node called ‘transformed_data_p’ which is identical in name to the output of the `preprocess_dag` but in reality can represent something else with the same name. This creates a dependency between DAGs. Now, let's try visualizing them in a single plot.

```python
plt.figure(figsize=(10, 6))

# Positions for the graphs to appear side by side
pos_preprocess = nx.spring_layout(preprocess_dag)
pos_postprocess = nx.spring_layout(postprocess_dag)

# shift x coordinates to the right to separate graphs
for node in pos_postprocess:
    pos_postprocess[node][0] += 3

# Draw preprocess DAG with blue nodes and edges
nx.draw(preprocess_dag, pos_preprocess, with_labels=True, node_color='lightblue', edge_color='blue', arrowsize=20, node_size=1000)

# Draw postprocess DAG with green nodes and edges
nx.draw(postprocess_dag, pos_postprocess, with_labels=True, node_color='lightgreen', edge_color='green', arrowsize=20, node_size=1000)

# Add a dashed edge to visualize connection between the graphs
nx.draw_networkx_edges([(list(preprocess_dag.nodes)[-1], list(postprocess_dag.nodes)[0])], pos= {**pos_preprocess, **pos_postprocess}, edge_color='black', arrowsize=20, style='dashed', width=2)

plt.title('Combined DAG Visualization with Spatial Separation')
plt.show()

```
In the above code, `spring_layout` generates positions for the nodes in each graph. Then, the second graph was shifted horizontally to visually separate it. I’ve added specific coloring for different DAGs, and added a dashed edge between the outputs of the preprocess and the beginning of the post process dag. This method does a decent job when graphs are relatively simple. However, the layout could still be improved. We may also consider graph drawing algorithms that explicitly support hierarchal layouts or node grouping which is usually easier to follow.

Now, suppose we require greater clarity and want to indicate that the nodes 'transformed_data' from `preprocess_dag` and 'transformed_data_p' from `postprocess_dag` are logically related though they exist in different DAGs. It’s not ideal to show them connected with an edge since they are not really part of the same dag.

In order to show a relationship between those, I may consider a visual technique such as node grouping by surrounding them in a box or similar construct. However, for this example, I will add an intermediate visual object, a “connector node” and introduce a slightly different visualization approach with a hierarchical layout which makes it more understandable when the complexity of the DAGs increase:
```python
import networkx as nx
import matplotlib.pyplot as plt

preprocess_dag = nx.DiGraph()
preprocess_dag.add_nodes_from(['raw_data', 'clean_data', 'transformed_data'])
preprocess_dag.add_edges_from([('raw_data', 'clean_data'), ('clean_data', 'transformed_data')])


postprocess_dag = nx.DiGraph()
postprocess_dag.add_nodes_from(['transformed_data_p', 'model_output', 'report'])
postprocess_dag.add_edges_from([('transformed_data_p', 'model_output'), ('model_output', 'report')])

combined_dag = nx.DiGraph()
combined_dag.add_nodes_from(preprocess_dag.nodes)
combined_dag.add_nodes_from(postprocess_dag.nodes)
combined_dag.add_edges_from(preprocess_dag.edges)
combined_dag.add_edges_from(postprocess_dag.edges)

# Add a virtual connector node
combined_dag.add_node('connector', shape='ellipse', color='grey')

# Add edges to indicate relationship to the connector
combined_dag.add_edges_from([('transformed_data', 'connector'),('connector', 'transformed_data_p')], style='dashed')


# Use a hierarchical layout for better visualization
pos = nx.multipartite_layout(combined_dag, subset_key="group", align="horizontal")
#Manually adjust layer order so connector appears in between
pos['connector'] = (pos['transformed_data'][0] + 1.5, pos['transformed_data'][1])

plt.figure(figsize=(12, 8))
nx.draw(combined_dag, pos=pos, with_labels=True, node_color=[
        'lightblue' if node in preprocess_dag else ('lightgreen' if node in postprocess_dag else 'grey')
        for node in combined_dag.nodes],
        edge_color=[
             'blue' if edge in preprocess_dag.edges else ('green' if edge in postprocess_dag.edges else 'black')
            for edge in combined_dag.edges
        ],
        arrowsize=20, node_size=1000, style=[combined_dag.edges[edge].get('style','solid') for edge in combined_dag.edges]
    )
plt.title('Combined DAG Visualization with Connector Node and Hierarchical Layout')
plt.show()
```

In this example, I create an aggregate graph `combined_dag`, I have added a virtual node 'connector' which shows relationship between nodes without introducing spurious edges that can confuse the viewer with an actual execution flow. Then I use `multipartite_layout` from `networkx` which arranges the nodes in layers which significantly improves the clarity of DAG visualization. The connector node's position is then adjusted to appear in between the two interconnected nodes. This technique is particularly helpful for complex graphs where node interactions need to be explicitly highlighted.

For a deeper dive into graph layout algorithms, look at the book "Graph Drawing: Algorithms for the Visualization of Graphs" by Giuseppe di Battista et al. It offers a solid foundation on different approaches. Another useful resource is the paper “Drawing Graphs with High Visual Quality,” by Peter Eades and Seok-Hee Hong, which has great insights into the evaluation criteria used in good quality graph layouts. When considering hierarchical layouts, the paper “A Technique for Drawing Directed Graphs” by Kozo Sugiyama et al. is a standard foundational paper that has influenced many existing layout methods.

In conclusion, visualizing two connected DAGs in a single graph is not only feasible but also frequently necessary. By applying a combination of spatial separation, color coding, virtual nodes, and sophisticated layout algorithms, we can create clear, informative, and technically accurate visual representations that effectively convey the relationships within and between these graphs. The best approach, as usual, will depend on your specific context and what insights you aim to extract from the visualization.
