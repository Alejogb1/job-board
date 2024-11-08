---
title: "Need a .NET Graph Library? Here's What I Found"
date: '2024-11-08'
id: 'need-a-net-graph-library-here-s-what-i-found'
---

```csharp
// Install-Package QuickGraph
using QuickGraph;
using QuickGraph.Algorithms;
using QuickGraph.Algorithms.Search;
using QuickGraph.Algorithms.ShortestPath;
using QuickGraph.Serialization;
using System;
using System.Collections.Generic;
using System.Linq;

// Create a graph
var graph = new AdjacencyGraph<string, Edge<string>>();

// Add vertices
graph.AddVertex("A");
graph.AddVertex("B");
graph.AddVertex("C");
graph.AddVertex("D");

// Add edges
graph.AddEdge(new Edge<string>("A", "B"));
graph.AddEdge(new Edge<string>("B", "C"));
graph.AddEdge(new Edge<string>("C", "D"));
graph.AddEdge(new Edge<string>("D", "A"));

// Find shortest path
var shortestPathAlgorithm = new DijkstraShortestPathAlgorithm<string, Edge<string>>(graph, e => 1);
shortestPathAlgorithm.Compute("A");

// Get the shortest path
var shortestPath = shortestPathAlgorithm.ShortestPaths.ToList();

// Print the shortest path
foreach (var vertex in shortestPath)
{
    Console.WriteLine(vertex);
}
```
