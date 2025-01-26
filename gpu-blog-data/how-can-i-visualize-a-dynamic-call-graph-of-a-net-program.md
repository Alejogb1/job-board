---
title: "How can I visualize a dynamic call graph of a .NET program?"
date: "2025-01-26"
id: "how-can-i-visualize-a-dynamic-call-graph-of-a-net-program"
---

Dynamic call graph visualization in .NET necessitates runtime instrumentation, as static analysis alone cannot capture the nuanced execution paths resulting from polymorphism, reflection, and dynamic code generation. I’ve found that successfully achieving this involves a combination of tracing, data aggregation, and appropriate graph rendering tools.

The core challenge lies in intercepting function calls at runtime. .NET provides several mechanisms for this, including the Common Language Runtime (CLR) Profiling API, which is my preferred method for detailed analysis, and simpler reflection-based techniques for specific situations. The CLR Profiling API offers low-level callbacks for various CLR events, including function entry and exit, providing the precise data required for call graph construction. This API requires writing a native C++ profiler DLL, which hooks into the .NET runtime. While more complex to set up, the performance overhead is generally lower than reflection-based methods, and it captures all managed code activity, regardless of reflection or dynamic invocation.

My typical workflow begins with constructing the profiler DLL in C++. The key interfaces to implement are `ICorProfilerCallback` or `ICorProfilerCallback2`, depending on the version of the .NET runtime being profiled. Within this profiler, specific callbacks like `FunctionEnter` and `FunctionLeave` are of primary concern. These callbacks provide method IDs and class IDs (the `FunctionID` and `ClassID`), which can be resolved into function and class names using `ICorProfilerInfo` interface methods such as `GetFunctionInfo` and `GetClassInfo`. The data collected within these callbacks forms the raw material for the call graph. For each function entry, the profiler records the caller and callee IDs, creating a directed edge in the graph. When a function exits, it helps confirm the return path. For more accuracy, it is important to account for thread IDs as well since .NET programs are often multithreaded, and a call on one thread may not be related to another thread's call.

Once the profiling data is collected, which I generally store in an intermediate format such as a custom binary file optimized for rapid parsing and minimal overhead, the visualization stage begins.  This phase requires careful post-processing to assemble the raw edge data into a coherent call graph structure. The basic graph representation is a set of nodes (methods) and edges (calls). It is vital to avoid adding duplicate calls, particularly in recursive scenarios, or the graph can become unmanageable. After constructing the graph, you need to choose a rendering library or application that can display the graph effectively. I prefer using graphviz, and its dot language for graph description, as it offers high customization options and can handle larger graphs well.

Here are a few concrete examples of how to approach certain aspects.

**Example 1: C++ Profiler Callbacks for Function Entry**

This code snippet demonstrates the `FunctionEnter` callback within the native profiler DLL.

```cpp
HRESULT FunctionEnter(FunctionID functionId)
{
  if (profilerInfo == nullptr) return S_OK;

  ClassID classId;
  ModuleID moduleId;
  mdToken methodToken;
  HRESULT hr = profilerInfo->GetFunctionInfo(functionId, &classId, &moduleId, &methodToken, 0, nullptr, nullptr);
  if (FAILED(hr)) return S_OK;

  WCHAR classNameBuffer[256];
  ULONG nameLength = 0;
  hr = profilerInfo->GetClassIDInfo(classId, 256, &nameLength, classNameBuffer, &moduleId, 0);
  if (FAILED(hr)) return S_OK;

  WCHAR methodNameBuffer[256];
  nameLength = 0;
  hr = profilerInfo->GetFunctionInfo(functionId, &classId, &moduleId, &methodToken, 256, &nameLength, methodNameBuffer);
  if (FAILED(hr)) return S_OK;

  // Get current thread ID.
  DWORD threadId = GetCurrentThreadId();

  // Obtain parent frame for the call using frame information if available in the profiler callback. 
  // Here, we will have to save the functionID and frame if available, or fall back to a simple call tracking method.
  // Store class name, method name, thread Id, and the parent function ID (if available) in the data stream.
  // This is an oversimplification, real-world scenarios also consider generic types, etc. 
  AddToDataStream(classNameBuffer, methodNameBuffer, threadId, functionId);

  return S_OK;
}
```
In this example, I am retrieving class and function information associated with the given `functionId`. I store this information, along with thread ID and parent function ID, into a data stream for later processing. The `AddToDataStream` function (not shown here) represents the method that would serialize this data in an efficient, thread-safe way. Error handling is also intentionally simplified here for clarity. In a production profiler, you would want more robust error checking, dynamic buffer allocation, and thread synchronization. This also does not show the `FunctionLeave` callback which is necessary for proper tracking of call graph nesting, but will use a similar structure.

**Example 2: Graph Post-processing in C#**

After profiling, the generated data needs processing to form a graph structure that can be used for visualization. The following code showcases a simplified approach to processing binary data and generating a graph object, in C#.

```csharp
using System;
using System.Collections.Generic;
using System.IO;

public class CallGraphBuilder
{
    public Dictionary<(string,string), List<(string, string)>> CallGraph {get;} = new Dictionary<(string, string), List<(string,string)>>();
    public void Process(string filename)
    {
        using (BinaryReader reader = new BinaryReader(File.OpenRead(filename)))
        {
           while (reader.BaseStream.Position < reader.BaseStream.Length)
            {
              //Read data as written during the Profiling stage 
               string className = reader.ReadString();
               string methodName = reader.ReadString();
               int threadId = reader.ReadInt32();
               int parentFunctionId = reader.ReadInt32();

                // Create a key from (class name, method name).
                var calleeKey = (className, methodName);

                //Check if parent function is available or is the starting point. 
                if(parentFunctionId > 0)
                {
                  //Lookup in the function database if we already stored its info. 
                   var callerKey = FindFunctionDataById(parentFunctionId); 

                   //Add this call graph link. 
                   if(callerKey != default){
                        if(!CallGraph.ContainsKey(callerKey)){
                          CallGraph[callerKey] = new List<(string,string)>();
                        }

                        CallGraph[callerKey].Add(calleeKey); 
                   }
                } else {

                  if(!CallGraph.ContainsKey(calleeKey)){
                      CallGraph[calleeKey] = new List<(string,string)>();
                  }
                }


            }
        }
    }


    private (string, string) FindFunctionDataById(int functionId)
    {
      //We need a functionId to name dictionary in the class. 
      //Lookup the function using functionId. 
      return default; 
    }
}
```
This code iterates through the binary file, reads the class, method name, thread id, and parent id from the file as previously written in the native profiler dll. It then builds a dictionary where the key is a method identifier (class name and method name) and value is list of calls to other methods. Here also we must consider error handling, thread safety, and data corruption. The `FindFunctionDataById` method is left as a stub as its implementation would depend on the specific data storage mechanism in the profiler.

**Example 3: Generating Dot Graph File**

Finally, I generate a DOT language file for rendering. Here's how I convert the C# graph structure to a text-based format suitable for `graphviz`.

```csharp
using System.IO;
using System.Text;

public class DotGraphGenerator
{
   public void GenerateDotFile(Dictionary<(string, string), List<(string, string)>> callGraph, string filename)
   {
        using (StreamWriter writer = new StreamWriter(filename, false, Encoding.UTF8))
        {
            writer.WriteLine("digraph callgraph {");
            writer.WriteLine("  rankdir=LR;"); //Layout the graph left to right
            foreach (var caller in callGraph)
            {
                 string callerName = $"{caller.Key.Item1}.{caller.Key.Item2}".Replace("<", "").Replace(">", "");
                foreach(var callee in caller.Value)
                {
                  string calleeName = $"{callee.Item1}.{callee.Item2}".Replace("<", "").Replace(">", "");
                   writer.WriteLine($"  \"{callerName}\" -> \"{calleeName}\";");
                }
            }
            writer.WriteLine("}");
        }
   }
}
```

This simple code generates a dot file that encodes the call graph in a format that can be processed by graphviz. String replacement is used to remove generic type markers and to make the node name valid. Each call is represented as a directed edge in the graph. This resulting DOT file can be processed by the dot command to generate SVG or PNG outputs.

For further investigation, I would recommend studying resources about the CLR Profiling API provided by Microsoft (specifically documentation and tutorials). Explore different graph layout options provided by Graphviz or alternative graph drawing libraries. Also, consider researching techniques for handling large, complex graphs. Performance is paramount when profiling, so it’s essential to minimize the impact of the profiler on the program being analyzed. Therefore, carefully selecting data storage mechanisms and keeping the profiler lightweight is essential. Lastly, keep a close eye on thread safety and proper resource management.
