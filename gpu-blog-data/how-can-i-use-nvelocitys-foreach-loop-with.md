---
title: "How can I use NVelocity's foreach loop with two lists?"
date: "2025-01-30"
id: "how-can-i-use-nvelocitys-foreach-loop-with"
---
Directly addressing the challenge of iterating over two lists simultaneously using NVelocity’s `foreach` directive requires a nuanced understanding of how Velocity templates handle data structures. Unlike some templating languages, NVelocity’s `foreach` operates primarily on a single list at a time; it does not provide native support for direct parallel iteration over two distinct lists. Therefore, solutions necessitate either pre-processing data to combine the two lists into a single iterable entity or leveraging specific Velocity context manipulation techniques. My experience with complex template generation systems, particularly within financial reporting platforms, has led me to frequently deal with this constraint.

The core issue is that NVelocity's `foreach` syntax, generally expressed as `#foreach ($item in $list)` only targets the `$list` variable. It expects `$list` to be an instance of an `IEnumerable` or equivalent capable of producing successive elements. There is no intrinsic mechanism within the directive itself to access or advance a second list concurrently with the first. This means a direct application of two lists in a single `foreach` loop is impossible and requires a preliminary or in-template approach. We must either join the data or access the lists by index within the loop.

My preference, based on past projects, leans towards preparing the data model appropriately before template rendering. This maintains template clarity and reduces reliance on complex in-template logic. However, in scenarios where pre-processing is cumbersome or impossible, NVelocity offers alternative strategies. I have found this flexibility valuable, especially in scenarios involving dynamic configurations where data structures could vary at runtime. I have handled scenarios where the data could be a list or a dictionary and pre-processing is more difficult. Let's consider three practical solutions.

**Solution 1: Pre-Processed Zipped List**

This approach is often the most straightforward and readable. The two separate lists are combined into a single list of compound elements, usually an object or a dictionary (hash map) that holds corresponding elements from each original list. This preparation occurs in the code before the NVelocity context is populated.

```csharp
using System.Collections.Generic;
using NVelocity;
using NVelocity.App;

public class Example1
{
    public static void Main(string[] args)
    {
        //Sample Data Lists
        List<string> names = new List<string> { "Alice", "Bob", "Charlie" };
        List<int> ages = new List<int> { 30, 25, 35 };

        //Zip List
        List<Dictionary<string, object>> zippedData = new List<Dictionary<string, object>>();
        for (int i = 0; i < names.Count; i++)
        {
            zippedData.Add(new Dictionary<string, object>() {
            { "Name", names[i] },
            { "Age", ages[i] }
            });
        }

        // Initialize Velocity Engine
        VelocityEngine velocityEngine = new VelocityEngine();
        velocityEngine.Init();

        // Create a context
        VelocityContext context = new VelocityContext();
        context.Put("people", zippedData);


        // Define the Velocity Template
        string templateString = @"
#foreach ($person in $people)
    Name: $person.Name, Age: $person.Age
#end";

        // Render the Template
        var template = velocityEngine.CreateTemplate(templateString);
        var writer = new StringWriter();
        template.Merge(context, writer);

        Console.WriteLine(writer.ToString());

    }
}
```

In the code sample, the two lists, `names` and `ages`, are “zipped” together before reaching the template. A new list, `zippedData`, is created, where each element is a dictionary containing a `Name` and `Age` property, taken in corresponding order from the original lists. The template is then able to iterate over this single, combined list using a regular `foreach` directive. This method has the advantage of being exceptionally clear, especially when the relationships between elements from different source lists are complex.

**Solution 2: Index-Based Access**

When pre-processing is not feasible or when lists are accessed indirectly via context lookups, we can utilize the `$velocityCount` variable within the template to access elements from the second list by index. This approach allows iteration without combining the lists beforehand.

```csharp
using System.Collections.Generic;
using NVelocity;
using NVelocity.App;
using System.IO;

public class Example2
{
    public static void Main(string[] args)
    {
        //Sample Data Lists
        List<string> names = new List<string> { "Alice", "Bob", "Charlie" };
        List<int> ages = new List<int> { 30, 25, 35 };

        // Initialize Velocity Engine
        VelocityEngine velocityEngine = new VelocityEngine();
        velocityEngine.Init();

        // Create a context
        VelocityContext context = new VelocityContext();
        context.Put("names", names);
        context.Put("ages", ages);


        // Define the Velocity Template
        string templateString = @"
#foreach ($name in $names)
    Name: $name, Age: $ages[$velocityCount-1]
#end";

        // Render the Template
        var template = velocityEngine.CreateTemplate(templateString);
        var writer = new StringWriter();
        template.Merge(context, writer);

        Console.WriteLine(writer.ToString());

    }
}
```

Here, the `names` and `ages` lists remain separate in the context. The template iterates over the `$names` list using the default `foreach`. Inside this loop, `$velocityCount` provides the current iteration index (starting from 1). Since list indexes are zero-based, we subtract 1 to access the corresponding age from the `$ages` list, via `$ages[$velocityCount-1]`. This solution is effective in instances where you must keep the two lists separated in the context and offers a direct, though less readable approach.

**Solution 3:  Custom Context Extension**

For more complex scenarios, consider extending the NVelocity context with custom functions or properties capable of handling dual-list iterations. This involves creating a C# class accessible to the template which then implements a custom way of iterating over the data. This would allow for a more customized and reusable approach.

```csharp
using System.Collections.Generic;
using NVelocity;
using NVelocity.App;
using System.IO;

public class Example3
{

    public class ListIterator
    {
        public List<string> Names { get; set; }
        public List<int> Ages { get; set; }

         public IEnumerable<Dictionary<string, object>> Iterate()
        {
            if (Names == null || Ages == null || Names.Count != Ages.Count)
            {
                yield break;
            }
            for (int i = 0; i < Names.Count; i++)
            {
               yield return new Dictionary<string, object>() {
            { "Name", Names[i] },
            { "Age", Ages[i] }
                };
            }
        }

    }

    public static void Main(string[] args)
    {
        //Sample Data Lists
        List<string> names = new List<string> { "Alice", "Bob", "Charlie" };
        List<int> ages = new List<int> { 30, 25, 35 };

       var iterator = new ListIterator { Names = names, Ages = ages };
        // Initialize Velocity Engine
        VelocityEngine velocityEngine = new VelocityEngine();
        velocityEngine.Init();

        // Create a context
        VelocityContext context = new VelocityContext();
        context.Put("iterator", iterator);


        // Define the Velocity Template
        string templateString = @"
#foreach ($person in $iterator.Iterate())
    Name: $person.Name, Age: $person.Age
#end";

        // Render the Template
        var template = velocityEngine.CreateTemplate(templateString);
        var writer = new StringWriter();
        template.Merge(context, writer);

        Console.WriteLine(writer.ToString());

    }
}
```

In this instance, we create a custom class `ListIterator` which accepts the lists and has an `Iterate` method that returns a dictionary of combined values. In the main program, this class is instantiated, and the instance is put in the context. The template can then directly iterate on `iterator.Iterate()`, treating it as any other list in velocity. While this involves a bit more overhead in the C# code, it offers more control in template logic and improved readability. It’s suitable for scenarios where similar operations are performed regularly or involve more complex transformations beyond simple indexing.

Regarding resource materials for further study, I recommend focusing on the official NVelocity documentation; the core library's source code can offer insights, although it is more technical. Additionally, studying best practices in template architecture within systems such as Django or Jinja2 may offer useful insights into structuring template logic, even though the syntax is different. Books on C# programming patterns can also be valuable for structuring the code which pre-processes the data before passing it to velocity. Finally, studying data structure usage within a particular operating system or environment where the velocity templates are implemented can also help provide additional ideas for manipulating and combining the data before rendering the template.
