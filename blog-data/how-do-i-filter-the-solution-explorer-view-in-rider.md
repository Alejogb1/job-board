---
title: "How do I filter the solution explorer view in Rider?"
date: "2024-12-23"
id: "how-do-i-filter-the-solution-explorer-view-in-rider"
---

Alright,  Filtering the solution explorer in Rider, as I've found in past projects, can really streamline your workflow, especially when dealing with large, sprawling solutions. It's a feature I've come to appreciate deeply over the years, particularly during that rather cumbersome enterprise application project where navigating the sheer volume of files was a constant struggle. Initially, I found myself relying heavily on manual scrolling, which, as you can imagine, quickly becomes tedious. The solution, however, is quite elegant and leverages Rider's robust filtering capabilities.

Essentially, Rider provides several avenues to accomplish this, and I often combine them for maximum efficiency. The most basic way is to simply begin typing in the search bar located at the top of the solution explorer window. This acts as a live filter; as you type, Rider will dynamically display only the nodes – projects, folders, and files – that match the entered text within their names. The matching algorithm is reasonably intelligent, considering partial matches and not strictly demanding an exact string. For instance, typing 'api' might show you 'MyWebApiProject,' 'ApiControllers,' and 'DataTransferObjects/ApiModels.' This immediate feedback is incredibly helpful.

But simple name filtering is often inadequate. We need more granular control. That's where pattern-based filters come into play. You can use wildcards like '*' and '?' to define more complex patterns. For example, '*test*' will display anything containing "test" in its name, while 'Config*.*' would only show files beginning with 'Config' and having any extension. I found that using the wildcard asterisk is particularly useful when dealing with multiple similar named projects that also reside in similar locations within the project structure. The question mark '?' matches a single character. So, 'MyFile?.cs' matches 'MyFile1.cs', 'MyFileA.cs', etc. but not 'MyFile12.cs' or 'MyFile.cs'.

The second, more advanced method involves using the "Filter" dropdown menu in the solution explorer's toolbar. Here, you’ll find several pre-configured filters like "Problems," which highlights files containing errors or warnings, and "Opened Files," which displays only the currently active files. There’s also a "Modified Files" filter to quickly jump to files you have changed since the last commit. These are fantastic for focusing your attention, and they’re customizable, which is something I really appreciate.

Finally, and perhaps the most powerful for complex solutions, is the ability to create your own custom filters. You can achieve this using the "Edit filter..." option under the "Filter" dropdown. This option opens a dialog where you can specify complex filter rules, combining various conditions based on name, file extension, directory location, or even the kind of project node. It’s possible to use logical operators like `and`, `or`, and `not` to create very specific filters. I vividly remember when I needed to rapidly check a specific collection of DTO objects, nested within subfolders and spread across several projects, this customizable filter option was indispensable.

To demonstrate these concepts, let’s consider a scenario with a solution structure that might be reminiscent of what you encounter in a typical development environment. Imagine you have a solution with the following (simplified) structure:

```
MySolution
├── MyProjectA
│   ├── Controllers
│   │   └── ApiController.cs
│   ├── Services
│   │   └── MyService.cs
│   └── Models
│       ├── MyModel.cs
│       └── AnotherModel.cs
├── MyProjectB
│   ├── Helpers
│   │   └── Utility.cs
│   ├── Tests
│   │   └── MyTests.cs
│   └── Configuration
│       └── ConfigSettings.json
└── Shared
    ├── DataTransferObjects
    │   └── SharedModel.cs
    └── SharedUtilities
        └── SharedFunction.cs
```

Here are three code snippet examples to illustrate different filtering scenarios:

**Example 1: Simple Name Filtering using Search Box**

Let's say I'm working on the API controller and want to quickly locate the file. By entering `ApiController` in the search bar at the top of the solution explorer, Rider will instantly filter the view to display only `ApiController.cs` within the `Controllers` directory of `MyProjectA`, while hiding everything else in the solution view.

```csharp
//  MyProjectA/Controllers/ApiController.cs (Shown with filter: ApiController)
using Microsoft.AspNetCore.Mvc;

namespace MyProjectA.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class ApiController : ControllerBase
    {
        [HttpGet]
        public IActionResult Get()
        {
            return Ok("API is working");
        }
    }
}
```

**Example 2: Wildcard Pattern Filtering using Search Box**

Now suppose I want to examine all files related to configuration within the project. By using the pattern `Config*` or `*Config*` in the search box, Rider will display `ConfigSettings.json` from `MyProjectB/Configuration`, and as long as there's a file with the word `config` within its name, it would also surface other relevant files if they were to exist within other folders, projects or the solution.

```json
// MyProjectB/Configuration/ConfigSettings.json (Shown with filter: config*)
{
  "ConnectionStrings": {
    "DefaultConnection": "Server=.;Database=MyDb;Trusted_Connection=True;"
  },
  "AppSettings": {
      "ApplicationName": "MyApp"
  }
}
```

**Example 3: Custom Filtering based on file type and folder location using "Edit Filter..." option**

Let's imagine I'm interested in inspecting only files with the extension '.cs' that reside within the `DataTransferObjects` folder (assuming it exists somewhere within the solution structure.)  I would use the "Edit Filter..." option and configure a filter with these conditions: 1) File extension equals '.cs'. 2) Folder path contains 'DataTransferObjects'. Upon applying the custom filter, only `SharedModel.cs` would be displayed, irrespective of which project the file is found in, and everything else would be hidden within the solution explorer view.

```csharp
// Shared/DataTransferObjects/SharedModel.cs (Shown with filter: File extension is .cs and Folder path contains 'DataTransferObjects')

namespace Shared.DataTransferObjects
{
    public class SharedModel
    {
        public int Id { get; set; }
        public string Name { get; set; }
    }
}
```

These examples are, of course, simplified. In real-world scenarios, filters can become arbitrarily complex to accommodate the nuances of your project.

For further reading, I recommend "Working Effectively with Legacy Code" by Michael Feathers, which while not specifically about Rider, covers practices that necessitate efficient file navigation and filtering. Furthermore, the official documentation for JetBrains Rider is your best resource for the most up-to-date and comprehensive information regarding Rider’s feature set. Specifically, look for the sections related to "Solution Explorer" and "Filtering and Search". You will also find the “ReSharper Power Features” section useful for advanced usages.

Mastering these filtering techniques will significantly reduce the time spent navigating large projects, allowing you to focus more on writing and debugging code. This has consistently proven to be valuable across different projects throughout my career, and I hope these insights will be useful to you as well.
