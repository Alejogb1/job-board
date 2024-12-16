---
title: "How can I filter the Rider Explorer Solution View?"
date: "2024-12-16"
id: "how-can-i-filter-the-rider-explorer-solution-view"
---

Okay, let's tackle this one. I've spent a fair amount of time navigating large codebases, and the ability to effectively filter the Rider Explorer Solution View is a lifesaver, plain and simple. It’s one of those subtle features that dramatically impacts productivity once you understand its capabilities. I recall a particularly convoluted project involving a microservices architecture, where the solution contained literally hundreds of projects. Without solid filtering, it felt like trying to find a specific grain of sand on a beach.

The beauty of Rider's filtering mechanism isn't just about reducing visual clutter; it’s about providing you with surgical precision when focusing on specific parts of your solution. It lets you surface only what's relevant to your current task. There are several methods and levels of refinement available, each serving a distinct purpose. I'll walk you through the core approaches I've found most effective, with code examples to illustrate how these techniques work with specific project structures.

Firstly, and most directly, Rider provides what I'd call **Basic String Matching**. You can start typing in the search box at the top of the solution explorer and Rider immediately begins filtering based on what you’ve entered. This filtering is ‘live’ meaning it instantly updates as you type. This is helpful for isolating projects with specific names. For example, if I'm looking for projects related to ‘authentication’ in a hypothetical e-commerce application, typing "auth" or even just "aut" into the filter box will quickly show projects such as `AuthenticationService`, `AuthDataAccess`, and so on. This utilizes a simple substring search and is case-insensitive.

Here's the first code example to illustrate a hypothetical project structure that this would work with:

```
Solution
│
├───ShoppingCart.Core
│       ├───Models
│           ├── Order.cs
│           └── Product.cs
│
├───ShoppingCart.Services
│    └──  OrderService.cs
│    └── ProductService.cs
│
├───Authentication.Api
│    └──  AuthenticationController.cs
│    └──  UserService.cs
│
└───UserInterface
       └── MainForm.cs
       └──  LoginView.cs
```

Using a simple string filter like ‘Auth’ would immediately highlight `Authentication.Api` and the contained files. It's rudimentary, but remarkably powerful for quick navigation.

However, basic string matching only scratches the surface. Rider’s filtering capabilities go beyond simple substring matching by incorporating what is essentially a pattern matching system, that allows for the use of wildcards and more advanced filtering expressions. This includes wildcard characters like * and ?, allowing you to search for files based on more flexible patterns. For example, let’s say you need to isolate all tests. In a well-structured project, you’d likely see test projects ending in "Tests" or "Test" For this scenario, filtering with something like `*Test` or `*Tests` is exceptionally efficient.

Let's use another hypothetical project structure, where tests are part of each module:

```
Solution
│
├───Calculator.Core
│   ├── Calculator.cs
│   └── CalculatorTests.cs
│
├───Payment.Services
│    ├── PaymentService.cs
│    └── PaymentServiceTests.cs
│
├───Reporting.Infrastructure
│    ├── DataReport.cs
│    └──  ReportTests.cs
│
└───Ui.Components
    └── UIComponent.cs
```

Here, the filter `*Test` or `*Tests` will select only `CalculatorTests.cs`, `PaymentServiceTests.cs`, and `ReportTests.cs`. This allows developers to isolate test files, even when those files are present within many different folders. Now, that's a substantial improvement over simple text filtering. This also becomes very useful in a project where you have both integration and unit tests structured using the same naming convention, where you can easily select all your test files for a particular type of test with a pattern such as `*IntegrationTests`.

Moving a step further, filtering in Rider also understands hierarchical elements, and it can leverage project and folder structures for precise filtering using what's essentially a path-based syntax. You can, for example, filter by both a part of the file name and its enclosing folder. Imagine you want all files named “Service” which exist directly under your `Services` or `API` directory. This kind of operation is very frequently necessary in larger projects, and Rider offers an intuitive way of doing this. Let's imagine the following folder structure:

```
Solution
│
├───CustomerModule
│   ├── Api
│      └── CustomerApiService.cs
│   └── Services
│       └── CustomerService.cs
│
├───ProductModule
│    ├── Api
│        └── ProductApiService.cs
│    └── Services
│        └── ProductService.cs
│
└───OrderModule
    ├── Api
    │    └── OrderApiService.cs
    └── Services
        └── OrderService.cs

```
To find all files which include ‘Service’ in their name and which are direct descendants of the folder `Services`, I can utilize `Services/Service`, or to specifically find all files with ‘Service’ inside ‘Api’ directories, I would use `Api/Service`. This becomes a very targeted search and greatly reduces the time taken to locate and navigate a particular resource. I have personally seen situations where even with good naming convention, file hierarchies are deep and complex, and this functionality makes it a game changer.

The documentation on the JetBrains website provides a wealth of additional detail on the full syntax and capabilities. Specifically, I highly recommend reading the Rider documentation section on ‘Solution Explorer’ and ‘Search Everywhere’. Beyond that, the book "Refactoring: Improving the Design of Existing Code" by Martin Fowler has relevant concepts about how project structures benefit from a clearly defined module structure which can be navigated with such tools, and is therefore highly relevant for getting the best results out of any IDE's filtering functionalities. Additionally, "Clean Code: A Handbook of Agile Software Craftsmanship" by Robert C. Martin gives more insights into organizing your project in a clean manner which benefits from targeted filtering when you need to access a specific set of related resources.

In closing, mastering Rider's filtering within the Solution Explorer is not a trivial matter, but it is a skill that repays the effort many times over. It provides a substantial boost to developer productivity, by facilitating quick and targeted navigation and reducing cognitive load, which allows for greater concentration on tasks at hand. The ability to swiftly filter, either by simple text matching, using wildcards, or through paths, drastically reduces time spent searching, allowing for significantly faster development and debugging. It’s a feature that truly distinguishes effective use of an IDE from mere usage.
