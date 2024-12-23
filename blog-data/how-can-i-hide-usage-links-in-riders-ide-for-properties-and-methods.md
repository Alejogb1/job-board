---
title: "How can I hide 'usage' links in Rider's IDE for properties and methods?"
date: "2024-12-23"
id: "how-can-i-hide-usage-links-in-riders-ide-for-properties-and-methods"
---

Okay, let's tackle this. It's a specific annoyance, isn't it? Those "usage" links, while helpful in some contexts, can quickly clutter the visual space, especially when you're working on very frequently accessed properties or methods. I remember once, a few years back, on a large .net project involving complex business logic and numerous interdependencies, the constant stream of usage indicators was actually hindering more than helping. I found myself spending precious mental cycles filtering them out rather than focusing on the code itself. After some rather deep configuration dives and a few chats with the JetBrains support team (they’re pretty sharp, by the way), I arrived at a decent solution.

Now, it’s not about completely removing the functionality, as that would throw out the baby with the bathwater. Instead, we're aiming for a selective approach to manage when and how these usages are displayed. The key is understanding the fine-grained control that Rider offers. It's not a straightforward checkbox option, but rather a combination of settings that you need to manipulate.

The primary mechanism we’ll utilize is the “Code Vision” feature. Code vision, in essence, is Rider's system for providing contextually relevant information directly within the editor window, and these usage links are one facet of it. Thankfully, it is quite customizable. Instead of a brute-force approach, we can target specific code elements.

Let's start with property usage links. I've found three effective ways of dealing with them. Firstly, and perhaps the simplest, is to globally disable the "usages" indicator for members (which includes properties) through the settings. You can navigate to `File > Settings (or Rider > Preferences on macOS) > Editor > Inlay Hints > C# > Code Vision`. Here, you'll find a section labeled “Usages” and a checkbox next to “Members.” Unchecking this will, globally, remove the number of usages from properties and methods. Note that this applies to members within the class itself. This is a very impactful change if you simply don't need it on that level.

However, the nuclear option of disabling them for all members is not always ideal; sometimes you *do* need them, just not on, say, those ubiquitous getter/setters. For that finer level of control, consider utilizing code analysis rules in conjunction with severity settings. This allows you to suppress the display of these usage links on a case-by-case basis, based on the code structure. We achieve this by creating a custom suppression that essentially tells Rider to not show the usage of specific property types or specific named properties. This approach is significantly more nuanced.

Finally, for a dynamic approach, you can create your own custom annotations and code inspections, though this involves a more in-depth understanding of Rider’s API. I'll walk through the first two more in detail.

**Example 1: Globally Disabling Member Usages**

As I've mentioned before, it's the most direct approach and involves no code changes. You just have to tweak the IDE's settings. As a very quick example of how it works:

1.  Go to `File > Settings` (or `Rider > Preferences` on Mac).
2.  Navigate to `Editor > Inlay Hints > C# > Code Vision`.
3.  Uncheck the checkbox next to "Members" under the "Usages" section.
4.  Click "Apply" or "OK."

This will disable the count of usages globally, as in, on all classes and members, which might be what you are looking for.

**Example 2: Targeted Suppression using Code Analysis**

This method provides more control by using Rider's code analysis features, which allow us to selectively suppress usage indicators. It might sound a bit complex at first, but it's actually quite powerful. To do this, you will need to configure an inspection profile that will target specific usage cases.

```csharp
// Example C# code:
public class DataContainer
{
    public int Id { get; set; } // We might want to hide usage on this one
    public string Name { get; set; } // This, however, we want to keep the usage count displayed.

    public void ProcessData()
    {
        Console.WriteLine(Id); // This usage of Id, of course, is still present and still functional.
        Console.WriteLine(Name);
    }
}
```

In the example above, I'll show you how to suppress usage links for the `Id` property while maintaining them for the `Name` property. Here's how:

1.  Go to `File > Settings` (or `Rider > Preferences` on macOS).
2.  Navigate to `Editor > Inspection Settings > Inspection Severity`.
3.  Search for `Unused member in type`. You'll likely want to create a copy profile first if you intend to change these settings permanently.
4.  Find `Unused parameter, field, or property` within the tree.
5.  In the right pane, look for the "Options" section.
6.  Click the plus (+) button in the "Suppress for:" section and create a new suppression.
7.  You can specify a custom rule. For instance, you might add a rule such as `Property: Id`. You can also specify by types, in case you have, for example, standard "Id" type across multiple classes.
8.  Then, change its severity to "Do not show."
9.  Click "Apply" or "OK".

Now, usage links will *not* show for the `Id` property, while usages on `Name` are still shown. Notice that the functionality of the property and the usage in the method remain completely unaffected; only the visualization of the usages is modified.

**Example 3: Custom Code Annotation (Advanced)**

This approach is considerably more involved and requires you to work with Rider's extensibility model. You'll need to create a custom plugin or use an existing one to define your own attributes or annotations and code inspections. This is beyond the scope of this response and is usually necessary when the built-in system is not enough or you need something very particular. However, this is the most powerful method and is something to keep in mind.

To delve further into this, I'd recommend reading JetBrains' official documentation on Rider's plugin API. Specifically, study how to create custom annotations, inspections, and settings pages. Also, "ReSharper Plugin Development" by Matt Ellis, if you are using Resharper rather than Rider as the backend, is a great resource. Another authoritative text for custom code analysis is "Code Analysis: The Static Analysis Toolkit" by O'Callahan.

In my experience, the second method, leveraging the suppression rules, provides the best balance of control and ease of implementation for the majority of cases. It gives you the ability to selectively hide usage links where they are disruptive, while retaining them in locations where they offer genuine value. It's always about finding that sweet spot where the tools work *for* you, not against you.

This way, I was able to clean up my editor view significantly and reduce the mental load. Remember, the goal of any tool is to enhance your productivity; when it becomes a distraction, it's time to reconfigure it. These strategies should hopefully offer you the granularity needed to manage those "usage" links effectively.
