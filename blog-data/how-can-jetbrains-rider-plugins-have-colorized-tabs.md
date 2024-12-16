---
title: "How can JetBrains Rider plugins have colorized tabs?"
date: "2024-12-16"
id: "how-can-jetbrains-rider-plugins-have-colorized-tabs"
---

Alright, let's talk about colored tabs in JetBrains Rider plugins. It's a feature I've had to implement a few times, most notably during a rather complex project involving microservice orchestration, where having visually distinct tabs for each service configuration file was invaluable. The visual organization significantly reduced context switching overhead, so it's definitely a worthwhile feature to implement.

The core concept revolves around the `EditorColorsScheme` and the `EditorFileTypeColorProvider` extension point within the Rider plugin architecture. Instead of modifying the base UI directly, which is generally not recommended and could break with future updates, we hook into the theming system to define how our custom file types should be rendered, including their tab colors.

I've often seen newer plugin developers try to directly manipulate the UI components, which, while tempting, usually leads to brittle code and potential conflicts. The established mechanism for this is to leverage the extension points, giving JetBrains control over its appearance while still allowing us the flexibility to define our custom integrations.

So, let's break down how you can achieve colored tabs, step-by-step. The process involves essentially two main components: defining a new file type and then assigning a color scheme to it. This is a consistent pattern across JetBrains IDEs, making it a transferable skill.

First, we need to define the file type our plugin will be working with. Let’s assume you're creating a plugin to manage some custom configuration files with a `.cfg` extension. You need to create an implementation of `FileTypeFactory` and `FileType`:

```csharp
using JetBrains.Application.PluginSupport;
using JetBrains.ProjectModel;
using JetBrains.ReSharper.Psi;

[PluginComponent]
public class CustomFileTypeFactory : FileTypeFactory
{
    public override FileType CreateFileType(string name)
    {
      if (name == "CustomFileType")
          return CustomFileType.Instance;
      return null;
    }
}


public class CustomFileType : FileType
{
  public static readonly CustomFileType Instance = new CustomFileType();

  private CustomFileType() : base("CustomFileType", "Custom CFG File") {}
    public override string DefaultExtension => "cfg";

    public override PsiLanguageType Language => PsiLanguageType.Unknown;
}

```

In this snippet, we've defined a new file type named `CustomFileType` with a default file extension of `.cfg`. Note that the `PsiLanguageType` is set to `Unknown`, since in this example we're not dealing with any specific language parsing, rather just the file type classification. If you have custom syntax, you can create your own language types instead.

Next, we'll implement `EditorFileTypeColorProvider`. This is where we actually define the color scheme for our custom file type.

```csharp
using JetBrains.Application.Settings;
using JetBrains.ProjectModel;
using JetBrains.TextControl.Document;
using JetBrains.TextControl;
using JetBrains.ReSharper.Feature.Services.Text;
using JetBrains.ReSharper.Feature.Services.Options;
using JetBrains.DataFlow;
using System.Drawing;
using JetBrains.UI.Theming;

[SolutionComponent]
public class CustomFileTypeColorProvider : EditorFileTypeColorProvider
{
    private readonly IContextBoundSettingsStore settingsStore;
    private readonly Lifetime lifetime;

    public CustomFileTypeColorProvider(Lifetime lifetime, IContextBoundSettingsStore settingsStore)
    {
        this.settingsStore = settingsStore;
        this.lifetime = lifetime;
    }

    public override bool IsApplicable(IDocument document, ITextControl textControl)
    {
        if (document == null) return false;
        var projectFile = document.GetProjectFile(settingsStore);

        return projectFile != null && projectFile.LanguageType == CustomFileType.Instance;
    }


    public override EditorColorsSchemeOverride? GetColorSchemeOverride(IDocument document, ITextControl textControl)
    {
        if (!IsApplicable(document, textControl)) return null;

        return new EditorColorsSchemeOverride(
          new EditorColorSchemeOverrides(
              new TextControlColors(new Color?(Color.FromArgb(204, 255, 204)), null)));
    }
}
```

Here, the `IsApplicable` method ensures that this color provider is only activated for files of our custom type. `GetColorSchemeOverride` then returns an `EditorColorsSchemeOverride`, which provides `TextControlColors`. The `TextControlColors` lets us specify the background color of the editor tab associated with files that are matched by the `IsApplicable` method. In this example, we’re setting the tab background color to a light green `(204, 255, 204)`.

It is important to note that the `EditorColorsSchemeOverride` is a fairly rich object that allows setting not only tab colors but also colors of text, carets, and other visual elements in a controlled way. The example here is focused on changing the tab color only.

For more granular control, you can make the color configurable via plugin settings, giving users the flexibility to customize the color to their liking. This also involves creating a settings class and a UI to edit the settings. This is, however, beyond the scope of this specific question, but a very common practice. The principle for achieving colored tabs still remains the same though.

And finally, you need to declare your component into the plugin xml file, in this case `plugin.xml`, like this:

```xml
<applicationComponents>
    <component>
        <implementation-class>YourPluginNamespace.CustomFileTypeFactory</implementation-class>
    </component>
</applicationComponents>
<solutionComponents>
    <component>
        <implementation-class>YourPluginNamespace.CustomFileTypeColorProvider</implementation-class>
    </component>
</solutionComponents>
```

This declares the two classes from the code examples as Rider components and makes them available to the Rider engine.

To reiterate, you can find more in-depth information about editor customization in JetBrains IDEs through the official documentation. For a broader understanding of the plugin development architecture, I'd highly recommend *Developing Plugins for JetBrains IDEs*, an official resource that goes into significant detail about the various facets of plugin creation. You can also benefit from exploring the ReSharper SDK samples, which often provide working examples of these mechanisms.

Implementing these changes, as I have demonstrated, provides you with a mechanism to not only differentiate different file types based on the kind of document they represent but also gives you the control to provide a more user-friendly IDE experience, in a way that integrates naturally within the existing JetBrains ecosystem. The above approach is a far more scalable and maintainable methodology, compared to direct UI manipulation, and something that I've come to rely on heavily over the years in my work with IDE extension development.
