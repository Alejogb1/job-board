---
title: "How can I add colorized tabs in a JetBrains Rider plugin?"
date: "2024-12-23"
id: "how-can-i-add-colorized-tabs-in-a-jetbrains-rider-plugin"
---

Alright, let’s tackle this. Colorized tabs in Rider plugins are more than just eye candy; they can significantly enhance usability, especially when you're juggling numerous projects or dealing with different file types. From my experience, maintaining several large codebases simultaneously, this becomes crucial for quickly locating the context you need. The key lies in manipulating Rider’s UI components, specifically utilizing the `EditorTab` and the `EditorFileType` associated with it. It’s a bit more involved than just setting a background color property, though. We need to get into the internals of the editor view.

The challenge isn't about finding an API call to magically change the color. Rather, it’s about correctly hooking into Rider’s event system and appropriately modifying the appearance of these UI elements. This involves extending Rider's internal interfaces and leveraging its component model. Now, there isn't one single class you target, rather a series of hooks, and it’s important to get them in the correct order. It’s analogous to setting up a complex event pipeline; if you don’t have the correct wiring, nothing happens.

The basic strategy is this: you’ll need to implement a service that listens for editor tab changes or creates, then, based on some logic (usually related to the `EditorFileType` of the open file), you'll adjust the appearance of the tab. This generally involves accessing the underlying `Presentation` object of the `EditorTab`. Let’s break it down into specific code snippets.

**Example 1: Basic Tab Colorization Based on File Extension**

This first example shows how to change the tab color based on the file extension. We’ll be targeting .cs files specifically here, and setting their tab background to a light blue.

```csharp
using JetBrains.Application.Components;
using JetBrains.Application.Settings;
using JetBrains.DataFlow;
using JetBrains.DocumentModel;
using JetBrains.Lifetimes;
using JetBrains.ProjectModel;
using JetBrains.ReSharper.Psi;
using JetBrains.ReSharper.Resources.Shell;
using JetBrains.TextControl;
using JetBrains.UI.Icons;
using JetBrains.UI.Presentation.Framework;
using JetBrains.UI.RichText;
using System;
using System.Windows.Media;

[SolutionComponent]
public class CustomTabColorizer : IDisposable
{
    private readonly Lifetime _lifetime;
    private readonly ITextControlManager _textControlManager;
    private readonly IUIApplication _uiApplication;

    public CustomTabColorizer(Lifetime lifetime, ITextControlManager textControlManager, IUIApplication uiApplication)
    {
        _lifetime = lifetime;
        _textControlManager = textControlManager;
        _uiApplication = uiApplication;
        _textControlManager.TextControls.View(lifetime, OnTextControlView);
    }


    private void OnTextControlView(Lifetime lt, ITextControl textControl)
    {
        if (textControl.Document == null) return;
        var sourceFile = textControl.Document.GetPsiSourceFile(Solution);
        if (sourceFile == null) return;

        lt.Bracket( () => {
            textControl.View.EditorTab.Presentation.ComputedBackground.Advise(lt, UpdateTabColor);
           }, () => {
            // do nothing
           } );

        UpdateTabColor(textControl.View.EditorTab.Presentation.ComputedBackground.Value);
    }

    private void UpdateTabColor(Color color)
    {
      using (ReadLockCookie.Create())
        {
            if (Solution is null) return;

            var textControl = _textControlManager.FocusedTextControl.Value;

            if (textControl == null) return;


            var sourceFile = textControl.Document.GetPsiSourceFile(Solution);
             if (sourceFile == null) return;

            if (sourceFile.LanguageType.Is<CSharpLanguage>())
            {
              _uiApplication.Dispatcher.InvokeAsync(() => textControl.View.EditorTab.Presentation.Background.Value = new SolidColorBrush(Colors.LightBlue));
           }
            else {
              _uiApplication.Dispatcher.InvokeAsync(() => textControl.View.EditorTab.Presentation.Background.Value = null);
            }


        }

    }
    public ISolution? Solution { get; set; }


    public void Dispose()
    {
      // do nothing
    }

}
```

In this snippet, we register a `SolutionComponent`. We subscribe to the textcontrolmanager's events, listening for `TextControls.View` event, when a new editor view is made available. When one is ready, we grab its `EditorTab` and then, using the `ComputedBackground` we subscribe to advise an action to change the tab color. We finally invoke the update method to set the color based on the detected file type, in this case a `.cs` file. Notice that we leverage the dispatcher to ensure we’re touching the UI on the UI thread. Also note that you need to set `Solution` before using this service, as I have marked as nullable and it will not work if it's null.

**Example 2: Using a Settings Layer for Configurable Colors**

Now, let's enhance our approach and make the colors configurable. This example assumes you have a settings schema already defined (which is a different topic, but essential for user customization). We are using settings to get the color values instead of hard coding them.

```csharp
using JetBrains.Application.Components;
using JetBrains.Application.Settings;
using JetBrains.DataFlow;
using JetBrains.DocumentModel;
using JetBrains.Lifetimes;
using JetBrains.ProjectModel;
using JetBrains.ReSharper.Psi;
using JetBrains.ReSharper.Resources.Shell;
using JetBrains.TextControl;
using JetBrains.UI.Icons;
using JetBrains.UI.Presentation.Framework;
using JetBrains.UI.RichText;
using System;
using System.Windows.Media;

[SolutionComponent]
public class ConfigurableTabColorizer : IDisposable
{
    private readonly Lifetime _lifetime;
    private readonly ITextControlManager _textControlManager;
    private readonly IUIApplication _uiApplication;
    private readonly IContextBoundSettingsStore _settings;

    public ConfigurableTabColorizer(Lifetime lifetime, ITextControlManager textControlManager, IUIApplication uiApplication, IContextBoundSettingsStore settings)
    {
        _lifetime = lifetime;
        _textControlManager = textControlManager;
        _uiApplication = uiApplication;
        _settings = settings;
        _textControlManager.TextControls.View(lifetime, OnTextControlView);
    }

  private void OnTextControlView(Lifetime lt, ITextControl textControl)
    {
        if (textControl.Document == null) return;
        var sourceFile = textControl.Document.GetPsiSourceFile(Solution);
        if (sourceFile == null) return;

        lt.Bracket(() => {
          textControl.View.EditorTab.Presentation.ComputedBackground.Advise(lt, UpdateTabColor);
        }, () => {
          // do nothing
        });

      UpdateTabColor(textControl.View.EditorTab.Presentation.ComputedBackground.Value);
    }

  private void UpdateTabColor(Color color)
  {
    using (ReadLockCookie.Create())
    {
      if (Solution is null) return;

      var textControl = _textControlManager.FocusedTextControl.Value;

      if (textControl == null) return;


      var sourceFile = textControl.Document.GetPsiSourceFile(Solution);
      if (sourceFile == null) return;

      if (sourceFile.LanguageType.Is<CSharpLanguage>())
      {
        var settingsValue = _settings.GetValue(MySettingsSchema.CSharpTabColor);
        _uiApplication.Dispatcher.InvokeAsync(() => textControl.View.EditorTab.Presentation.Background.Value = new SolidColorBrush(settingsValue));

      }
      else
      {
          _uiApplication.Dispatcher.InvokeAsync(() => textControl.View.EditorTab.Presentation.Background.Value = null);

      }


    }
  }

    public ISolution? Solution { get; set; }

    public void Dispose()
    {
      // do nothing
    }
}
```

Here, we inject `IContextBoundSettingsStore` and retrieve the color from the settings using `_settings.GetValue(MySettingsSchema.CSharpTabColor)`. The `MySettingsSchema` is assumed to be the settings schema you have defined. This makes the plugin more versatile. You’ll need to adjust `MySettingsSchema` and the name of the setting according to your implementation. You will also need a settings schema defined, but that is out of the scope of this answer.

**Example 3: Differentiating Tab Colors by Project**

Finally, let’s consider a scenario where different projects need different tab colors, which can be really helpful when navigating across multiple open projects. This uses project guids to differentiate the colors.

```csharp
using JetBrains.Application.Components;
using JetBrains.Application.Settings;
using JetBrains.DataFlow;
using JetBrains.DocumentModel;
using JetBrains.Lifetimes;
using JetBrains.ProjectModel;
using JetBrains.ReSharper.Psi;
using JetBrains.ReSharper.Resources.Shell;
using JetBrains.TextControl;
using JetBrains.UI.Icons;
using JetBrains.UI.Presentation.Framework;
using JetBrains.UI.RichText;
using System;
using System.Windows.Media;

[SolutionComponent]
public class ProjectSpecificTabColorizer : IDisposable
{
  private readonly Lifetime _lifetime;
  private readonly ITextControlManager _textControlManager;
  private readonly IUIApplication _uiApplication;

  public ProjectSpecificTabColorizer(Lifetime lifetime, ITextControlManager textControlManager, IUIApplication uiApplication)
  {
    _lifetime = lifetime;
    _textControlManager = textControlManager;
    _uiApplication = uiApplication;
    _textControlManager.TextControls.View(lifetime, OnTextControlView);
  }

  private void OnTextControlView(Lifetime lt, ITextControl textControl)
  {
    if (textControl.Document == null) return;
    var sourceFile = textControl.Document.GetPsiSourceFile(Solution);
    if (sourceFile == null) return;

    lt.Bracket(() =>
    {
      textControl.View.EditorTab.Presentation.ComputedBackground.Advise(lt, UpdateTabColor);
    }, () =>
    {
      //do nothing
    });
    UpdateTabColor(textControl.View.EditorTab.Presentation.ComputedBackground.Value);

  }
  private void UpdateTabColor(Color color)
  {
    using (ReadLockCookie.Create())
    {
      if (Solution is null) return;

      var textControl = _textControlManager.FocusedTextControl.Value;

      if (textControl == null) return;

      var sourceFile = textControl.Document.GetPsiSourceFile(Solution);
      if (sourceFile == null) return;

      var project = sourceFile.GetProject();

      if (project == null) return;

      Color tabColor = GetColorForProject(project.Guid);

      _uiApplication.Dispatcher.InvokeAsync(() => textControl.View.EditorTab.Presentation.Background.Value = new SolidColorBrush(tabColor));

    }
  }


  private Color GetColorForProject(Guid projectGuid)
  {
      // Here, you can implement logic to map project guids to different colors
      if (projectGuid == new Guid("Your-Project-Guid-Here"))
      {
          return Colors.LightGreen;
      }
      else if (projectGuid == new Guid("Another-Project-Guid-Here"))
      {
          return Colors.LightSalmon;
      }
      else
      {
          return Colors.White; // Default color
      }
  }


  public ISolution? Solution { get; set; }
  public void Dispose()
  {
    // do nothing
  }
}
```
This snippet adds a `GetColorForProject` method that takes the project’s `Guid` and maps it to a specific color, using a simple `if-else` for illustrative purposes. This can be expanded using a more complex mapping such as a Dictionary. Remember to replace the dummy Guids with your actual project guids.

**Key takeaways and recommendations:**

*   **Performance:** Be careful not to perform too much work on the UI thread. Do your file type and project checking under a read lock, or in a non ui thread and only update the UI when needed with the dispatcher.
*   **Event Handling:** Always make sure to properly dispose of subscriptions to ensure no memory leaks or performance issues. Use `Lifetime.Bracket` to ensure the clean disposal of event subscriptions.
*   **Settings Integration:** For real-world plugins, always make color customization options available to the user via Rider's settings dialog.
*   **Error Handling:** Handle edge cases such as null checks for document, source file, and project references.
*   **Further Reading:** To deepen your understanding, I strongly recommend consulting the following resources:
    *   **"JetBrains Platform SDK Documentation"**: This is the primary source for all things related to the IntelliJ platform development, including Rider plugin development.
    *   **"Effective Java" by Joshua Bloch**: Though not specific to plugin development, this book will teach you best practices, especially around resource management and thread safety that applies to any software design including plugins.
    *   **"Design Patterns: Elements of Reusable Object-Oriented Software" by Erich Gamma et al**: Understanding design patterns, particularly the observer pattern will help you structure your code better when subscribing to event streams of the rider framework, which the snippets above are using.

Implementing colorized tabs adds a powerful visual cue to your Rider plugin, greatly improving user experience. It does take a bit of effort to understand the internals and correctly hook into the correct events, but the result is well worth the effort. You can enhance this approach further based on the project, file type and any other business logic you have, provided you have a firm understanding of how to interact with Rider's UI and events system.
