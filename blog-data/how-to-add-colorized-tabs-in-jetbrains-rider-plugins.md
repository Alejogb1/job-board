---
title: "How to add colorized tabs in JetBrains Rider plugins?"
date: "2024-12-16"
id: "how-to-add-colorized-tabs-in-jetbrains-rider-plugins"
---

, let's talk about colorized tabs in JetBrains Rider plugins. I’ve spent a fair amount of time in Rider’s plugin ecosystem, and this specific feature request comes up quite frequently, often from developers wanting to improve their workflow or visually distinguish project elements. The process isn't terribly complicated, but it does involve navigating Rider’s plugin API and understanding how to leverage its UI component system. I’ll guide you through it, leaning heavily on some practical code that I’ve used in the past to achieve this.

Essentially, the goal is to modify the appearance of the tabs within Rider’s editor, likely in the form of adding a color indication. The key here is that we are not directly manipulating the tab’s visual presentation itself. Instead, we’re interacting with Rider's infrastructure, which decides what colors to render for various components. What we can influence is the ‘data’ that Rider uses for rendering. We'll use a `FileEditorManagerListener` to achieve this.

A quick note before diving in - this isn’t about custom drawing or rendering at a low level. We are adhering to the API provided by JetBrains, which means we have to work within their constraints. This approach also keeps the solution stable across different versions of Rider. Directly manipulating UI components is usually a recipe for disaster when software updates arrive.

The general approach is as follows:

1.  **Implement `FileEditorManagerListener`:** This listener allows you to intercept events related to the opening and closing of editors. This is where we will trigger our colorization logic.
2.  **Identify Target Files/Tabs:** We'll use file information or other contextual data to decide which tabs should have a custom color.
3.  **Set Editor Background Color:** We'll use the `EditorColorsManager` and `TextAttributesKey` to add a color to the tab. This isn't changing the text color, but adding a background indicator to the editor tab.

Now let’s take a look at some code snippets. First, we’ll set up our listener. In this instance, I'm assuming we are working in Kotlin since it's the dominant language for Rider plugins, but the same principles will apply for java.

```kotlin
import com.intellij.openapi.editor.colors.EditorColorsManager
import com.intellij.openapi.editor.colors.TextAttributesKey
import com.intellij.openapi.fileEditor.FileEditor
import com.intellij.openapi.fileEditor.FileEditorManager
import com.intellij.openapi.fileEditor.FileEditorManagerListener
import com.intellij.openapi.project.Project
import com.intellij.openapi.vfs.VirtualFile
import java.awt.Color

class ColorizedTabListener(private val project: Project) : FileEditorManagerListener {

    override fun fileOpened(source: FileEditorManager, file: VirtualFile) {
        updateTabColor(file)
    }

    override fun fileClosed(source: FileEditorManager, file: VirtualFile) {
        // Clear the color when the file is closed (optional)
        clearTabColor(file)
    }


    private fun updateTabColor(file: VirtualFile) {
            val colorKey = TextAttributesKey.createTextAttributesKey("MY_CUSTOM_TAB_COLOR")
            val editorColorsManager = EditorColorsManager.getInstance()
            val scheme = editorColorsManager.globalScheme
            val attributes = scheme.getAttributes(colorKey)


            if (shouldColorize(file)) {
                attributes.backgroundColor = Color(200, 230, 200) // Light green
                editorColorsManager.globalScheme.setAttributes(colorKey, attributes)

            }
          else{
                attributes.backgroundColor = null // Clear if file is no longer target
                editorColorsManager.globalScheme.setAttributes(colorKey, attributes)

            }
             // Force a repaint to see changes instantly
             FileEditorManager.getInstance(project).allEditors.forEach {
                 it.component.repaint()
             }
    }

    private fun clearTabColor(file: VirtualFile) {
        val colorKey = TextAttributesKey.createTextAttributesKey("MY_CUSTOM_TAB_COLOR")
        val editorColorsManager = EditorColorsManager.getInstance()
        val attributes = editorColorsManager.globalScheme.getAttributes(colorKey)
        attributes.backgroundColor = null // Remove the background color
        editorColorsManager.globalScheme.setAttributes(colorKey,attributes)

        // Force a repaint to see changes instantly
        FileEditorManager.getInstance(project).allEditors.forEach {
            it.component.repaint()
        }
    }

    private fun shouldColorize(file: VirtualFile): Boolean {
      // example: only color files ending with ".config"
      return file.name.endsWith(".config", ignoreCase = true)
    }
}
```

The above code implements a basic listener. Note the creation of `TextAttributesKey`, this is the way you will identify your color modification. `fileOpened` and `fileClosed` methods are the two relevant to use. The `updateTabColor` method, which we use on the opening, checks using the shouldColorize() example, if we should be changing the background color or not. We are also repainting the editors to show the changes without waiting for a UI refresh. The `clearTabColor` method reverses this, removing the color. Note the `shouldColorize` method is the place where you will implement the logic that drives what files you want to be colored. This could be based on file extension as seen here, or perhaps something more complex.

Next, we need to register our listener with the application:

```kotlin
import com.intellij.openapi.components.ProjectComponent
import com.intellij.openapi.fileEditor.FileEditorManager
import com.intellij.openapi.project.Project

class ColorizedTabProjectComponent(private val project: Project) : ProjectComponent {

    override fun projectOpened() {
        val listener = ColorizedTabListener(project)
        project.messageBus.connect().subscribe(FileEditorManagerListener.FILE_EDITOR_MANAGER, listener)
    }

     //Optional - but good practice to clean up after project closure
    override fun projectClosed() {
       project.messageBus.disconnect()
    }
}
```

This `ProjectComponent` handles the registration of our listener at the opening of the project, and we should also clean it up when the project is closed. This component will also need to be registered in the `plugin.xml` configuration file of the plugin. The exact XML may vary, but a snippet should look something like this:

```xml
<extensions defaultExtensionNs="com.intellij">
      <projectConfigurable groupId="tools" instance="com.example.plugin.ColorizedTabSettingsConfigurable" displayName="Colorized Tabs"/>
      <projectService serviceImplementation="com.example.plugin.ColorizedTabProjectComponent" />
</extensions>
```

This setup gives us a basic, functional, colorization of the tabs, but it's still fairly basic, and we might want to refine it. For example, we could make the color customizable.

Let’s add a simple settings configuration to allow users to choose their tab color:

```kotlin
import com.intellij.openapi.components.PersistentStateComponent
import com.intellij.openapi.components.State
import com.intellij.openapi.components.Storage
import com.intellij.openapi.options.Configurable
import com.intellij.openapi.options.SearchableConfigurable
import com.intellij.ui.ColorPanel
import com.intellij.util.xmlb.XmlSerializerUtil
import java.awt.Color
import javax.swing.JComponent
import javax.swing.JPanel
import java.awt.FlowLayout

@State(name = "ColorizedTabSettings", storages = [Storage("colorized_tabs.xml")])
class ColorizedTabSettings : PersistentStateComponent<ColorizedTabSettings>, Configurable, SearchableConfigurable {

    var tabColor: Color = Color(200, 230, 200) // Default light green


    override fun getState(): ColorizedTabSettings {
        return this
    }

    override fun loadState(state: ColorizedTabSettings) {
        XmlSerializerUtil.copyBean(state, this)
    }

    override fun createComponent(): JComponent {
        val panel = JPanel(FlowLayout(FlowLayout.LEFT))
        val colorPanel = ColorPanel()
        colorPanel.selectedColor = tabColor
        colorPanel.addActionListener {
            tabColor = colorPanel.selectedColor
        }

        panel.add(colorPanel)
        return panel
    }


    override fun isModified(): Boolean {
        return true // always modified as ColorPanel triggers actions even if no real change
    }

    override fun apply() {
        //This is actually done by the apply() of the ColorPanel
    }

    override fun getDisplayName(): String {
        return "Colorized Tabs"
    }


    override fun getId(): String {
      return "colorized.tabs"
    }
    override fun getHelpTopic(): String? {
         return null
     }
}

```

And finally, we must modify our `ColorizedTabListener` to read this setting, instead of a hardcoded color:

```kotlin
import com.intellij.openapi.editor.colors.EditorColorsManager
import com.intellij.openapi.editor.colors.TextAttributesKey
import com.intellij.openapi.fileEditor.FileEditorManager
import com.intellij.openapi.fileEditor.FileEditorManagerListener
import com.intellij.openapi.project.Project
import com.intellij.openapi.vfs.VirtualFile
import com.intellij.openapi.components.service

class ColorizedTabListener(private val project: Project) : FileEditorManagerListener {

    override fun fileOpened(source: FileEditorManager, file: VirtualFile) {
        updateTabColor(file)
    }

    override fun fileClosed(source: FileEditorManager, file: VirtualFile) {
        // Clear the color when the file is closed (optional)
        clearTabColor(file)
    }


    private fun updateTabColor(file: VirtualFile) {
            val colorKey = TextAttributesKey.createTextAttributesKey("MY_CUSTOM_TAB_COLOR")
            val editorColorsManager = EditorColorsManager.getInstance()
            val scheme = editorColorsManager.globalScheme
            val attributes = scheme.getAttributes(colorKey)
            val settings = project.service<ColorizedTabSettings>()

            if (shouldColorize(file)) {
                attributes.backgroundColor = settings.tabColor
                editorColorsManager.globalScheme.setAttributes(colorKey, attributes)

            }
          else{
                attributes.backgroundColor = null // Clear if file is no longer target
                editorColorsManager.globalScheme.setAttributes(colorKey, attributes)

            }
             // Force a repaint to see changes instantly
             FileEditorManager.getInstance(project).allEditors.forEach {
                 it.component.repaint()
             }
    }

    private fun clearTabColor(file: VirtualFile) {
        val colorKey = TextAttributesKey.createTextAttributesKey("MY_CUSTOM_TAB_COLOR")
        val editorColorsManager = EditorColorsManager.getInstance()
        val attributes = editorColorsManager.globalScheme.getAttributes(colorKey)
        attributes.backgroundColor = null // Remove the background color
        editorColorsManager.globalScheme.setAttributes(colorKey,attributes)

        // Force a repaint to see changes instantly
        FileEditorManager.getInstance(project).allEditors.forEach {
            it.component.repaint()
        }
    }

    private fun shouldColorize(file: VirtualFile): Boolean {
      // example: only color files ending with ".config"
      return file.name.endsWith(".config", ignoreCase = true)
    }
}
```

This gives a more complete solution. We now have a setting we can configure in the plugin settings, which gets applied to our listener.

For deeper understanding, I recommend exploring the JetBrains Plugin documentation. Specifically look at `com.intellij.openapi.fileEditor` (for the `FileEditorManager` and related classes), `com.intellij.openapi.editor.colors` (for the coloring mechanism), and `com.intellij.openapi.options` (for the settings API), and of course the excellent examples on the IntelliJ Github repository. Also for a deeper understanding of UI components and how they work in this environment, “Swing” is the underlying library so a deeper understanding of its workings will help.

Keep in mind this solution utilizes the `EditorColorsManager`, which might affect other color settings within the IDE in some edge cases. Always test your implementation thoroughly, especially when sharing the plugin, and make sure any changes to editor colorizations have a valid reason behind them.
