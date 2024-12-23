---
title: "How does the JetBrains Marketplace determine IDE version compatibility?"
date: "2024-12-23"
id: "how-does-the-jetbrains-marketplace-determine-ide-version-compatibility"
---

Alright, let's break down how JetBrains Marketplace tackles the ever-present challenge of plugin compatibility with varying IDE versions. It's a complex dance, and I've certainly seen my fair share of missteps during my time building and maintaining plugins, which gives me a good perspective on this. It's not simply a matter of checking a version number; there's quite a bit more going on under the hood.

Fundamentally, the process revolves around several layers of metadata and validation that JetBrains utilizes both during plugin submission and during the installation phase on a user's IDE. This system is designed to ensure a reasonable level of stability and prevent those dreaded "plugin incompatible" error messages from appearing too frequently.

First, let's talk about how compatibility is *specified*. At the heart of each plugin is the `plugin.xml` file. This file contains a plethora of information, and crucial to our discussion is the `<idea-version>` tag. This tag uses attributes to specify compatible versions. You typically see one or more of these attributes:

*   `since-build`: Specifies the lowest build number that the plugin is compatible with. This is crucial for ensuring a plugin isn't attempting to use APIs that don't exist in older versions.
*   `until-build`: Defines the highest build number the plugin supports. This prevents installation on IDE versions where significant API changes could lead to problems.
*   `platform-version`: This attribute is less commonly used but specifies a particular platform version, mainly when you're targetting specific IDEs. It's useful for ensuring plugins are installed only on the right IDE (e.g., IntelliJ vs. PyCharm).

These attributes don’t just check major version numbers like 2023.3 or 2024.1, but rather uses internal build numbers. For example, `233.11799.241` or similar build number which can then be matched appropriately. Each IDE release uses the `<since-build>` and `<until-build>` number in its own core configuration.

Now, let me give you a practical example from when I developed a plugin for code analysis. Let's say a certain API feature I was utilizing was introduced in build number `201.6668` and had breaking changes in `202.6948`. The relevant part of the plugin.xml would look similar to this:

```xml
<idea-plugin>
    <id>com.example.mycodeanalyzer</id>
    <name>My Code Analyzer</name>
    <version>1.0.0</version>
    ...
    <idea-version since-build="201.6668" until-build="202.6947" />
    ...
</idea-plugin>
```

In this example, `since-build="201.6668"` indicates the plugin requires build number `201.6668` or any later build number within the same branch. The `until-build="202.6947"` means any IDE build numbered `202.6947` or before is also compatible, but not `202.6948` or any later build number in this example. This is crucial since the changes introduced in build `202.6948` could break the plugin completely.

When a user attempts to install a plugin through the Marketplace, the Marketplace client (integrated in the IDE) examines this `<idea-version>` tag. It compares the IDE’s build number with the specified ranges in the plugin’s `plugin.xml`. If the IDE’s build number falls outside of the defined range, the plugin is flagged as incompatible. This happens *before* the plugin is even downloaded, meaning the user is alerted before potentially destabilizing their IDE installation.

Another crucial element of this process occurs during the *plugin verification* stage before plugin publication to the JetBrains Marketplace. JetBrains runs a series of compatibility checks automatically based on the given `<since-build>` and `<until-build>` numbers within plugin’s descriptor. If the declared versions are in conflict or there is an internal issue with the plugin’s structure that may affect functionality and stability, then plugin verification might fail.

This prevents developers from publishing plugins that have the potential to break user IDE installations. The verification process also catches plugin authors who might be tempted to skip setting the ranges correctly or who are not aware of breaking API changes in newer versions.

Here's another practical example. Suppose I was working on a plugin targeting multiple different IDEs and their platforms, perhaps the previous code analysis plugin, and I was supporting older and newer versions, my `plugin.xml` would contain multiple `<idea-version>` elements to handle various platforms:

```xml
<idea-plugin>
    <id>com.example.mycodeanalyzer</id>
    <name>My Code Analyzer</name>
    <version>1.0.1</version>
    ...
    <idea-version since-build="193.3722" until-build="201.6667" platform-version="IC"/>
    <idea-version since-build="201.6668" until-build="202.6947" platform-version="IC"/>
    <idea-version since-build="201.6668" until-build="202.6947" platform-version="PY"/>
	<idea-version since-build="202.6948" platform-version="IC"/>
    ...
</idea-plugin>
```

Here, the plugin supports multiple IDEs. `platform-version="IC"` specifies IntelliJ IDEA, and `platform-version="PY"` is for PyCharm. Notice that the `until-build` is absent in the last entry, as this specifies that the plugin is compatible up to the latest IDE version based on `since-build`, this covers everything greater than `202.6948` for IntelliJ IDE builds. This granularity is essential to managing complex compatibility scenarios.

Another layer to this is the handling of major IDE upgrades. During a major IDE update, JetBrains provides a grace period where plugins are generally marked as "compatible" if they don't explicitly declare an incompatibility with that major release. However, there is no guarantee the plugin will function properly because JetBrains cannot automatically identify breaking changes in plugin behavior.

This is a complex system, and it is the responsibility of each plugin author to understand API changes that may break their plugins. JetBrains relies on a combination of automated checks and good practice on the part of plugin developers to maintain stability.

The system isn't foolproof, though. Incorrectly configured `<idea-version>` tags can lead to users either being unable to install the plugin on a compatible version or, perhaps worse, allowing plugins to install on an incompatible version leading to errors. Thus plugin developers must thoroughly test their plugin on a range of IDE versions and be diligent in updating the `<idea-version>` tag. For example if a plugin makes use of a new IDE API it must properly declare this in the plugin.xml by updating the `<since-build>` attribute.

Here’s a final example illustrating what happens with API deprecation and necessary versioning changes. Assume my code analysis plugin has been updated to use a new improved API and as such, my old plugin is not compatible with the latest IDE updates.

```xml
<idea-plugin>
    <id>com.example.mycodeanalyzer</id>
    <name>My Code Analyzer</name>
    <version>1.1.0</version>
     ...
    <idea-version since-build="241.14494.150"/>
   ...
</idea-plugin>
```

In this case the code analysis plugin now only supports IDE build numbers `241.14494.150` and greater, assuming that any earlier versions of the IDE are now incompatible with the changes introduced in this version, likely due to a breaking API change. Therefore, plugin authors must carefully monitor release notes for new IDE features and update their plugin to support such changes.

For further reading on this subject, I'd highly recommend referring to the official JetBrains Plugin Development documentation which contains sections on the `plugin.xml` file, including those specifically discussing the `<idea-version>` tag attributes and verification processes. It is also highly beneficial to study "Effective Java" by Joshua Bloch to understand robust API design principles and versioning, which are core aspects of avoiding compatibility issues in the first place.

In summary, the compatibility mechanism implemented by the JetBrains Marketplace is a robust system but it requires careful configuration on the part of plugin authors and developers. It is a constantly evolving ecosystem and it takes time and effort to maintain compatibility.
