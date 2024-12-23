---
title: "How do I exclude files from change lists in JetBrains?"
date: "2024-12-16"
id: "how-do-i-exclude-files-from-change-lists-in-jetbrains"
---

Alright,  I've seen this exact issue crop up quite a few times across different projects, and while JetBrains IDEs offer a fairly robust change list system, it's definitely not always intuitive how to effectively exclude files. Especially when you’re dealing with automatically generated files or temporary build artifacts cluttering your diffs. It's a common annoyance, and thankfully, solvable with a few different approaches. I’ll walk you through the methods that have served me well over the years.

The key thing to understand is that JetBrains IDEs, like IntelliJ IDEA, PyCharm, or WebStorm, manage changes through change lists and the underlying version control system (typically Git, but also Mercurial or others). Excluding a file from a change list does *not* mean it's removed from your project or from Git's tracking, but rather that its modifications are not included in the specific changelist you are currently working with. This allows you to keep your commits focused, isolating work on different features or bug fixes, and avoiding accidentally staging changes you don't want to commit.

First, there's the straightforward way: directly from the "Changes" tool window. When you have the file listed under ‘Default’ or another changelist, you can right-click it. You'll see an option to “Move to another changelist” (or similar phrasing). If you haven't created any specific changelists, you'll see that you can create a new one. I often create a dedicated ‘Excluded’ or ‘Ignore’ changelist for files I generally don't intend to commit—things like temporary configuration files, IDE project files specific to my machine, or frequently modified build products. It's a useful way of visual management and keeps the primary change lists clean. This is usually the go-to method for a small number of files that you've noticed by eye while reviewing your changes.

However, what if you have hundreds, or even thousands, of files that follow a specific pattern? Manually selecting them becomes a nightmare. That’s when you need to leverage the power of patterns, specifically using `.gitignore` rules (or their equivalent, depending on the version control system you’re using) *and* the "Ignore files and folders" settings within the IDE itself. While `.gitignore` tells Git what not to track in the first place, the IDE's settings influence what it considers a "change," regardless of git’s tracking status, which is crucial. This interplay between these mechanisms is where things sometimes get tricky.

For a practical example, consider the following scenario. Let's say in my last project, a node.js application, the build process generated a large `dist` folder containing various JavaScript bundles. These would constantly appear in the change lists, making it hard to see actual changes. While we could have added it to `.gitignore`, we also wanted to see any changes to the contents of `dist` if for example, our builds started failing, but not include them on our standard changes lists.

Here’s what I did to exclude these files from my active changes:

```javascript
// Example 1: Pattern matching in the IDE settings
// Navigating to Settings -> Editor -> File Types -> Ignore files and folders
// I would add the following string (or strings) to this section:
//   dist/*;
//   build/*;
//   .idea/*;
//   node_modules/*;
//   *.log;

//This tells the IDE to not even track the changes in these files
```

This pattern will tell the IDE to not consider files or folders matching these patterns as changes in your main view. They won't be part of your diffs and will not appear under the “Unversioned files” section, which can help with general noise reduction even before committing. This is separate from Git's tracking. Think of this as a visual filter in the IDE, not a modification of your repository's tracking behavior.

Now let's say that sometimes we *do* need to review changes in those files, but we don't want them constantly clogging up our primary change lists. This is where specific change lists become handy. I'd create a change list called 'Build Changes' for the sole purpose of grouping changes to files in the `dist` directory when necessary. This would involve either staging the changed files directly to this change list manually, or, using the third method – working with a combination of gitignore and staging files.

For files not explicitly ignored by .gitignore but which you want to keep separate, you will have to manually select and move them to a specific change list. This is generally done after the fact, by selecting a file and using the right-click menu option "move to another changelist," as mentioned above.

Here's a snippet of how to handle situations where you might accidentally have a set of files in the 'Default' change list which you'd rather keep separate:

```python
# Example 2: moving files to another changelist. Pseudo Code
#
#  1. Open the "Changes" tab in the IDE
#  2. Expand the "Default" or any other existing changelist
#  3. Select the files matching, for instance dist/* and build/*
#  4. Right click selected files
#  5. Select "Move to Another Changelist" from the context menu
#  6. In the dialog prompt select either an existing changelist,
#     for example "Build Changes" or create a new one.
#
# This approach separates these files and keeps your primary change list focused.
```

The IDE does not provide an easy way to do this dynamically based on file patterns after they are added to the default change list. If the files have already been staged, then you have to unstage and restage them into the correct change list, which can be annoying. This is why properly configuring the ‘Ignore files’ patterns upfront is essential.

Finally, let's consider that I also used to work on a large Java project where build artifacts were copied from `target/` directory to a server for testing. Here, similar to the example above, I'd want to ignore the `target` directory most of the time, but sometimes I would need to review the build products to diagnose issues. This is where I'd make use of `.gitignore` in combination with change lists.

```java
// Example 3: .gitignore and IDE change list handling
// in the .gitignore file at project root:
//
//  target/*
//  .idea/*
//
// When I needed to actually see changes inside the `target/classes/` folder,
// I would need to first stage the particular files in the correct changelist.
// This ensures that changes to these files will only be commited when I explicitly add them to a specific changeset
//
// The settings under 'Editor -> File Types -> Ignored Files and Folders'
// will help reduce noise in the 'Changes' window by not tracking these changes
// in the first place. It is, in effect, a superset of the behaviour defined
// by `.gitignore` - files which are not tracked by git at all won't even appear
// as changes. Files that git *does* track, but you might not want to include
// on certain commits (for example those that are generated by the build system),
// will appear as changes that you will then have to consciously allocate to another
// change list if required.
```

To really understand the interplay between `.gitignore`, IDE ignore settings, and change lists, I highly recommend spending time with the official Git documentation, especially the sections on tracking files, staging, and the ignore file mechanism. Also, take a close look at the documentation for your specific JetBrains IDE; look for sections on change lists, ignored files, and the version control settings. You’ll find that the IDE’s help is quite thorough in this respect. Also, delve into the inner workings of Git, I personally found *Pro Git* by Scott Chacon and Ben Straub to be a fantastic resource. The book *Version Control with Git* by Jon Loeliger is another excellent option for a deeper dive.

In summary, effective exclusion of files from change lists is achieved through a combination of carefully configured `.gitignore` rules, pattern matching in the IDE's "Ignore files and folders" settings, and strategic use of specific change lists. By leveraging these mechanisms, you can keep your work organized, and your commit history clean and focused. The key is to understand that these are different tools working in tandem, and that proper initial setup and a clear understanding of their interplay are crucial.
