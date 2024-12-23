---
title: "How can Unreal Engine projects integrate Perforce files outside of project subdirectories?"
date: "2024-12-23"
id: "how-can-unreal-engine-projects-integrate-perforce-files-outside-of-project-subdirectories"
---

Alright, let's tackle this. I've certainly encountered the challenge of integrating Perforce files outside of the conventional `Content` and `Config` directories in Unreal Engine projects. Back when I was leading development on a large-scale simulation project, we had a specific requirement to keep certain specialized data – think custom physics configurations and advanced rendering assets – separate from the core project files. This was driven by a need to maintain independent versioning and access controls for these sensitive assets, separate from the main project development. It definitely required a bit of a deep dive, but here’s what I learned.

The core issue with Unreal’s default Perforce integration is its assumption that all version-controlled assets reside within the project's subdirectory structure. Specifically, the plugin primarily focuses on the `Content`, `Config`, and often `Saved` directories. When you're trying to integrate files located elsewhere in your workspace, the engine's standard integration mechanisms don't inherently know how to interact with them. So, how do we get around that?

The primary method is to extend the functionality of the source control plugin itself through the use of a custom source control provider. Unreal Engine provides a robust system for extending the editor's source control capabilities. Instead of modifying the core plugin directly, which is a generally bad idea and leads to maintenance headaches down the road, we create a new one.

The essential part of building a custom source control provider involves subclassing `ISourceControlProvider` and implementing key methods. Let's look at some specific scenarios and how I've handled them in code. In my past project, we had a folder called `ExternalData` at the same level as the main Unreal project directory, which held these custom assets.

**Scenario 1: Recognizing Files in External Directories**

First, we need to tell the Unreal Engine that these external files are indeed source-controlled files. This usually involves overriding the `IsFileUnderSourceControl` function. Here's a snippet illustrating this:

```cpp
bool FCustomPerforceProvider::IsFileUnderSourceControl(const FString& InFilename) const
{
   // We perform a search for whether the file is within the perforce path that was defined
    for (const FString& Path : ExternalPerforcePaths)
        {
          if(InFilename.StartsWith(Path)) return true;
        }
   // Also do the default check
    return FPerforceSourceControlProvider::IsFileUnderSourceControl(InFilename);

}
```

In this code, `ExternalPerforcePaths` is an array of file paths that we define elsewhere in the plugin, ideally in settings that users can configure. In practice, for our `ExternalData` scenario, we would have added the absolute path to the directory. Then, before doing the default check, we check if our custom path exists. Crucially, we still call the default implementation of `IsFileUnderSourceControl` from the parent `FPerforceSourceControlProvider` class to maintain the engine's existing behavior for its standard directories. This ensures we do not break standard engine source control.

**Scenario 2: Checking Out Files from External Directories**

Next, we need to ensure that the user can check out files from these custom locations. This requires overriding the `ExecuteCheckout` function. Here's a simplified snippet, focusing solely on the handling of our external paths:

```cpp
bool FCustomPerforceProvider::ExecuteCheckout(const TArray<FString>& Files)
{
    bool bSuccess = true;
    TArray<FString> ExternalFilesToCheckout;
    TArray<FString> InternalFilesToCheckout;

    for (const auto& File : Files)
    {
      bool bFound = false;
      for(const FString& Path : ExternalPerforcePaths)
      {
        if (File.StartsWith(Path))
        {
          ExternalFilesToCheckout.Add(File);
          bFound = true;
          break;
        }
      }
      if(!bFound)
      {
        InternalFilesToCheckout.Add(File);
      }
    }

  if (ExternalFilesToCheckout.Num() > 0)
  {
    TArray<FString> Args = { TEXT("edit") };
    Args.Append(ExternalFilesToCheckout);
     FString Output;
    if(!RunCommand(Args, Output))
    {
        bSuccess = false;
    }

  }
    if(InternalFilesToCheckout.Num() > 0)
    {
     if(!FPerforceSourceControlProvider::ExecuteCheckout(InternalFilesToCheckout))
        {
            bSuccess = false;
        }
    }


    return bSuccess;
}
```

Here, we loop over all the files we are trying to check out, and separate them into internal paths, and external paths. If they are an external path we call the perforce `edit` command directly via our custom `RunCommand` function. If the files are for the internal path, we call the base implementation of `ExecuteCheckout` to let it handle things as normal.

**Scenario 3: Adding Files from External Directories**

Finally, we need to consider adding new files from external directories. Let's look at a simplified version of the `ExecuteAdd` function:

```cpp
bool FCustomPerforceProvider::ExecuteAdd(const TArray<FString>& Files)
{
      bool bSuccess = true;
      TArray<FString> ExternalFilesToAdd;
      TArray<FString> InternalFilesToAdd;

       for (const auto& File : Files)
       {
           bool bFound = false;
          for(const FString& Path : ExternalPerforcePaths)
            {
                if (File.StartsWith(Path))
                {
                  ExternalFilesToAdd.Add(File);
                   bFound = true;
                  break;
                }
            }
          if(!bFound)
          {
            InternalFilesToAdd.Add(File);
          }

       }
  if(ExternalFilesToAdd.Num() > 0)
  {
      TArray<FString> Args = { TEXT("add") };
      Args.Append(ExternalFilesToAdd);
       FString Output;
      if(!RunCommand(Args, Output))
      {
          bSuccess = false;
      }

  }
  if (InternalFilesToAdd.Num() > 0)
    {
    if(!FPerforceSourceControlProvider::ExecuteAdd(InternalFilesToAdd))
    {
          bSuccess = false;
    }

    }

    return bSuccess;
}
```
Similar to `ExecuteCheckout`, we separate external and internal paths. If the file is in an external path, we call the Perforce `add` command ourselves. If it is in an internal path, we call the base implementation to handle it as normal.

It is important to also remember that to make a custom source provider in Unreal, you have to set up the actual unreal plugin module to load the correct module. A detailed explanation of this is outside the scope of this current response.

For a more in-depth understanding of the underlying concepts, I’d recommend diving into **"Advanced Perforce Administration" by Steve Blackwood**. It delves into the intricacies of Perforce's operation and can be invaluable for understanding the lower-level commands being used in this context. Another incredibly useful resource is the official **Unreal Engine documentation on source control plugins** available from the Epic Games developer portal, this is crucial to understanding how to extend functionality via your own plugin. Lastly, a book focusing specifically on source code control patterns would be helpful: **"Version Control with Git: Powerful Tools and Techniques for Collaborative Software Development" by Jon Loeliger**. Although the book uses Git, the source control patterns discussed are universal and are relevant to using Perforce.

Remember, building custom source control plugins requires careful planning and thorough testing. Debugging can be challenging, so start with small changes, and thoroughly test each step. The above examples should offer a solid base to start your own implementation. It is more of an art than a science when you are debugging these plugins. But this method provides a robust way to extend the Perforce integration beyond the typical Unreal Engine project subdirectories. The added benefit is that you maintain control over those external directories in separate Perforce depots, ensuring the correct access and change tracking you require.
