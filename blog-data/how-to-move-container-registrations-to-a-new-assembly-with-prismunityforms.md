---
title: "How to Move container registrations to a new assembly with Prism.Unity.Forms?"
date: "2024-12-14"
id: "how-to-move-container-registrations-to-a-new-assembly-with-prismunityforms"
---

alright, so, you're facing the classic prism.unity.forms container migration headache, huh? i’ve been there, multiple times, and it’s never *fun*. i've spent way more hours than i care to count staring at registration errors. it usually starts when your project grows and you realize putting everything in the main application assembly was a terrible idea (we all do it, don’t judge too hard). let's get into it.

the core issue is that prism’s unity container needs to know where all your types are so it can do its dependency injection magic. when you start splitting your registrations across different assemblies, unity gets confused. prism has some ways to deal with this, but they’re not always obvious.

first, forget the idea of moving all your registrations to one other assembly. this is often a bad design idea. the better pattern, generally is to move the registrations related to a *feature* into the assembly that contains the feature. this will reduce coupling and make the project more maintainable in the long run.

so, you've got your main application project, let’s call it `mycoolapp`. it used to have all your view models, services, etc., and the unity registrations inside. now, you've created a new assembly, `mycoolapp.featureA`, for some specific functionality. you want to register the types defined in `mycoolapp.featureA` with unity. the straightforward, but often problematic approach, is just trying to get unity to scan the new assembly. that typically ends in pain.

let me give you some background, during a project many years ago where i was responsible for the modularization of a huge legacy application, i made the terrible choice to try to do it all at once. the result was an absolute unholy mess of registrations, and i had to spend days debugging the resulting circular dependencies. it turns out it’s better to modularize in smaller chunks, i learned that the hard way.

anyway, let's talk code. usually when you start you have something like this in your main project:

```csharp
public class App : PrismApplication
{
    protected override void RegisterTypes(IContainerRegistry containerRegistry)
    {
        containerRegistry.RegisterForNavigation<MainPage, MainPageViewModel>();
        containerRegistry.RegisterSingleton<IAnalyticsService, AppCenterAnalyticsService>();
        containerRegistry.Register<IDataService, JsonDataService>();
        // ... tons more registrations
    }
}
```

now you want move some of the registration that belong to `mycoolapp.featureA` into its own code.

there are multiple ways to do it. but here’s my approach (one of the two most common methods). you want to have the feature register its own dependencies. and the way to do this is to create a `iModule` implementation.

first, in your `mycoolapp.featureA` project:

```csharp
using Prism.Ioc;
using Prism.Modularity;
using mycoolapp.featureA.ViewModels;
using mycoolapp.featureA.Services;

namespace mycoolapp.featureA
{
   public class FeatureAModule : IModule
   {
        public void OnInitialized(IContainerProvider containerProvider)
        {
            // usually there is nothing to do here
        }
        public void RegisterTypes(IContainerRegistry containerRegistry)
        {
             containerRegistry.RegisterForNavigation<FeatureAPage, FeatureAPageViewModel>();
             containerRegistry.Register<IFeatureAService, FeatureAService>();
        }
    }
}
```

this module class handles the registrations that are specific to feature a. it registers the view and the viewmodel and a local service. now, you need to tell prism to load this module. this is done in the `App.cs` of the main project (the one that inherits from prismapplication) this is the second step where we declare the module using `ConfigureModuleCatalog`:

```csharp
using Prism.Ioc;
using Prism.Modularity;
using Prism.Unity;
using mycoolapp.featureA;

namespace mycoolapp
{
    public partial class App : PrismApplication
    {
        public App()
        {
        }

        protected override void ConfigureModuleCatalog(IModuleCatalog moduleCatalog)
        {
            moduleCatalog.AddModule<FeatureAModule>();
            base.ConfigureModuleCatalog(moduleCatalog);
        }

        protected override void RegisterTypes(IContainerRegistry containerRegistry)
        {
            containerRegistry.RegisterForNavigation<MainPage, MainPageViewModel>();
            containerRegistry.RegisterSingleton<IAnalyticsService, AppCenterAnalyticsService>();
            containerRegistry.Register<IDataService, JsonDataService>();
        }
    }
}
```

that’s the basic idea. you need to ensure you have the required nuget packages installed `prism.unity.forms` and `prism.module`. and make sure the `mycoolapp.featureA` has the correct references to prism and any common dependencies. now when the app loads the unity container is configured with the types of `mycoolapp.featureA` that are declared in the module class.

but it’s not always that simple, is it? sometimes you might need to load modules conditionally. for example, you might have a ‘premium’ feature in another assembly, and you only want to load it if the user has a certain license. or, you could have plugins loaded from external files, a classic move for extensible applications. in these cases, you’d need to configure your `iModuleCatalog` manually in code, using a `directorymodulecatalog`, `aggregate modulecatalog` or a custom module catalog. these catalog methods have the ability to read module definitions from external configurations and/or files and it's better than hardcoding modules. you will find this in the documentation for prism.

another case where you could run into problems is the use of interfaces and abstractions. you should register implementations of interfaces that are declared in different assemblies. sometimes it can be a pain debugging these, especially when the implementations are not correctly registered and your ioc is failing due to missing bindings. if this happens usually you should check the lifetime of your instances using `Register` versus `RegisterSingleton`. if you want more information about this check the unity documentation about lifetime management. they are many more details about this issue there.

a more complex scenario is when you start injecting types from the main application in types from other modules (for example `IAnalyticsService`). this works by default, but you need to be careful about cyclic dependencies (a classic ioc problem). if you're injecting stuff in module a from module b, and the opposite is happening it means that your architecture has a big design problem, which should be addressed before you even start the ioc registration, modularization should always consider the separation of concerns and the architectural design of the solution.

now, some tips from the trenches:

*   **always double-check your references**. a missing reference is the cause of 90% of the issues in these kinds of projects. ensure each project is referencing all the required nuget packages and project dependencies.
*   **use the ioc debug tools**. prism and unity usually have internal logging that can help debug problems, look for binding errors, and other issues. the unity documentation itself is a good starting point.
*   **start simple**. don’t try to move everything at once. move one or two types at a time, testing after each change. it is the way to go.
*   **modularize in a proper way**. don't move things into modules just to move things. make sure each module represents a feature that can be decoupled from the rest of the app.

finally, i’m kidding with this, but remember kids, always test your code before pushing it to production. unless you *really* like debugging runtime exceptions on your release build. i once spend 3 days debugging a runtime null exception, that was simply caused by a missing dependency, so, if this can happen to me it can happen to anyone.

also remember that prism has plenty of documentation about this subject which should be your starting point. look for the `iModule` and `iModuleCatalog` and unity’s documentation about dependency injection and lifetime. those are your bread and butter when solving issues about ioc registrations in prism.

anyway, that’s my experience. let me know if i can help you with any other issue.
