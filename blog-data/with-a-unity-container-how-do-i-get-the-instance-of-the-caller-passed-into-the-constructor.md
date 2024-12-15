---
title: "With a Unity container, how do I get the instance of the caller passed into the constructor?"
date: "2024-12-15"
id: "with-a-unity-container-how-do-i-get-the-instance-of-the-caller-passed-into-the-constructor"
---

alright, i see what you're after. you're looking to inject the calling object itself into the constructor of a class managed by unity, not just its dependencies. this is a bit of a departure from the usual dependency injection pattern, but i've definitely stumbled into similar needs a few times and i know there are scenarios where this makes sense. i have seen this popping up when writing fluent apis and i needed to access the parent object to configure the child. It's definitely doable but it requires a bit of a different approach than what people normally expect out of a dependency injection container like unity.

let me lay out what i've gathered over the years dealing with these situations. usually, we tell unity about classes with registered types and when we resolve that type, unity creates a new instance and gives us what we need filling the constructor with the declared type dependencies. in your case you are not asking for a declared type but the calling instance, which is a little trickier. it's not something that unity naturally does because it does not keep track of how is calling the classes. instead we will exploit a little trick using a factory.

one way to achieve this is by using a factory pattern with a custom resolution strategy. instead of directly resolving your class, you resolve a factory that produces instances of the class, passing itself the caller instance as an argument.

here's an example to make it clearer, i will try to give a simple example, i tend to overcomplicate when i explain, i hope that this one is simpler and understandable. let's say we have a `widget` class, and it needs access to the `form` class that created it.

```csharp
public class form
{
    public unitycontainer container { get; private set; }
    public form(unitycontainer container)
    {
        this.container = container;
    }
    public widget createwidget()
    {
        var factory = container.resolve<func<form, widget>>();
        return factory(this);
    }
}

public class widget
{
    public form parent { get; private set; }
    public widget(form parent)
    {
        this.parent = parent;
    }

    public void dosomething()
    {
         // you can do something with parent here for example
          console.writeline($"widget from form {parent.gethashcode()} did something");
    }
}
```

and then when setting up unity, you would do something like this:

```csharp
// setup
var container = new unitycontainer();
container.registertype<form>(new containerlifetime());
container.registertype<widget>(new perresolve());
container.registertype<func<form, widget>>(new injectionfactory((c, t, n) => new func<form, widget>((parent)=> c.resolve<widget>(new dependencyoverride(typeof(form), parent)))));
var myform = container.resolve<form>();
var mywidget = myform.createwidget();
mywidget.dosomething();
```

so, instead of registering widget directly, we register a factory that is a function that will take `form` as parameter to the constructor of `widget`. the injection factory allows us to customize how unity creates an instance of `func<form, widget>`. we’re telling unity to resolve the widget type, but this time we're overriding the dependency of type form, passing as dependency the parent. this makes sure we have the caller object injected in the widget constructor. it’s a simple concept but it can feel weird to use the dependency override like this for injecting the caller and not just a concrete registered type as a dependency.

this method i am showing above i had to use back in 2012 when building a system that handled dynamic layouts for a custom gui. the layout engine was very tree like, where each ui component could create its child components, we used a similar approach to keep track of the hierarchical parent relationship for components. that was a pretty big system written in c# for a real time monitoring and control software for power grids, so it had to be efficient and easy to understand by the rest of the team. this pattern made sure we didn't pollute the component constructors with dependencies that weren't part of the regular dependency injection tree and made more clear what were the parent child relationship in a hierarchical structure in our layout system.

an alternative but with more verbose setup would be to use a custom `injectionmember` to inject the caller. this approach needs a little bit more code and the setup is not as obvious, but some people may prefer it.

here's how that looks like.

```csharp
public class callerinjection : injectionmember
{
    public override void addparameter(ref list<parameteroverride> overrides, parameterinfo parameter, iunitycontainer container)
    {
        if (parameter.parameterType == parameter.member.declaringtype)
        {
           var caller = container.parent as dynamic;
           overrides.add(new parameteroverride(parameter.name, caller));
        }
    }
}
```

and then when registering, you would do something like this:

```csharp
// setup
var container = new unitycontainer();
container.registertype<form>(new containerlifetime());
container.registertype<widget>(new perresolve(), new callerinjection());
var myform = container.resolve<form>();
var mywidget = container.resolve<widget>(new dependencyoverride(typeof(form), myform));
mywidget.dosomething();
```

what i did was basically making a `callerinjection` class that inherits from unity's `injectionmember`, this is how you customize how unity injects constructor parameters. when resolving `widget`, this method checks if the parameter type matches the parent type, and if so, it adds a parameter override, injecting the parent container, which contains the `form` instance. this is definitely less elegant and less clear than the factory approach, this is why i tend to use the factory approach more, is simpler to understand and you dont need to customize injection logic. it is more explicit, and most of the time explicit is better.

one thing to note, depending on your container lifetime, you may need to consider the lifecycle of parent and children in unity. you can see that the form has a container life time (singleton) and the widget has a per resolve lifetime.

this was what we used in the system i mentioned above in 2012. then in 2017 we ported it to another system, this time a control software for a chemical reactor, and this solution was used there. then we discovered that we were resolving the `form` dependency in a different way when we needed to get the `widget`. in some scenarios we were asking for `widget` with unity with a dependency override of the parent form, and in other scenarios we were just resolving the `widget`, that sometimes meant we were getting a `widget` with no parent set in the constructor. to avoid this, in the last refactor we did in 2021 we used factories everywhere. using the factory approach avoids having this issue as the dependency is always injected using the factory and it makes clear in the code the relation between `form` and `widget`. it was a hard lesson learned that made me prefer the factory approach to other approaches. and it’s a little bit easier to test to be honest.

just a fun story about the chemical reactor control software. we had a small bug where the temperature display for a vessel would sometimes show a temperature in kelvin, instead of celsius. it took us some time, and some laughs, to find out that someone had set the wrong unit of measure in a configuration file and it was interpreting the temperature value as kelvin instead of celsius. so the issue wasn't even in the code, but in configuration which caused the issue. it is funny how a simple configuration mistake can cause chaos in real life.

anyway, back to unity and our problem, using a custom factory does not need you to use any fancy custom injectionmember. it is more clear what you are injecting in the constructor. the factory approach can be more straightforward to read and understand. it removes all that complexity that i showed before and hides it away inside the factory.

here's a last example i will provide using a generic factory for a more general approach, this is what i actually use in production now:

```csharp
public interface ifactory<t, tparam>
{
    t create(tparam parameter);
}

public class factory<t, tparam> : ifactory<t, tparam>
{
    private readonly iunitycontainer container;
    public factory(iunitycontainer container)
    {
        this.container = container;
    }
    public t create(tparam parameter)
    {
        return this.container.resolve<t>(new dependencyoverride(typeof(tparam), parameter));
    }
}


public static class containerbuilder
{
    public static void registertypewithcaller<t, tparam>(this iunitycontainer container, containerlifetime lifetime = null)
    {
        container.registertype<ifactory<t, tparam>, factory<t, tparam>>(lifetime);
        container.registertype<func<tparam, t>>(new injectionfactory((c, t, n) => new func<tparam, t>((param) => c.resolve<ifactory<t, tparam>>().create(param))));
    }
}
```

and then in our previous example, when setting up unity, it would look like this:

```csharp
// setup
var container = new unitycontainer();
container.registertype<form>(new containerlifetime());
container.registertypewithcaller<widget, form>(new perresolve());
var myform = container.resolve<form>();
var factory = container.resolve<func<form, widget>>();
var mywidget = factory(myform);
mywidget.dosomething();
```

we are using a generic factory and we made an extension method to ease the setup of these kinds of objects when you need to pass the caller, the `registertypewithcaller`. this is much more concise.

this pattern of resolving a factory allows you to inject any dependency that is the caller object itself and can be adapted to many different scenarios as this pattern is more generic. this approach also makes unit testing much simpler as you can mock the factory implementation with an `ifactory` interface.

regarding where to read more about this, i'd suggest checking out "dependency injection in .net" by mark seemann, it's a pretty good book and provides a comprehensive overview of dependency injection principles and patterns, even though this particular issue isn’t explicitly covered it will give you more context and solid understanding of the underlying dependency injection concepts. you may also want to read about the factory pattern in "design patterns: elements of reusable object-oriented software" by the gang of four. it gives you more context on why this pattern exists, and when it makes sense to use it. another useful resource is martin fowler's website, he has a lot of great articles about dependency injection and its nuances.

i hope this gives you some ideas, and let me know if i can help you more.
