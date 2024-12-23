---
title: "c# unsubscribe event lambda event handler?"
date: "2024-12-13"
id: "c-unsubscribe-event-lambda-event-handler"
---

 so you're asking about unsubscribing from C# events specifically when the event handler is a lambda well I've been there man countless times it's a common pitfall so let's dive in I'm gonna talk from personal experience you know things I learned the hard way

See the problem is lambda expressions when used as event handlers create anonymous delegates these delegates don't have a direct name you can use for unsubscribing like a normal method you'd hook up with += and unhook with -= That direct name is crucial for that -= operation It's a core part of how C# event system works

First things first let's look at the basic scenario suppose you have a class Publisher that raises an event and you want another class Subscriber to handle that event using a lambda expression

```csharp
public class Publisher
{
    public event EventHandler MyEvent;

    public void RaiseEvent()
    {
        MyEvent?.Invoke(this EventArgs.Empty);
    }
}

public class Subscriber
{
    public Subscriber(Publisher publisher)
    {
        publisher.MyEvent += (sender EventArgs e) =>
        {
            Console.WriteLine("Event received via lambda");
        };
    }
}
```

So far so good right you create a Publisher instance a Subscriber attaches to that publishers event with a lambda expression and everything works as expected events get triggered the lambda gets executed but now the problem emerges How do you get that subscriber to disconnect because you can't directly do this

```csharp
//This wont work!!
publisher.MyEvent -= (sender EventArgs e) =>
{
    Console.WriteLine("Event received via lambda");
};
```
C# won't let you just use an equivalent lambda expression it's not about the lambda looking the same the compiler generates a new anonymous delegate each time even if the code is identical It's like trying to remove a sticker by a duplicate sticker but the real sticker is the one underneath

This is a gotcha and people fall for it all the time So how do we get around it? Well the simplest way and what I usually do now is to stash the lambda in a variable a delegate variable I should be specific. Instead of adding the anonymous delegate we create an actual variable of the right type and assign the lambda to it

```csharp
public class Subscriber
{
    private EventHandler _myHandler;

    public Subscriber(Publisher publisher)
    {
        _myHandler = (sender EventArgs e) =>
        {
            Console.WriteLine("Event received via lambda assigned to a delegate variable");
        };

        publisher.MyEvent += _myHandler;
    }

    public void Unsubscribe(Publisher publisher)
    {
        publisher.MyEvent -= _myHandler;
    }
}
```

Now you can properly unsubscribe by referencing that `_myHandler` variable which you could and should also make a private field for encapsulation purposes this variable allows us to unsubscribe from the event without problem It's the specific delegate instance we added earlier

I remember once I was working on this old legacy project it was a real spaghetti code fest and events were everywhere I was tracking a memory leak and I discovered there were event handlers added using anonymous lambda's all over the place The issue was that they were never getting unsubscribed it was a real mess of events firing all the time even after they were supposed to stop being needed It took me hours to trace the issue back to the lack of named delegate variables for each handler It was my "aha" moment understanding exactly what was going on and it's why I'm so particular about this now

The other alternative which I would only use in cases that your handler logic is very small and where I don't want to create a delegate variable and I don't mind the small overhead of the garbage collector is just to implement some sort of IDisposable interface where the unsubscription is done inside the dispose method usually this also would need storing the event publisher in a field variable but that's a common practice anyways

```csharp
public class Subscriber : IDisposable
{
    private Publisher _publisher;
    
    public Subscriber(Publisher publisher)
    {
        _publisher = publisher;
        _publisher.MyEvent += (sender EventArgs e) =>
        {
            Console.WriteLine("Event received via lambda using IDisposable");
        };
    }
    
    public void Dispose()
    {
        _publisher.MyEvent -= (sender EventArgs e) =>
        {
            Console.WriteLine("Event received via lambda using IDisposable");
        };
        _publisher = null;

    }
}
```
Now you'll notice here I am using the same lambada expression here as when subscribing this is ok this will work due to how Dispose is called usually it's not good practice to create the lambda expression like this inside the `Dispose` method but because of the time the Dispose method is called it will ensure that the correct handler will be removed this should only be used in simple use cases where you can guarantee the object that is subscribing is indeed being disposed when no longer needed It's another way to handle the lambda but more specific scenarios

So there you have it basically you should avoid anonymous lambda expressions as event handlers if you plan on unsubscribing from them without a reference to the event handler delegate. Delegate variables are your best friend and IDisposable interface could be useful on very specific cases if you can manage that and there is no performance penalty

Remember you need that specific delegate instance you added so you can remove it later it is not about the lambda code itself being exactly the same but about the instance of the delegate itself. And that's why it is so important to store that delegate somewhere and to use that stored delegate when unsubscribing from the event

If you want to get deeper into these concepts check out "C# in Depth" by Jon Skeet it really lays out the event system in detail and if you are interested on event driven patterns check out "Event-Driven Architecture" by Martin Fowler these are not directly C# books but they will give you more theory and best practices
One final tip I've learned the hard way from past experience never ignore warnings about potential memory leaks your future self will thank you seriously

And before I forget my favorite joke of a programmer: Why do programmers prefer dark mode? Because light attracts bugs!
