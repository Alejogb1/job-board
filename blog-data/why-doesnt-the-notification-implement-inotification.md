---
title: "Why doesn't the notification implement INotification?"
date: "2024-12-23"
id: "why-doesnt-the-notification-implement-inotification"
---

Alright, let's tackle this one. It's a common point of confusion, and I've certainly spent my fair share of time debugging similar scenarios in the trenches. Specifically, the question at hand is: why doesn’t a typical notification implementation directly implement an interface like `INotification`? To understand this, we need to move beyond the surface and examine the underlying architectural patterns at play.

In many systems, notifications aren’t just simple data carriers; they often involve a complex lifecycle. Think about it. Notifications usually need to be dispatched to multiple handlers, might require asynchronous processing, and could potentially trigger cascading events. Directly implementing a basic interface on the core notification payload would lock it into a very rigid structure and dramatically limit its flexibility, especially as the system evolves. It would be akin to using a single pipe for all your plumbing needs, rather than a system with varying diameters, valves, and connections.

I remember a particularly tricky project I worked on a few years back. We were building a real-time analytics platform that involved a deluge of event notifications. Initially, I thought, "Hey, let's make each event implement a basic notification interface." It sounded clean and simple at the outset, but we quickly ran into performance bottlenecks and issues with extending the system's functionality. We needed the ability to add audit logging, retry mechanisms, and batch processing later on. Directly coupling all those responsibilities to the notification data itself was clearly a dead end.

The solution, and a best practice I've observed repeatedly, is the separation of concerns. The notification itself should be a simple data transfer object (dto), essentially a vessel holding relevant information about an event. The actual *processing* of this notification, involving dispatching and handling, should be handled by a separate mechanism, frequently employing patterns like the mediator pattern, event aggregator pattern, or a publish/subscribe model. This decoupling allows for independent development, testing, and more straightforward maintenance of different components within the system.

Consider a simple scenario. We have a notification for a user updating their profile. Instead of our `UserProfileUpdatedNotification` object implementing `INotification`, it’s simply a data holder. We use a separate notification handler to process this:

```csharp
// Data transfer object for user profile update
public class UserProfileUpdatedNotification
{
    public int UserId { get; set; }
    public string NewDisplayName { get; set; }
    public string UpdatedTimestamp {get; set;}
}


// Example of a simplified handler interface

public interface IMessageHandler<TMessage>
{
    void Handle(TMessage message);
}

// Concrete handler for user profile update notification
public class UserProfileUpdatedHandler : IMessageHandler<UserProfileUpdatedNotification>
{
     private readonly ILogger _logger;
     private readonly IUserDataService _userDataService;
    public UserProfileUpdatedHandler(ILogger logger, IUserDataService userDataService)
    {
       _logger = logger;
       _userDataService = userDataService;
    }

    public void Handle(UserProfileUpdatedNotification notification)
    {
      _logger.LogInformation($"User Profile Updated, User ID: {notification.UserId}, Updated Timestamp: {notification.UpdatedTimestamp}");

       //Update profile in Database
       _userDataService.UpdateUserDisplayName(notification.UserId, notification.NewDisplayName);
    }
}

```

In this example, `UserProfileUpdatedNotification` doesn't directly implement anything specific. It's just a plain object containing information. The `UserProfileUpdatedHandler` is responsible for the actual processing logic. This follows the single responsibility principle – the notification data holds data, and the handler processes data. This promotes clean, maintainable code.

Now, let’s consider a more complex scenario where we use a mediator pattern for dispatching. The mediator is responsible for receiving the notification and distributing it to the appropriate handlers. We’ll define a simple mediator and a dispatcher.

```csharp
//  Simple Mediator Interface
public interface IMediator
{
    void Publish<TNotification>(TNotification notification);
    void RegisterHandler<TNotification>(IMessageHandler<TNotification> handler);
}


// Basic implementation of a mediator
public class Mediator: IMediator
{
     private readonly Dictionary<Type, List<object>> _handlers = new Dictionary<Type, List<object>>();


     public void RegisterHandler<TNotification>(IMessageHandler<TNotification> handler)
     {
         if(!_handlers.ContainsKey(typeof(TNotification)))
         {
             _handlers[typeof(TNotification)] = new List<object>();
         }

         _handlers[typeof(TNotification)].Add(handler);
     }


    public void Publish<TNotification>(TNotification notification)
    {
        if(_handlers.TryGetValue(typeof(TNotification), out var handlers))
        {
            foreach (var handler in handlers.OfType<IMessageHandler<TNotification>>())
            {
                handler.Handle(notification);
            }
        }

    }
}
```

In this structure, the `Mediator` manages the dispatching process to any registered handlers. The notification object remains completely decoupled from the handling process, which greatly increases the flexibility and testability of the code.

Finally, let's demonstrate a slightly more involved use case using event aggregators, where the notification itself might be used to update various parts of a complex UI. We can employ an event aggregator pattern to decouple the components even further.

```csharp
// Generic interface for event publishing and subscribing
public interface IEventAggregator
{
    void Publish<TEvent>(TEvent @event);
    void Subscribe<TEvent>(Action<TEvent> action);
}

// Basic implementation of the event aggregator
public class EventAggregator : IEventAggregator
{
    private readonly Dictionary<Type, List<Delegate>> _subscriptions = new Dictionary<Type, List<Delegate>>();

    public void Publish<TEvent>(TEvent @event)
    {
        if (_subscriptions.TryGetValue(typeof(TEvent), out var actions))
        {
            foreach (var action in actions)
            {
                (action as Action<TEvent>)?.Invoke(@event);
            }
        }
    }


    public void Subscribe<TEvent>(Action<TEvent> action)
    {
        if (!_subscriptions.ContainsKey(typeof(TEvent)))
        {
            _subscriptions[typeof(TEvent)] = new List<Delegate>();
        }
        _subscriptions[typeof(TEvent)].Add(action);
    }
}

// Sample UI component subscribing to a specific event.
public class UserProfileView
{
    private readonly IEventAggregator _eventAggregator;

    public UserProfileView(IEventAggregator eventAggregator)
    {
        _eventAggregator = eventAggregator;
        _eventAggregator.Subscribe<UserProfileUpdatedNotification>(OnProfileUpdate);
    }

    private void OnProfileUpdate(UserProfileUpdatedNotification notification)
    {
        Console.WriteLine($"UI update: Display name changed to {notification.NewDisplayName} for User {notification.UserId}");
        // UI update code here based on the payload.
    }

}


```

Here, the `EventAggregator` decouples the publisher (the place where `UserProfileUpdatedNotification` might originate) from the subscribers (e.g., `UserProfileView`). The notification itself remains a pure data container. This allows for an extremely flexible and modular system, as new subscribers can be added without modifying any of the core classes.

In summary, the reason why a typical notification doesn’t directly implement an interface like `INotification` boils down to sound architectural practices: separation of concerns, flexibility, and maintainability. The notification should be a simple data structure, while its processing should be handled by dedicated components. The mediator pattern, event aggregator patterns, or even a direct implementation of handlers all provide a better solution.

If you want to dive deeper into these topics, I'd highly recommend exploring these resources. For comprehensive understanding of design patterns, "Design Patterns: Elements of Reusable Object-Oriented Software" by the Gang of Four remains a cornerstone text. For a deeper understanding of message-driven architectures, "Enterprise Integration Patterns" by Gregor Hohpe and Bobby Woolf offers extensive coverage. And to round it out, a look at "Clean Architecture: A Craftsman's Guide to Software Structure and Design" by Robert C. Martin can help you solidify your understanding of separation of concerns. These resources are gold when it comes to mastering the architecture of complex applications.
