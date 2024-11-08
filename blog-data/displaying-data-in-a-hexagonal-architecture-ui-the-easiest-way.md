---
title: "Displaying Data in a Hexagonal Architecture UI:  The Easiest Way"
date: '2024-11-08'
id: 'displaying-data-in-a-hexagonal-architecture-ui-the-easiest-way'
---

```csharp
// Domain
public class Worker
{
    // ... domain logic
}

public interface IWorkerRepository
{
    // ... methods for interacting with Workers
}

// Application
public class CreateWorkerCommand
{
    // ... command properties
}

public interface ICommandHandler<TCommand>
{
    void Handle(TCommand command);
}

public class CreateWorkerCommandHandler : ICommandHandler<CreateWorkerCommand>
{
    private readonly IWorkerRepository _repository;

    public CreateWorkerCommandHandler(IWorkerRepository repository)
    {
        _repository = repository;
    }

    public void Handle(CreateWorkerCommand command)
    {
        // ... create Worker and persist to repository
    }
}

// Infrastructure
public class WorkerRepository : IWorkerRepository
{
    // ... implementation using a database or other persistence mechanism
}

// UI
public class WorkerViewModel
{
    // ... properties for displaying Worker data in the UI
}

public class WorkerViewModelFactory
{
    private readonly IWorkerRepository _repository;

    public WorkerViewModelFactory(IWorkerRepository repository)
    {
        _repository = repository;
    }

    public WorkerViewModel Create(int workerId)
    {
        // ... fetch Worker data from repository
        // ... create WorkerViewModel from fetched data
        return new WorkerViewModel();
    }
}
```
