---
title: "How can I use IAmbientDbContextLocator with EF Core / .NET 6?"
date: "2024-12-14"
id: "how-can-i-use-iambientdbcontextlocator-with-ef-core--net-6"
---

alright, so you're diving into the world of ambient contexts with entity framework core and .net 6. i get it, i've been there, staring at the screen wondering why my db context isn't magically appearing where i need it. it's a common enough issue, and while ef core itself doesn't offer a direct `iambientdbcontextlocator` out of the box, we can absolutely whip up something that works nicely.

first off, let’s talk about *why* you’d want this thing. the main problem it solves is the hassle of constantly passing around your db context instance, especially when you have nested services or operations that need access to the database. it makes your code less cluttered and more focused on the core logic, which is always a good thing. imagine a scenario with a transaction that spans multiple layers: without a locator, each method call would need to receive and pass on the context, making code feel more verbose.

i remember this one project, a crud-heavy web app where every single request needed to touch the database. passing the `dbcontext` through layers was driving me bonkers – the constructor injection was getting out of control, it was a dependency injection spaghetti western. that's when i started hunting for a better way, and that journey led me to something similar to `iambientdbcontextlocator`.

now, about the implementation. the core idea is to use the power of `async local` to store the db context on a per-execution-flow basis. that means that when you start an operation that uses the database, you create the context, put it in an `asynclocal` variable, and then anyone within the same async flow can access it. it’s like a local variable, but it magically works across async calls, which is what makes it ideal for our needs.

here’s a basic implementation of an `iambientdbcontextlocator` interface, plus a concrete implementation.

```csharp
public interface iambientdbcontextlocator
{
    mydbcontext? getcontext();
    void setcontext(mydbcontext context);
    void clearcontext();
}

public class ambientdbcontextlocator : iambientdbcontextlocator
{
    private static readonly asynclocal<mydbcontext?> context = new asynclocal<mydbcontext?>();

    public mydbcontext? getcontext() => context.value;
    public void setcontext(mydbcontext newcontext) => context.value = newcontext;
    public void clearcontext() => context.value = null;
}
```

this is the core mechanism: `asynclocal<mydbcontext?> context`. it holds our database context, and because it’s static, it’s shared throughout the application, but the *value* within it is tied to each individual async flow. it’s really nice and simple when you break it down.

for `mydbcontext`, it’s nothing special, just your regular ef core context. let’s assume it's a context with some simple user and blog entities as example:

```csharp
public class user
{
    public int id { get; set; }
    public string? name { get; set; }
    public string? email { get; set; }
    public ienumerable<blog>? blogs { get; set; }
}
public class blog
{
    public int id { get; set; }
    public string? title { get; set; }
    public string? content { get; set; }
    public int userid { get; set; }
    public user? user { get; set; }
}
public class mydbcontext : dbcontext
{
    public mydbcontext(dbcontextoptions<mydbcontext> options) : base(options) { }

    public dbset<user> users { get; set; }
    public dbset<blog> blogs { get; set; }

    protected override void onmodelcreating(modelbuilder modelbuilder)
    {
        modelbuilder.entity<user>().hasmany(x => x.blogs).withone(x => x.user).hasforeignkey(x=>x.userid);
    }
}
```

to really make the `ambientdbcontextlocator` useful, you’ll usually integrate it into a unit of work or repository pattern. the idea is that when you start an operation, you create a new database context, set it in the locator, and when the operation completes (or throws an exception), you clean up (save changes and dispose).

now, let's look at a potential implementation of the unit of work pattern:

```csharp
public class unitofwork : iunitofwork
{
    private readonly iambientdbcontextlocator _contextlocator;
    private readonly mydbcontext _context;
    private bool _disposed = false;
    public unitofwork(ambientdbcontextlocator contextlocator, mydbcontext context)
    {
        _contextlocator = contextlocator;
        _context = context;
    }

    public async task begin()
    {
        if (_disposed)
        {
            throw new objectdisposedexception(nameof(unitofwork), "unit of work has already been disposed.");
        }

        _contextlocator.setcontext(_context);
        await _context.database.beginTransactionasync();
    }
    public async task commit()
    {
        if (_disposed)
        {
            throw new objectdisposedexception(nameof(unitofwork), "unit of work has already been disposed.");
        }
        await _context.savechangesasync();
        await _context.database.committransactionasync();
        _contextlocator.clearcontext();
    }

    public async task rollback()
    {
        if (_disposed)
        {
            throw new objectdisposedexception(nameof(unitofwork), "unit of work has already been disposed.");
        }
        await _context.database.rollbacktransactionasync();
        _contextlocator.clearcontext();
    }
    public async value task disposeasync()
    {
        if (_disposed)
        {
            return;
        }
        if (_contextlocator.getcontext() != null)
        {
            _contextlocator.clearcontext();
        }
        await _context.disposeasync();
        _disposed = true;
    }
    public void dispose()
    {
        if (_disposed) return;
        if (_contextlocator.getcontext() != null)
        {
            _contextlocator.clearcontext();
        }
        _context.dispose();
        _disposed = true;
    }
}
public interface iunitofwork : iasyncdisposable, idisposable
{
    task begin();
    task commit();
    task rollback();
}
```

this is how i tend to use it now. it makes the code much more maintainable, the unit of work creates the context, sets the locator, and ensures it’s cleaned up. i tend to have the repositories retrieve the context by using the `iambientdbcontextlocator` implementation and no longer need to have the context injected in every constructor.

now, it might sound like this is a lot of plumbing but in reality it simplifies your code a lot. you just ensure that an operation begins with a unit of work and the rest of your code can access the db context anywhere. this is what it looks like on the application level:

```csharp
public class myapplicationservice
{
    private readonly iuserrepository _userrepository;
    private readonly iunitofwork _unitofwork;

    public myapplicationservice(iuserrepository userrepository, iunitofwork unitofwork)
    {
        _userrepository = userrepository;
        _unitofwork = unitofwork;
    }
    public async task doinsertuser(user newuser)
    {
        await _unitofwork.begin();
        try
        {
            await _userrepository.insertuser(newuser);
            await _unitofwork.commit();
        }
        catch
        {
            await _unitofwork.rollback();
            throw;
        }
    }
}
public class userrepository : iuserrepository
{
    private readonly iambientdbcontextlocator _contextlocator;

    public userrepository(iambientdbcontextlocator contextlocator)
    {
        _contextlocator = contextlocator;
    }
    public async task insertuser(user newuser)
    {
        var context = _contextlocator.getcontext();
        if (context == null)
        {
            throw new exception("no db context found. this should never happen.");
        }
        context.users.add(newuser);
        await context.savechangesasync();
    }
}

public interface iuserrepository
{
    task insertuser(user newuser);
}

```

that's the long and short of it. the key points are to use `asynclocal` for the context storage, create a locator service to access the context, and utilize it in a unit of work pattern.

for further reading, i’d recommend looking into the “domain-driven design” book by eric evans. it covers a lot about repositories and unit of work, even though it doesn’t directly touch on ambient contexts, the design principles are extremely useful. also check out books and papers discussing the async patterns in .net, as understanding `asynclocal` fully can be critical. for ef core specific implementation knowledge there is the book by jon smith called “programming entity framework core” although that does not directly cover ambient context either.

one important thing: like any other pattern, ambient contexts aren't the magic bullet for every single project. overuse can lead to complications too, especially if you introduce a bunch of thread-related weirdness. and i once spent a whole day debugging a case where some asynchronous task was not using `await`, so the db context was not created at the top of the flow and things were not working as expected - not fun. so, use the tool carefully and remember the fundamentals.
