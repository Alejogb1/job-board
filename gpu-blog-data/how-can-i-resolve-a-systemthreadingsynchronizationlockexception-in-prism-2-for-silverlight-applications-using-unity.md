---
title: "How can I resolve a System.Threading.SynchronizationLockException in Prism 2 for Silverlight applications using Unity?"
date: "2025-01-26"
id: "how-can-i-resolve-a-systemthreadingsynchronizationlockexception-in-prism-2-for-silverlight-applications-using-unity"
---

Encountering a `System.Threading.SynchronizationLockException` within a Prism 2 for Silverlight application, particularly when using Unity for dependency injection, often points to a concurrency issue related to accessing UI elements from background threads. Silverlight's UI objects are inherently single-threaded, accessible only from the UI thread. Failure to adhere to this restriction results in the aforementioned exception, typically during attempts to update data bound to UI controls or performing operations that modify the visual tree.

The core problem lies in the interaction between background tasks, likely spawned via asynchronous operations or background workers, and UI updates. Unity, in itself, is not the source of the concurrency issue but rather a catalyst. When Unity resolves types, it may inadvertently inject objects or services that later trigger background work impacting the UI. For example, a ViewModel might utilize an injected service that fetches data asynchronously, and upon completion, this data attempts to update properties bound to the view. If this update happens outside the UI thread, `SynchronizationLockException` arises.

Let's dissect the common causes and resolutions. A primary scenario involves data fetching within a service. Consider the following pseudo-code:

```csharp
// Problematic Service
public class DataService : IDataService
{
  public async Task<List<string>> GetDataAsync()
  {
      // Simulate long-running network operation
      await Task.Delay(1000);
      return new List<string> { "Item 1", "Item 2", "Item 3"};
  }
}
```

Here, `GetDataAsync` is asynchronous but the consuming code might not correctly marshal back to the UI thread when the operation completes, causing a direct UI update. The ViewModel, consuming the `IDataService`, could be implemented thus:

```csharp
// Problematic ViewModel
public class MyViewModel : ViewModelBase
{
    private readonly IDataService _dataService;
    private List<string> _items;

    public List<string> Items
    {
      get { return _items; }
      set { SetProperty(ref _items, value); }
    }

    public MyViewModel(IDataService dataService)
    {
      _dataService = dataService;
      LoadData();
    }

    private async void LoadData()
    {
      Items = await _dataService.GetDataAsync(); // Potential threading issue here!
    }
}
```

The `LoadData` method, although asynchronous, assigns the result directly to the `Items` property. Assuming that `Items` is bound to a Silverlight control, the `SetProperty` method, called by the `Items` property setter, will likely attempt a UI update from a thread other than the UI thread. This precisely is where the exception manifests.

The solution hinges on marshaling execution back to the UI thread for UI-related updates. This can be achieved through the `Dispatcher` object, which every `System.Windows.DependencyObject` (including the view and view model) possesses. I frequently implement this marshaling operation via a reusable helper extension method.

```csharp
// UI Thread Helper
public static class DispatcherExtensions
{
  public static void InvokeOnUIThread(this Dispatcher dispatcher, Action action)
  {
    if (dispatcher.CheckAccess())
    {
       action();
    }
    else
    {
       dispatcher.BeginInvoke(action);
    }
  }
}
```

This `InvokeOnUIThread` extension allows you to execute an `Action` on the UI thread regardless of which thread it is called from. It first verifies if the caller is the UI thread. If so, the action is executed directly. If not, it dispatches the action to the UI thread using `BeginInvoke`. A revised version of the ViewModel, incorporating the above helper is shown:

```csharp
// Corrected ViewModel
public class MyViewModel : ViewModelBase
{
    private readonly IDataService _dataService;
    private List<string> _items;

    public List<string> Items
    {
      get { return _items; }
      set { SetProperty(ref _items, value); }
    }

    public MyViewModel(IDataService dataService)
    {
      _dataService = dataService;
      LoadData();
    }

    private async void LoadData()
    {
        var data = await _dataService.GetDataAsync();

        Dispatcher.InvokeOnUIThread(() =>
        {
           Items = data;
        });
    }
}
```

Here, the UI update for the `Items` property is enclosed within a lambda that's passed to the `Dispatcher.InvokeOnUIThread` method, which ensures the operation runs on the UI thread.

Another frequent source of this error can stem from event handlers or background worker callbacks which, similar to the service scenario, directly attempt to modify UI components. Suppose the `IDataService` has events that propagate data; for example, consider a `DataUpdated` event within the `DataService`:

```csharp
// Problematic Event Source
public class DataService : IDataService
{
    public event EventHandler<List<string>> DataUpdated;

    public async Task FetchInitialData()
    {
      await Task.Delay(1000);
      OnDataUpdated(new List<string> { "Initial A", "Initial B"});
    }

    protected virtual void OnDataUpdated(List<string> data)
    {
        DataUpdated?.Invoke(this, data);
    }
}
```
And let's assume that, the `MyViewModel` subscribes to this event:
```csharp
// Problematic Event Handler
public class MyViewModel : ViewModelBase
{
    private readonly IDataService _dataService;
    private List<string> _items;

    public List<string> Items
    {
      get { return _items; }
      set { SetProperty(ref _items, value); }
    }

    public MyViewModel(IDataService dataService)
    {
        _dataService = dataService;
        _dataService.DataUpdated += OnDataUpdated;
        _dataService.FetchInitialData();
    }

    private void OnDataUpdated(object sender, List<string> data)
    {
      Items = data; // Potential exception here!
    }
}
```

The event callback, `OnDataUpdated`, may execute on a thread different from the UI thread; hence, modifying `Items` directly is unsafe. The fix mirrors the previous example; it necessitates using `Dispatcher.InvokeOnUIThread` to modify the UI from the event handler:

```csharp
// Corrected Event Handler
public class MyViewModel : ViewModelBase
{
    private readonly IDataService _dataService;
    private List<string> _items;

    public List<string> Items
    {
      get { return _items; }
      set { SetProperty(ref _items, value); }
    }

    public MyViewModel(IDataService dataService)
    {
        _dataService = dataService;
        _dataService.DataUpdated += OnDataUpdated;
        _dataService.FetchInitialData();
    }

    private void OnDataUpdated(object sender, List<string> data)
    {
        Dispatcher.InvokeOnUIThread(() =>
        {
           Items = data;
        });
    }
}
```

Correctly handling concurrency, specifically ensuring all UI updates are performed on the UI thread, is critical when working with Prism and Unity within Silverlight. Failing to do so consistently leads to the `SynchronizationLockException`.

For further study, I recommend focusing on resources that discuss:

1.  **Silverlight Threading Model:** Understanding the single-threaded nature of the UI and the role of the `Dispatcher`.
2.  **Asynchronous Programming in .NET:** Comprehending the `async` and `await` keywords and how they interact with threading.
3.  **Data Binding in Silverlight:** Gaining a detailed view of how data binding interacts with the UI thread and how changes to bound properties trigger updates.
4.  **Prism and MVVM architecture:** A deep study of the patterns, particularly where long running tasks may be initiated outside of the View or ViewModel. This is important, as the View itself is a DispatcherObject and has the tools to perform marshaling, whereas other objects may not.

By understanding these concepts, one can systematically avoid `SynchronizationLockException` and create more robust Silverlight applications with Prism.
