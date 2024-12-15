---
title: "How to implement a MVVM Navigation service, and how to create a navigate method that waits for another view model to be closed?"
date: "2024-12-15"
id: "how-to-implement-a-mvvm-navigation-service-and-how-to-create-a-navigate-method-that-waits-for-another-view-model-to-be-closed"
---

alright, so you're tackling mvvm navigation with a twist – making it wait for a viewmodel to close. i get it, i've been there, staring at the screen wondering why my flow is all kinds of messed up. its a common situation with asynchronous navigation. i spent a good chunk of my early career messing this up in a project for an inventory management system. ended up with a bunch of memory leaks and a very unhappy client. lets break this down into simple parts.

the core of this whole thing is building a navigation service that actually controls transitions between different views. think of it as the air traffic controller for your application. in mvvm, viewmodels shouldn't be directly fiddling with the ui, that includes navigation. the service is the middleman. you'll need some interfaces for this, it makes swapping implementations and unit testing easier, trust me on that one.

first, the interface for our navigation:

```csharp
public interface inavigation service
{
   void navigate to<tviewmodel>(tviewmodel viewmodel) where tviewmodel : viewmodelbase;
   task<tresult> navigate to<tviewmodel, tresult>(tviewmodel viewmodel) where tviewmodel : viewmodelbase;
   void goback();
   bool can goback { get; }
}

```

this interface is generic, notice the `tviewmodel` type parameter, allows to handle any kind of viewmodel. the `navigate to<t>` method takes a viewmodel, it does not returns anything because is just navigation to other view, in contrast the `navigate to<t,tresult>` its for navigations where you want the viewmodel to return some result when closed. the `goback` is self explanatory, and also the `can goback`. simple stuff.

now, a basic implementation, without the waiting, for navigation:

```csharp
public class navigationservice : inavigation service
{
    private readonly iframe _frame;
    private readonly dictionary<type, type> _viewmodelmapping = new dictionary<type, type>();
    private readonly stack<type> _navigationhistory = new stack<type>();


    public navigationservice(iframe frame)
    {
        _frame = frame;
    }

     public void register<tviewmodel, tview>() where tviewmodel : viewmodelbase where tview : page
    {
      if(!_viewmodelmapping.containskey(typeof(tviewmodel)))
      {
           _viewmodelmapping.add(typeof(tviewmodel), typeof(tview));
      }
    }


    public void navigate to<tviewmodel>(tviewmodel viewmodel) where tviewmodel : viewmodelbase
    {
        var viewtype = _viewmodelmapping[viewmodel.gettype()];
        _frame.navigate(viewtype, viewmodel);

        _navigationhistory.push(viewtype);
    }

    public void goback()
    {
       if (_navigationhistory.count <= 1)
           return;

       _navigationhistory.pop();
       var prevviewtype = _navigationhistory.peek();

       _frame.go back();
    }


    public bool can goback => _navigationhistory.count > 1;

    //note this navigation doesn't wait
      public task<tresult> navigate to<tviewmodel, tresult>(tviewmodel viewmodel) where tviewmodel : viewmodelbase
     {
        throw new notimplementedexception();
     }

}

```

this code assumes you have a `frame` (could be a uwp frame or similar), and you inject it into the service. the `register<tviewmodel, tview>` helps you register the viewmodel with its corresponding view. notice the `_navigationhistory` stack, used to handle the back navigation. navigation is just finding the view from the viewmodel, and ask the frame to display it, and update the navigation history. pretty basic.

the `navigate to<tviewmodel,tresult>` method throws a notimplementedexception because it is not implemented yet, thats where the waiting part comes into play.

so, how do we make that `navigate to<t,tresult>` method wait? it involves using `taskcompletionSource<tresult>`. it's a bit like a promise. the viewmodel being navigated to will set the result when it's closed, and the `taskcompletionsource` will resolve.

here is the implementation of the `navigate to<t,tresult>` method, with the waiting part:

```csharp
  public class navigationservice : inavigation service
    {
        private readonly iframe _frame;
        private readonly dictionary<type, type> _viewmodelmapping = new dictionary<type, type>();
        private readonly stack<type> _navigationhistory = new stack<type>();
        private readonly dictionary<type, taskcompletionsource> _pendingtasks = new dictionary<type, taskcompletionsource>();

        public navigationservice(iframe frame)
        {
            _frame = frame;
        }

         public void register<tviewmodel, tview>() where tviewmodel : viewmodelbase where tview : page
        {
          if(!_viewmodelmapping.containskey(typeof(tviewmodel)))
          {
               _viewmodelmapping.add(typeof(tviewmodel), typeof(tview));
          }
        }


        public void navigate to<tviewmodel>(tviewmodel viewmodel) where tviewmodel : viewmodelbase
        {
            var viewtype = _viewmodelmapping[viewmodel.gettype()];
            _frame.navigate(viewtype, viewmodel);

            _navigationhistory.push(viewtype);
        }


        public task<tresult> navigate to<tviewmodel, tresult>(tviewmodel viewmodel) where tviewmodel : viewmodelbase
        {
           var viewtype = _viewmodelmapping[viewmodel.gettype()];
           var tcs = new taskcompletionsource<tresult>();

            _pendingtasks[viewtype] = tcs;
             _frame.navigate(viewtype, viewmodel);
            _navigationhistory.push(viewtype);
           return tcs.task;
        }

        public void setnavigationresult<tresult>(tresult result)
        {
          var currentpage = _frame.currentpage.gettype();

          if (_pendingtasks.containskey(currentpage))
          {
              var tcs = _pendingtasks[currentpage];
              tcs.setresult(result);
              _pendingtasks.remove(currentpage);
          }
        }

        public void goback()
        {
           if (_navigationhistory.count <= 1)
               return;

           _navigationhistory.pop();
           var prevviewtype = _navigationhistory.peek();

           _frame.go back();
        }


        public bool can goback => _navigationhistory.count > 1;
    }

```

notice the `_pendingtasks` dictionary. it stores a `taskcompletionsource` object for each pending navigation. when you call the `navigate to<tviewmodel, tresult>` method, a `taskcompletionsource` is created, its task is returned and the view is displayed.

then, when the viewmodel is closing it must call the `setnavigationresult<tresult>`, the dictionary is checked with the current view, it finds the right `taskcompletionsource`, and sets the result. this, unblocks the navigation task and the result is returned.

the viewmodels need a way to communicate back with the navigation service when closed to deliver the results. a simple way to do this is inject the `inavigationservice` in the constructor of the viewmodel. for example.

```csharp
public class modalviewmodel : viewmodelbase
{
  private readonly inavigation service _navigation service;

  public string Result {get;set;}

  public modalviewmodel(inavigation service navigation service)
  {
      _navigation service = navigation service;
  }


   public void closewithresult()
   {
       _navigation service.setnavigationresult(result);
       //optional goback();
   }

}
```

so this viewmodel receives the navigation service, and when close it calls the `setnavigationresult` method passing the result to be returned.

this basic structure should give you a solid base for building a navigation system that can wait for results from navigated views. remember that this is a simplified implementation, real-world applications may have more intricate navigation structures.

now, some things to keep in mind:

*   **error handling:** you'll want to add try-catch blocks, perhaps setting `exception` in the `taskcompletionsource` if navigation fails. it is important to add handling to the `goback` and navigation methods, in case the view model or the view are not registered.
*   **multiple navigation requests:** you'll need to decide how to handle concurrent navigation. you might want to queue requests or cancel previous navigations. i had a fun time figuring out that one in a side project where i tried to implement a single-page application but had several buttons that could trigger navigation, it was like a crazy navigation loop.
*   **passing parameters:** you might want to pass parameters during navigation, you can add parameters to the navigate method, but you should always use viewmodel properties to transmit data, using the method parameters is not a very good pattern.
*   **dialogs/popups:** you might want a specific navigation service for dialogs or popups, as they behave slightly differently. in one project, i used to use the main window to display all the dialogs, that was a horrible mistake, everything was coupled to the main window and all the business logic. i do not recommend that practice.
*   **unit testing:** having a interface for navigation allows to easily test the viewmodels and the navigation service without having to open the window, just mock the interface. i have seen a lot of developers skip unit testing navigation, it is a very bad idea as it can be very difficult to debug and fix.

for more in-depth details on this stuff, i'd recommend taking a look at the "patterns of enterprise application architecture" by martin fowler for architecture, and for the async part "concurrency in c# cookbook" by stephen cleary.

remember: the devil is in the details. this base implementation will serve you well, but you will need to adapt it to your specific needs and use case. its not a 'one size fits all' situation.

one last thing before i wrap up: why don't scientists trust atoms? because they make up everything. just kidding! it was a mandatory bad joke to help you release some dopamine, now get back to coding!
