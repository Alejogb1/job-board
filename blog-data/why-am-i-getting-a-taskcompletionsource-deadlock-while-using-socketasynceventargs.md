---
title: "Why am I getting a TaskCompletionSource deadlock while using SocketAsyncEventArgs?"
date: "2024-12-15"
id: "why-am-i-getting-a-taskcompletionsource-deadlock-while-using-socketasynceventargs"
---

alright, so you're hitting a classic taskcompletionsource deadlock when using socketasynceventargs, huh? i've been there, felt that pain. it's like threading the eye of a needle while riding a unicycle – tricky, and one wrong move and you're faceplanting. let me break down what's probably happening, based on my own less-than-stellar experiences, and how i eventually got it sorted out.

the core problem usually boils down to a misunderstanding of how `taskcompletionsource` and asynchronous operations with `socketasynceventargs` interact. it’s tempting to think that when you call `socket.receiveasync(args)`, the operation magically happens in parallel and the `args.completed` event fires in some separate thread. but the reality is, that's not quite how it works, especially with the asynchronous pattern.

let's rewind to the mid-2000s – yeah, i'm that old. i was working on this real-time data streaming application. imagine thousands of clients connecting to a server, constantly sending sensor data. we decided to use the `socketasynceventargs` pattern for maximum efficiency because, well, we needed all the performance we could get. we thought, "hey, async, non-blocking, what could possibly go wrong?" oh, the naive enthusiasm of youth.

what we did was, in our client connection handler, we would create a new `taskcompletionsource<int>` and then attach it to the `socketasynceventargs` object. the `completed` event handler, the one that supposedly gets called when data arrives, it would call `setresult` on this `taskcompletionsource` to signal completion of a read operation. it was all textbook, based on the msdn documentation, or so we thought.

`socketasynceventargs`, under the hood, usually executes the completion event on an i/o thread pool thread or in the thread that issued the asynchronous call if the operation is immediately completed. the `setresult` method of the `taskcompletionsource` will return immediately if the associated task has not yet been awaited or is finished and so this call would simply end, but if the task was awaited in the current thread that fired this `completed` event, the call will complete synchronously on the same thread which will block the completion callback.

here's where we messed up, and likely where you're stumbling too. the code looked something like this:

```csharp
public async Task<int> ReceiveDataAsync(Socket socket)
{
  var args = new SocketAsyncEventArgs();
  args.SetBuffer(new byte[1024], 0, 1024);
  var tcs = new TaskCompletionSource<int>();
  args.Completed += (sender, e) =>
    {
      if (e.SocketError != SocketError.Success)
      {
        tcs.SetException(new SocketException((int)e.SocketError));
      }
      else
      {
        tcs.SetResult(e.BytesTransferred);
      }
    };

  if(!socket.ReceiveAsync(args)) {
    tcs.SetResult(args.BytesTransferred);
  }

  return await tcs.Task;
}
```

the problem lies in the line `return await tcs.Task;`. see, `await` will capture the context, usually the ui thread or some synchronization context, and then tries to resume on the same context. the `completed` event handler, as discussed, is often called on a thread pool thread. and this can cause a deadlock if the thread the asynchronous operation starts in also gets blocked trying to `await`. a race condition ensues, where the `completed` event wants to complete on a thread that is waiting on the `task`, and the `task` is waiting for the `completed` to finalize. they're both waiting for each other, like two kids in a playground squabbling over the last piece of cake, and neither will give in. we are stuck forever.

we did not understood that we were using `await` on a synchronization context that could cause the deadlock. the `await tcs.task;` call is waiting for completion, but that completion is blocked and we are stuck forever on a single thread blocking itself.

i remember spending several late nights scratching my head and trying to figure out what went wrong. we tried everything, from changing thread pool settings to doing voodoo rituals, nothing helped. at one point i thought i was going crazy, it was truly some bizarre edge case.

a colleague, who was more grizzled than me (and apparently a bit smarter), pointed out our error. the key, he explained, was to detach from the captured context of await, or to avoid it all together, in the completion event, and to ensure that the `completed` event handler runs on a thread that will not try to complete the `task` on the same thread as the `await` call.

here's one way we fixed it, it uses `task.run` to queue the completion on the thread pool, detaching the completion event handler from the synchronization context and preventing the deadlock scenario, but this implies extra overhead, but you must use this if you are unsure how the calling context is behaving.

```csharp
public async Task<int> ReceiveDataAsync(Socket socket)
{
    var args = new SocketAsyncEventArgs();
    args.SetBuffer(new byte[1024], 0, 1024);
    var tcs = new TaskCompletionSource<int>();
    args.Completed += (sender, e) =>
    {
        Task.Run(() =>
        {
          if (e.SocketError != SocketError.Success)
          {
              tcs.SetException(new SocketException((int)e.SocketError));
          }
          else
          {
              tcs.SetResult(e.BytesTransferred);
          }
        });
    };
    if(!socket.ReceiveAsync(args)) {
      tcs.SetResult(args.BytesTransferred);
    }

    return await tcs.Task;
}
```

here's another way to achieve a deadlock-free implementation, it uses `unsafe` threading logic, this method is more efficient because it does not use `task.run`, but it requires more care, the main trick here is to create the `taskcompletionsource` with `taskcreationoptions.runcontinuationsasynchronously` which ensure that continuations from `setresult` will happen in a different thread than the original context, this is a faster implementation, but harder to understand.

```csharp
public async Task<int> ReceiveDataAsync(Socket socket)
{
  var args = new SocketAsyncEventArgs();
  args.SetBuffer(new byte[1024], 0, 1024);
  var tcs = new TaskCompletionSource<int>(TaskCreationOptions.RunContinuationsAsynchronously);
  args.Completed += (sender, e) =>
  {
      if (e.SocketError != SocketError.Success)
      {
        tcs.SetException(new SocketException((int)e.SocketError));
      }
      else
      {
        tcs.SetResult(e.BytesTransferred);
      }
  };
  if (!socket.ReceiveAsync(args))
  {
    tcs.SetResult(args.BytesTransferred);
  }
  return await tcs.Task;
}
```

and there is another way, the way that i prefer to work with, it does not use the `socketasynceventargs.completed` event at all, it directly await on the operation instead using an abstraction of a class that simplifies usage of the operations of the socket:

```csharp
    public async Task<int> ReceiveDataAsync(Socket socket, byte[] buffer)
    {
        if (socket == null) throw new ArgumentNullException(nameof(socket));
        if (buffer == null) throw new ArgumentNullException(nameof(buffer));
        if(buffer.Length == 0) return 0;


        int totalBytesReceived = 0;
        while (totalBytesReceived < buffer.Length)
        {
           var bytesReceived = await socket.ReceiveAsync(buffer, totalBytesReceived, buffer.Length - totalBytesReceived, SocketFlags.None);
            if(bytesReceived <= 0) break;
             totalBytesReceived += bytesReceived;

        }
       return totalBytesReceived;
    }

```

this last implementation has the benefit of not requiring the usage of `taskcompletionsource` and the usage of `socketasynceventargs` making the logic easier to reason about and to not produce deadlocks.

the key takeaway here is understanding that asynchronous operations with `socketasynceventargs` involve multiple threads, and the completion events will be triggered in a different thread than the main thread, the one doing the `await`. using `task.run` on the continuation or configuring the task to run asynchronously will detach the execution from a blocked context, or just avoiding the usage of the `taskcompletionsource` or the `socketasynceventargs.completed` event like on the last example above will prevent this type of deadlock.

now i always double check my asynchronous code, paying special attention to the thread contexts and completion callbacks. i've learned that with asynchronous code, a little extra care goes a long way.

for further reading, i would suggest checking out "concurrency in c# cookbook" by stephen cleary, and "programming c#" by jesse liberty, those are classics that helped me a lot at the time. i also recommend searching for "synchronization contexts c#" online, there are many good articles detailing how they work.

i hope this helps you untangle this knot, let me know if there's anything else i can clarify. happy coding and watch for those deadlocks, they can be tricky little fellas, even though sometimes it can feel like they are just mocking our coding skills, it can even be fun when you finally defeat them.
