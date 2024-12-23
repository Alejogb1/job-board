---
title: "Why is `someFuture.onSuccess()` returning null instead of a future in Vert.x v4.2.5?"
date: "2024-12-23"
id: "why-is-somefutureonsuccess-returning-null-instead-of-a-future-in-vertx-v425"
---

Alright, let's unpack this. The behavior you're observing with `someFuture.onSuccess()` seemingly returning null, instead of a new future, is a classic gotcha that I've bumped into a few times, particularly when things get a little complex with asynchronous flows in Vert.x. It's not a bug, as such, but rather a misunderstanding of how `onSuccess` (and its counterparts) are designed to function within the context of Vert.x’s asynchronous model and composability.

In essence, the core issue isn't that `onSuccess` is *failing* to return anything; it's that it’s *not supposed to return a future directly*. What it actually does is register a callback to be executed when the original future completes successfully. The return type of `onSuccess` itself is void, which means it doesn’t yield a new future you can chain off of directly in the conventional manner where you expect subsequent asynchronous steps. This is a subtle but important distinction.

Let's illustrate with a bit of a contrived example, based on some rather messy code I had to refactor years back: Imagine I was building a service that fetches user data, enriches it with profile information from another source, and then stores it in a database. Initially, I had this, or something very similar:

```java
// Incorrect approach
Future<JsonObject> fetchUserData() {
  return Future.succeededFuture(new JsonObject().put("userId", 123));
}

Future<JsonObject> fetchUserProfile(JsonObject userData) {
  return Future.succeededFuture(new JsonObject().put("profileId", 456));
}


Future<JsonObject> storeEnrichedData(JsonObject enrichedData) {
  return Future.succeededFuture(new JsonObject().put("status", "success"));
}


public void processUser() {
  Future<JsonObject> userDataFuture = fetchUserData();

  // Expecting a Future<JsonObject>, but onSuccess returns void
  Future<JsonObject> profileDataFuture = userDataFuture.onSuccess(userData -> {
     return fetchUserProfile(userData);
  });

  profileDataFuture.onSuccess(profileData -> {
      //This will result in a null pointer exception if you call anything on profileData
      JsonObject enrichedData = new JsonObject().mergeIn(userDataFuture.result()).mergeIn(profileData);
      storeEnrichedData(enrichedData);
    });

    // This approach doesn't work as expected
    // profileDataFuture is not the next future in the chain
    // subsequent processing depends on userDataFuture result, it does not chain

   //...and this causes problem because profileDataFuture will be void after the first call
}
```
In the code snippet, I was incorrectly assuming that `userDataFuture.onSuccess(...)` would somehow transform into a new future representing the result of `fetchUserProfile()`. That assumption is not correct. The problem lies in the misunderstanding of void return value from onSuccess, which does not chain the async flow in a sequential manner and instead tries to run the asynchronous function in a separate event loop, without affecting the main chain that was started with `userDataFuture`.

This led to the common problem of seeing `profileData` as null in the next `.onSuccess` as it was never the return value of previous `onSuccess`. Instead it was triggered by the completion of  `userDataFuture`'s event loop. The function within `onSuccess` was executed asynchronously and did not directly return a value used by the next call of `onSuccess` in the chain. That is the core of the issue here.

To resolve this, we should use `compose`. `compose` is specifically designed for chaining asynchronous operations where the result of one operation becomes the input to the next, providing a new future that represents the result of the entire chain. Let’s look at a revised version:

```java
// Correct Approach with compose
public void processUserCorrected(){
    Future<JsonObject> userDataFuture = fetchUserData();

    Future<JsonObject> combinedFuture = userDataFuture.compose(userData -> fetchUserProfile(userData))
    .compose(profileData -> {
        JsonObject enrichedData = new JsonObject().mergeIn(userDataFuture.result()).mergeIn(profileData);
       return  storeEnrichedData(enrichedData);
    });


    combinedFuture.onSuccess(result -> {
        // This will be executed if all composing operations succeed
        System.out.println("Data processing completed successfully:" + result);
    }).onFailure(err -> {
        // This will be executed if anything goes wrong during composing
         System.err.println("Error during data processing: " + err.getMessage());
    });
}
```

Here, `compose` acts as the correct way of continuing the asynchronous execution chain. Each `compose` takes the result of the previous future and passes it to its provided function that returns a future, effectively chaining operations sequentially. The return value of the previous function passed to the next future, making the chain work. This results in `combinedFuture` being the actual resulting future of the composition of `fetchUserData`, `fetchUserProfile` and `storeEnrichedData`, correctly chaining the results from each to the next one in the pipeline.  This prevents the issue of `null` values where we expected futures.

Another related mistake I've observed is trying to perform multiple operations that should happen in parallel without correctly accounting for their potential concurrency. Let's say we had two different user profile enrichments to perform:

```java
public void processParallelUserData() {

    Future<JsonObject> userDataFuture = fetchUserData();


    //Incorrect use of onSuccess for parallel operations
    Future<JsonObject> profile1Future = userDataFuture.onSuccess(userData -> fetchUserProfile(userData));
    Future<JsonObject> profile2Future = userDataFuture.onSuccess(userData -> fetchAdditionalProfile(userData));

    //Attempting to combine results without waiting properly will cause null pointers again
    profile1Future.onSuccess(profile1 -> profile2Future.onSuccess(profile2 -> {
         JsonObject merged = new JsonObject().mergeIn(userDataFuture.result()).mergeIn(profile1).mergeIn(profile2);
         System.out.println("Data Merged: " + merged);
    }));

}
```
In the above scenario, we're still using `onSuccess` incorrectly for parallel operations, creating a similar void type problem, but also potentially introducing race conditions. The intended behavior is that both `fetchUserProfile` and `fetchAdditionalProfile` run in parallel, but the incorrect use of `onSuccess` with subsequent calls makes this fail again. The `profile1Future` and `profile2Future` are still void and the code will likely execute out of order, causing unexpected null pointer exceptions.

The correct way to do this is to use `CompositeFuture` to wait for both of them to complete, before combining results:

```java
//Correct approach with CompositeFuture for parallel operations
public void processParallelUserDataCorrected() {

    Future<JsonObject> userDataFuture = fetchUserData();

    Future<JsonObject> profile1Future = fetchUserProfile(userDataFuture.result());
    Future<JsonObject> profile2Future = fetchAdditionalProfile(userDataFuture.result());

    CompositeFuture.all(profile1Future, profile2Future).onSuccess(result -> {
        JsonObject merged = new JsonObject().mergeIn(userDataFuture.result())
          .mergeIn(profile1Future.result())
          .mergeIn(profile2Future.result());
         System.out.println("Data Merged: " + merged);
    }).onFailure(err -> {
      System.err.println("Error during parallel operation: " + err.getMessage());
    });

}

Future<JsonObject> fetchAdditionalProfile(JsonObject userData) {
    return Future.succeededFuture(new JsonObject().put("additionalId", 789));
}
```

Using `CompositeFuture.all()`, we create a new future that will complete when both `profile1Future` and `profile2Future` complete successfully (or fail if one or both fail). We also need to make sure that we call the `fetchUserProfile` and `fetchAdditionalProfile` with `userDataFuture.result()` instead of just on the `onSuccess` callback, otherwise, the futures will never be correctly created. This method allows us to correctly perform parallel operations, wait for them to finish and then compose our resulting object once they both are done, correctly avoiding unexpected null pointer errors in asynchronous operations.

In summary, the key takeaway is that `onSuccess`, while useful for side effects that don’t impact the main async flow, *does not return a new future suitable for chaining*. For sequential asynchronous flows, `compose` is what you need; for parallel execution of multiple futures, `CompositeFuture` is your friend. These are essential concepts when dealing with asynchronous programming in vert.x and, as i've seen time and time again, a lack of understanding of how they work leads to these frustrating null-related issues.

For deeper exploration, I'd recommend reviewing the Vert.x documentation, particularly the sections on Futures and Composition. Also, reading the excellent book "Reactive Systems in Java" by Kenny Bastani will give you a more nuanced view of reactive programming concepts that underline how futures work and why they behave as they do in Vert.x. Another useful resource is “Java Concurrency in Practice” by Brian Goetz, which, although it’s not specific to Vert.x, provides fundamental understanding of concurrency that helps immensely when working with Vert.x's asynchronous model. The official documentation does an excellent job, but having these additional resources in your toolkit can only make you more effective when using Vert.x.
