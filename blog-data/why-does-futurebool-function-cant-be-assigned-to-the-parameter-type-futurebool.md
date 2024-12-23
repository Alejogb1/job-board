---
title: "Why does 'Future<bool> Function()' can't be assigned to the parameter type 'Future<bool>?'?"
date: "2024-12-15"
id: "why-does-futurebool-function-cant-be-assigned-to-the-parameter-type-futurebool"
---

hey there, i’ve definitely bumped into this one before. it’s a classic case of nullable types and function signatures in asynchronous programming, something that trips up quite a few people, especially when dealing with futures. let's break it down.

the core issue here boils down to the difference between a *function that returns a future* and a *nullable function that returns a future*. the type `future<bool> function()` represents a function that, when invoked, will definitely return a `future<bool>`. this future, when it completes, will yield a boolean value. it's a guarantee. on the other hand, `future<bool>?` describes a variable that might hold a `future<bool>` or might hold nothing (null).

now, when we talk about a parameter of type `future<bool>?`, we're talking about a parameter that accepts either a future that produces a bool, or it can accept a null value. a function that *always* returns a future is not the same thing as *sometimes* not returning anything at all. it's not about the future itself being nullable, but the whole function being absent.

this is a common misunderstanding that happens when we mix synchronous function types with the notion of nullable objects. consider this from the compiler's perspective. it needs to ensure type safety and that there are no surprises at runtime. when a function takes a `future<bool>?` as a parameter, it's saying "i might not even have a function to call." if you pass it a non-nullable `future<bool> function()`, there is no way to ensure that the receiver will have a function that returns a `future<bool>`, it will *always* get a function. the receiving function is expecting that it could be `null` and you are not providing that option, not providing a possible null function.

i remember when i first encountered this a few years ago. i was working on a distributed system where different services were sending each other requests asynchronously. i had a worker service that was supposed to process these requests. the worker service had an interface that took a `future<string>? function(string)` which was meant to be an optional function that would pre-process the request.

```dart
// (this is actually dart but it shows the problem well)
typedef Preprocessor = Future<String> Function(String);

void processRequest(String request, Preprocessor? preprocessor) async {
  if (preprocessor != null) {
    String processedRequest = await preprocessor(request);
    print("processed request: $processedRequest");
  } else {
    print("original request: $request");
  }
}

Future<String> myPreprocessor(String request) async {
  // some async work...
  return 'preprocessed: $request';
}

void main() async {
  processRequest("initial request", myPreprocessor); //this will fail with the error you asked about

  Preprocessor? optionalPreprocessor = myPreprocessor; //ok
  optionalPreprocessor = null; //also ok

  processRequest("initial request", optionalPreprocessor); //this will work

  //example with an anonymous function
  processRequest("initial request", (String input) async {
    return "anonymous preprocessed: $input";
  });
}

```

i had a function that was a straight-up `future<string> function(string)`. i tried passing it directly to the processRequest function, and the compiler went ballistic, as it should. i kept trying to cast it, thinking it was some sort of variance issue, and then, after hours of pulling my hair out and quite a few google searches (stack overflow was key), i finally got it. the issue was the non-nullability of the function signature. i was effectively trying to force a square peg into a round hole. the solution was to make my `preprocessor` variable optional.

it's not enough to just have the *return* of the function be a future. the *function itself* has to be nullable to match the parameter type. imagine a function that sometimes just isn't there. that’s what a `future<bool>?` as a function parameter signifies. it’s not just about the result of the future, but the presence of the function itself. we're not talking about a future that can resolve to null but a function that can be null.

let me give you another code example. let's say we have a function that handles user authentication. it could take a function as a parameter to handle a more complex authentication flow. this function may or may not be used.

```csharp
//(csharp here)
using System;
using System.Threading.Tasks;

public class AuthenticationService
{
    public async Task<bool> AuthenticateUser(string username, string password, Func<string, string, Task<bool>>? customAuthenticator = null)
    {
        if (customAuthenticator != null)
        {
            Console.WriteLine("Using custom authenticator");
            return await customAuthenticator(username, password);
        }
        else
        {
            Console.WriteLine("Using default authenticator");
           //imagine doing some kind of actual authentification here
            return Task.FromResult(username == "user" && password == "password").Result;
        }
    }
}

public class Program
{
    public static async Task<bool>  MyCustomAuthenticator(string username, string password)
    {
      //imagine doing some complex authentication with calls to databases etc
       await Task.Delay(100);
      return  username.StartsWith("user") && password.StartsWith("pass");
    }

    public static async Task Main(string[] args)
    {
        AuthenticationService authService = new AuthenticationService();

         // this will fail:
        // bool result = await authService.AuthenticateUser("user", "password", MyCustomAuthenticator);

        //this is the right way of doing it:
        Func<string, string, Task<bool>>? optionalCustomAuth = MyCustomAuthenticator;
        bool result = await authService.AuthenticateUser("user", "password", optionalCustomAuth);
        Console.WriteLine("Authentication result: " + result);

        optionalCustomAuth = null;
        result = await authService.AuthenticateUser("user", "password", optionalCustomAuth);
        Console.WriteLine("Authentication result: " + result);


        //also you can use an anonymous function
        result = await authService.AuthenticateUser("user", "password", async (u,p) => {
          await Task.Delay(100);
          return u.StartsWith("user") && p.StartsWith("pass");
        });
          Console.WriteLine("Authentication result: " + result);

    }
}
```
here again the issue is the same. passing a non-nullable function where a nullable one is expected. the solution is either to make the function itself nullable using `?` or create a nullable variable with the function as its value. and that's how you get your compiler to stop yelling at you.

this distinction is essential in any language that has null safety or optional types. the compiler is not being difficult, it is trying to prevent runtime errors, it is trying to prevent you from calling a function on `null` which, as most of us know, is not fun.

i sometimes find it helpful to think of function types in terms of sets. `future<bool> function()` is a set of functions that will always return a future. `future<bool>?` as a parameter type allows the parameter to be either a member of that set or the `null` value. if you try passing in just a function `future<bool> function()` that can only be a member of that set but you need a value that is either a member of that set *or null*, it just doesn't work. it's like mixing up your socks. you can't put the left sock on your right foot. , maybe that was a little bit of analogy but not a terrible one, right?

this stuff can get a bit complicated so a resource i found useful is the "types and programming languages" by benjamin pierce. it has a very thorough treatment of type systems and how they work. for more concrete examples in real world use i also recommend "effective c#" by bill wagner. he gives a lot of tips on how to handle nulls correctly in c#. while it focuses on c#, the underlying principles apply to many languages.

i hope that helps explain why a `future<bool> function()` can't directly be assigned to a parameter of type `future<bool>?`. it’s about the nullability of the function itself, not just the future it returns. if you ever want to really nerd out on this, i can always point you to some type theory research papers, but for now i think you got it.
