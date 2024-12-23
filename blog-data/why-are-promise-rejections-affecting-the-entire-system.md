---
title: "Why are promise rejections affecting the entire system?"
date: "2024-12-23"
id: "why-are-promise-rejections-affecting-the-entire-system"
---

Alright,  I remember one particularly challenging project a few years back—a distributed e-commerce platform built on a microservices architecture. We were pushing updates daily and things were humming, then bam, cascading failures. Rejection after rejection, seemingly without rhyme or reason. Eventually, we traced it all back to poorly handled promises. It's a lesson I've never forgotten, and it’s a core issue when dealing with asynchronous operations.

The problem isn’t usually the rejection itself; it’s how that rejection propagates, or more accurately, how we fail to *contain* that propagation within our system. Promises, while elegant for handling asynchronous tasks, come with a crucial caveat: unhandled rejections can lead to catastrophic system-wide issues. Think of it like a broken circuit in a complex electrical grid. One faulty component can trigger a chain reaction, shutting down entire sections.

The fundamental problem often lies in a lack of proper error handling. When a promise rejects, that rejection travels upwards, searching for a `.catch()` handler. If it doesn't find one, depending on the runtime environment, you'll encounter different outcomes. In JavaScript, for instance, this will often result in an 'unhandled promise rejection' warning or error, but crucially, *it doesn't stop the JavaScript engine from continuing to execute*. This means you might have functions relying on a failed promise attempting to proceed, leading to unpredictable behavior. That behavior will cascade if these functions don’t have error handlers themselves.

Furthermore, some rejection behaviors are less obvious, and it's here where things often spiral. If you’re using a system that isn't purely synchronous (like most real-world applications), the rejection might not be visible immediately. That failed database query inside a promise might trigger later issues when that data is needed by another service or operation, but by then it's difficult to trace it back to the source. This is particularly true with complex chains of promises or when using `async`/`await` without the appropriate try/catch mechanisms.

Consider this simple example in javascript, exhibiting the core problem:

```javascript
async function fetchUserData(userId) {
  const response = await fetch(`/api/users/${userId}`);
  if (!response.ok) {
     throw new Error(`Failed to fetch user data: ${response.status}`);
  }
  return response.json();
}

async function processUserData(userId) {
  const userData = await fetchUserData(userId);
  // Assume this function fails if userData is not present
  return userData.name.toUpperCase();
}

async function displayUserName(userId) {
  const userName = await processUserData(userId);
  console.log(`User's name: ${userName}`);
}

displayUserName(123); // If user 123 doesn't exist, fetchUserData throws an error
```

Here, if `fetchUserData` fails, it throws an error, which is fine *if* `processUserData` has a catch handler, which it doesn’t. The rejection will propagate all the way to `displayUserName`, and potentially halt any processes downstream, *and* cause an unhandled rejection warning in the console which should indicate an issue. This can be easily fixed using try/catch blocks or `.catch()`.

Now, let's illustrate a more resilient approach, showcasing best practices in promise error handling:

```javascript
async function fetchUserData(userId) {
  try {
    const response = await fetch(`/api/users/${userId}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch user data: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Error fetching user data:", error);
    throw new Error("Data fetch error"); // rethrow
  }
}

async function processUserData(userId) {
  try {
    const userData = await fetchUserData(userId);
    // Assume this function fails if userData is not present
    return userData.name.toUpperCase();
  } catch (error) {
    console.error("Error processing user data:", error);
    throw new Error("Data processing error"); // rethrow for the caller
  }
}


async function displayUserName(userId) {
  try {
      const userName = await processUserData(userId);
      console.log(`User's name: ${userName}`);
  } catch (error) {
      console.error("Error displaying user name:", error);
  }
}

displayUserName(123); // Now errors are caught and logged, preventing cascading failure
```

In this updated example, each async function has its try/catch block which enables the logging of errors, and the rethrowing of a new, context-appropriate error. This pattern, when applied system-wide, becomes key to resilience. We're not ignoring errors; instead, we're logging and rethrowing errors in a controlled manner, allowing upstream callers to handle the specific failures more elegantly.

In the real world you'd also want a centralized error handler to perform additional actions such as logging specific user errors, triggering alerts, or displaying error messages to the client, but that is outside of the scope of the question.

Finally, consider a scenario using standard `.then()` and `.catch()` methods to showcase an alternative but equally valid approach:

```javascript
function fetchDataFromApi(url) {
    return fetch(url)
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .catch(error => {
        console.error("Error fetching data:", error);
        throw new Error("Data fetch error"); // Rethrow for the caller
      });
  }


function processApiResponse(url) {
    return fetchDataFromApi(url)
      .then(data => {
        if(!data || !data.results || data.results.length === 0){
            throw new Error("Invalid data")
        }
        return data.results[0].name
      })
      .catch(error => {
          console.error("Error processing API response:", error)
          throw new Error("Api response error")
      });
  }

  function showFirstApiResultName(url) {
      processApiResponse(url)
      .then(name => {
          console.log(`Name: ${name}`);
      })
      .catch(error => {
          console.error("Error displaying name:", error);
      })
  }

  showFirstApiResultName('https://api.example.com/items');
```

This example uses standard promises and `.then()` and `.catch()` callbacks. It is fundamentally the same as the async/await example, but instead of try/catch, it uses `.catch()` to handle errors at each step. Again, errors are logged and rethrown (or a new one is created).

The key takeaway here is that *uncaught* promise rejections are the real problem, not promises themselves. These examples are not just theoretical; they are based on real-world scenarios I've encountered numerous times. The issue isn't about the mechanism but how we, as developers, interact with it.

To further solidify your understanding, I highly recommend exploring these resources: "Effective JavaScript" by David Herman, which covers asynchronous programming and best practices very well. Also, "You Don't Know JS: Async & Performance" by Kyle Simpson is an absolute must-read for anyone trying to understand asynchronous programming deeply. For a more formal theoretical perspective, the ECMA specification for promises is also highly useful, albeit heavy reading. Finally, studying real-world examples in popular open-source libraries is another great way to see best practices in action.

In conclusion, when promise rejections start rippling through your system, it's a signal to examine your error handling strategies. Focus on containment, logging, and ensuring that rejections are not ignored, but processed at the most appropriate layer of your application. Do that, and you’ll have a far more resilient, stable application.
