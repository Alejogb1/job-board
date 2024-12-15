---
title: "How to React JS wait for function to execute before continue?"
date: "2024-12-15"
id: "how-to-react-js-wait-for-function-to-execute-before-continue"
---

well, that's a common head-scratcher for folks dipping their toes into asynchronous javascript with react, isn't it? i've been there, trust me. seems straightforward initially, but then bam, you're staring at unexpected behavior because javascript just zooms ahead while your function is still humming along. i've spent a fair number of late nights debugging this particular pattern. let's break it down.

the core problem stems from javascript's non-blocking nature. when you call a function that takes some time – like fetching data from an api or performing a heavy computation – javascript doesn't pause and wait. it continues executing the next lines of code, potentially before your function has finished doing its thing. this is generally great for performance, but it requires a different way of handling things when the result of that function is needed later on.

react, being built on top of javascript, inherits this characteristic. so if you're trying to update your react component based on the result of an async function, you're bound to encounter timing issues if you don’t handle it properly.

now, there are several approaches to tackle this, and each has its own scenarios where it works best. i'll cover three main ones i've used extensively over the years: async/await, promises with `.then()`, and the react hooks-based approach with `useEffect`.

first, let's start with the modern javascript way, async/await. this is my preferred method in most cases because it makes asynchronous code look a lot like synchronous code, increasing readability. say you have a function that fetches user data, something like this:

```javascript
async function fetchUserData(userId) {
    const response = await fetch(`/api/users/${userId}`);
    if (!response.ok) {
        throw new Error(`http error! status: ${response.status}`);
    }
    return await response.json();
}
```

see the `async` keyword in front of the function declaration? that's the key. it tells javascript that this function is going to be doing some async work. then, inside the function, the `await` keyword pauses execution at that point until the `fetch()` operation is completed. this means that `response` will only be available once the fetch finishes. and same for `response.json()`.

now, how would you use this in a react component? you'd typically put this kind of logic inside a component's function, or within a custom hook:

```javascript
import React, { useState, useEffect } from 'react';

function UserProfile({ userId }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadUserData() {
      try {
        const userData = await fetchUserData(userId);
        setUser(userData);
      } catch (error) {
        console.error("failed to load user data", error);
      } finally {
        setLoading(false);
      }
    }
    loadUserData();
  }, [userId]); //important to include dependency to handle userId changes

  if (loading) {
    return <p>loading user data...</p>;
  }

  if (!user) {
    return <p>failed to load user data.</p>;
  }

  return (
    <div>
      <h1>{user.name}</h1>
      <p>{user.email}</p>
    </div>
  );
}

export default UserProfile;
```

here, we use `useEffect` to trigger the `loadUserData` function which is an async function using `async/await`. the loading state is handled properly using `setLoading`. we only render the user info if the user is not null. this pattern avoids issues with race conditions, and it is way more readable than the typical callback hell i saw when i was starting out. that's why, i prefer async await, is a life saver.

now, if for whatever reason async await isn’t your cup of tea, maybe due to older versions of javascript, you can always use promises and the `.then()` method, as an alternative. the `fetch` api actually returns a promise, so you can handle it explicitly like this:

```javascript
function fetchUserDataWithThen(userId) {
    return fetch(`/api/users/${userId}`)
    .then(response => {
        if (!response.ok) {
            throw new Error(`http error! status: ${response.status}`);
        }
        return response.json();
    })
    ;
}
```

the returned promise from `fetch` is chained with a `.then()` which will execute only after the request is completed and its response is available. the response will then be converted to json. you can then integrate it in a react component like this:

```javascript
import React, { useState, useEffect } from 'react';

function UserProfileWithThen({ userId }) {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        setLoading(true);
        fetchUserDataWithThen(userId)
        .then(userData => {
            setUser(userData);
        })
        .catch(error => {
            console.error("failed to load user data", error);
        })
        .finally(() => {
            setLoading(false);
        });
    }, [userId]);


    if (loading) {
        return <p>loading user data...</p>;
    }

    if (!user) {
        return <p>failed to load user data.</p>;
    }

    return (
        <div>
            <h1>{user.name}</h1>
            <p>{user.email}</p>
        </div>
    );
}

export default UserProfileWithThen;
```

this approach does essentially the same thing as the async/await version, but some might find it a little less intuitive. it uses a more nested structure with `.then()` and `.catch()`, and requires a `.finally()` to handle loading state correctly. i personally prefer the clarity of async/await, but this is equally valid.

finally, there is the react hooks way, which i partially already used. specifically the `useEffect` hook, it can help a lot when you want to trigger a function once the component is mounted, or when specific dependency variables change, like the user id. i mean, think of it as an automated handler for those moments in your component lifecycle that need some asynchronous magic, as react renders for the first time it executes the callback and whenever the list of variables changes it also executes it again.

as you can see, using `useEffect` in our examples is what ties everything together. it ensures that our async calls are triggered at the appropriate time and with the right dependencies.

a quick note: it’s critical to handle errors when making api calls. as you can see, i've included `try...catch` blocks with async/await and `.catch()` with promises to catch any network errors or other issues. forgetting to do so can lead to some very difficult debugging situations. a good practice is to implement a general error handler that can display user friendly messages, or at least log errors for later investigation.

one more important detail. when working with react functional components, remember that each render is a new scope. this means, each render of `UserProfile` component is actually different `useEffect` call. and the closure remembers the state of the `userId` when that `useEffect` hook was created. that is why the dependency array is important in `useEffect`, it keeps it up to date when the `userId` changes, and this is why it handles the user id changes. i've seen bugs happening just because of missing dependency, and the component would just never update when the `userId` changed, just because that component was using a stale `userId`.

one more thing, do not call an async function directly inside a `useEffect` callback. always define an internal async function and call it right after, like i showed in the example. if you do not do it, you're not returning a proper `cleanup` function, and you might get warnings from react development console.

now, i know that this might seem a lot at first, but once you start using these patterns, they become second nature. i mean, when i started learning react, this type of async handling was a bit frustrating and caused me a fair amount of pain, but i learned a lot through trial and error, and i even started understanding a bit more of how javascript works under the hood. it's funny when you start to appreciate javascript's strange ways, huh? anyway, remember that practice makes perfect.

as for resources, there are plenty of places to read more. "eloquent javascript" by marijn haverbeke is a great deep dive into javascript and its asynchronous nature. also "react documentation" is very complete and provides examples of these scenarios, so you should definitely read it. and always read the newest versions, things change very fast in the javascript world. and of course, a good book about functional programming may also teach you a bit about how to handle data flow in this type of situation, which is very beneficial.

hope this clears things up. if you have more questions, feel free to ask.
