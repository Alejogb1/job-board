---
title: "How does React JS wait for a function to execute before continue?"
date: "2024-12-15"
id: "how-does-react-js-wait-for-a-function-to-execute-before-continue"
---

alright, so you're asking about how react handles asynchronous operations, specifically how it waits for a function to complete before moving on to the next thing, it's a pretty common question, and it's crucial for building functional react apps, it's not like react just halts everything until a function finishes, it's more subtle than that.

fundamentally, javascript itself is single-threaded, which means it can only do one thing at a time. but we frequently deal with things like network requests or timeouts that take some time, and we don't want our whole app to freeze while it's waiting, react builds upon this. it relies on javascript’s asynchronous mechanisms like promises and async/await, and react’s state management and rendering process to achieve this “waiting” behavior. it doesn’t really wait in the blocking sense, more like “observes”.

let's dive into some specifics. first off, it's important to understand that react components generally re-render when their state changes or their props changes, this re-rendering is how react updates the ui. now, when we have an asynchronous function like a network request, we need a way to tell react that something has changed, and it should re-render. that’s where state management and the various react hooks come into play, particularly `usestate` and `useeffect`.

i've seen countless developers try to directly use the result of asynchronous function calls in the render part of their component. this never works the way one would intuitively think, because the asynchronous call is likely pending during the first render, leading to either undefined or outdated data being displayed. i recall a particularly messy issue back when i was working on that chat application, where the message timestamps weren’t always updating properly because the fetch call was being made directly in the render part of the component rather than properly using a hook. it was like watching the same wrong time over and over, that was very frustrating.

let's get into a practical example, consider we have a function called `fetchdata` which retrieves data from an api, if we try to call `fetchdata` directly in a component we'll have a problem:

```javascript
import React from 'react';

function MyComponent() {
  const data = fetchData(); // <-- this is problematic

  return (
    <div>
      {data && <p>Data: {data.message}</p>}
    </div>
  );
}

export default MyComponent;
```

this is not going to work because `fetchdata` is asynchronous, we need to use some state to hold the results:

```javascript
import React, { useState, useEffect } from 'react';

async function fetchData() {
  // simulate a network request
  await new Promise(resolve => setTimeout(resolve, 1000));
  return { message: 'data loaded!' };
}

function MyComponent() {
  const [data, setData] = useState(null);

  useEffect(() => {
    async function loadData() {
      const result = await fetchData();
      setData(result);
    }

    loadData();
  }, []);

  return (
    <div>
      {data && <p>Data: {data.message}</p>}
    </div>
  );
}

export default MyComponent;
```
here's the breakdown:
we use `usestate` to create a piece of state called data, which is initialized to null, then we use `useeffect` to call our asynchronous function `fetchdata` which is wrapped in another async function, `loaddata`, this ensures that `fetchdata` is executed only after the first component render (because of the empty dependency array `[]`), then when the promise from `fetchdata` resolves the `.then` part is executed and `setdata` is called, which updates the state and re-renders the component. this update of the state makes react re-render the component using the new state value, which will not be `null`. now the component will display the data.

notice the `async/await`, this makes working with promises much easier, instead of `then` chains, we can write code that looks more synchronous, but it’s important to remember it still is asynchronous, it just makes it more manageable. it basically pauses the execution of the `loaddata` function until the promise returned by `fetchdata` resolves, react doesn't wait, the javascript engine is the one waiting for the promise to resolve, then it calls the next line which sets the state and the react engine will use that state change to trigger a render.

another example where handling state is crucial is with loading states:

```javascript
import React, { useState, useEffect } from 'react';

async function fetchData() {
  await new Promise(resolve => setTimeout(resolve, 1000));
  return { message: 'data loaded!' };
}

function MyComponent() {
  const [data, setData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
        setIsLoading(true);
      const result = await fetchData();
      setData(result);
        setIsLoading(false);
    }
    loadData();
  }, []);

  return (
      <div>
          {isLoading ? <p>loading data...</p> : data && <p>Data: {data.message}</p>}
      </div>
  );
}

export default MyComponent;
```
in this example we added another state to handle the loading, `isloading`, we set it to `true` when the loading starts and `false` when it’s done, and show a “loading data…” message when it’s true, providing immediate feedback to the user, these loading states are essential in real-world applications to prevent users from thinking that the app is broken. in my past job, at that company that made all the 'best of' lists on tech blogs, we had a case where the loading state was not properly used and users would think that the app crashed when they loaded a page with a lot of data to fetch, the fix was implementing a very simple loading screen which greatly improved the user experience.

it's also common to use other hooks to manage asynchronous operations, `usecallback` and `usememo` are great for optimizing components with asynchronous operations to avoid unnecesary re-renders. for instance, if you have a slow calculation that you only want to perform when a specific input changes, `usememo` will memoize the result and only recalculate if the inputs change. if you have a function that's passed as a prop to a child component and you only want the function to change if the inputs change `usecallback` is the way to go.

there are libraries that are very popular to manage asynchronous side-effects, like `redux-saga` or `redux-thunk`, they are commonly used to handle side-effects like network calls, web socket connections and other async tasks. they give more structure and make the code easier to test. i tend to prefer `redux-saga` over `redux-thunk`, it is easier to reason about async tasks with sagas, i used to see lots of callbacks with `redux-thunk` which, while doable, can create a kind of pyramid of code that is hard to maintain and debug.

if you want to dive deeper into this kind of topic i would highly recommend checking out resources on javascript event loop, promises and async/await like "you don't know javascript: async and performance" by kyle simpson. there are also lots of good books about react architecture, such as the "fullstack react" book, these resources will help solidify your understanding and also expose you to patterns and best practices when dealing with asynchronous code in react.

in summary, react doesn't "wait" like you would expect in a synchronous environment, instead, it relies on asynchronous javascript features to manage its state and renders, react components are rendered and re-rendered based on state updates, and the way you handle asynchronous data updates depends on react hooks, if you're like me you’ll probably say that the trick is knowing how to set states and how react renders based on changes in the state, so try it a few times, it’s not like your code will start running backward if you mess up, i'm just kidding, of course.
