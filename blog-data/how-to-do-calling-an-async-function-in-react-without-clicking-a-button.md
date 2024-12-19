---
title: "How to do Calling an Async function in React without clicking a button?"
date: "2024-12-15"
id: "how-to-do-calling-an-async-function-in-react-without-clicking-a-button"
---

so, you're asking how to trigger an async function in react without a button click, right? been there, done that, got the t-shirt. it’s a common hurdle when you’re past the basics and start dealing with data fetching or other asynchronous operations. let me walk you through it, based on my scars, i mean, *experience*.

the core issue here is react’s rendering lifecycle. components mount, render, update, and sometimes unmount. if you want to fire off an async task *automatically*, you need to hook into these phases. a naive approach would be to simply call the function directly in the component's body – like this:

```javascript
function MyComponent() {
  async function fetchData() {
     //some api call here
     await new Promise(resolve => setTimeout(resolve, 1000));
     console.log("data fetched")
  }

  fetchData(); // <-- bad idea!

  return (
    <div>
      {/* ... component content ... */}
    </div>
  );
}
```

this, my friend, will trigger an infinite loop. every time `mycomponent` renders, `fetchdata()` gets called, which causes a state update, which triggers a re-render, and so on. it's a react equivalent of that hamster wheel meme. the key is to use react’s effects, specifically `useeffect`. `useeffect` lets you perform side effects in your functional components. it runs *after* every render, but we can control when exactly.

the basic way you would call an async method automatically would be like this:

```javascript
import React, { useEffect } from 'react';

function MyComponent() {
  async function fetchData() {
    // pretend this is a real api call, ok
    await new Promise(resolve => setTimeout(resolve, 1000));
    console.log("data fetched")
  }

  useEffect(() => {
    fetchData();
  }, []); // <-- this empty dependency array is key
  
  return (
    <div>
       {/* ... component content ... */}
    </div>
  );
}
```

now, with the `useeffect` hook, we’re telling react to execute the `fetchdata` function only once, after the initial render because of that empty dependency array `[]`. it means that the effect will only run when the component mounts, as the dependencies never change. notice how i created the async function `fetchdata` inside the component, and then called it within the effect, there are a lot of ways to structure your async code, i just like this one, it makes it cleaner for my mental model when i have to debug complex components. the async keyword lets me use `await` inside the `fetchdata` function and i don’t have to worry about promises directly.

this works, but there are some gotchas. when dealing with async functions inside `useeffect` there can be some nuances, what if you need to do some updates in the state? you will use `setstate` so there could be changes on the state and re renders, in that scenario its better to create a wrapper method within the useeffect that can handle the state updates. let me show you what i mean:

```javascript
import React, { useState, useEffect } from 'react';

function MyComponent() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  async function fetchData() {
    // simulate a failed api call
     try {
      setLoading(true);
      await new Promise((resolve, reject) => setTimeout(() => {
           const apiResponse = Math.random() > 0.5;
           if(apiResponse){
             resolve("some api data")
           }else{
              reject("api failed");
           }
      }, 1000));
        setData("some real data") //mocked it, it's ok.
     } catch(err){
        setError(err);
     } finally {
         setLoading(false);
     }

  }

  useEffect(() => {
      // wrapper function
      const loadData = async () => {
          await fetchData();
      }
     loadData();
  }, []);

    if (loading) {
     return <div>loading...</div>
    }

   if(error){
    return <div>error: {error}</div>
   }
   
  return (
    <div>
       {/* ... component content with data ... */}
      {data}
    </div>
  );
}
```
in this more elaborate example, i've introduced loading and error states. you can see the `try...catch...finally` block, it allows me to elegantly handle api errors and set the states accordingly and finally set `loading` to `false`. it’s important to set loading to false in the finally block in case any error occurs to make sure your components renders correctly. now the `loadData` is just a simple wrapper over `fetchData`. it’s more a stylistic choice than anything, but it helps keep my code more readable, at least i find it more readable, it separates state updates from the async call itself.

the empty dependency array makes this happen only after the initial render. now you might be thinking, “what if i *do* need to rerun the async function based on some prop or state changes?” that’s where that dependency array comes into play. if you put a variable in it, the effect will run *every time* that variable changes. for example, if you have a user id prop, and you want to refetch data every time the user id changes, you would do something like this:

```javascript
import React, { useState, useEffect } from 'react';

function MyComponent({ userId }) {
  const [data, setData] = useState(null);

  async function fetchData(userId) {
     await new Promise(resolve => setTimeout(resolve, 1000));
     console.log(`data fetched for user ${userId}`)
        setData({name:`user-${userId}`});

  }
  useEffect(() => {
     fetchData(userId);
  }, [userId]); // <-- dependency array!

  return (
    <div>
     {data ? data.name : "no user yet"}
    </div>
  );
}

export default function App(){
   const [userId, setUserId] = useState(1);
   return (
   <div>
      <button onClick={()=>setUserId(userId+1)}>change user</button>
       <MyComponent userId={userId}></MyComponent>
    </div>
   )
}
```

in this last example `fetchdata` is called when the component mounts initially and then each time `userid` changes in the parent component. so, if i press the button, the data re-fetches.
a very important point when you work with asynchronous code is race conditions. what if a new request is sent before the previous has resolved? this could cause stale data or overwriting the wrong state. there are various ways to tackle this, but the simplest is to abort previous requests if a new one is fired. it can be implemented with an abort controller if your fetch api supports it.

now, about resources, there's no single source of truth for this, since the react ecosystem is always evolving. but i’d recommend focusing on books that explain the javascript asynchronous model in general, for example, "eloquent javascript" by marijn haverbeke or "you don't know js: async & performance" by kyle simpson. they are great to understand the foundations. also, diving deeper into react’s official documentation on hooks, mainly `useeffect` is vital.

one last thing, i’ve been doing this long enough to know that react will not always solve all of your problems. sometimes, using a state management tool like redux or zustand or mobx can make your data flow logic much cleaner, specially in larger applications. i will leave that for another time. this is basically the way i do it, remember this code has not been tested and is just a simple explanation, in real world scenarios you will have much more complex conditions to take care of, like different ways to handle errors or loading screens or different layouts, but that is left to each individual implementation.
i've been there a million times myself, and there isn’t a single perfect solution, each use case might require a different approach, so just keep trying and reading the code of others, that is the best way. i hope this helps!
