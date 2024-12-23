---
title: "How to effectively use useEffect with Axios and async/await?"
date: "2024-12-23"
id: "how-to-effectively-use-useeffect-with-axios-and-asyncawait"
---

Okay, let’s tackle this. I've seen my fair share of component lifecycles go sideways, especially when asynchronous operations like API calls enter the picture. `useEffect` with `axios` and `async/await` is a common pattern, but it's also a place where things can get messy if not handled carefully. Over the years, I've developed a few strategies to keep the code clean, predictable, and efficient. It’s more than just slapping `async` in front of a function; it’s about understanding the nuances of React’s rendering cycle and the implications of asynchronous updates.

The primary issue stems from the way `useEffect` behaves. Without a proper cleanup function, you could end up with stale closures, memory leaks, or even infinite loops, especially when dealing with changes in component props or state. We need to ensure that our API requests are properly managed and that we avoid performing updates on unmounted components.

Let's break this down into a few key areas:

1.  **Structuring the Effect:**
    The basic setup involves using `useEffect` with an asynchronous function inside it. It's critical to define this function internally to the hook, as declaring it outside can cause problems with stale closures. You also want to ensure you are using a valid dependency array to avoid unnecessary re-renders.

2.  **Error Handling and Loading States:**
    API requests aren't always successful, so you need robust error handling and a mechanism to manage the loading state. Displaying a loading spinner or an error message is vital to maintain a good user experience.

3.  **Cleanup Functions:**
    The cleanup function in `useEffect` is critical. It allows you to cancel pending requests when the component unmounts or when the dependencies change. Failure to do so can result in errors and performance issues.

Now, let’s see these points in action with some examples, showcasing real-world scenarios I've encountered.

**Example 1: Basic Fetch and State Update**

This is probably the most common scenario: fetching data and setting it in state. Here’s how I typically handle it:

```javascript
import React, { useState, useEffect } from 'react';
import axios from 'axios';

function UserList() {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);


  useEffect(() => {
     const fetchData = async () => {
      setLoading(true); // start loading
      setError(null); //clear any previous errors
      try {
        const response = await axios.get('https://jsonplaceholder.typicode.com/users');
        setUsers(response.data);
      } catch (err) {
        setError(err); //set the error
      } finally {
        setLoading(false); //end loading state
      }
    };

    fetchData();

  }, []); // empty dependency array for initial fetch only

  if (loading) return <p>Loading...</p>;
  if (error) return <p>Error: {error.message}</p>;

  return (
    <ul>
      {users.map(user => (
        <li key={user.id}>{user.name}</li>
      ))}
    </ul>
  );
}

export default UserList;
```
This example is fairly straightforward. We use `useState` to manage our data, loading state, and potential errors. The `useEffect` hook defines an internal `async` function `fetchData` and calls it. Importantly, the dependency array is empty (`[]`), so this effect only runs once after the initial render. I include the `loading` and `error` state to provide user feedback. The `finally` block ensures loading state is set to false whether the request succeeded or not.

**Example 2: Fetching Based on Props and Cleanup**

Now let’s move onto a slightly more complicated scenario: fetching data based on a prop and properly handling component unmounting. This is where cleanup becomes incredibly important.

```javascript
import React, { useState, useEffect } from 'react';
import axios from 'axios';

function UserDetails({ userId }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);


  useEffect(() => {
    let isMounted = true;
      const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await axios.get(`https://jsonplaceholder.typicode.com/users/${userId}`);
          if (isMounted) { // check if component is still mounted
             setUser(response.data);
          }
      } catch (err) {
        if (isMounted) {
            setError(err); // only set error if component is mounted
        }

      } finally {
        if (isMounted) {
            setLoading(false); //only set loading state if component is mounted
        }
      }
    };

    fetchData();
     return () => {
      isMounted = false; // set isMounted to false on unmount to prevent updates on unmounted component
    };

  }, [userId]); // dependency array: userId, fetch data when userId changes

  if (loading) return <p>Loading...</p>;
  if (error) return <p>Error: {error.message}</p>;
  if (!user) return <p>User not found</p>;


  return (
    <div>
      <h2>{user.name}</h2>
      <p>Email: {user.email}</p>
    </div>
  );
}
export default UserDetails;
```
Here, the `useEffect` hook depends on the `userId` prop. Each time the `userId` changes, the effect runs and fetches new data. Notice the introduction of the `isMounted` variable and the associated check in both the `try` and `catch` block, this is crucial to avoid setting state on an unmounted component. The cleanup function sets `isMounted` to `false`, which effectively prevents updates if a component is unmounted during an API call. I find this method to be more efficient and less error-prone than using a separate `cancelToken` on every request.

**Example 3: AbortController for Canceling Requests**

There may be instances where you need to cancel a request explicitly. This is often the case when a user quickly navigates away from a page or initiates another request before the current one is complete. For that, the `AbortController` is very helpful.

```javascript
import React, { useState, useEffect } from 'react';
import axios from 'axios';

function SearchResults({ query }) {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const abortController = new AbortController();
    const { signal } = abortController;
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await axios.get(
            `https://api.example.com/search?q=${query}`,
          { signal }
          );
          setResults(response.data);
      } catch (err) {
        if(err.name !== "AbortError"){
            setError(err);
        }
      } finally {
        setLoading(false);
      }
    };


     if (query) {
         fetchData();
     } else {
        setResults([]); // reset the results when the query is empty
     }
      return () => {
       abortController.abort(); // cancel ongoing request
    };

  }, [query]); // dependency array: query, fetch data when query changes

  if(loading) return <p>Loading results...</p>;
  if (error) return <p>Error: {error.message}</p>;
  if (results.length === 0) return <p>No results found</p>

  return (
    <ul>
      {results.map(result => (
        <li key={result.id}>{result.title}</li>
      ))}
    </ul>
  );
}
export default SearchResults;

```

In this example, we create an `AbortController` at the beginning of the `useEffect` and associate its `signal` with the `axios.get` request. The cleanup function then calls `abortController.abort()`, which cancels any pending request. The error is specifically checked to make sure that AbortErrors are handled properly and not displayed to the user.

For deeper understanding, I highly recommend reviewing “You Don’t Know JS: Async & Performance” by Kyle Simpson for foundational knowledge of asynchronous JavaScript. Additionally, diving into React’s official documentation on hooks, particularly the useEffect section, is essential. Also, the Fetch API documentation on MDN is also a good read. These resources provide the theoretical background that, combined with practical experience, will make working with `useEffect` and asynchronous operations smoother.

In conclusion, using `useEffect` with `axios` and `async/await` isn’t difficult once you understand the underlying principles. Clean and effective code requires careful management of loading states, errors, and the lifecycle of the component. These examples should help in navigating the nuances of asynchronous programming in React effectively, and I’ve found these approaches to be effective in many production environments.
