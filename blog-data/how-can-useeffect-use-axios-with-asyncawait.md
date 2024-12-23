---
title: "How can useEffect use Axios with async/await?"
date: "2024-12-16"
id: "how-can-useeffect-use-axios-with-asyncawait"
---

Alright,  I remember back in '17, I was working on a real-time dashboard project – a classic example of needing to fetch data from an api on component mount and update the view. Using `useEffect` with `axios` and `async/await` felt clunky initially, but after some refinement, I landed on a pattern that’s been pretty reliable since. The core issue stems from `useEffect`'s design: it doesn't directly support asynchronous functions as its callback. This is because `useEffect` expects a synchronous function or a cleanup function (returned from the effect) – and an async function, inherently, returns a promise, not a cleanup callback directly.

To get around this, you essentially need to define an asynchronous function *inside* the `useEffect` callback and call it. This isn't a hack, but a necessary construction based on how React's effect system is structured. Let’s break down a common approach and see how it works in practice.

The most basic version would look something like this:

```javascript
import React, { useState, useEffect } from 'react';
import axios from 'axios';

function DataComponent() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const response = await axios.get('https://api.example.com/data');
        setData(response.data);
        setError(null); // Clear any previous error
      } catch (err) {
        console.error('Error fetching data:', err);
        setError('Failed to load data. Please try again.');
        setData(null); // Reset data in case of failure
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []); // Empty dependency array, runs only once on mount

  if (loading) {
      return <p>Loading...</p>;
  }

  if(error){
      return <p>Error: {error}</p>;
  }

  if(!data) {
     return <p>No Data Available</p>;
  }
  return (
    <div>
      {/* Render your data here */}
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </div>
  );
}

export default DataComponent;
```

Here, I've introduced a few crucial elements. The `fetchData` function is declared as `async`. Inside it, we initiate the request using `axios.get` and use `await` to handle the promise resolution. The `try...catch` block handles potential errors gracefully, and importantly, a `finally` block ensures that the `loading` state is always set to `false`. Note the empty dependency array `[]` passed as the second argument to `useEffect`; this ensures the effect runs only once after the component mounts (effectively mimicking `componentDidMount` from class components).

Now, imagine our project gets more complicated. Perhaps you need to perform a different action depending on the component's props. Or maybe you have several similar requests in one component. We can refactor to make things more maintainable and less repetitive by extracting the `fetchData` logic and using a custom hook. Let’s look at the second snippet showcasing this:

```javascript
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const useFetch = (url) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
      let isMounted = true;

    const fetchData = async () => {
        setLoading(true);
      try {
        const response = await axios.get(url);
        if(isMounted){
          setData(response.data);
          setError(null);
        }

      } catch (err) {
        if(isMounted){
            console.error('Error fetching data:', err);
            setError('Failed to load data. Please try again.');
            setData(null);
        }

      } finally {
        if (isMounted){
          setLoading(false);
        }

      }
    };
     fetchData();
     return () => { isMounted = false; };

  }, [url]);

  return { data, loading, error };
};

function DataComponent({ apiUrl }) {
  const { data, loading, error } = useFetch(apiUrl);

  if (loading) {
    return <p>Loading...</p>;
  }
  if(error){
    return <p>Error: {error}</p>;
  }
  if(!data) {
      return <p>No Data Available</p>;
  }

  return (
      <div>
        {/* Render your data here */}
          <pre>{JSON.stringify(data, null, 2)}</pre>
      </div>
  );
}

export default DataComponent;
```

This version shows us encapsulating the data fetching logic within a reusable custom hook called `useFetch`. Now, a component can call this hook by passing an API URL. Notice the additional `isMounted` flag and cleanup function within the effect. This is crucial to avoid issues with setting state on unmounted components - a common pitfall when dealing with asynchronous requests. If the component unmounts before the request is resolved, `isMounted` will be set to `false`, preventing the asynchronous callback from attempting to update state.

Finally, let's explore a slightly more complex scenario. Consider a situation where you need to cancel an ongoing request when the component unmounts. This is essential for performance and preventing memory leaks, especially when components that perform fetches are often rendered and unrendered frequently. `axios` provides a mechanism for this using cancellation tokens:

```javascript
import React, { useState, useEffect } from 'react';
import axios from 'axios';

function DataComponent() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const source = axios.CancelToken.source();

    const fetchData = async () => {
      setLoading(true);
      try {
        const response = await axios.get('https://api.example.com/data', {
          cancelToken: source.token,
        });
        setData(response.data);
        setError(null);
      } catch (err) {
         if (axios.isCancel(err)) {
             console.log("Request cancelled", err.message)
             //Request was cancelled so don't set error or re-render
             setError(null)
         } else {
            console.error('Error fetching data:', err);
            setError('Failed to load data. Please try again.');
            setData(null);
        }

      } finally {
        setLoading(false);
      }
    };

    fetchData();

    return () => {
        source.cancel('Component unmounted, request cancelled.');
    };
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }
  if(error){
      return <p>Error: {error}</p>;
  }

    if(!data) {
      return <p>No Data Available</p>;
    }

  return (
    <div>
      {/* Render your data here */}
        <pre>{JSON.stringify(data, null, 2)}</pre>
    </div>
  );
}

export default DataComponent;
```

This enhanced version incorporates an `axios.CancelToken` allowing the `useEffect`'s cleanup function to cancel any ongoing request if the component is unmounted. The `axios.isCancel(err)` check within the `catch` block ensures we don’t accidentally set an error state when a request is intentionally cancelled. This pattern is essential when your application is very reactive and many components are loading and unmounting frequently.

These examples are not exhaustive, but should provide a solid starting point on the general approaches to use `useEffect` with `axios` and `async/await`. For deeper insights into asynchronous JavaScript I highly recommend exploring “You Don't Know JS: Async & Performance” by Kyle Simpson. To enhance understanding of react hooks and effects, refer to the official React documentation thoroughly or explore “Learning React” by Alex Banks and Eve Porcello. These resources provide the theoretical underpinning and advanced techniques crucial to mastering asynchronous data fetching in React. Remember that handling state, errors, and potential cancellations is vital for creating a robust and reliable application. Always aim for modularity and code that's easy to reason about, especially when dealing with complex interactions between asynchronous operations and the user interface.
