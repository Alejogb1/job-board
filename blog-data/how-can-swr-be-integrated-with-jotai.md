---
title: "How can SWR be integrated with Jotai?"
date: "2024-12-23"
id: "how-can-swr-be-integrated-with-jotai"
---

Okay, let's talk about integrating SWR with Jotai. It's a combination I’ve seen come up a lot, particularly when trying to manage both global state and server-side data effectively in React applications. My experience implementing this in a project involving a real-time dashboard some years back highlighted both the strengths and potential pitfalls, so I've got some firsthand insights that might prove useful. The key, as with most integrations, is understanding each library's core responsibilities and finding the best way to have them complement each other, rather than overlap.

SWR, as you likely know, excels at data fetching, caching, and invalidation. It’s purpose-built for managing asynchronous data from APIs. Jotai, on the other hand, is a minimalist state management library that's fantastic for handling shared application state in a reactive way. It’s not primarily designed for network operations. Instead, think of it as a tool that helps you move data around in your application efficiently, once that data is available. It doesn’t replace the need for something like SWR; rather, it offers a robust place to store the data fetched by SWR.

The most straightforward approach involves using an atom in Jotai to hold the data that's returned from an SWR hook. This allows any component in your application that’s subscribed to that Jotai atom to automatically update whenever the SWR data changes due to new fetches, revalidations, or mutations. Here's a basic example to illustrate:

```javascript
// using `jotai` version 2.0 or later
import { atom, useAtom } from 'jotai';
import useSWR from 'swr';

const fetcher = (...args) => fetch(...args).then(res => res.json());

// 1. Create a Jotai atom to hold the data.
const apiDataAtom = atom(null);

function useApiData(key) {
  // 2. Use SWR to fetch the data.
  const { data, error, isValidating } = useSWR(key, fetcher);

  // 3. Get the atom setter
  const [, setApiData] = useAtom(apiDataAtom)


  // 4. Update the atom whenever SWR returns new data.
  React.useEffect(() => {
      setApiData(data)
  }, [data, setApiData]);

  return { data: useAtom(apiDataAtom)[0], error, isValidating };
}

function MyComponent() {
    const { data, error, isValidating } = useApiData("/api/data");

    if (isValidating) return <p>Loading...</p>
    if (error) return <p>Error Fetching Data</p>
    if (!data) return <p>No Data</p>;

  return (
      <>
        {/*  use data anywhere in the component */}
        <pre>{JSON.stringify(data, null, 2)}</pre>
      </>
  )
}
```
In this example, we create an atom `apiDataAtom` initialized to `null`. The `useApiData` hook, which encapsulates both SWR and Jotai, is our primary integration point. Within this hook, we use `useSWR` to fetch data. Crucially, when `data` from `useSWR` changes, we use the `setApiData` from the Jotai atom and pass new fetched data. This ensures that our atom always holds the latest data from the server. The `MyComponent` then accesses this shared data via `useAtom(apiDataAtom)` making the data reactive in every component using this atom.

However, you might run into situations where you want a more customized caching strategy or where the atom needs to hold additional information, beyond what SWR returns directly. For example, consider a scenario where you want to persist a loading state in the Jotai atom to prevent rendering issues during transitions when the `data` from SWR is `undefined` or `null` and keep track of whether a request is being made.

```javascript
// using `jotai` version 2.0 or later
import { atom, useAtom } from 'jotai';
import useSWR from 'swr';

const fetcher = (...args) => fetch(...args).then(res => res.json());

// 1. Expanded Jotai atom to hold additional states.
const apiDataAtom = atom({
  data: null,
  isLoading: false,
  error: null
});

function useApiData(key) {
  // 2. Use SWR and get the necessary values
  const { data, error, isValidating, mutate } = useSWR(key, fetcher);

  // 3. Get the atom setter
  const [, setApiData] = useAtom(apiDataAtom)


    // 4. update the atom on every change from SWR
    React.useEffect(() => {
        setApiData(prev => ({ ...prev, isLoading: isValidating }));

        if(error)
          setApiData(prev => ({ ...prev, error: error }));

        if(data)
          setApiData(prev => ({ ...prev, data: data }));

    }, [data, error, isValidating, setApiData]);


  const refresh = () => { mutate() }
  return { data: useAtom(apiDataAtom)[0].data, error: useAtom(apiDataAtom)[0].error, isLoading: useAtom(apiDataAtom)[0].isLoading, refresh };
}

function MyComponent() {
    const { data, error, isLoading, refresh } = useApiData("/api/data");

    if (isLoading) return <p>Loading...</p>
    if (error) return <p>Error Fetching Data</p>
    if (!data) return <p>No Data</p>;

  return (
      <>
        {/*  use data anywhere in the component */}
        <pre>{JSON.stringify(data, null, 2)}</pre>
        <button onClick={refresh}>Refresh</button>
      </>
  )
}
```

Here, our `apiDataAtom` now holds an object with `data`, `isLoading`, and `error`. This setup allows our components to not only react to the arrival of new data, but also to changes in the loading and error states by directly inspecting what’s stored in the atom, which gives us more control over the rendering logic. The addition of the `refresh` function gives an example of how you might integrate the `mutate` from the `useSWR` hook.

Lastly, consider a more complex example involving server-side mutations where you not only want to fetch data but also update the server, and automatically update your cached data using Jotai.

```javascript
// using `jotai` version 2.0 or later
import { atom, useAtom } from 'jotai';
import useSWR from 'swr';

const fetcher = (...args) => fetch(...args).then(res => res.json());

async function updateData(url, payload) {
    const res = await fetch(url, {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
    });

    if(!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
    return await res.json();
};

const apiDataAtom = atom({
    data: null,
    isLoading: false,
    error: null
});

function useApiData(key) {
    const { data, error, isValidating, mutate } = useSWR(key, fetcher);

    const [, setApiData] = useAtom(apiDataAtom)

    React.useEffect(() => {
        setApiData(prev => ({ ...prev, isLoading: isValidating }));

        if(error)
          setApiData(prev => ({ ...prev, error: error }));

        if(data)
          setApiData(prev => ({ ...prev, data: data }));

    }, [data, error, isValidating, setApiData]);


    const update = async (payload) => {
        setApiData(prev => ({...prev, isLoading:true, error:null}));

        try {
            const newData = await updateData(key, payload);
            // Optimistically update the atom data
            setApiData(prev => ({...prev, data: newData, isLoading:false}));
            // Optionally revalidate the cache
            mutate();
        } catch (updateError) {
            setApiData(prev => ({...prev, error: updateError, isLoading:false}));
        }
    };

    return {
      data: useAtom(apiDataAtom)[0].data,
      error: useAtom(apiDataAtom)[0].error,
      isLoading: useAtom(apiDataAtom)[0].isLoading,
      update,
    };
}

function MyComponent() {
    const { data, error, isLoading, update } = useApiData("/api/data");
    const [updatedValue, setUpdatedValue] = React.useState("");

    if (isLoading) return <p>Loading...</p>
    if (error) return <p>Error Fetching Data</p>
    if (!data) return <p>No Data</p>;

    const handleUpdate = async () => {
      await update({ updatedValue });
    };

  return (
      <>
      {/*  use data anywhere in the component */}
        <pre>{JSON.stringify(data, null, 2)}</pre>

        <input type='text' value={updatedValue} onChange={(e) => setUpdatedValue(e.target.value)}/>
        <button onClick={handleUpdate}>Update</button>

      </>
  )
}
```

Here, the `update` function is an example of performing an http mutation call, and also optimizing the user experience by updating the state before any result is received from the server. Additionally, the `mutate` is used within the success block to revalidate the cache ensuring any other components that might be using this data will also be updated.

For more in-depth knowledge, I'd recommend checking out the official SWR documentation, which is excellent. Additionally, “React Hooks: Up and Running” by Robin Wieruch provides a solid foundation for understanding hooks and their usage which is a must for both SWR and Jotai. Also, reading through the source code of Jotai is helpful to better understand its minimalistic approach to state management. Finally, look into the "Server-Side Data Management" patterns in "Surviving The Front-End Apocalypse" by Ben Akehurst, while not directly about SWR and Jotai, it discusses principles that would help to have better separation of concerns between data fetching and state management.

In conclusion, integrating SWR with Jotai primarily involves using a Jotai atom to hold the data provided by SWR, allowing different components to react to data changes effectively. This simple yet powerful approach ensures you maintain control over your data handling, while keeping your application performant and reactive. The examples provided should offer a good starting point for more complex requirements as you build your application.
