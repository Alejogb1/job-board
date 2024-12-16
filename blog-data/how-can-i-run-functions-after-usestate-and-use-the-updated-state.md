---
title: "How can I run functions after useState and use the updated state?"
date: "2024-12-16"
id: "how-can-i-run-functions-after-usestate-and-use-the-updated-state"
---

Alright, let's tackle this. I've seen this particular hurdle countless times in react projects – the timing dance between `useState` and wanting to immediately utilize that updated state. It's a common point of confusion, and getting it nailed down is absolutely key for smooth, predictable behavior. Let me walk you through it based on a few situations I’ve encountered personally, and hopefully, it'll clear up any ambiguity you might have.

The core issue stems from the asynchronous nature of state updates in React. When you call a state setter function from `useState`, like `setState(newValue)`, React doesn't instantly change the state and immediately re-render the component. Instead, it schedules this update. This scheduling enables React to batch multiple updates, optimizing performance and preventing unnecessary re-renders. The result is that if you try to access the state immediately after calling its setter, you’ll still get the previous value. We need to use a different mechanism to ensure code executes after the state has actually updated and the component re-rendered.

One scenario that always pops up, and something I dealt with extensively during a large-scale data dashboard project, involves fetching data based on a filter selected by the user. Let's say we have a `filter` state managed by `useState`, and we want to perform an api request to update the displayed data whenever the filter changes. Trying to fetch data immediately after the filter state update will use the old filter value since `setState` is asynchronous.

To handle this, we leverage React's `useEffect` hook, which is perfect for managing side effects. The trick lies in using the state as a dependency to `useEffect`, this tells react when to re-execute the effect function.

Here's how you could structure your code:

```jsx
import React, { useState, useEffect } from 'react';

function DataDisplay() {
  const [filter, setFilter] = useState('all');
  const [data, setData] = useState([]);

  useEffect(() => {
    async function fetchData() {
      try {
        const response = await fetch(`/api/data?filter=${filter}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const result = await response.json();
        setData(result);
      } catch (error) {
        console.error("Failed to fetch data:", error);
      }
    }

    fetchData();
  }, [filter]); // filter is the dependency, effect runs when it changes

  const handleFilterChange = (newFilter) => {
    setFilter(newFilter);
  };

  return (
    <div>
      <div>
        <button onClick={() => handleFilterChange('all')}>All</button>
        <button onClick={() => handleFilterChange('active')}>Active</button>
        <button onClick={() => handleFilterChange('inactive')}>Inactive</button>
      </div>
      <ul>
        {data.map(item => (
          <li key={item.id}>{item.name}</li>
        ))}
      </ul>
    </div>
  );
}

export default DataDisplay;
```

In this example, `useEffect` takes the `filter` state as a dependency. React will execute the `fetchData` function inside `useEffect` after the component has re-rendered and the `filter` state has been updated. Every time the filter is updated using the `handleFilterChange` function, `useEffect` will trigger the data fetch. The code ensures that the API call will be made with the most recent filter value and this behavior is reliably ensured by React’s rendering cycle and `useEffect`’s dependency array, so we can be certain to get the intended results.

Another common scenario I’ve faced frequently, was handling derived state. Sometimes, you need a piece of state that’s computed directly from another state. Say you’ve got a list of items and want to display only those that meet certain criteria based on the `filter` state, you can't just update a state variable inside of the rendering function as it will trigger infinite re-renders. This is where the power of `useMemo` comes into play and it can be incredibly useful when dealing with derived state from multiple updates.

Here’s how we can use `useMemo`:

```jsx
import React, { useState, useMemo } from 'react';

function FilteredList({ items }) {
  const [filterText, setFilterText] = useState('');

  const filteredItems = useMemo(() => {
      if (!filterText) return items;
    return items.filter(item =>
        item.name.toLowerCase().includes(filterText.toLowerCase())
      );
  }, [items, filterText]); // dependencies: items and filterText

  return (
    <div>
      <input
        type="text"
        placeholder="Filter by name"
        value={filterText}
        onChange={(e) => setFilterText(e.target.value)}
      />
      <ul>
        {filteredItems.map(item => (
          <li key={item.id}>{item.name}</li>
        ))}
      </ul>
    </div>
  );
}

export default FilteredList;
```

Here, `useMemo` calculates `filteredItems` based on `items` and `filterText` and the filter logic within the `useMemo` callback function ensures the list of items are filtered based on the provided `filterText` which is modified by the user using the `<input>` component. The key thing to note here is that the computation is only done when one of the dependencies, `items` or `filterText`, change ensuring the app doesn’t repeatedly filter a list, providing a more optimized experience. This is crucial in situations where the `items` array might be large and filtering it could be expensive.

Finally, there are situations where you might be dealing with more complex state transitions and you'd need to be absolutely certain that all the updates have been applied. I have seen this come up while managing complex form states where each input could trigger updates that cascade into other derived states, or when you’re managing UI state transitions which are reliant on multiple state changes. For these scenarios, relying on the dependency array on `useEffect` or `useMemo` might not always be the clearest way. This is when the functional form of `setState` can be handy.

Consider this contrived example:

```jsx
import React, { useState } from 'react';

function Counter() {
    const [count, setCount] = useState(0);

    const increment = () => {
        setCount(prevCount => prevCount + 1);
        setCount(prevCount => prevCount + 1);
        setCount(prevCount => {
          console.log(`Count is now: ${prevCount + 1}`); // Correctly logs the final count.
          return prevCount + 1;
        });
    };

    return (
      <div>
        <p>Count: {count}</p>
        <button onClick={increment}>Increment</button>
      </div>
    );
}

export default Counter;
```

In this code snippet, when we call `setCount` three times in quick succession it’s not going to immediately render after each call. Instead, React will batch them and apply them in a single pass. What’s important to understand here is that by using a functional form of `setCount` `(prevCount => prevCount + 1)` and not using the direct value of count, we are guaranteed to perform the state update based on the latest, correct, value of the `count` variable. Within each setter call, `prevCount` will represent the latest `count` value instead of what it was when each setter was created. This pattern can be incredibly valuable in situations where you’re doing multiple updates in a row to guarantee consistency of updates, but more importantly for the purposes of the question asked, you can use it to log the final updated value by accessing `prevCount` within a functional form of `setCount` call.

In summary, understanding the asynchronous nature of `useState` is fundamental when writing React applications. Leveraging `useEffect` with the appropriate dependency array for side effects, `useMemo` for derived state, and the functional form of `setState` for complex updates ensures reliable and predictable component behavior. For a more theoretical understanding on react’s reconciliation process I would recommend reading “Thinking in React” on the official React documentation and “Advanced React” by Kent C. Dodds, which will give you a clearer picture on how and why these patterns work. They’ll certainly help solidify these concepts and get a handle on the nuances of state management in React.
