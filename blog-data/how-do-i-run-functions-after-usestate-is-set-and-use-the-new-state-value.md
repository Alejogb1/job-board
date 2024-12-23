---
title: "How do I run functions after `useState` is set and use the new state value?"
date: "2024-12-23"
id: "how-do-i-run-functions-after-usestate-is-set-and-use-the-new-state-value"
---

Alright, let's tackle this. I've seen this dance play out countless times in react projects, especially when developers are just getting their feet wet with hooks. The problem of executing code *after* a `useState` update and using the *new* value isn’t always immediately obvious, but there’s a very logical pattern to follow once you understand how React’s rendering lifecycle works.

The crux of the issue is that `useState`’s setter is asynchronous. It doesn’t immediately modify the state; instead, it schedules a re-render, at which point the state will be updated. Attempting to access the new state value immediately after the setter will only give you the old value. Think of it as submitting a request to react to update, and you're only going to see the effect once the update is processed and rendered.

My first brush with this was back on a project where I was building a real-time collaboration tool. We were using a `useState` hook to manage the user’s cursor position, and we needed to update the cursor of other users whenever a change was detected. I incorrectly thought I could just set the new cursor position with the setter and then *immediately* use that value to send a websocket message. Needless to say, it led to some very wonky cursor movement for everyone involved.

The key to resolving this is understanding the role of `useEffect`. This hook isn't just for side effects; it's fundamental to handling state updates that trigger other actions. A `useEffect` hook is triggered *after* react has completed rendering, after the state changes it references. So, to accomplish your goal, you need to have `useEffect` depend on the state in question and execute your function inside that hook.

Let me illustrate with some code examples:

**Example 1: Basic `useEffect` with state dependency**

```javascript
import React, { useState, useEffect } from 'react';

function ExampleComponent() {
    const [count, setCount] = useState(0);

    const handleIncrement = () => {
        setCount(prevCount => prevCount + 1);
    };

    useEffect(() => {
        console.log(`The count is now: ${count}`);
        // Here, you would perform any logic that depends on the NEW count
        // For example: make an API call, send a message, etc.
       // simulate an API call:
        setTimeout(() => {
          console.log("simulated API call triggered with new count:", count)
        }, 100)

    }, [count]); //The array of dependencies here is critical; only trigger when 'count' changes

    return (
        <div>
            <p>Count: {count}</p>
            <button onClick={handleIncrement}>Increment</button>
        </div>
    );
}

export default ExampleComponent;
```

In this scenario, the `handleIncrement` function updates the `count` using the setter. The `useEffect` hook, with `count` in its dependency array, then logs the updated value of count to the console *after* react has re-rendered the component. Critically, the function in the `useEffect` is called only *after* `count` has been updated by react and reflected in the render of the component. If you were to put the `console.log` right after the `setCount`, it would log the old value. This principle holds true for more complex logic, like initiating network requests or triggering animations.

**Example 2: Performing complex calculations with new state value**

```javascript
import React, { useState, useEffect } from 'react';

function ComplexCalculationComponent() {
    const [input, setInput] = useState('');
    const [result, setResult] = useState(null);


    const handleInputChange = (event) => {
        setInput(event.target.value);
    };

    useEffect(() => {
        if (input){
          const parsedValue = parseInt(input, 10);
          if (!isNaN(parsedValue)) {
              const calculatedResult = parsedValue * 2 + 10;
              setResult(calculatedResult);
              console.log("calculated result:", calculatedResult);
            // simulate a database write operation:
            setTimeout(() => {
              console.log("simulated database write triggered with value:", calculatedResult)
            }, 150);
          } else {
             setResult("invalid input")
             console.log("invalid input");
          }
        }

    }, [input]);

    return (
        <div>
            <input
                type="text"
                placeholder="Enter a number"
                value={input}
                onChange={handleInputChange}
            />
            {result !== null &&  <p>Result: {result}</p>}
        </div>
    );
}

export default ComplexCalculationComponent;
```

Here, we have an input field, and we want to perform a calculation and display the result every time the input changes. The `useEffect` monitors the `input` and does not trigger until the input has been updated and rendered by react. The `parseInt` function ensures the input is a number; if it is, the hook calculates a value, uses another state setter to update the value of the result, and simulates a database write. This demonstrates that multiple state updates are acceptable within the `useEffect` hook, as long as react has already completed the initial state update that caused the hook to be triggered.

**Example 3: Handling race conditions (simplified)**

```javascript
import React, { useState, useEffect } from 'react';

function RaceConditionExample() {
    const [userId, setUserId] = useState(null);
    const [userData, setUserData] = useState(null);


    const handleLogin = (id) => {
      setUserId(id);
    };


    useEffect(() => {
        if (userId) {
            fetch(`https://api.example.com/users/${userId}`)
                .then(response => response.json())
                .then(data => {
                    setUserData(data);
                    console.log("user data loaded:", data);
                })
                .catch(error => {
                    console.error("Error fetching user data:", error);
                });
        }
        else {
          setUserData(null);
        }
    }, [userId]);

    return (
        <div>
            <button onClick={() => handleLogin(123)}>Login User 123</button>
            <button onClick={() => handleLogin(456)}>Login User 456</button>
           {userData && <p>User Data: {JSON.stringify(userData)}</p>}
        </div>
    );
}

export default RaceConditionExample;

```

In a scenario involving network requests, `useEffect` becomes extremely useful for handling the response. We set the `userId`, which triggers the `useEffect`. The `useEffect` then handles the request and appropriately sets `userData` based on response. While this simplified code doesn't *fully* demonstrate race condition scenarios, the core idea of using a `useEffect` triggered on state updates to handle asynchronous operations is displayed. The user data will load and update the UI only after React has completed it's update cycle from the call to `setUserId`.

To delve deeper, I'd recommend focusing on resources that explain the intricacies of the react rendering lifecycle. The official React documentation, particularly the section on “Thinking in React” is invaluable. Another worthwhile resource would be *The Road to React* by Robin Wieruch, which provides an excellent step-by-step explanation of react hooks and their best practices. A deeper dive into the react source code itself can also be very enlightening if you're comfortable with low-level details.

In closing, using `useEffect` with the appropriate dependency array is the reliable way to execute code after a `useState` update and access the newly updated value. It’s a common pattern once you grasp the asynchronous nature of react’s state updates and a critical building block for writing clean and maintainable React code.
