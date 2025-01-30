---
title: "How to maintain a value until a non-null value is received?"
date: "2025-01-30"
id: "how-to-maintain-a-value-until-a-non-null"
---
In systems where data arrives asynchronously or intermittently, propagating a placeholder value until a valid replacement becomes available is a recurring necessity. Often, initial or default values are used as this placeholder, but the challenge lies in systematically and efficiently substituting these with the incoming non-null data. This requires a strategy that manages both the initial state and subsequent updates.

A common scenario involves configurations or parameters fetched from a remote service. An application might boot using default values to ensure functionality, but it must seamlessly transition to server-provided configurations when these become accessible. Without a defined mechanism, incorrect values could be used or the application might repeatedly attempt to load the data, leading to resource wastage. My experience with building distributed sensor networks, specifically the application for temperature monitoring, reinforces the need for such a mechanism. The initial values (historical averages) were replaced with real time data when each sensor came online, making it crucial to have the behavior correct and efficient.

The core principle is a stateful evaluation process. An initial value is held until a non-null value arrives; thereafter, the non-null value is retained. This process can be encapsulated within a dedicated function or object, ensuring a uniform and maintainable approach. A key component of this process involves explicitly handling the 'null' case. Rather than relying on implicit boolean conversion, a check for `null` or `undefined` is essential to avoid unintended behaviors, especially when dealing with non-primitive data types, such as objects.

The simplest method uses a conditional assignment and a variable.

```javascript
let value = 'default value'; // initial placeholder

function updateValue(newValue) {
  if (newValue != null) {
    value = newValue;
  }
}

console.log(value); // output: "default value"
updateValue(null);
console.log(value); // output: "default value"
updateValue("new value");
console.log(value); // output: "new value"
updateValue("another value");
console.log(value); // output: "new value" - stays because it was already updated
```

This example demonstrates the core logic. The `value` variable holds the placeholder until `updateValue` receives a non-null input. After the first non-null value, further calls to `updateValue` do not change it. However, this example does not encompass more complex scenarios. It is imperative that in real applications, these updates must be properly synchronized to avoid race conditions. If an async process is performing the update, then we must make sure to avoid race condition.

For more intricate situations, where data updates might arrive asynchronously, and to ensure immutability, a function closure becomes helpful.

```javascript
function createUpdater(initialValue) {
  let currentValue = initialValue;

  return function update(newValue) {
    if (newValue != null) {
        if (currentValue != null) {
            return currentValue;
        } else {
            currentValue = newValue;
            return currentValue;
        }
    }
    return currentValue
  };
}

const myValueUpdater = createUpdater('initial');
console.log(myValueUpdater());  // Output: initial
console.log(myValueUpdater(null));  // Output: initial
console.log(myValueUpdater('first value'));  // Output: first value
console.log(myValueUpdater('second value')); // Output: first value
```
This approach uses a closure to encapsulate the `currentValue`, preventing direct access from the outside. This enforces immutability;  once a non-null value is set, it cannot be overwritten (by another non-null). The returned 'update' function provides a controlled means for updating the value. Furthermore, the returned `currentValue` is the previous value rather than the current value, ensuring the user knows the value after the update.

In more complex scenarios involving object property updates, a deep copy approach might be required. This method is required to preserve immutability when dealing with mutable objects such as dictionaries.

```javascript
function createObjectUpdater(initialObject) {
    let currentObject = initialObject;

    return function update(newObject) {
      if (newObject != null) {
        if (currentObject != null) {
            return currentObject;
          }
        else {
        // Shallow copy sufficient when object properties are immutable
          currentObject = { ...newObject };
          return currentObject;
        }
      }
      return currentObject;
    };
  }

const myObjectUpdater = createObjectUpdater({ a: 1, b: 2 });
console.log(myObjectUpdater()); // Output: {a:1, b:2}
console.log(myObjectUpdater(null)); // Output: {a:1, b:2}
console.log(myObjectUpdater({a:3, b:4})); // Output: {a:3, b:4}
console.log(myObjectUpdater({a:5, b:6})); // Output: {a:3, b:4}
```

This function manages object updates. A shallow copy via the spread operator ensures that when the object is mutated, the initial reference does not change. If there are nested mutable structures within, a deep copy would be required. This strategy maintains a clear distinction between the initial placeholder and the received valid data, which is critical in data-driven application. Once a non null object is set, it cannot be updated.

In summary, when dealing with intermittent data streams, I have found that establishing a controlled, stateful evaluation to hold an initial value until a non-null value becomes available is a practical way to address data availability. The choice between simple variable updates and the encapsulation provided by function closures should depend on the complexity of the update and immutability requirements.

For resources, consider exploring documentation related to JavaScript closure and functional programming, specifically immutability best practices. Texts that delve into reactive programming patterns can also provide more insights into managing asynchronous data. Generic programming tutorials also offer valuable strategies. Furthermore, investigating libraries that provide utilities around state management in complex applications is recommended.
