---
title: "How to store a function as a jotai atom?"
date: "2024-12-14"
id: "how-to-store-a-function-as-a-jotai-atom"
---

alright, so you're looking at storing a function within a jotai atom, huh? been there, done that, got the t-shirt (and the accompanying stack overflow post with a very low score). it's not as straightforward as shoving a primitive in there but it's definitely doable and i've spent my share of time figuring out the cleanest way to go about it.

first, let's clarify something fundamental: jotai atoms are primarily designed to hold _data_. they're essentially reactive containers for values. functions, while technically data (in javascript at least), often represent behavior or logic. storing a function directly might not be the ideal way in all cases, but there are scenarios where it's useful. i had to use this pattern in my past when i was making a simple crud app and needed to change how the data is filtered based on which component was actively showing. back then jotai was pretty new and i was fighting with state management for a week so i feel your pain.

so here's a very simple example: let's say you have a function to add two numbers and you want to store that.

```javascript
import { atom, useAtom } from 'jotai';

// define function, this could be any function of course
const add = (a, b) => a + b;


const functionAtom = atom(add);

function MyComponent() {
  const [func] = useAtom(functionAtom);

    // use function
    const result = func(2,3)


  return (
   <div>
     <p>result is: {result}</p>
    </div>
  );
}


export default MyComponent;
```

in this example, `functionAtom` holds the `add` function itself. inside `mycomponent`, `useAtom` retrieves it, and we can call it. that's basic case, but now lets explore why this might be useful.

the above example might be to simple so lets dive deeper. now imagine you have a more complex scenario. perhaps you're developing a ui component and based on different interactions with a user you want to change the function that is run. consider this real world type scenario where you might change the filter of a list depending on which button is clicked:

```javascript
import { atom, useAtom } from 'jotai';
import { useCallback } from 'react';

const filterByEven = (data) => data.filter(item => item % 2 === 0);
const filterByOdd = (data) => data.filter(item => item % 2 !== 0);

const filterFunctionAtom = atom(filterByEven);


function MyListComponent({ data }) {

  const [filterFunction, setFilterFunction] = useAtom(filterFunctionAtom);
  const filteredData = filterFunction(data);

  const handleSetEvenFilter = useCallback(() => {
    setFilterFunction(filterByEven);
  }, [setFilterFunction]);

    const handleSetOddFilter = useCallback(() => {
        setFilterFunction(filterByOdd);
      }, [setFilterFunction]);


  return (
    <div>
        <button onClick={handleSetEvenFilter}>even</button>
        <button onClick={handleSetOddFilter}>odd</button>

      <ul>
        {filteredData.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </div>
  );
}


export default MyListComponent;
```

here, our `filterFunctionAtom` starts off holding `filterbyeven`, but we can switch it with `setfilterfunction` which uses the atom's update function. then we pass in our data and this data is dynamically filtered. this pattern is useful because it allows you to change the behaviour of other components by only controlling what function is passed by jotai. i’ve used this pattern extensively in projects where i want to keep things highly flexible and avoid deeply nested if statements. there is a drawback, as you might have noticed: if the function is recalculated on every render, jotai will trigger component updates even when the function itself hasn't changed in functionality which is bad for performance. that brings us to `usecallback` and how to memoize these functions before putting them into an atom. this is important for the performance of larger applications.

let's tweak our previous example to add memoization. it's not always needed but it’s a practice i’ve found useful. imagine your filtering function is computationally intensive or involves external api calls. if it is recalculated at every render, you might see a performance problem. i have faced this issue more than once and it is annoying to figure it out the first time, especially with complex logic:

```javascript
import { atom, useAtom } from 'jotai';
import { useCallback } from 'react';

const filterByEven = useCallback((data) => data.filter(item => item % 2 === 0), []);
const filterByOdd = useCallback((data) => data.filter(item => item % 2 !== 0), []);


const filterFunctionAtom = atom(filterByEven);

function MyListComponent({ data }) {

  const [filterFunction, setFilterFunction] = useAtom(filterFunctionAtom);
  const filteredData = filterFunction(data);

   const handleSetEvenFilter = useCallback(() => {
    setFilterFunction(filterByEven);
  }, [setFilterFunction]);

  const handleSetOddFilter = useCallback(() => {
    setFilterFunction(filterByOdd);
  }, [setFilterFunction]);


  return (
    <div>
         <button onClick={handleSetEvenFilter}>even</button>
        <button onClick={handleSetOddFilter}>odd</button>
      <ul>
        {filteredData.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </div>
  );
}


export default MyListComponent;
```

now, `filterByEven` and `filterByOdd` are memoized using `useCallback`, so they're only recreated if their dependencies change, in this case, they never change, hence they're memoized. in fact, now you can even store these functions in a different module and import them and you won't have to worry about them being re-created. it looks a bit longer, but it's a crucial habit to build for performance-sensitive applications. and believe me, you’ll thank me later. this small change saves you from a world of pain when your app starts to have hundreds of elements on the screen. i remember when i had to refactor an old app that used to freeze for a couple of seconds because of a badly written filter function on a huge dataset.

also one last point, and this is the only joke that i will make during this explanation: storing functions like this is kind of like giving your component a remote control that can change its own brain. it can be useful but also dangerous if you don’t know what you’re doing.

resources wise, i would say, forget the blogs, i never found any help there. you need to take a look at the fundamentals, so i highly recommend reading "structure and interpretation of computer programs" by abelson, sussman, and sussman. it might sound intimidating at first but is one of the best sources to understand how functions are first-class citizens in many languages and why storing them can be useful. after that, you can explore "effective javascript" by david herman for more javascript specific tips. also, if you want a deeper look into react and state management patterns, check out "thinking in react" on the official react docs, they have a couple of good insights on how to organize state and avoid unnecessary recalculations. these resources are a deep investment but will pay off.

so, in summary, storing a function in a jotai atom is possible, and in certain cases it's even necessary. make sure that the function is memoized, and that you understand that storing a function inside an atom changes the behaviour of your react code. be wary and happy coding.
