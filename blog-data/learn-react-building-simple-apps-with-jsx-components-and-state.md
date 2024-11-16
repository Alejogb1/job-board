---
title: "Learn React: Building Simple Apps with JSX, Components, and State"
date: "2024-11-16"
id: "learn-react-building-simple-apps-with-jsx-components-and-state"
---

dude so i watched this totally rad video on, like,  building a simple react app and man oh man was it a trip  it was all about teaching total newbies how to get started with react which is this super popular javascript library for building user interfaces basically you know how websites and apps have all those buttons and stuff that react makes it easier to manage all that jazz the whole thing was aimed at people who are like totally clueless about coding  and trust me i was right there with them at the start

the setup was pure comedic gold this guy the instructor right he starts off looking all serious and professional like he's about to teach quantum physics or something then bam he slips and almost falls over his keyboard  it was the most relatable thing ever instantly i was like yeah dude i get it coding is a chaotic mess sometimes  that's what i loved it set the tone for the whole video  it was gonna be fun easygoing and not some stuffy lecture


one of the key moments was when he explained jsx it's this weird thing in react where you kinda mix javascript and html  it's like  "what even is this sorcery" at first but once it clicks it's amazing  imagine you're building a button right normally you'd use pure html like this


```jsx
<button>click me</button>
```

super simple right but in react using jsx you can do something way cooler like this


```jsx
function MyButton({ text }) {
  return (
    <button onClick={() => alert('button clicked!')}>
      {text}
    </button>
  );
}

<MyButton text="click me!"/>

```


see the difference this lets you embed javascript directly into the html so you can make things dynamic like adding that alert when the button is clicked  it's like magic i tell ya  this guy in the video explained it using the analogy of lego bricks each brick is a component and you snap them together to build a whole website or app  so clever


another key idea was components dude  components are like the building blocks of every react app they are reusable bits of code that do one specific thing  like a button a text input or a picture you can reuse components in multiple places saving you tons of time and making your code way more organized  think of it like a factory  you've got a button factory and a text box factory both make their pieces and they're used all across the website it's super efficient. the video showed examples of making a simple header component that displayed text and a navigation bar component  it really helped me visualize how components fit together


and another totally mind blowing moment the instructor showed this thing called state  state is basically the data that your react app uses  it's like the app's memory  imagine a to-do list app the state would be the list of tasks  when you add a new task the state updates and the app re-renders to show the new task  the video showed a simple counter example using `useState` hook which is a special react function that manages state


```jsx
import React, { useState } from 'react';

function Counter() {
  const [count, setCount] = useState(0);

  return (
    <div>
      <p>You clicked {count} times</p>
      <button onClick={() => setCount(count + 1)}>
        Click me
      </button>
    </div>
  );
}

export default Counter;
```

this code snippet shows a counter that starts at 0 and increases by 1 each time you click a button  the `useState(0)` creates a state variable called `count` and initializes it to 0 `setCount` is the function that updates the value of count the key part is how react automatically re-renders the component whenever the `count` state changes  it's like magic again  that was some serious enlightenment for me


then theres the part about props props are like arguments that you pass to components they are the ways you pass information from one component to another  think of it like this you have a button component and you want to change the text on the button  you would pass the text as a prop  it's like saying "hey button component here's the text you should display"


```jsx
function Greeting(props) {
    return <h1>Hello, {props.name}</h1>;
}

<Greeting name="Alice" />
```

this snippet shows a simple greeting component that takes a `name` prop and displays a personalized greeting  it's such a simple yet powerful concept


the whole video was super visual too i remember this one part where he used these colorful animations to show how data flowed between components it was way easier to understand than just reading code  and he even made a funny joke about debugging  he said something like "debugging is like hunting for a tiny gremlin in your code" that’s exactly how i felt  he made the whole process less intimidating


finally the resolution of the video was all about building a simple todo list app using all the concepts he covered  it wasn't some massive complex app but it was perfect for demonstrating how everything worked together  it was a pretty solid way to solidify everything he taught that way i could literally see how all these concepts – components jsx state and props – interacted with each other to build a functioning app it totally blew my mind to see how easily you could build something useful using react


so yeah dude that video was a total game changer for me it took react from some scary mystical thing to something i feel like i can actually wrap my head around now  i mean i still have tons to learn but i feel way more confident about getting started with react  it's all about those small steps and this video was the perfect first step   highly recommend checking it out if you're even remotely curious about web development  you won't regret it trust me  it's not as scary as it looks honestly   you'll be building rad websites in no time  just remember the lego bricks and the gremlins  those two analogies alone really helped me stick with it  you got this dude  let me know if you find it !
