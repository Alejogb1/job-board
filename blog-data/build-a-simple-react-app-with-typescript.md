---
title: "Build a Simple React App with TypeScript"
date: "2024-11-16"
id: "build-a-simple-react-app-with-typescript"
---

 dude so we're gonna break down this video thing right  it's all about this crazy idea called "building a simple react app with typescript" sounds boring i know but trust me it gets wild  basically the whole point is to show you how to make a little web app using react which is like this super popular javascript library for building user interfaces think of it as the lego blocks of websites and then typescript which is like...javascript but with superpowers  it helps you catch mistakes before they even happen  it's like having a super-powered spell checker for your code  the video's aim is to build something simple but show you the core concepts so you can build way cooler stuff later

so the setup is this the dude in the video starts with a completely blank slate an empty project  he's like "let's build something from scratch" which is pretty hardcore  it's a bit intimidating at first cause you just see a bunch of files and folders but he walks you through it step by step  one thing i remember clearly is him installing all these packages using npm  remember npm  that's the node package manager  it's basically the app store for javascript  you tell it what you need and it downloads it for you  it’s like ordering pizza but instead of pepperoni you get react and typescript libraries

a key moment was when he started setting up the typescript stuff  he had to configure this `tsconfig.json` file  think of it as the instruction manual for typescript it tells the compiler how to behave  it was pretty technical but the dude explained it simply  he mentioned stuff like `target` which is like the version of javascript you want to generate and `include` which tells the compiler which files to look at this was my favorite part because it makes everything way easier to comprehend

another major part was when he showed how to build a basic component in react using typescript  this is where things get really fun  react components are basically reusable pieces of your app  think buttons  forms  lists  anything that shows up on the screen  so he makes a super simple component let's call it the `greeting` component  it just displays "hello world" which is like the classic "hello world" program in programming but it’s in react  the cool part was seeing how he typed the props  props are like the inputs of the component  it was like this:

```typescript
interface GreetingProps {
  name: string;
}

const Greeting: React.FC<GreetingProps> = ({ name }) => {
  return <h1>Hello, {name}!</h1>;
};

export default Greeting;
```

see how it says `interface GreetingProps` that's typescript telling you "hey this component needs a `name` which must be a string" it's awesome because if you accidentally try to give it a number it’ll yell at you before your app crashes it's like having a super-strict but helpful butler

and then things went crazy when he started showing state management  that’s where you deal with data changing in your app like when you click a button or type in a text box  he used a simple `useState` hook which is a built-in react feature  it's basically a way to store data in your component and update it  it's deceptively simple but incredibly powerful   here's a little snippet of how that looked  this is a counter example:

```typescript
import React, { useState } from 'react';

const Counter: React.FC = () => {
  const [count, setCount] = useState(0);

  return (
    <div>
      <p>You clicked {count} times</p>
      <button onClick={() => setCount(count + 1)}>
        Click me
      </button>
    </div>
  );
};

export default Counter;
```

see how easy it is to increment the count?  the `useState` hook handles all the complicated stuff behind the scenes and you just focus on the logic

then he touched on jsx which is this weird mix of javascript and html  it's how you tell react what to display on the screen  it's like writing html but with javascript superpowers   i really liked this part it felt pretty intuitive

another cool part was when he showed how to use typescript interfaces with react components  remember that `GreetingProps` interface  that's a good example  typescript interfaces enforce the data types ensuring your components receive the correct kind of data  it helps prevent runtime errors and makes debugging a breeze   it's like having a safety net

and here's another code snippet showing how to use an interface with a form:

```typescript
interface User {
  name: string;
  email: string;
}

const MyForm: React.FC = () => {
  const [user, setUser] = useState<User>({ name: '', email: '' });

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setUser({ ...user, [event.target.name]: event.target.value });
  };

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    console.log(user);
  };

  return (
    <form onSubmit={handleSubmit}>
      <input type="text" name="name" value={user.name} onChange={handleChange} />
      <input type="email" name="email" value={user.email} onChange={handleChange} />
      <button type="submit">Submit</button>
    </form>
  );
};

export default MyForm;
```

see how the `user` state is typed as a `User` interface  that makes sure the form only accepts the right kind of data  it's a game changer for preventing errors

so the resolution the whole video basically showed you how to build a simple react app with typescript and the benefits of doing so it highlighted how typescript improves code quality makes debugging easier and prevents errors before they happen it's like having a safety net a spell checker and a personal assistant all in one  the video wasn't about building a super complex app it was about grasping the fundamental concepts so you can go on to build amazing stuff  it was a really effective tutorial  and frankly the dude explaining it all was really entertaining  i felt like i was building something cool the whole time  highly recommend checking it out  it's like leveling up your web dev skills  super fun and informative

i hope this was helpful  let me know if you have any other questions  i'm always down to geek out about coding stuff
