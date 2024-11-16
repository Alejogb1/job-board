---
title: "Build Simple React Apps: A Beginner's Guide"
date: "2024-11-16"
id: "build-simple-react-apps-a-beginners-guide"
---

dude so i watched this totally rad video about building a simple web app with react and it was like a rollercoaster of emotions and code  it was all about making a to-do list app which sounds super boring but trust me it was anything but  the whole point was to show you how easy it is to build something that actually works even if you're a total noob which i kinda am so it was perfect

the setup was super chill  the guy started with the basics like "why react" and all that jazz but he didn't get too preachy  he just showed us this awesome little to-do list app in action like a magic trick  it was already finished but then he went back and built it from scratch  he was showing off his coding skills but also demonstrating that building a cool little app isnt a mythical beast

one of the first key moments was when he talked about jsx  you know like how you can write html-like code inside your javascript  this blew my mind cause i thought you had to keep them super separate  but he showed how writing `<li>{item.text}</li>` just *works* and adds list items dynamically  it looked so much cleaner than the usual document.getelementbyid nonsense that i’m used to  it was like magic  seriously

another mind-blowing thing was how he handled state  remember that thing where you need to remember what the user is doing and keep track of it  he used react's `useState` hook which i initially thought was some super-advanced witchcraft  but it turned out to be pretty straightforward  it's like a little magic box that holds your app's data and lets you update it whenever something changes like adding a new task  it was so slick i wanted to cry

here’s a snippet showing a super basic example of `useState`  this is how you keep track of the to-do items

```javascript
import React, { useState } from 'react';

function ToDoList() {
  const [items, setItems] = useState([]);
  const [newItem, setNewItem] = useState('');

  const addItem = () => {
    if (newItem.trim() !== '') {
      setItems([...items, { text: newItem, done: false }]);
      setNewItem('');
    }
  };

  const toggleDone = (index) => {
    const newItems = [...items];
    newItems[index].done = !newItems[index].done;
    setItems(newItems);
  };

  return (
    <div>
      <input type="text" value={newItem} onChange={e => setNewItem(e.target.value)} />
      <button onClick={addItem}>Add Item</button>
      <ul>
        {items.map((item, index) => (
          <li key={index} style={{ textDecoration: item.done ? 'line-through' : 'none' }}>
            <input type="checkbox" checked={item.done} onChange={() => toggleDone(index)} />
            {item.text}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default ToDoList;
```

so basically `useState`  gives you this magical array `items`  and a function `setItems` to change that array  every time you add an item `setItems` updates the screen instantly

then there was this whole thing about components  which at first felt a little overwhelming  but it was basically like breaking down the app into smaller chunks  like instead of having one giant blob of code he made separate components for the input field the list of items and the button  it was like assembling lego pieces  each component is responsible for one thing and they’re all working together

here’s a glimpse of what a simple component might look like

```javascript
function AddItem({ onAddItem }) {
  const [newItem, setNewItem] = useState('');
  return (
    <div>
      <input type="text" value={newItem} onChange={e => setNewItem(e.target.value)} />
      <button onClick={() => onAddItem(newItem)}>Add Item</button>
    </div>
  );
}
```

see  it takes a function `onAddItem` as a prop  making it reusable  this is called prop drilling which i’m starting to love even though it sounds scary  but it’s actually way more manageable than having one big ball of mud of a program


he also talked about events  like how the app knows when you click the button or type something in the input field  it’s all handled by those `onChange` and `onClick` attributes  and these attributes are hooked up to functions that update the state  it was almost like watching a tiny symphony orchestra of functions working together  it was so elegant

this part was pure magic  i’m still trying to wrap my head around how it works but it was beautiful

```javascript
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

this is a super simple counter that shows how `onClick` updates the state `count` every time you click the button

the resolution was pretty straightforward  he showed us a fully functional to-do list app built using react  it was super simple  super clean  and super effective  it proved that building complex things can actually be easy  provided you break it down into manageable pieces  use the right tools  and of course  have a healthy dose of patience and maybe some caffeine


the whole thing was like a super fun tutorial  but it also made me realize how much i still need to learn  but hey  that's half the fun right  i’m already excited to start building my own little react projects and maybe even make a super awesome shopping list app or something  who knows  this video definitely ignited a fire in me  i’m ready to dive deeper  react is actually really cool  it’s not as terrifying as it initially sounded

so yeah that's my super casual and probably slightly incoherent breakdown of this amazing react video  hope you enjoyed my rambling  let me know if you have any questions  cause i’m still processing all this information myself haha
