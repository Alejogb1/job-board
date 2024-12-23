---
title: "ambiguous indirect export default error?"
date: "2024-12-13"
id: "ambiguous-indirect-export-default-error"
---

 so you're hitting the "ambiguous indirect export default" thing yeah I've been there more times than I care to admit Man that one's a classic pain in the rear

let's break this down from a position of been-there-done-that I remember back in my early days working on this big React project that was supposed to be the next big thing before it totally fizzled out because of over-engineering you know the type We had a sprawling component library and the whole "export default" thing was causing all sorts of chaos It was like trying to herd cats Seriously

The core issue as you might be kinda seeing is when you're trying to export a module's default export indirectly through another module It's like trying to pass a message through three people one person mumbles then other one misshears it and the third one tells it totally wrong to the recepient Javascript kinda freaks out at this indirectness because it doesn't know what the heck the actual underlying default export is supposed to be

Let's say you got moduleA js which has like this

```javascript
// moduleA.js
const myValue = "Hello from A"
export default myValue;
```

And then moduleB js tries to re-export it like so

```javascript
// moduleB.js
export { default } from './moduleA';
```
Finally moduleC js tries to import it

```javascript
// moduleC.js
import something from './moduleB';

console.log(something); // Will not work sometimes because its ambiguous
```

Now sometimes that actually works but that's the problem It's inconsistent Depending on your bundler webpack rollup or whatever and its configuration you might not hit that error every time but its lurking there always a potential time bomb

The problem is the bundler loses the track of the underlying concrete default export. It sees an indirect export it scratches its head and goes "Uhh I have no clue what the actual thing being exported is" This is why you are hitting that ambiguous indirect export default error. It’s javascript’s way of saying “Dude I am not psychic”

So what's the fix I mean besides rewriting everything which no one wants to do right

Well you have a few good options all depending on your setup and your preferences The first and often most direct approach is to just to re-export with a named export and then re-export default from that named export. This is how I've always done it in personal projects and its the easiest and it solves that ambiguity problem I swear to my code editor

Here is that last code example re written with the named export approach

```javascript
// moduleA.js
const myValue = "Hello from A"
export default myValue;
```

```javascript
// moduleB.js
import valueA from './moduleA';

export { valueA as myValueA };

export default valueA;
```

```javascript
// moduleC.js
import something from './moduleB';

console.log(something);
```
See we made it easy for the bundler to understand by making it explicit that we are taking the valueA from moduleA and re-exporting it using a named export and then using that named export to re-export the default export Its verbose but that is often what you need to make bundlers happy.

Another approach which I've used in bigger teams is to avoid indirect defaults altogether as much as possible I mean its good practice anyway Always prefer named exports when feasible It forces you to be explicit and reduces a lot of potential confusion down the line So If you have more control over the whole project prefer to not use indirect default exports at all its much better to do named exports and just name your import as the same as the default export

So instead of this

```javascript
// moduleA.js
const myValue = "Hello from A"
export default myValue;
```

```javascript
// moduleB.js
export { default } from './moduleA';
```

You just name the export and re export that

```javascript
// moduleA.js
const myValue = "Hello from A"
export { myValue as default };
```

```javascript
// moduleB.js
export { default as myValue } from './moduleA';
```
```javascript
// moduleC.js
import { myValue } from './moduleB';

console.log(myValue); // Should work much better now
```

This is a much more robust approach as it makes the default export more explicit This removes the ambiguity from the equation If you don't want to do named export just do a straight import default and re export with a default export that is also explicit That always works as well. Just don't do indirect default exports if you can avoid it

Now remember this is mostly a bundling and tooling issue at the end of the day it depends on how your bundler handles that situation I have seen inconsistencies between different versions of the bundlers and their respective plugins You gotta dive into their documentation and even look into their source code sometimes to understand what is going on

One more thing to remember this sometimes happens with Typescript If you have types that are not being imported explicitly and they are being referenced in a indirect export situation then Typescript might also complain I'm not a big fan of Typescript but I have dealt with that as well

So for solid resources I wouldn't tell you to go to those silly Stackoverflow discussions I mean they help in some level but you need to dig deeper You want to check out stuff like "Javascript Modules: A Deep Dive" by Axel Rauschmayer it really breaks down the module system and the ins and outs of imports and exports That book saved my behind countless times And also look at the documentation of your bundler webpack rollup or parcel you need to know those like the back of your hand

Don't blindly copy and paste the code examples you should analyze them and then apply them to your specific situation There is no magic bullet and always be prepared to debug the heck out of your code When in doubt just log it all and try to understand what is going on behind the scenes

Oh and one last thing I was debugging a similar issue once and realized that I had just copy pasted some code I didn’t understand from some guy on the internet it made me look at my own code much better lol So make sure you are not copy pasting blindly and not really understanding what you are doing I had my laugh about that experience.

So to summarize we went through the problem of ambiguous indirect default exports and the problems they cause we talked about how they are a common pain point especially when dealing with module bundling And then we covered a few solutions named exports explicit defaults and a general preference to avoid them we also covered tooling issues and the fact that it might come from other places

So go forth and conquer and if you are hitting this again remember what we talked about here it will save you some headaches along the way I’m sure of it

Good luck with that
