---
title: "How many prototype objects are in the chain for the following array? let arr = '';"
date: "2024-12-15"
id: "how-many-prototype-objects-are-in-the-chain-for-the-following-array-let-arr--"
---

let's talk about prototype chains, specifically for arrays, and how many prototype objects we're dealing with. it's a common area that seems simple on the surface but has some depth if you look closer. that `let arr = [];` you posted? it looks basic but under the hood, it’s got a lineage.

when you create an array like that in javascript, it doesn't just spring into existence as a bare list of elements. it's an object, and like all objects in javascript, it inherits properties and methods from its prototype. so, to answer your direct question, there are *two* prototype objects in that array's prototype chain. let me unpack that.

first, let's get really down to basics. in javascript, everything that's not a primitive value (like numbers, strings, booleans, null, undefined, and symbols) is an object. and arrays? they're objects too. special kinds of objects, with some extra built-in behaviors, but still objects.

now, for all objects, we have this hidden property called `__proto__` (or you can use the more standardized `object.getprototypeof()`). this `__proto__` property points to the object's prototype. it's the key to the prototype chain.

when you call a method on an object (or access a property) javascript first checks if that method (or property) exists on the object itself. if it doesn't find it directly, it climbs up the `__proto__` chain, checking each prototype object in the sequence until it finds the method (or property) or reaches the end of the chain. that's the prototype chain mechanism in a nutshell.

so, what about our empty array, `let arr = [];`?

1. the array object itself: it's an instance of `array`. this array object, behind the scenes, has its `__proto__` property.
2. the `array.prototype`: this is the object that all arrays inherit from. it's where all those useful array methods like `push`, `pop`, `map`, `filter` and so on come from. your `arr` array, even though empty, has access to all these because its `__proto__` points to this prototype.
3. the `object.prototype`: finally, `array.prototype` itself is also an object, and guess what? it also has a `__proto__`. it points to the `object.prototype`. this `object.prototype` is the ultimate ancestor of all objects in javascript. it's where general purpose methods like `tostring` or `hasownproperty` come from.

therefore, if we count, `arr` has one direct prototype object (`array.prototype`), which itself has its own prototype (`object.prototype`).

let's look at the code so you can visualize this:

```javascript
let arr = [];

// check the direct prototype of the array (array.prototype)
console.log(object.getprototypeof(arr) === array.prototype); // this should output: true

// check the prototype of array.prototype (object.prototype)
console.log(object.getprototypeof(array.prototype) === object.prototype); // this should output: true

// check the prototype of object.prototype (should be null because it is the top)
console.log(object.getprototypeof(object.prototype) === null); // this should output: true
```
this confirms my statement of 2 prototype objects.

to be even more explicit:

```javascript
let arr = [];
console.log(arr.__proto__ === array.prototype); // true
console.log(arr.__proto__.__proto__ === object.prototype); // true
console.log(arr.__proto__.__proto__.__proto__ === null); // true
```
it is worth mentioning, the `__proto__` is often discouraged to use directly in code. using `object.getprototypeof()` is generally a better way of dealing with prototypes programmatically.

if i was to be even more nitpicky about this, let’s go back some years. i remember when i was first learning javascript, and i was confused by this. i thought `array` was a built-in class, like in java or c++. this misconception kept me confused for a while. then i spent some time reading about prototypes, and i came across douglas crockford’s work on javascript. his writings on prototypal inheritance were a big eye-opener for me. i spent so much time trying to make it fit into class-based thinking. (and the joke here, if you get it, is there are no classes). but eventually i got the idea, that `array` is not a class at all it’s just a function that returns objects that are already prebuilt to behave like arrays.

another point where people get lost, is when the try to modify the prototype. let's say for some reason you wanted to add your custom method for all arrays. you could do something like this:

```javascript
array.prototype.myCustomMethod = function() {
  console.log("this is my custom method, and this array has " + this.length + " elements");
};

let arr1 = [1, 2, 3];
let arr2 = [];
arr1.myCustomMethod(); // outputs: this is my custom method, and this array has 3 elements
arr2.myCustomMethod(); // outputs: this is my custom method, and this array has 0 elements

```
that code will work, but i really advice you to not pollute built-in prototypes. it can cause issues down the line with other libraries. there are almost always better ways of doing what you want without polluting the `array.prototype`.

if you want to go deeper into how prototypes work, i suggest checking out "javascript: the good parts" by douglas crockford (mentioned above). it might be a bit dated, but the sections on prototypal inheritance are still totally relevant. another good book, more recent, is "eloquent javascript" by marijn haverbeke, which also covers prototypes very well. there are a few papers on the subject on academic databases as well, but those 2 books are a great start.

so, again, to summarize: the prototype chain for your empty array goes `arr` -> `array.prototype` -> `object.prototype` which is 2 prototypes, and then `null` which ends the chain. no magic, just prototypes.
