---
title: "accessors are only available when targeting ecmascript 5 higher?"
date: "2024-12-13"
id: "accessors-are-only-available-when-targeting-ecmascript-5-higher"
---

Alright so this old chestnut huh Been there done that got the t-shirt and probably a few compiler warnings too Let me tell you about the time I wrestled with this accessor thing back in the day It was the early days of JavaScript frameworks you know pre-React pre-Vue pre-everything It was a wild west of prototypes and weird behaviors I was tasked with building this data binding library it was gonna be the next big thing I swear but yeah that didnt quite pan out

Anyway this data binding thingy needed to automatically update the UI when the data changed Sounds simple right ha Yeah it was everything but I started by simply using objects and properties You'd update it and things would update that was the whole goal But then I stumbled upon this little gem the world of get and set accessors and boy did things get complicated

The problem was I was supporting this old Internet Explorer version which didn't have them it was like IE8 or something I cant quite remember which cursed IE version it was But I quickly discovered that these `get` and `set` keywords were exclusive to ECMAScript 5 and later You know `get` and `set` accessors are what allow you to intercept property access and property assignment which is perfect for stuff like data binding

Basically back then the browser I was supporting didn't know what to do with that syntax it would throw errors left and right causing complete chaos and browser crashes It was just a mess pure coding madness So I had to get creative I couldn't just ignore all users using older browsers I had to come up with an alternative solution that could work across the board

The simplest answer was the old Object defineProperty and a bunch of extra code which is how I did it back then

Here's the basic concept of how I tackled it back then you need to use `Object.defineProperty` in these situations It's not as convenient as `get` and `set` but it gets the job done

```javascript
function defineObservableProperty(obj, propertyName, initialValue) {
  let _value = initialValue;
  Object.defineProperty(obj, propertyName, {
    get: function() {
      console.log("Getting", propertyName, _value)
      return _value;
    },
    set: function(newValue) {
      console.log("Setting", propertyName, "old value", _value, "new value", newValue)
      _value = newValue;
        // in reality you trigger some other stuff like updating the UI or whatever here
        // for example lets say we have a callback function that we want to use here
        if(obj.onChange){
            obj.onChange(propertyName, newValue, _value);
        }
    },
      enumerable: true,
      configurable: true // this is important to allow redefinition later
  });
}
// Example
const myObj = {};
defineObservableProperty(myObj, 'name', 'John');
myObj.name = 'Jane'; // outputs set message in console
console.log(myObj.name); // outputs get message in console
```

This is a bit verbose yes but it's the only way you can achieve the same behavior of get and set if you do not support ES5 or more It works even on the ancient browsers I supported back then It basically creates the getter and setter using `Object.defineProperty` which does exist in older JavaScript versions

You see the key here is the `Object.defineProperty` method. It's an older API that lets you define properties on objects with more control than just plain assignment. You can specify things like whether the property is enumerable or configurable which I show above in the code but most importantly you can provide the `get` and `set` functions to achieve the same behavior as the `get` and `set` accessors from ECMAScript 5 and above. It's like the granddad of the newer syntax. It works just fine. Just a lot more code

The `enumerable` flag makes sure that the property shows up in `for...in` loops and stuff and `configurable` allows you to redefine the property later. It was important in my use case because I was dynamically creating these properties a lot.

Now back to my old library nightmare This simple example was the main building block but I had to abstract this so that it would be easy to use across multiple properties and multiple objects So I would create my own custom helper function that would wrap this `Object.defineProperty` thingy in a function that could easily be re-used

And to complicate things even more my code was supposed to be backward compatible with older browsers It was a pain. I had to handle situations where there was no `Object.defineProperty` too which at that point it was an awful javascript version but there was not much I could do about it

So I had to use some feature detection logic I had to check if this method existed if so I'd use my custom wrapper and if not then I would have to use an old plain object which wouldn't really have any of the data binding features or some fake feature to simulate it that would not be that efficient but it had to be done

Here's roughly what that detection and fallback logic looked like at the core of my old library.

```javascript
function createObservable(obj) {
  if (Object.defineProperty) {
      console.log("using Object.defineProperty")
    for (const key in obj) {
      if (obj.hasOwnProperty(key)) {
        defineObservableProperty(obj, key, obj[key]);
      }
    }
  }
    else {
        console.log("using a simple old fallback")
        // very old browser here doing nothing much in reality
        // i would implement a fallback here that did the most basic feature to make it somewhat work
    }
  return obj;
}
// Example
const myObj2 = createObservable({name:'Bob',age:42});
myObj2.name = "Robert";
console.log(myObj2.name);
```

This was how I had to do it. The browser detection was primitive I know but it had to do the trick in a quick and dirty way. It was what it was. But this illustrates my approach for handling different browsers with different levels of support. If you know what I mean you know

The cool thing about this approach though is that you could have multiple properties with their own `get` and `set` behaviors all using the same function. You can do a lot with it. Even when I had to support old browsers that wouldn't just cooperate. Its a great example of how we had to make the best of old browser compatibility issues

The truth is that nowadays you can pretty much get away with only using the standard `get` and `set` accessors most of the time which is great because these old days were really difficult and full of workarounds and hacks. It was definitely a headache.

Now let's get back to the original question If you are encountering this error `accessors are only available when targeting ecmascript 5 or higher` you're most likely trying to use the `get` or `set` keywords in an environment that does not support them or in other words an environment that only support older versions of javascript The solution is either

1 Change the environment if possible if not

2 transpile your code using a tool like Babel which will convert the `get` and `set` keywords to the `Object.defineProperty` equivalent that I showed earlier or

3 Implement the old `Object.defineProperty` as a fallback solution for those old environments in your specific project

The solution number two which is using Babel is the most common solution nowadays because it is easier to handle and less error-prone than writing this code manually so you just configure Babel to transpile the newer syntax into older one and it would work just fine in most cases.

And just so you know, even when you transpile, some features might not work exactly as intended. It's not a perfect process. For instance, you might lose some performance, debugging can be harder, or a tiny feature might just work a bit different in old browsers than in modern ones

And one last tip for you. If you are working with a very old environment then be sure to test your code in that environment as much as possible because bugs do popup where you least expect them. I remember once I worked so much to try and get a feature working that I forgot to use the correct versions in my test browser and it didn't work as expected when I deployed it to production. Never again would I do that mistake again. We all learn one way or another right? *laughs sarcastically*. But for real testing is really important

And for resources I wouldn't recommend any specific links because links die you know It's better to study books or standards instead. If you want to learn more about these kinds of stuff I recommend reading the ECMAScript specification directly. It's a bit dense but it's the definitive source of truth for JavaScript. I also recommend reading books like "JavaScript: The Definitive Guide" by David Flanagan it's a great resource to understand how JavaScript works in general and how these details like accessors work in practice If you are diving deep you should look for some old javascript books too. It's surprising how some of the details that are not as popular as they used to be but they still make sense in some situations

And that's pretty much it. Accessors and browser compatibility. A tale as old as time for us developers. It's all part of the game. I hope that this helped you in one way or another
