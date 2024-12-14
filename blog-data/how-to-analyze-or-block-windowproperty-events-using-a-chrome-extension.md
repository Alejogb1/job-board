---
title: "How to analyze or block window.property events using a chrome extension?"
date: "2024-12-14"
id: "how-to-analyze-or-block-windowproperty-events-using-a-chrome-extension"
---

hey there,

so, you're looking into intercepting `window.property` events with a chrome extension, huh? i've been down that rabbit hole myself, and it's a bit more nuanced than it initially seems. it's not as straightforward as hooking into dom events. let me walk you through what i've learned.

basically, direct event listeners on `window` properties aren't a thing. you can't just do `window.someProperty.addEventListener('change', ...)` and expect it to work. those aren't event emitters. instead, you need to get creative with javascript's object model and proxies, and that's where things become interesting.

the main goal is to intercept when a property on the `window` object is being read or written. and that's how javascript proxies come to our aid. we can't modify the `window` object directly from an extension's content script (for very good reasons, security being primary). but we can use a proxy to create a virtual `window` object and reroute access to it while monitoring it.

first, let's talk about how you might actually do it. the core of this is going to be a content script. a content script can access the dom, and the `window` object of the page but it also needs some extra considerations about isolation. here's the general plan:

1.  we'll create a proxy around the original `window` object. this proxy will essentially act as a "man in the middle".
2.  when any property is accessed on the proxy, we'll check if it's the specific property we want to monitor.
3.  if it is, we'll log it (or block it, depending on your needs).
4.  then, and only then, we'll forward the request (get or set) to the original `window` object.

here is a basic implementation for a content script called `content.js`:

```javascript
(function() {
  const originalWindow = window;
  const monitoredProperty = 'someProperty'; // the property you want to track.

  const windowProxy = new Proxy(originalWindow, {
    get(target, property, receiver) {
      if (property === monitoredProperty) {
        console.log(`window property read: ${monitoredProperty}`);
      }
      return Reflect.get(target, property, receiver);
    },
    set(target, property, value, receiver) {
      if (property === monitoredProperty) {
        console.log(`window property set: ${monitoredProperty} to ${value}`);
        // optionally block the set operation if needed, if you like do not do the next line
         // return false to stop the setting
      }
       return Reflect.set(target, property, value, receiver);
    }
  });
  // we should also modify globalThis
  globalThis.window = windowProxy;
  globalThis.globalThis = windowProxy;

})();
```

this code snippet will log to the console each time `someProperty` is read from or set onto the global window object of the page, it acts like a `console.log` method for `window` object properties. if you want to block setting, you could return `false` instead of calling `reflect.set()` in the set function and add extra conditions depending on the value, you could even create a more complex logic, but i kept the example simple.

now, you might be thinking, "great, but what about blocking things?". well, to block, you'd simply avoid forwarding the `set` operation. in the example, that is already done as comment. but be very careful with blocking things in an extension as it can break the web page experience. a more refined blocking might involve checking the value and deciding whether to block or not. for example, imagine that the property you want to track is a `token` value, and you want to make sure that the user can not modify this specific `token` to an invalid value, or to a value that you know is not good for your needs.

here's how you could implement the blocking logic:

```javascript
(function() {
    const originalWindow = window;
    const monitoredProperty = 'authToken'; // the property you want to track

  const windowProxy = new Proxy(originalWindow, {
    get(target, property, receiver) {
        if (property === monitoredProperty) {
          console.log(`window property read: ${monitoredProperty}`);
        }
        return Reflect.get(target, property, receiver);
      },
    set(target, property, value, receiver) {
      if (property === monitoredProperty) {
        console.log(`Attempt to set ${monitoredProperty} to`, value);

        // this is where the logic for blocking is added
        if (typeof value === 'string' && value.startsWith('bad')) {
           console.warn(`blocked: setting of ${monitoredProperty} to "${value}" is blocked by extension`);
           return false; // blocks the setting
        }
      }
      return Reflect.set(target, property, value, receiver);
      }
  });
  // we should also modify globalThis
  globalThis.window = windowProxy;
  globalThis.globalThis = windowProxy;

})();
```

in this example, if you try to set `authToken` to any string that begins with "bad", the setting will be blocked. if you want to do a blocking of a non valid value you would need to validate based on your use case. it’s very important that you do this kind of blocking responsibly.

but here’s a personal anecdote, i was working on a project that was using a very old javascript library that for security reasons i was not authorized to update. i needed to block a particular call to `window.eval`. you can imagine what i had to go through to understand what they were doing, and they were doing some quite tricky stuff. i mean, at one point i thought they were trolling me. and honestly, a lot of people think that `eval` is the devil so, yeah, it was definitely something that i had to block, and it was very tricky to do it without breaking anything.

so i created the extension, and the code was basically this with the `eval` check:

```javascript
(function() {
  const originalWindow = window;
  const windowProxy = new Proxy(originalWindow, {
    get(target, property, receiver) {
      return Reflect.get(target, property, receiver);
    },
    set(target, property, value, receiver) {
      if (property === 'eval') {
        console.warn('window.eval called! blocked by extension.');
        return false; // blocks the setting of eval
      }
      return Reflect.set(target, property, value, receiver);
    }
  });
  // we should also modify globalThis
  globalThis.window = windowProxy;
  globalThis.globalThis = windowProxy;
})();
```

the main lesson i learned during this experience is that when you override object properties you need to be careful about recursion. for example in `set` you should use `reflect.set` and in get you should use `reflect.get` to avoid infinite loop.

and that's basically it! now, a couple of things to keep in mind.

*   this approach uses `proxy`, and that only works in modern browsers. if you need to support older browsers, you might need to look into `object.defineproperty`. that approach is more verbose and less elegant.

*   be extremely careful when modifying or blocking page functionality. a badly implemented extension can completely break a page. test thoroughly before deploying.

*   remember that you also have to modify `globalThis` to be the proxied version of `window` to avoid bypasses.

*   for advanced blocking or monitoring you might need a different approach based on your needs, for example, you might need to use a background script, to use web requests api, and other tools, this is just a simple example of an initial simple proxy for object interception.

*   also, these techniques may be bypassed by very clever code and for security needs you should consider other techniques.

for resources, i’d recommend a good book like "javascript: the definitive guide" by david flanagan. it has a great section about objects, properties and their manipulation. also, the mdn web docs are very helpful, especially the entries on `proxy`, `reflect` and `object.defineproperty`. also i recommend diving deeper into the ecma specifications if you want to understand how `javascript` work in depth.

i've given you the basic tools you need, it's up to you to use it effectively. good luck with your project and feel free to ask if you get stuck.
