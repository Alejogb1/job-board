---
title: "How can I include two versions of the same module in a Rails 7 importmap?"
date: "2024-12-15"
id: "how-can-i-include-two-versions-of-the-same-module-in-a-rails-7-importmap"
---

alright, so, you've run into the classic "two versions of the same thing" problem in rails, specifically with importmap. i've been there, trust me. it's like finding out your favorite library has a twin you never knew existed, and both want to live in your application. this isn't actually super rare; i encountered something similar way back when i was building a reporting tool that needed different chart libraries for different types of reports. it became a real puzzle.

here's the gist of what we're dealing with: importmap, rails' built-in tool for managing javascript dependencies, normally assumes each dependency has a single, canonical version. when you try to declare the same module twice, it gets confused, understandably so. it doesn't know which version to actually import when your code calls for it.

the key here isn't about *tricking* importmap, but rather, it's about giving each version a unique identity, so they won't step on each other's toes and the application can load them both separately when they are needed. we're not changing the module itself (we'll get to that later), just changing how importmap sees it.

here's the basic approach i've used in the past, broken down step-by-step:

**1. aliasing the module**

the most straightforward method is to use an alias. you won't be modifying the modules themself just changing the import path. let's say, for the sake of this example, we have two versions of a module, called `my_module`, located in, say, `/vendor/javascript/my_module_v1.js` and `/vendor/javascript/my_module_v2.js`. your initial `importmap.rb` would probably look something like this:

```ruby
# config/importmap.rb
pin "my_module", to: "my_module.js"
```

this, of course, won't work for both versions. instead, we are going to be more specific:

```ruby
# config/importmap.rb
pin "my_module_v1", to: "my_module_v1.js"
pin "my_module_v2", to: "my_module_v2.js"
```
*note, that we don't need the full path, just the name of the file as the importmap uses the javascript directory from the app/javascript folders to resolve relative paths*.
this changes the way you reference them in your javascript. instead of just `import myModule from 'my_module'`, you will need to be specific:

```javascript
// app/javascript/my_file.js

import myModuleV1 from 'my_module_v1'
import myModuleV2 from 'my_module_v2'

console.log("v1", myModuleV1.someFunction())
console.log("v2", myModuleV2.someFunction())

```
i used this approach many times, it is clean and simple to understand, and also it has served well to many people as i've seen online.

**2. namespaced imports**

sometimes you don't want to change the import name. and this may lead to a bad experience. we can solve this by namespacing the original module names and keep our imports as they were before. instead of adding `_v1` and `_v2` we can create a namespace that allows us to keep the original import name and reference it as we intend. this is often a better approach in my opinion when you need to import them both in the same file without the risk of them colliding and overriding the same variable.

let's create two subdirectories inside our vendor directory `vendor/javascript/v1/my_module.js` and `vendor/javascript/v2/my_module.js`.
we can modify the importmap to recognize them as the same module under different namespaces by using the `as:` attribute. it might seem like a complex addition but trust me it will make your code more maintainable.

```ruby
# config/importmap.rb
pin "my_module", to: "v1/my_module.js", as: "v1"
pin "my_module", to: "v2/my_module.js", as: "v2"
```

then we can import it as following:

```javascript
// app/javascript/my_file.js
import * as myModuleV1 from 'v1/my_module';
import * as myModuleV2 from 'v2/my_module';

console.log("v1", myModuleV1.someFunction())
console.log("v2", myModuleV2.someFunction())
```
with this approach, you can keep the same module name `my_module` but under a different namespace, `v1` and `v2`.

**3. module modification (be careful!)**

now, there might be situations where you absolutely *need* both versions to use the same import name. this is where it gets a bit tricky. and please, proceed with caution. you can modify the module itself to provide different behaviors depending on the import call. i never really liked doing that, as it goes against the core concept of separating concerns, and also is not really easy to debug if not done carefully.
let's say that in `my_module_v1.js` and `my_module_v2.js` you'll add a global variable called `version`.

```javascript
// vendor/javascript/my_module_v1.js
window.my_module_version = 'v1';

export function someFunction() {
   return 'some data from ' + window.my_module_version;
}
```

```javascript
// vendor/javascript/my_module_v2.js
window.my_module_version = 'v2';

export function someFunction() {
   return 'some data from ' + window.my_module_version;
}

```

now, in our importmap, we only pin it to one version.
```ruby
# config/importmap.rb
pin "my_module", to: "my_module_v1.js"
```
and in our file, we can import the module, and we can use this variable to determine what module has been loaded:

```javascript
// app/javascript/my_file.js
import myModule from 'my_module'

console.log(window.my_module_version)
console.log(myModule.someFunction())

import 'my_module_v2.js'
console.log(window.my_module_version)
console.log(myModule.someFunction())

```

this approach is not ideal. as it goes against what we try to achieve when we import modules. however, there are situations when you need to do it. i would always use one of the two other options if possible. modifying the module can be hard to maintain if not done correctly.

**important notes**

*   **path resolution:** rails' importmap uses the app/javascript directory as the base for resolving paths. if you place your javascript files inside the /vendor directory, you can place them inside vendor/javascript as a best practice.
*   **caching:** when dealing with different versions of the same thing, browser caching can be a real pain. make sure you have your asset pipeline setup to use different cache busting strategies to avoid stale code. if you are using the sprockets asset pipeline or something similar make sure you've got a good way to manage your assets.
*   **naming conventions:** it's tempting to try to keep the same file names. but believe me, it's better to be explicit and avoid confusion. use clear and descriptive names for your aliases, and always be extra careful when using the same names under different namespaces.

**resources for deeper dives**

if you want to go deeper into the topic of javascript modules and dependency management, here are some resources that helped me over the years. i recommend reading them.

*   "javascript modularity" by addy osmani: this is a seminal book on the topic of modularity in javascript. it covers a lot of the fundamentals and best practices.
*   "webpack: the complete guide" by sean larkin: if you want to know how module bundlers work under the hood, this book is invaluable. even if you use importmaps, understanding how webpack works will help you understand the challenges of handling different versions of dependencies. it's also helpful if you need to move from importmap to a bundler at some point.

*   a good documentation of es modules is also something you must read to better understand how javascript handles them. try to see the es modules specifications as the source of truth.

dealing with multiple versions of the same module can be tricky, but it's definitely not impossible. by carefully aliasing, namespacing or modifying the module itself, you can make importmap handle it well. and remember, clear naming conventions, and knowing what's going on under the hood will save you a lot of time and headache in the long run.

i hope this helps, and let me know if you have other questions, i'll do my best to answer based on my experience. also, it turns out that two javascript modules walking into a bar isn't as interesting as it sounds. they just started talking about scope and it was honestly super boring.
