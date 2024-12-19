---
title: "Why is Rails 7 Turbolinks giving an error on a fetch page?"
date: "2024-12-15"
id: "why-is-rails-7-turbolinks-giving-an-error-on-a-fetch-page"
---

alright, so, rails 7 and turbolinks giving fetch headaches, right? i've been there, more times than i care to remember. it's a classic case of the 'javascript not playing nice with turbolinks' dance. let's break this down, from the trenches, so to speak.

the core issue here, and i bet a shiny nickel on this, is that turbolinks doesn’t behave like a traditional page reload. it intercepts link clicks and form submissions, swapping out the `<body>` content via ajax. this is great for speed, but it wreaks havoc when your javascript expects a full page lifecycle. specifically, when you're doing `fetch` requests, especially on what turbolinks considers a “page” transition within the same view, things can get… well, not predictable. it's basically a conflict between turbolinks' partial page reload logic and the assumption of a complete page refresh when you initiate fetch requests.

back in my early rails days, i remember building a complex search interface. users would click on filter buttons, and i’d use `fetch` to pull down the updated results, injecting them into the page. everything was grand until i added turbolinks for that “snappy” feel. suddenly, those ajax updates became ghosts, flickering once and disappearing. i was pulling my hair out, trust me. i spent hours `console.log`ing everything imaginable, convinced that the backend had spontaneously combusted. but no, it was good old turbolinks being too clever for its own good.

the first culprit is probably the javascript event listeners. when turbolinks navigates to a new page (or a new page section as it thinks), it doesn't refresh the entire js scope. if you've attached event listeners directly to document or window elements during the `DOMContentLoaded` event or similar, those listeners might not be re-bound when turbolinks reloads the page content. they may refer to dom elements that were destroyed and recreated during turbolinks’ swap of the page body. now, they are pointing to nothing or worse, old nodes. so when your `fetch` completes, and tries to update a dom node that javascript is tracking, it can’t find it. it's like handing a note to someone who moved out and hoping they get it.

here's a simplified example of the kind of setup that commonly fails:

```javascript
document.addEventListener('DOMContentLoaded', function() {
  const button = document.querySelector('#my-button');
  if(button){
    button.addEventListener('click', function() {
        fetch('/my_data', { method: 'GET' })
          .then(response => response.json())
          .then(data => {
            document.querySelector('#my-data-container').innerHTML = data.content;
          });
    });
  }
});
```

this code will often work fine on a full page refresh, but turbolinks will trip it. the `DOMContentLoaded` event only fires once per full page load, not on subsequent turbolinks visits. the event listener is attached once, and after turbolinks swaps content in the page, the dom referenced by `button` and `my-data-container` is gone, and replaced by new html nodes, leaving our listener hanging on to ghost elements.

the solution? well, there isn't one single cure, and it depends on your specific setup, but we can get around this common problem. usually, you need to hook into turbolinks' own events instead of the browser ones. try the `turbolinks:load` event. it's fired every time turbolinks replaces the page content. it's turbolinks’ version of the `DOMContentLoaded` event. instead of `DOMContentLoaded` use `turbolinks:load`. this allows you to reliably attach your event listeners to the fresh content.

here's how that same code would look like with the fix:

```javascript
document.addEventListener('turbolinks:load', function() {
  const button = document.querySelector('#my-button');
  if(button){
    button.addEventListener('click', function() {
        fetch('/my_data', { method: 'GET' })
          .then(response => response.json())
          .then(data => {
            document.querySelector('#my-data-container').innerHTML = data.content;
          });
    });
  }
});
```

this solves the binding issue, but we aren’t done yet! in addition, you should always check if elements exist before binding listeners or manipulating them. sometimes turbolinks loads a page with that element, and sometimes does not. for example, you might have a controller with multiple actions using the same layout, sometimes one action renders the element `my-button` and sometimes it does not. in that case, your javascript would throw an error because `document.querySelector('#my-button')` would return null, and on the next line, you try to bind an event to it. the `if(button)` check will guard that error from happening and avoid headaches when tracking errors on the production logs.

another common mistake i've seen (and done myself, not that i am proud) involves managing multiple ajax calls and their updates. let's say you have two or more fetch requests fired at the same time. if you assume that they return in the order you sent them, well, you're in for a bad surprise. the server can take different times to process each one, and when turbolinks changes the page content mid-fetch, the older fetches can end up updating the wrong section of the new dom, or, worse, elements that have been removed. here, the problem isn't with binding events anymore but rather, the data-updating part of it.

imagine i was trying to update a table with results of two different queries. one query for the list of products and another for a list of comments, and both `fetch` calls happen almost at the same time. the comments endpoint could have more load on the backend server than the products, so the product endpoint response would be served first, and then the comments. if turbolinks changed the page after the products fetch, when the comments fetch finishes, it would update a non-existent element in the page. i would be left scratching my head.

here’s a way to mitigate these problems, use a flag to prevent updates if the dom changes mid-request. this is particularly helpful for asynchronous operations.

```javascript
document.addEventListener('turbolinks:load', function() {
    const button = document.querySelector('#my-button');
  if(button){
    button.addEventListener('click', function() {
        let pageUpdated = false;
        document.addEventListener('turbolinks:before-render', () => {
              pageUpdated = true;
        });

        fetch('/my_data', { method: 'GET' })
          .then(response => response.json())
          .then(data => {
             if(!pageUpdated){
               document.querySelector('#my-data-container').innerHTML = data.content;
            }
          });
        });
  }
});
```

this approach adds a flag that is set to true when turbolinks is about to swap content. now, when the `fetch` resolves, we only update the dom if `pageUpdated` is false. this is a more robust approach.

i should mention also the importance of using the right event to remove listeners when turbolinks navigates away from the page. you can attach listeners to the `turbolinks:before-cache` event. that event is triggered when turbolinks saves a cached version of the current page, usually when navigating away to a new page. it's the place to clean up references, prevent memory leaks, and also prevent javascript from firing on the wrong version of a page after you return to it using back-navigation on the browser.

in essence, debugging these issues often requires a careful analysis of your javascript lifecycle alongside turbolinks' actions.

it reminds me of when i was asked to debug an app with weird random failures, and i found out that there was a rogue server process that, on top of being outdated, was only printing funny ascii-art when it failed. i was half mad, half laughing when i found it. i’ve always said that sometimes, coding feels like trying to teach a cat how to use a computer.

regarding resources, instead of links, i would recommend delving into *stimulus.js* as an alternative to manually binding events. also, check out the book *rails 7 javascript with import maps* by noel rappin, it delves into modern javascript setups in rails. *turbo 8 cookbook* by andrew cumming is also a must read. even if you are using turbolinks, it will give you insights into its workings as turbo and turbolinks share much of their underlying mechanisms. lastly, the source code of turbolinks is very well written and not that complex if you wish to really dive into the details of how it works.

that's about all i got for you. it's a finicky problem, but if you pay attention to turbolinks events, keep your code clean, and don't make assumptions about element availability, you’ll conquer it. good luck with that!
