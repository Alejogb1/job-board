---
title: "Why isn't Turbo Drive intercepting links when the `turbo:load` event is firing?"
date: "2024-12-23"
id: "why-isnt-turbo-drive-intercepting-links-when-the-turboload-event-is-firing"
---

Let's tackle this one. It’s a situation I've debugged more times than I’d like to count, often late at night when a client project was hitting an unexpected snag. The core issue – why Turbo Drive might seem to ignore links precisely when the `turbo:load` event is firing – typically boils down to a misunderstanding of the timing and lifecycle of Turbo Drive’s interaction with the DOM, and how event handlers are registered.

The crux of the matter lies not necessarily in Turbo Drive *failing* to intercept links, but in when your event listeners are actually attached relative to when the initial page load (and thus `turbo:load`) is completed. It's a timing dance, and if you're not precisely on tempo, things will go awry.

Turbo Drive’s fundamental behavior is to replace the `<body>` content without reloading the entire page when navigating between pages. It does this by intercepting click events on links (`<a>` tags with `href` attributes) and forms. When a link is clicked, Turbo fetches the new page, replaces the existing body with the fetched content, and then fires a `turbo:load` event.

Now, if you're attempting to add event listeners to your `<a>` tags *within* the scope of a `turbo:load` event handler, that logic is being executed *after* the body has already been replaced. Consequently, any handlers you attach to *elements* in the old body will effectively be discarded with it. Your newly added handler is operating on a *new* page body. If your specific logic is then relying on, say, data attributes that were in the *old* body, it might not behave as expected. This manifests as a failure to intercept.

Here's a simplified example to clarify this. Imagine your initial HTML page:

```html
<!DOCTYPE html>
<html>
<head>
  <title>Initial Page</title>
  <script src="/turbo.js"></script>
</head>
<body>
  <a href="/nextpage">Go to next page</a>
  <script>
      document.addEventListener('turbo:load', () => {
          console.log('turbo:load triggered');
          document.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', (event) => {
              event.preventDefault();
              console.log('click intercepted!');
              // Custom logic here
            });
          });
      });
  </script>
</body>
</html>
```

In this scenario, the initial `turbo:load` fires, and the click handler is correctly attached to the anchor tag. But if navigation occurs to `/nextpage`, the *entire* body gets swapped. The *new* body will *not* have these event handlers attached unless the script is re-executed on the new page load.

Let's examine how we might address this incorrectly. In the following, I'm making the mistake of adding click handlers inside another `turbo:load`:

```html
<!DOCTYPE html>
<html>
<head>
  <title>Initial Page</title>
  <script src="/turbo.js"></script>
</head>
<body>
    <a href="/nextpage">Go to next page</a>
  <script>
    document.addEventListener('turbo:load', () => {
      console.log('Initial turbo:load triggered');

      document.addEventListener('turbo:load', () => {
        console.log('Nested turbo:load triggered');
          document.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', (event) => {
                event.preventDefault();
                console.log('Incorrect click handler');
            });
        });
      });
    });
  </script>
</body>
</html>
```

Here, the nested `turbo:load` will never trigger for the initial load, because it’s inside the handler for the *initial* `turbo:load`. On subsequent navigation, however, a *new* nested event handler will be registered, leaving the click handler only functioning on links present in that newly loaded body, while missing those present in previous page loads. This will likely cause intermittent issues that are tricky to debug.

The correct way to handle this scenario is to delegate the event at a higher, unchanging level, such as the `document` itself. This pattern leverages event bubbling, meaning that click events originating on child `<a>` elements will bubble up to the document element where you've attached your handler. This ensures your handler always intercepts the event, regardless of how many times the `<body>` is replaced. This approach ensures event listeners remain functional following Turbo Drive’s page replacements:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Initial Page</title>
    <script src="/turbo.js"></script>
</head>
<body>
    <a href="/nextpage">Go to next page</a>
    <script>
      document.addEventListener('click', (event) => {
        if (event.target && event.target.matches('a[href]')) {
            event.preventDefault();
            console.log('Delegated click intercepted');
            // Custom logic here
        }
      });
    </script>
</body>
</html>
```

In this corrected example, I’m attaching a single click event listener to the entire document. This handler examines whether the clicked element is an anchor tag with an `href`. If it is, then it intercepts it. This technique ensures the click handler persists across Turbo Drive navigations. This approach is far more resilient to Turbo Drive’s behavior of swapping out the body.

Further, if you're dealing with more complex scenarios involving custom data attributes or specific behaviors, the delegated handler ensures those attributes remain accessible. Moreover, it is significantly more efficient than constantly attaching and detaching event listeners on each navigation. This strategy greatly improves performance and predictability in single-page applications using Turbo Drive.

To delve deeper into this subject, I recommend carefully reading the official Turbo documentation, which is usually very clear and concise. Additionally, “Eloquent JavaScript” by Marijn Haverbeke is invaluable for developing a solid understanding of event handling in Javascript. Understanding the DOM event model, specifically event bubbling and capturing, is fundamental to solving this kind of issue. The section on event handling in the Mozilla Developer Network (MDN) documentation is also an excellent resource. Furthermore, "Single Page Apps in Depth" by Mikito Takada provides a comprehensive understanding of the challenges, including event management, that are faced in such applications. Finally, "Thinking in Systems: A Primer" by Donella H. Meadows, while not purely technical, helps you understand how dynamic systems behave, which can provide insight into the timing nuances of Turbo Drive's behavior. Mastering these concepts and resources will greatly help in addressing these subtle but impactful nuances of working with Turbo and other similar technologies. In summary, timing is everything, particularly with Turbo Drive, and proper event delegation is paramount.
