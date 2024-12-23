---
title: "How can Rails active links be styled within Turbo Frames?"
date: "2024-12-23"
id: "how-can-rails-active-links-be-styled-within-turbo-frames"
---

,  It’s a topic I've spent quite a bit of time on, particularly after that rather perplexing incident with a client's navigation system. We were using Turbo frames extensively, and the active link styling just wasn't behaving as expected; it was a classic case of the dynamic nature of Turbo interfering with our carefully laid-out css. Let’s dive into how we can effectively manage active link styling within Turbo frames.

The challenge primarily stems from the way Turbo handles page transitions. When a link within a Turbo frame is clicked, only the content inside that frame is updated, rather than a full page reload. This selective update means that conventional css selectors relying on page-level information—such as those targeting an ‘active’ class on a navigation element based on the current url—may not trigger or update correctly after the frame is replaced. Your typical `current_page?` helper in Rails might work fine for the initial load, but falls short once the frame content is updated through Turbo.

The core issue is this: The browser isn’t navigating away from the current page url from its perspective, it’s just replacing frame content with new html. Therefore, methods that depend solely on comparing the current url to a specific route will fail to pick up the active class on the frame’s updated html.

What we need are approaches that are reactive to frame updates and can dynamically adjust the styling of active links, keeping our navigation consistent as users navigate through the application without the full page reloads. Here are a few solutions I’ve found to be quite reliable, which can be implemented in various scenarios depending on the project specifics.

**Solution 1: Javascript-Based Active Class Management**

One effective strategy involves leveraging javascript to detect the active link and dynamically add or remove the active class. The advantage here is that we're reacting directly to the update event of the turbo frame.

Here's a working snippet demonstrating this approach. We'll use the `turbo:frame-load` event, which is fired after a Turbo frame finishes loading:

```javascript
document.addEventListener('turbo:frame-load', function(event) {
  const frame = event.detail.target;
  const currentPath = window.location.pathname;
  const links = frame.querySelectorAll('a[href]');

  links.forEach(link => {
    if (link.getAttribute('href') === currentPath) {
      link.classList.add('active');
    } else {
      link.classList.remove('active');
    }
  });
});
```

Here’s a breakdown of what happens:
-   We attach an event listener to `turbo:frame-load`.
-   We obtain the frame that has been updated through the `event.detail.target` property.
-   We derive the `currentPath` from the browser's location.
-   We collect all the `a` tags with `href` attributes within the updated frame using `querySelectorAll`.
-   We iterate through each of these links and check if their `href` attribute matches the `currentPath`. If it does, we add the ‘active’ class; otherwise, we remove it.

**Solution 2: Server-Side Conditional Rendering with `turbo-stream`**

While javascript-based solutions are effective, sometimes you might prefer server-side control. In that case, we can leverage `turbo-stream` to dynamically update the navigation whenever a turbo-frame is updated. This requires some server-side logic to be executed when processing the request for a given frame update.

Here's a simplified example, focusing just on updating the link, but you’d obviously extend this to all links that might need an `active` class:

First, imagine a partial, `_nav_link.html.erb`:

```erb
<%= link_to name, path, class: ('active' if active_link?(path)) %>
```

We define a helper method called `active_link?(path)` in our application helper (or any relevant helper):

```ruby
def active_link?(path)
  request.fullpath == path
end
```

Now, in our controller action that handles the turbo-frame content render, we’d include this:

```ruby
def my_controller_action
    respond_to do |format|
      format.html
      format.turbo_stream {
        render turbo_stream: [
          turbo_stream.update('my_nav', partial: 'my_nav')
          # other frame updates
        ]
      }
    end
  end

```

In this scenario, we're updating the entire navigation partial every time a frame update occurs. This is server-driven, avoiding reliance on client-side javascript. While it’s a bit heavier than the javascript approach, it can be quite effective for keeping the navigation consistent and is easier to reason about sometimes.

**Solution 3:  Hybrid Approach with Data Attributes and Javascript**

A good middle ground is leveraging data attributes and minimal javascript. We'll add data attributes to the navigation links and toggle classes based on that attribute and the frame's content location.

Here’s what the initial html might look like:

```html
<nav>
  <a href="/dashboard" data-path="/dashboard" >Dashboard</a>
  <a href="/reports" data-path="/reports">Reports</a>
  <a href="/settings" data-path="/settings">Settings</a>
</nav>
```
And here’s the javascript:

```javascript
document.addEventListener('turbo:frame-load', function(event) {
    const frame = event.detail.target;
    const currentPath = frame.querySelector('[data-current-path]').dataset.currentPath; // Ensure frame has the data attribute

   const links = frame.querySelectorAll('nav a');

    links.forEach(link => {
        if(link.dataset.path === currentPath){
            link.classList.add('active');
        } else {
            link.classList.remove('active');
        }

    });
});
```

Note that we will include `data-current-path` in the turbo frame when rendering it from the server with its current path:

```erb
  <turbo-frame id="my_frame" data-current-path="<%= request.fullpath %>">
     ... contents of frame...
  </turbo-frame>
```

This approach avoids relying directly on `window.location.pathname`, and takes the path information directly from the frame itself, reducing the risk of mismatch with what's being displayed.

**Resource Recommendations**

For further deep dives, I highly recommend the following resources:

1.  **"Rails 7 Cookbook" by Stefan Wintermeyer**: This book provides very practical recipes for common rails problems, including turbo frame usage and how to handle active links within this context. It covers many nuances that might not be obvious from the official Rails documentation.

2.  **"Hotwire Handbook" by Joe Masilotti:** This handbook is a great resource for everything related to turbo, including detailed explanations of frame interactions and techniques for combining javascript and server-side rendering effectively.

3.  **The official Turbo documentation:** Though I didn't link it directly, exploring the latest version of the official turbo documentation, found on the Hotwire documentation site, is vital. They provide in-depth insight into all lifecycle events, like `turbo:frame-load`, that you need to master when using the framework.

These solutions, and the resources above, should get you on the right path to reliably manage active link states within your turbo-frame setup. Always consider the tradeoffs each technique presents and pick what fits best for your specific needs and the desired maintenance profile. Good luck!
