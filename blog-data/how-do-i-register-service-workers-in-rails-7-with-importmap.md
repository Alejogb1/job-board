---
title: "How do I register service workers in Rails 7 with importmap?"
date: "2024-12-23"
id: "how-do-i-register-service-workers-in-rails-7-with-importmap"
---

Alright, let's unpack this service worker integration with importmap in Rails 7. I've seen this come up in a few different projects, and it's definitely a case where the default setup can feel a little… incomplete. It’s not that Rails 7 isn't equipped for it; it just requires a few specific steps to get those service workers playing nicely with importmap. The issue, as I've observed, is typically with the way assets are managed and how we ensure the service worker script is loaded correctly.

When I first encountered this, it was in a progressive web app project for a small e-commerce client back in 2022. We needed offline capability for product browsing, and naturally, service workers were the route. Initially, the service worker was failing to register and throw obscure errors. This was largely because we hadn't paid enough attention to the nuances of how importmap handles relative paths, and how that affects service worker registration. So, let’s get into the details.

The first key area is getting your service worker code to be accessible through importmap, which essentially means it should be a module. You should begin by making sure your service worker code resides in your `app/javascript` directory. Let’s say you have a file named `service_worker.js`. It could be something straightforward to begin with, like this:

```javascript
// app/javascript/service_worker.js
console.log("Service worker script loaded");

self.addEventListener('install', (event) => {
  console.log('Service worker installed');
  event.waitUntil(self.skipWaiting());
});

self.addEventListener('activate', (event) => {
  console.log('Service worker activated');
  event.waitUntil(self.clients.claim());
});
```

Now, in your `config/importmap.rb` file, you'll need to map this module correctly. Add an entry for your service worker, like this:

```ruby
# config/importmap.rb
pin "application", preload: true
pin "service_worker", to: "service_worker.js", preload: true
```

This tells importmap how to load our service worker. `preload: true` is helpful here as it loads the service worker at initial request, rather than on-demand, which can be important for initial service worker setup.

Now that the script is mapped, the challenge shifts to registering it within your application. Typically you’d do this in your main application javascript file, `application.js`, by accessing the URL provided by the importmap. Here is the revised version for better integration:

```javascript
// app/javascript/application.js
import * as serviceWorker from "service_worker"

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/assets/service_worker.js', { scope: '/' })
      .then((registration) => {
        console.log('Service worker registered successfully:', registration);
      })
      .catch((error) => {
        console.error('Service worker registration failed:', error);
      });
  });
}
```

Notice a key change in the registration URL: Instead of using relative paths like `/service_worker.js`, we're now using `/assets/service_worker.js`. This is because importmap typically places the compiled javascript inside assets. Also, observe the import line `import * as serviceWorker from "service_worker"`, this is crucial to resolve that `service_worker` module name from importmap as it's now resolved from this import. It doesn't require anything from that import to be invoked or used but instead only to resolve its module location. The `scope: '/` ensures your service worker applies to your entire domain.

The 'if' block check `if ('serviceWorker' in navigator)` confirms that service workers are supported by the browser, before trying to register it. This prevents errors in browsers that lack the API. The load event listener also ensures that the service worker tries to register only after all resources on the page are loaded. This is a small, but important detail that can prevent errors.

It’s important to emphasize, that you *must* use the assets path when registering the service worker. This is where importmap makes a crucial shift: we aren't registering a file at our root `/service_worker.js`, but we are registering the compiled javascript bundle within the `/assets` folder. This is one of the primary causes of failed registrations.

Let's look at a more advanced example involving caching. Imagine we wanted to cache static assets. Here's how you might modify the `service_worker.js` file:

```javascript
// app/javascript/service_worker.js
const CACHE_NAME = 'my-site-cache-v1';
const urlsToCache = [
  '/',
  '/assets/application.js',
  //add the full paths to any static assets
  '/assets/path_to_your_css.css',
  '/assets/path_to_your_image.png',
  //add any api endpoints used
   '/api/your_api_endpoint'
];

self.addEventListener('install', (event) => {
  console.log('Service worker installed');
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
          console.log('Cache opened');
        return cache.addAll(urlsToCache);
      })
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
    console.log('Service worker activated');
    const cacheWhitelist = [CACHE_NAME];
    event.waitUntil(
      caches.keys().then((cacheNames) => {
        return Promise.all(
          cacheNames.map((cacheName) => {
              if (cacheWhitelist.indexOf(cacheName) === -1){
                  return caches.delete(cacheName);
              }
          })
        );
      })
      );
    self.clients.claim();

});

self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request)
      .then((response) => {
        // Cache hit - return response
        if (response) {
          return response;
        }
        // No cache, return fetch response
        return fetch(event.request);
      })
  );
});
```

In this version, we've added a basic caching mechanism. Note how `/assets/application.js` is included in `urlsToCache`. This is important because it ensures that even if network access is down, the service worker can return the cached version of our main javascript file allowing the app to load. Likewise, remember that any assets you wish to cache must include full paths, relative to your rails assets path which is generally `/assets/`.

A final example, demonstrating how you might handle offline pages, might include the following code in your `service_worker.js` file:

```javascript
// app/javascript/service_worker.js
const CACHE_NAME = 'my-site-cache-v1';
const OFFLINE_PAGE = '/offline'; // the route for your offline page

self.addEventListener('install', (event) => {
  console.log('Service worker installed');
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
        console.log('Cache Opened');
        return cache.addAll([OFFLINE_PAGE]);
    })
  );
    self.skipWaiting();
});

self.addEventListener('activate', (event) => {
    console.log('Service worker activated');
    const cacheWhitelist = [CACHE_NAME];
    event.waitUntil(
        caches.keys().then((cacheNames) => {
          return Promise.all(
              cacheNames.map((cacheName) => {
                if(cacheWhitelist.indexOf(cacheName) === -1){
                  return caches.delete(cacheName);
                }
              })
          )
        })
    );
    self.clients.claim();
});


self.addEventListener('fetch', (event) => {
    event.respondWith(
      caches.match(event.request).then((response) => {
          if(response){
              return response;
          }
          return fetch(event.request).catch((err) => {
            console.error('Fetch error:', err);
            if (event.request.mode === 'navigate'){
              return caches.match(OFFLINE_PAGE);
            }
          });
      })
    )
});

```

In this setup, we cache the content at the `/offline` route and serve that cached page when the user loses connection. Note that the page must already exist in the app and be present in the `urlsToCache` and in the app already, for this approach to work correctly.

For further study, I'd suggest diving into "Offline Web Applications" by Jake Archibald; it’s a goldmine of information on service workers. Additionally, the documentation on "Service Workers API" from Mozilla Developer Network (MDN) is indispensable. Also reading up on "Progressive Web Apps" by Google is a good idea to understand the context around service workers. Understanding the Fetch API and Cache API is crucial as well.

In summary, setting up service workers with importmap in Rails 7 primarily requires careful attention to paths. Ensure your service worker script is correctly mapped in `importmap.rb`, register the service worker using `/assets/service_worker.js`, and thoroughly grasp the caching mechanisms and their implications. The three examples above should serve as a solid starting point, but there's always more to explore, so keep experimenting.
