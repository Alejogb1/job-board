---
title: "Why aren't images displaying on my GitHub Pages site after adding a custom domain?"
date: "2024-12-23"
id: "why-arent-images-displaying-on-my-github-pages-site-after-adding-a-custom-domain"
---

,  I’ve seen this issue crop up more times than I care to count, and it's usually a subtle combination of factors rather than one glaring mistake. When you’re moving a GitHub Pages site over to a custom domain, image display problems are often tied to how paths are being interpreted relative to the new base url. Let's unpack that.

The core problem usually stems from a mismatch between how your project references its assets, specifically images, and the way those paths are interpreted once the custom domain is in place. Think of it as navigating a city with familiar street names and suddenly those names change, requiring you to recalibrate your entire internal map. When you deploy on GitHub Pages directly (username.github.io/repo), the base path is straightforward. However, once a custom domain like `mydomain.com` is set up, that base path changes and your image paths may not reflect that.

In the early days of a project I worked on, we ran into this exact snag. Our images were referenced using relative paths—something like `<img src="images/my_image.png">`— which worked perfectly fine with our GitHub Pages URL. The moment we pointed the custom domain at it, suddenly all images went missing. I realized that the browser, when requesting the image, was not fetching from `mydomain.com/images/my_image.png` but was looking for that path within a directory on the custom domain that often didn’t exist.

To get a clear picture, let’s break down some of the likely scenarios. First, your image paths might be fully relative (as in our original issue). Second, you might be using root-relative paths, or absolute paths referencing the wrong domain. Lastly, and somewhat less likely, there can be caching issues causing the browser to load older versions of the page or having difficulties loading resources.

Let’s tackle these systematically with some practical examples, showing how you might encounter and fix the path issues, using code snippets to illustrate our point.

**Scenario 1: The Case of Fully Relative Paths**

If your images are called like this in your html file:

```html
<!-- example.html -->
<html>
<head>
  <title>My Page</title>
</head>
<body>
  <img src="images/logo.png" alt="My Logo">
  <p>Some Content Here</p>
</body>
</html>
```

And your directory structure is something like:

```
my-project/
├── index.html
└── images/
    └── logo.png
```

With `myusername.github.io/my-project`, the browser correctly retrieves the image located at `myusername.github.io/my-project/images/logo.png`. However, when `mydomain.com` is configured to point to the same site, the browser tries to retrieve the image at `mydomain.com/images/logo.png`. Usually, the image is actually under `mydomain.com/my-project/images/logo.png`, and that leads to a 404 error.

To rectify this, the fix is not to change relative paths because relative paths are the most resilient approach, but to ensure the root directory for deployment matches the custom domain’s configuration when a repository name is present. This scenario often occurs when you're deploying to a repository that isn't the main project page or user site page on github. The solution is that your deployment process must ensure that the repository subdirectory is included in the output folder.

**Scenario 2: Misconfigured Root-Relative Paths**

You might think root-relative paths (starting with a `/`) are a safe bet. However, they require a bit more care. Here’s an example:

```html
<!-- another_example.html -->
<html>
<head>
  <title>My Page</title>
</head>
<body>
  <img src="/images/product.jpg" alt="Product Image">
  <p>More Content Here</p>
</body>
</html>
```

With github pages, this might work when the site is accessed through `myusername.github.io/my-project` because the root directory is `myusername.github.io`, but when we use a custom domain, that's incorrect. The browser will be attempting to load `mydomain.com/images/product.jpg`, regardless of your directory structure. This can cause issues in multiple scenarios where your local directory is not your root directory when using a custom domain.

The solution in this case is two-fold:

1.  **If your site is not hosted in a project with a repository name**. You can correct the root paths by adding a base path in the meta tags of your html file as follows:

    ```html
    <!-- corrected_example.html -->
    <html>
    <head>
    <title>My Page</title>
     <base href="/my-project/">
    </head>
    <body>
     <img src="/images/product.jpg" alt="Product Image">
     <p>More Content Here</p>
    </body>
    </html>
    ```

2. **If your site is hosted in a project with a repository name**. You should, in general, avoid the use of root paths in the code. In this case, using relative paths in html code, `src="images/product.jpg"` is a better option.

**Scenario 3: Caching Issues and CDN Considerations**

Less frequently, caching can also be a culprit. Sometimes the browser is loading old CSS, which might not reference images correctly, or an outdated version of the webpage where image paths are incorrect. Even if the paths are now correct, you may be viewing cached content. A hard refresh (usually `Ctrl+Shift+R` or `Cmd+Shift+R`) can force a fresh load. Additionally, if you are utilizing a CDN, it's essential to purge the CDN cache after updating your image resources. I encountered this once when using Cloudflare: it cached a broken version of my site, requiring me to manually purge the cache to display the new image files.

Also, be mindful if you're utilizing a build process, such as with webpack or parcel, that can alter asset paths during build. Make sure that your build configurations properly account for the custom domain setup and that the base path of all assets is correct. In practice, this may involve setting a configuration parameter for the build tool which is used when deploying the site to the server. It is crucial to ensure that the paths in the final deploy folder match up with what the server expects from a request to the domain.

**Recommendations for further study**

To get deeper into these topics, I recommend delving into resources that cover web deployment strategies and URL resolution. Specifically, the following are good resources:

*   **"HTTP: The Definitive Guide" by David Gourley and Brian Totty:** This book is a comprehensive resource for all things related to HTTP and how it affects web development. Chapter 24, "Caching," and chapter 12, "URL Syntax and Semantics," are particularly relevant.
*   **The MDN web docs on URLs:** The Mozilla Developer Network (MDN) has extensive documentation on how URLs work. Search for ‘URL’ to learn the formal syntax and how they are used in different situations.
*   **The official github pages documentation:** It provides insights regarding how to use custom domains and is an essential reference.

In summary, image display issues after setting up a custom domain on GitHub Pages are usually rooted in path mismatches, which can stem from fully relative paths, misused root-relative paths, and caching. By systematically addressing these points, verifying that the deploy folder has all of the proper resources, and ensuring your pathing is accurate, you should be able to get your images working correctly. Always be methodical in your troubleshooting, check your browser’s developer tools for 404 errors related to image requests, and examine your build process to ensure everything is configured correctly for your custom domain.
