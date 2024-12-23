---
title: "Why is Nginx slow to render static Rails pages?"
date: "2024-12-23"
id: "why-is-nginx-slow-to-render-static-rails-pages"
---

Alright, let's delve into the nuances of why Nginx might seem sluggish when serving static Rails pages. It’s a topic I’ve encountered more times than I’d prefer, often in situations where the initial gut reaction is to blame Rails itself, when, more often than not, the culprit resides somewhere in the interplay between Nginx and the way assets are handled. My past projects, especially ones involving high-traffic e-commerce platforms, have repeatedly forced me to optimize this specific area.

The core issue rarely stems from Nginx's intrinsic capabilities, which are, let’s be frank, incredibly performant for static file serving. Instead, the bottleneck often arises from misconfigurations, inefficient content delivery pipelines, or unintended interactions with Rails' asset pipeline. What I've frequently seen boils down to these common points: incorrect location block configurations, lacking gzip compression, unnecessary processing by Rails, and poorly optimized caching strategies.

Firstly, let's consider how Nginx *should* handle static files. Ideally, we want Nginx to directly serve them without involving the Rails application server (typically Unicorn, Puma, or similar). This means avoiding proxying the request to Rails when a static asset (like an image, css file, or javascript) is requested. The most common mistake is when location blocks in the Nginx configuration are not set up to intercept requests for static files before they reach the Rails application server.

Here’s a look at a typical, *problematic*, location block that's configured to handle static assets – something I've sadly seen in production setups:

```nginx
location /assets {
    proxy_pass http://rails_app;  #wrong approach. Avoid proxying to Rails for static files
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
}

location / {
    proxy_pass http://rails_app;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
}
```

In this scenario, every request to `/assets` is still being forwarded to your Rails application server – entirely defeating the purpose of separating static serving from dynamic request handling. This is where we lose performance, as the Rails server is now handling static file requests which it shouldn’t be. This results in unnecessary overhead.

The correct way is to define a location block that directs requests for files within your `public/assets` directory directly to the file system:

```nginx
location /assets {
    root /path/to/your/rails/app/public;
    expires 30d; # add proper caching
    try_files $uri =404; # avoid letting it fall through to Rails
}

location / {
    proxy_pass http://rails_app;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
}

```

Here, the `root` directive points to the `public` directory of your Rails application, and Nginx directly handles the request. `try_files $uri =404;` ensures that if a file isn't found, it will immediately return a 404 error without involving the Rails application. We also set an `expires` directive to enable browser-side caching, further reducing load. The `expires` header helps reduce the overall traffic significantly by leveraging the browser's cache for repeated requests.

The next crucial optimization involves gzip compression. This is often overlooked but can dramatically reduce the size of transferred data, leading to faster load times for your assets. A significant portion of our efforts often centers around this, ensuring the correct compression settings are in place. Here's how we can enable gzip compression:

```nginx
gzip on;
gzip_types text/css application/javascript application/x-javascript text/plain application/xml application/json image/svg+xml;
gzip_comp_level 6;
gzip_min_length 1000;
gzip_vary on;
```
These directives enable gzip compression for appropriate file types, adjust the compression level for resource usage trade-off, sets minimum file size to compress and include the `vary: Accept-Encoding` header to serve pre-compressed version when the browser specifies support for `gzip`.

Furthermore, even with Nginx serving static files directly, the Rails asset pipeline itself can introduce slowdowns if it’s not configured properly. Especially if you are not pre-compiling your assets and you try to handle those on every request. For example in development it's common not to precompile, but in production, that has to change for optimized speed.

For instance, if the assets are not precompiled and are being served dynamically by Rails during each request in production environments, it will severely impact the performance. Instead, the precompiled assets should be served directly from disk by Nginx. This ties into the first point regarding proper Nginx location block configuration. Ensuring that assets are precompiled (using `rails assets:precompile`) is vital in production.

Also ensure, that you remove unnecessary middleware or features. For instance if you aren't using turbolinks, or sprockets you should remove them. Any middleware you do not actively utilize, might be slowing things down. Always lean towards the minimal needed for speed and stability.

Now, let’s touch upon caching. As touched upon briefly above, implementing proper caching is crucial. This includes both server-side caching using techniques like `proxy_cache` and browser-side caching using `expires` directives in Nginx. Setting appropriate `Cache-Control` headers in Nginx allows browsers and proxy servers to store copies of your assets, reducing subsequent request times. In other words, don't let browser always download your resources. Browser cache is fast and powerful. Utilize it. For static files we don’t need complex cache invalidation logic either, often just expiring after reasonable time such as 30 days, or even longer for stable resources.

To dive deeper into these topics, I’d recommend focusing on the following resources. *“High Performance Web Sites”* by Steve Souders provides invaluable insights into frontend optimization techniques, including caching and gzip compression. The official Nginx documentation is of course essential for understanding its configuration directives, especially in relation to file serving and proxying. Also, consider reviewing *“Programming Ruby”* by Dave Thomas, et al, which is a comprehensive guide to Rails, and will help in properly managing Rails asset pipeline and understanding how it works. Lastly, for a more theoretical understanding of web performance, refer to papers on HTTP caching and Content Delivery Networks (CDNs) available on scholarly websites like ACM Digital Library.

In summary, while Nginx is highly efficient, its performance in serving static Rails pages often comes down to careful configuration and an understanding of how it interacts with Rails. By directly serving static files with Nginx, enabling gzip compression, precompiling assets and implementing proper caching strategies, we can significantly improve page load times and resource utilization. Ignoring these key components can easily create a bottleneck where there shouldn’t be one, causing Nginx to appear slower than it should. The devil is, as usual, in the details, and a thorough review of the nginx configuration and the Rails asset pipeline is often required to achieve optimal performance.
