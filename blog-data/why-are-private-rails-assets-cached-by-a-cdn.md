---
title: "Why are private Rails assets cached by a CDN?"
date: "2024-12-16"
id: "why-are-private-rails-assets-cached-by-a-cdn"
---

, let's address this. It's a situation I've encountered firsthand on more than one occasion, and it always leads to interesting debugging sessions. We're talking about private assets in a Rails application—think user-specific files or content behind an authentication layer—and the seemingly paradoxical issue of them being cached by a Content Delivery Network (CDN). The problem isn't that it *shouldn't* happen in principle; it's that it *does* happen, and it creates serious security and privacy concerns if not managed correctly. The core issue isn't with the CDN itself acting maliciously, but rather with how caching rules and authentication are handled at various levels of the request-response lifecycle, and where those layers meet.

The primary reason why these assets end up in a CDN cache revolves around two major factors: incorrect cache-control headers and a failure to properly authenticate requests at the CDN level. Let me elaborate. When a browser (or any client) requests an asset, the server responds not just with the content itself but also with HTTP headers that tell downstream systems, like CDNs and proxies, how to handle that content. Among these headers, the `Cache-Control` header is crucial. If your Rails application serves private assets without explicitly setting `Cache-Control: private` or `Cache-Control: no-cache, no-store`, or an equivalent that forces revalidation with the origin server, the CDN will likely interpret it as cacheable content, defaulting to the common behavior of caching anything it can. This can happen even if your application has robust authentication mechanisms, because CDNs typically operate further downstream, examining only the raw HTTP headers as they pass through. They do not, by default, understand or care about the specific application-level authentication procedures used.

The authentication failure point often stems from the fact that CDNs act as reverse proxies. When a client requests an asset, the request first goes to the CDN, which checks its cache. If the asset is there and valid according to its cache rules, the CDN serves it directly, never even reaching the Rails application. However, if the CDN does reach the origin (your rails application) for the asset, it uses the same URL the browser requested. It lacks any understanding of application-level session cookies or authorization headers that would verify the user’s right to access the content. Hence, if authentication is handled solely within the Rails application, the CDN, while performing its proxying duties, isn't equipped to honor those conditions.

Now, let's examine some code snippets to clarify these points. First, a very common, and unfortunately problematic, setup where private assets become cacheable:

```ruby
# app/controllers/private_assets_controller.rb
class PrivateAssetsController < ApplicationController
  before_action :authenticate_user!

  def show
    @asset = current_user.assets.find(params[:id])
    send_data @asset.data, type: @asset.content_type, disposition: 'inline'
  end
end
```

In this scenario, the `authenticate_user!` method correctly secures the access at the application level. However, no special `Cache-Control` headers are being set. This means the default HTTP response headers are often cacheable, and if passed through a CDN, they'll be cached. The next client attempting to get the same URL—regardless of their authentication status—will likely receive the cached content from the CDN if they hit that cache.

The solution, in principle, is straightforward: send appropriate `Cache-Control` headers from your Rails application. Here is how this might be accomplished:

```ruby
# app/controllers/private_assets_controller.rb
class PrivateAssetsController < ApplicationController
  before_action :authenticate_user!

  def show
    @asset = current_user.assets.find(params[:id])
    response.headers['Cache-Control'] = 'private, no-cache, no-store, must-revalidate'
    send_data @asset.data, type: @asset.content_type, disposition: 'inline'
  end
end
```

By adding `response.headers['Cache-Control'] = 'private, no-cache, no-store, must-revalidate'`, we're instructing the CDN (and other intermediaries) *not* to cache the response. Specifically, `private` indicates that the response is not meant to be cached by intermediary caches, `no-cache` and `no-store` prohibit the response from being stored, and `must-revalidate` forces the cache to check with the server to see if the content is still valid before serving it.

There’s also another more sophisticated approach one might consider, especially if fine-grained control over caching is needed or the CDN offers some custom configuration options.

```ruby
# app/controllers/private_assets_controller.rb
class PrivateAssetsController < ApplicationController
  before_action :authenticate_user!

  def show
    @asset = current_user.assets.find(params[:id])
    expires_now
    send_data @asset.data, type: @asset.content_type, disposition: 'inline'
  end
end

```

Here we use `expires_now` a helper method in Rails that handles setting the necessary HTTP headers to prevent caching. This often compiles to the same `Cache-Control` directive we saw earlier but using this method can be a preferred way to declare that this content should not be stored in caches.

However, even these solutions are insufficient when dealing with sophisticated caching setups. To prevent CDN caching entirely for private resources, several further options may need to be considered beyond the `Cache-Control` header, including:

1.  **CDN-Level Authentication**: If your CDN supports authentication mechanisms (such as signed URLs, JWT verification, or access tokens) configured on a per-route basis, use them. This method shifts the authentication check to the CDN level and can effectively prevent unauthorized access even before the request reaches the Rails server. Be careful, however, about how you implement this authentication: it still needs to follow the principle of least privilege.
2.  **Custom Cache Rules**: Most CDNs allow you to define custom rules for caching. These can be based on request headers, paths, or query string parameters. These rules will often allow for bypassing the cache entirely for certain URL patterns which may include your private assets.
3.  **Dynamic Assets with Unique URLs**: A last-resort approach is to generate unique URLs for each private asset, or each request. This can include appending a query parameter with a unique identifier for each request, or even incorporating a timestamp into the URL itself. It is important to note that this strategy is generally less performant and must be handled carefully to avoid potential security vulnerabilities.

To deepen your understanding, I'd highly recommend reviewing the sections on caching from "High Performance Browser Networking" by Ilya Grigorik, which covers caching mechanics in exhaustive detail. For a more detailed analysis on HTTP headers, including `Cache-Control`, "HTTP: The Definitive Guide" by David Gourley and Brian Totty is an invaluable resource. Additionally, the RFC documents for HTTP (particularly RFC 7234 for caching) are essential to understanding the underlying protocols driving caching mechanisms.

In practice, I’ve discovered that while straightforward in theory, ensuring that your private assets are never cached by a CDN requires a blend of correct server-side header configuration combined with the appropriate CDN-specific configurations. It’s a multi-layered problem that requires paying close attention to both the principles of caching and the specific capabilities and configurations of your chosen CDN solution. Ignoring this will not only result in a loss of privacy and data security but can also create the kind of issues that can be very tricky to debug when dealing with the distributed nature of the request-response cycle.
