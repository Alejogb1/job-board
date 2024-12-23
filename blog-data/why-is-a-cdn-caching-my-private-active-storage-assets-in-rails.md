---
title: "Why is a CDN caching my private Active Storage assets in Rails?"
date: "2024-12-23"
id: "why-is-a-cdn-caching-my-private-active-storage-assets-in-rails"
---

Alright, let's unpack this. I've seen this specific headache rear its head more than a few times in my career, and it's almost always rooted in a subtle misunderstanding of how CDNs interact with Rails and, crucially, Active Storage. The short answer is: your CDN isn't *intentionally* caching your private assets; it's doing what it's designed to do—cache based on headers. The real culprit is often a misconfigured caching policy or an inadvertent exposure of your private asset URLs that we need to address.

First, let’s set the stage. Active Storage, by default, generates presigned URLs when you access private assets. These URLs are time-limited and should, theoretically, only grant access within their validity window. However, most CDNs, being content-agnostic, primarily operate on HTTP caching headers (like `Cache-Control` and `Expires`) and the `ETag`. If these headers are present, the CDN will happily cache the response, regardless of whether the URL is supposed to be temporary. Herein lies the issue. Let's explore the typical scenarios that lead to this.

One of the earliest incidents I encountered involved a rather sprawling e-commerce platform. They had a seemingly straightforward setup: users upload images, some are private (user profile pics, etc.), and others public (product images). The CDN was in place, optimized for speed, and everything seemed fine until… user profile pictures started showing up in other users' browsers. Turns out, the initial problem lay in a poorly configured `public_url` method. While we were generating a presigned URL for accessing the asset, that method was also returning an image tag that did not include the correct cache disabling headers. Here's a snippet that shows a scenario for this:

```ruby
# initial (incorrect) method in our active storage model
def public_url
  Rails.application.routes.url_helpers.rails_blob_url(self, only_path: true)
end

# and here is how we had it displayed in a view.
# <img src="<%= model.public_url %>" >
```

The `rails_blob_url` by itself doesn't dictate caching behavior. Browsers, and by extension, CDNs, are free to cache these resources. The fix? We needed to explicitly tell them not to. This requires ensuring the `Cache-Control` header is set properly for private assets when serving them. The `ActiveStorage::Blob#service_url` method, which is usually used behind the scenes, does not itself enforce any `Cache-Control` headers. You must do that on your end. We then added a custom method to explicitly set the headers.

```ruby
# Correct method in our Active Storage Model
def presigned_url(expires_in: 5.minutes)
    Rails.application.routes.url_helpers.rails_blob_url(
      self,
      expires_in: expires_in,
      disposition: 'inline' # You can set to 'attachment' if needed
    )
end
def url_with_no_cache(expires_in: 5.minutes)
    url = presigned_url(expires_in: expires_in)
    "#{url}?cache_control=no-store" # a simple trick to bust cached versions.
end
```
and in our controller

```ruby
def show
    @user = User.find(params[:id])
    if can_view_profile(@user)
       url = @user.avatar.url_with_no_cache
       redirect_to url
    else
      head :not_found
    end
end

```

What I learned was this: using `Cache-Control: no-store` is a potent tool for preventing caching, but sometimes, a workaround like appending a query parameter can be effective in some scenarios. The `no-store` directive, when properly set, should prevent intermediaries like CDNs from caching that resource. However, it may not be honored by all caching implementations, so I have found sometimes appending query parameters effective. It's crucial to understand that, while Active Storage provides robust controls, managing caching remains your responsibility.

Another challenging scenario involved a SaaS platform for educational institutions. They were delivering learning resources, including PDFs and videos, via Active Storage. They had set up presigned URLs and used a CDN, but again, caching issues surfaced, specifically when resource permissions changed. A teacher would update a document’s accessibility, but students with cached versions of the URL would still access the old version. In this case, it wasn’t the initial URL generation, but how the URL was used across their platform that caused the problem. The issue stemmed from the application's caching strategy around the `service_url`. Here's how their initial (problematic) code looked (simplified for brevity):

```ruby
# Somewhere in a service class, doing this
def fetch_resource_url(resource)
  resource.document.service_url
end

# used in the controller like so
def show
  @resource = Resource.find(params[:id])
  url = fetch_resource_url(@resource)
  redirect_to url
end
```

The service layer was directly fetching the service url from the active storage blob. This, combined with client-side caching, was the cause. We found that the same url was being generated, even after permissions were changed. The correct way to approach this was to use a presigned url *each time* the controller action was hit. This method regenerates a valid presigned url:

```ruby
def fetch_resource_url(resource)
  Rails.application.routes.url_helpers.rails_blob_url(
      resource.document,
      expires_in: 5.minutes, # Set appropriate expiry,
      disposition: 'inline'
    )
end

# Controller method
def show
    @resource = Resource.find(params[:id])
    url = fetch_resource_url(@resource)
    redirect_to url
end
```
This approach ensures a new URL is generated every time. We also, as before, used a custom method that added a `cache_control=no-store` query parameter to further prevent CDN caching. This pattern is very common when it comes to private resources, where access can change frequently. We should also note that if the presigned url expires during download, that can cause an access error, so we need to set the expiration high enough for expected downloads.

Finally, one less obvious case I recall was related to misconfigured CDN settings. In another instance, we had configured our CDN to ignore certain headers, *including* the `Cache-Control`, which was in place with the `no-store` parameter! This essentially told the CDN: "Hey, ignore what Rails is telling you about caching, and just use our default settings" which resulted in all private assets being cached aggressively! It goes without saying that checking your CDN's configuration alongside your application’s is an absolute must.

For deeper reading, I would strongly recommend checking out *High Performance Browser Networking* by Ilya Grigorik. It provides a comprehensive overview of browser caching mechanisms and their interactions with proxies and CDNs. Also, for a solid understanding of HTTP caching and headers, the HTTP specification documents on the W3C website are invaluable. Finally, reviewing the Active Storage source code itself on Github can be beneficial. Pay particular attention to the implementations of `service_url` and `rails_blob_url` functions. This gives you an understanding at the source about the underlying mechanics.

In summary, the caching of private Active Storage assets by your CDN is rarely a deliberate act of malice. It is generally a consequence of incorrect caching headers or a reliance on the `service_url` without realizing that it's not configured to actively disable caching. The key is to generate a fresh, presigned URL with a `Cache-Control: no-store` header or a cache busting parameter each time you need to serve a private asset. Always test with a staging environment to observe the behavior of your CDN in action. Remember, caching, while a powerful tool for performance, can be a double-edged sword when not managed correctly. So, keep those headers in check, and you'll avoid these frustrating episodes.
