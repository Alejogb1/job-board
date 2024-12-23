---
title: "How frequently does a CDN update cached content?"
date: "2024-12-23"
id: "how-frequently-does-a-cdn-update-cached-content"
---

Alright, let’s tackle this question about CDN update frequencies. It's a nuanced area, and I've certainly seen my fair share of headaches trying to get it just right, particularly back in my days working on the global e-commerce platform where keeping content fresh was critical for both user experience and inventory accuracy. The short answer, of course, is "it depends," but let's unpack that.

The frequency with which a Content Delivery Network (CDN) updates its cached content isn't governed by a single, universal rule. It's a combination of several factors including, primarily, the cache-control headers set by the origin server, CDN configurations, and the specific CDN vendor’s internal logic. In essence, a CDN acts as a sophisticated intermediary, storing copies of your data closer to users to reduce latency. However, if that cached data is stale, the benefits are negated, and users can have an inaccurate or frustrating experience.

At the heart of the update process are http cache headers. The `cache-control` header is the most crucial. It dictates how long a resource can be considered fresh and thus, served directly from the cache without revalidation with the origin. The most pertinent directives here are `max-age` and `s-maxage`.

*   `max-age`: Defines the maximum time in seconds a resource can be considered fresh by *any* cache (browsers, proxies, cdns).
*   `s-maxage`: Similar to `max-age`, but specifically applicable to *shared* caches like CDNs. It usually overrides `max-age` when present. This means you can specify a different caching policy for your visitors' browsers and for intermediary caches like the CDN.

If these directives aren't present, or if the specified time has elapsed, the CDN will then attempt a validation with the origin server. This is typically done through what's known as conditional requests. Essentially, the CDN sends a request with headers like `if-modified-since` or `if-none-match` containing timestamps or entity tags (etags), respectively. If the origin server responds with a 304 not modified, then the CDN will reuse its cached copy. If the origin server responds with a 200 ok and new data, then the CDN will cache the new response and update accordingly.

It's not *just* about time-based expirations. CDNs also have invalidation capabilities. This allows you to proactively remove a cached version of a resource and force the CDN to fetch a fresh copy from the origin. Typically, vendors offer apis or management interfaces for invalidations. I used to regularly utilize these to update the pricing data immediately after product price updates – otherwise, we’d have confused customers.

Let me show you how this might work in practice with a few scenarios and code examples (using Python with the `requests` library, but the concepts translate to other environments).

**Example 1: Basic Caching with max-age**

Let's say you're serving a static image that doesn't change often. You can set a `max-age` of, say, 3600 seconds (1 hour).

```python
import requests
from datetime import datetime, timedelta

# simulate an origin server response with cache-control headers
def mock_origin_response(url):
  now = datetime.now()
  response_headers = {
      "cache-control": f"max-age=3600",
      "date": now.strftime('%a, %d %b %Y %H:%M:%S GMT')
  }

  # assume this would return some data, and for brevity this will be some placeholder text
  return { 'status_code': 200, 'headers': response_headers, 'content': "some image data" }


def get_content_from_cdn(url, cached_response = None):
    if cached_response:
      # Simulate checking if the cache is valid based on the header
      response_date_str = cached_response['headers']['date']
      response_date = datetime.strptime(response_date_str, '%a, %d %b %Y %H:%M:%S GMT')
      max_age_str = cached_response['headers'].get('cache-control', "").replace("max-age=", "")
      max_age = int(max_age_str)
      expiration_time = response_date + timedelta(seconds=max_age)
      if expiration_time > datetime.now():
        print ("Serving from cache")
        return cached_response

    print("Fetching from origin.")
    origin_response = mock_origin_response(url)
    print ("Caching at cdn")
    return origin_response # assuming CDN would cache here

# Usage:
url = "https://example.com/image.jpg"

# First request: cache miss
cached_resp1 = get_content_from_cdn(url)
print(f"Response headers: {cached_resp1['headers']}")
print(f"Content: {cached_resp1['content']}")

# Second request within an hour: Cache hit
cached_resp2 = get_content_from_cdn(url, cached_resp1)
print(f"Response headers: {cached_resp2['headers']}")
print(f"Content: {cached_resp2['content']}")

```

In this simplified example, `get_content_from_cdn` function serves as a simulated CDN, the `mock_origin_response` acts as the origin server, creating a response with a max-age directive. As long as the next request occurs within the `max-age` interval, our simplified CDN would serve a cached version, otherwise, a request is made to the origin server and the cache is updated.

**Example 2: Using s-maxage for CDN-Specific Caching**

Now let's consider a situation where you need different caching rules for browsers and cdns. You might want a longer caching period for the CDN, but shorter period for browsers to make sure your user always has the newest data.

```python
import requests
from datetime import datetime, timedelta

def mock_origin_response_smaxage(url):
  now = datetime.now()
  response_headers = {
    "cache-control": "max-age=60, s-maxage=600", # browser cache of 60s and cdn of 600s
     "date": now.strftime('%a, %d %b %Y %H:%M:%S GMT')
  }
  return { 'status_code': 200, 'headers': response_headers, 'content': "important data" }

def get_content_from_cdn_smaxage(url, cached_response = None):
    if cached_response:
      # Simulate checking if the cache is valid based on s-maxage and if s-maxage is not available, default to max-age
      response_date_str = cached_response['headers']['date']
      response_date = datetime.strptime(response_date_str, '%a, %d %b %Y %H:%M:%S GMT')
      s_maxage_str = cached_response['headers'].get('cache-control', "").replace("max-age=", "").replace("s-maxage=", "").split(",")[1]
      if s_maxage_str:
        s_maxage = int(s_maxage_str)
        expiration_time = response_date + timedelta(seconds=s_maxage)
      else:
        max_age_str = cached_response['headers'].get('cache-control', "").replace("max-age=", "").split(",")[0]
        max_age = int(max_age_str)
        expiration_time = response_date + timedelta(seconds=max_age)

      if expiration_time > datetime.now():
        print ("Serving from cache")
        return cached_response

    print("Fetching from origin.")
    origin_response = mock_origin_response_smaxage(url)
    print ("Caching at cdn")
    return origin_response

# Usage:
url = "https://example.com/data.json"

# first time, cache miss
cached_resp1 = get_content_from_cdn_smaxage(url)
print(f"Response headers: {cached_resp1['headers']}")
print(f"Content: {cached_resp1['content']}")

# Request within 10 minutes: cache hit
cached_resp2 = get_content_from_cdn_smaxage(url, cached_resp1)
print(f"Response headers: {cached_resp2['headers']}")
print(f"Content: {cached_resp2['content']}")

```

Here, the origin is configured with both `max-age=60` and `s-maxage=600`. This means that our simplified CDN will check `s-maxage` before `max-age` when serving the content from the cache. Our example doesn't take into account the browser which would utilize `max-age=60`. It would have to revalidate much sooner than the simplified CDN.

**Example 3: Cache Invalidation**

Finally, let's look at a situation where the content needs immediate invalidation. This is handled via a separate action, not just relying on time. I won’t simulate the api calls here but just demonstrate how the CDN would invalidate the cached copy.

```python

import requests

def mock_origin_response_invalidation(url):
   response_headers = {
       "cache-control": f"max-age=3600",
    }
   return { 'status_code': 200, 'headers': response_headers, 'content': "old content" }

def get_content_from_cdn_invalidation(url, cached_response = None):
    if cached_response:
      print ("Serving from cache")
      return cached_response

    print("Fetching from origin.")
    origin_response = mock_origin_response_invalidation(url)
    print ("Caching at cdn")
    return origin_response


def invalidate_cache(url, cached_response):
    cached_response = None # this is equivalent to the cdn invalidating the cached entry
    print ("Invalidating Cache")
    return cached_response


url = "https://example.com/data.txt"

# Initially we pull the old data from origin
cached_resp1 = get_content_from_cdn_invalidation(url)
print(f"Content: {cached_resp1['content']}")


# Assuming a change at the origin and we want to trigger a invalidation
cached_resp2 = invalidate_cache(url, cached_resp1)

# next request will fetch fresh copy from the origin
cached_resp3 = get_content_from_cdn_invalidation(url, cached_resp2)
cached_resp3['content'] = "new content" # lets assume updated content
print(f"Content: {cached_resp3['content']}")

```

In this example, we simulate invalidation by simply setting the cached_response to `None`. The next call to `get_content_from_cdn_invalidation` will force the simplified CDN to fetch fresh data. In actual production, the invalidation mechanism would be a request to the CDN's api, with the URL that needs to be cleared.

The frequency at which the CDN refreshes its content is not a hard number, it's the result of a carefully orchestrated dance between origin server headers, CDN configurations, and invalidation methods. For a deeper understanding of http caching, I'd highly recommend reading "High Performance Web Sites" by Steve Souders. It is an older book, but provides an excellent foundation on the subject. Also the official http caching RFC (RFC7234) is very detailed for those who really want to dive deep. For details specific to CDN implementations, I suggest looking into the vendor-specific documentation, because each vendor has its own proprietary algorithms for content handling and updates, which is why it’s crucial to understand their specific recommendations. Remember, understanding caching is as critical as any performance tuning technique, so spending time to get it right is a very worthwhile investment.
