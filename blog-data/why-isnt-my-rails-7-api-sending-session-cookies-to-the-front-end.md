---
title: "Why isn't my Rails 7 API sending session cookies to the front-end?"
date: "2024-12-23"
id: "why-isnt-my-rails-7-api-sending-session-cookies-to-the-front-end"
---

Alright, let's talk session cookies in Rails 7 APIs; I've been down this rabbit hole more times than I care to count. It's a common hiccup, particularly when dealing with decoupled front-ends that aren't running on the same domain as your API. It's not usually a 'Rails is broken' situation, but rather a configuration and understanding issue of how cookies, particularly session cookies, are handled.

In my experience, the root cause almost always boils down to a combination of the following culprits. Let's dissect them.

First, *CORS (Cross-Origin Resource Sharing)* deserves a thorough check. In the default setup, Rails 7 APIs, by design, are protective. They will not generally send session cookies to a different origin, which is exactly the case when your front-end is on, say, `localhost:3000` and your API is on `localhost:3001`. The browser itself enforces these security measures. Without explicit permission, your front-end requests will not receive the session cookie, effectively leaving you unauthenticated. Now, I’ve seen some developers try to bypass this via JavaScript with tricks that seem to work in simpler cases but fall apart dramatically when under scrutiny or when deployed. Don't. Go the CORS route, it’s the only sustainable path.

Second, the *`same_site` attribute of the session cookie* plays a major role. Rails, by default, now sets this to `lax`. This attribute instructs browsers when to send cookies in cross-site requests. `lax` provides a good balance for preventing common CSRF vulnerabilities. It still allows cookie sending in same-site requests, i.e. navigation and cross-site top-level navigations (think clicking on a link). However, for most AJAX requests originating from a different origin, a `same_site: lax` cookie will *not* be included. What we often need when the front-end makes requests to an API is for session cookies to be included even on cross-site requests made via js fetch calls or using axios.

Third, and often overlooked, is the *`secure` attribute*. This attribute ensures that the cookie is only sent over HTTPS. If your API is not running on HTTPS, but your browser is accessing your front-end with it, the cookie will generally not be sent. This seems obvious now, but I’ve seen more than one case where this simple fact was overlooked on the local development environment. In development, you might sometimes skip HTTPS, but it's a critical aspect to remember.

Now, let’s explore a few scenarios and how to address them, using some working code snippets.

**Scenario 1: Basic CORS Configuration**

Let's say your React front-end is on `http://localhost:3000`, and your Rails API is on `http://localhost:3001`. You're not seeing your session cookies being sent when making API calls. Here’s how to address it:

```ruby
# config/initializers/cors.rb
Rails.application.config.middleware.insert_before 0, Rack::Cors do
    allow do
      origins 'http://localhost:3000'

      resource '*',
        headers: :any,
        methods: [:get, :post, :put, :patch, :delete, :options, :head],
        credentials: true
    end
  end

# config/initializers/session_store.rb
Rails.application.config.session_store :cookie_store, key: '_your_app_session', same_site: :none, secure: Rails.env.production?
```

*Explanation*:

*   The `Rack::Cors` middleware is configured to allow requests from the specified origin.
*   `credentials: true` is *crucial* to allow the sending of cookies.
*   The session store is modified to use `:none` for `same_site`.  This explicitly allows the cookie to be sent on cross-site requests, as long as it's over HTTPS or the `secure: false` is set for development. In production, you want to always use `:secure` and the cookie sent over HTTPS
*  If running locally, and not using HTTPS, I am using a conditional `Rails.env.production?` that sets `secure` to false in `development` and `true` in `production`. This ensures we can test without HTTPS in development and forces it in production.

**Scenario 2: More Complex Origins (e.g. subdomains)**

Let's assume your front-end might be accessed via multiple subdomains like `app.localhost:3000`, `staging.localhost:3000`, or even a different development port. We need to configure `origins` more robustly:

```ruby
# config/initializers/cors.rb
Rails.application.config.middleware.insert_before 0, Rack::Cors do
    allow do
        origins do |source, env|
          /^(https?:\/\/)?(app|staging)?\.?localhost(:[0-9]+)?$/.match?(source)
        end

        resource '*',
          headers: :any,
          methods: [:get, :post, :put, :patch, :delete, :options, :head],
          credentials: true
    end
end

# config/initializers/session_store.rb
Rails.application.config.session_store :cookie_store, key: '_your_app_session', same_site: :none, secure: Rails.env.production?
```

*Explanation*:

*   Instead of hardcoding origins, we're using a regular expression to match a pattern. This provides much more flexibility. It matches any origin starting with `http` or `https` followed by optional `app` or `staging` followed by an optional dot, `localhost` and then optionally port numbers.
* The rest remains the same as the previous example.

**Scenario 3: API-only with JWT in Headers but Cookies for Session (Hybrid Approach)**

You might be using JWT tokens for most API endpoints, but still want to use session cookies for some routes like authentication handling with a browser redirect after login. In that case you can selectively enforce CORS only on certain requests via the configuration of the `resource` block

```ruby
# config/initializers/cors.rb
Rails.application.config.middleware.insert_before 0, Rack::Cors do
  allow do
    origins 'http://localhost:3000'

      resource '/users/session',
        headers: :any,
        methods: [:post],
        credentials: true
    end
end

# config/initializers/session_store.rb
Rails.application.config.session_store :cookie_store, key: '_your_app_session', same_site: :none, secure: Rails.env.production?
```
*Explanation*:
* The `resource` block is restricted to the specific endpoint `/users/session`. This ensures that CORS and credential sending are only enabled for this route. Any other requests that do not match this path will be unaffected by these CORS rules.
* This shows the power of granular control using `resource` in `rack-cors`

**Things to Double-Check**

1.  *Browser Developer Tools*: Use them religiously. Inspect the network tab to check the `Set-Cookie` header in the response and the `Cookie` header in subsequent requests. See if your session cookies are even being sent by the server. Also, look for any CORS errors in the console.
2.  *HTTPS*: Make absolutely sure you are using it in production, and when using `same_site: none`. Or that you are configuring the `secure` flag correctly.
3. *Frontend configuration*: Verify that in your frontend request configuration `credentials` is set to `include`. For example, in fetch, it should look like `fetch('url', {credentials: 'include'})`.

**Recommended Reading**

For a deep dive, check out these resources:

*   *HTTP: The Definitive Guide* by David Gourley and Brian Totty. A comprehensive explanation of the HTTP protocol, including cookies.
*   *Mozilla Developer Network (MDN) documentation* on CORS and the `Set-Cookie` header. It provides up-to-date information and best practices. Search terms such as "CORS", "SameSite", "Set-Cookie".
*   *RFC 6265* for detailed information about HTTP cookies. It's a heavy read, but the ultimate source.

These solutions should cover a large percentage of the reasons you are not seeing the session cookies sent to your front-end. Remember, this is a tricky area, so it's essential to be thorough and test your setup on multiple environments. I've certainly learned it the hard way over several projects. The key, as always, is to understand the mechanics at play rather than relying on quick fixes.
