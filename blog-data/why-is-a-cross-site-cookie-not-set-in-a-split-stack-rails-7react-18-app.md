---
title: "Why is a Cross-site cookie not set in a split-stack Rails 7/React 18 app?"
date: "2024-12-14"
id: "why-is-a-cross-site-cookie-not-set-in-a-split-stack-rails-7react-18-app"
---

alright, let’s tackle this cross-site cookie issue. i've definitely been down this road before, and it can be a real head-scratcher, especially when you're mixing rails on the backend and react on the frontend. the problem you're seeing with the cookie not being set when it's a cross-site request is usually down to a few common culprits in the browser security mechanisms.

first, the core concept here is that browsers have really upped their game regarding security, and cross-site request forgery (csrf) is a big concern. basically, the browser tries to prevent a website from making requests to another website in a way that the user is unaware or doesn't consent to. cookies are often used in authentication, so they’re part of that picture.

when we talk about "cross-site", we mean your rails app and react app are running on different domains or subdomains. for example, your rails api might be on `api.example.com` and your react app on `app.example.com` or even `localhost:3000` (react) and `localhost:3001` (rails). that’s cross-site to the browser even if the ports are different for development purposes on the same machine.

the usual fix is to get the browsers to understand it's a legit cross-site request, not a malicious one. there's a combination of settings on both the rails and react sides. let's break it down:

**1. rails server setup:**

this part’s crucial. rails needs to send the correct headers in its responses, particularly the `set-cookie` header. the important attributes here are `secure`, `samesite`, and `domain` attributes

*   `secure`: should be `true` in production, meaning cookies are only sent over https. for development on localhost, you will probably have to use http for development but it is really better to configure it with https even for localhost, you can do this with something like mkcert for example.
*   `samesite`: this is where things get interesting. `samesite=none` is often needed for cross-site requests but also means you *must* use secure cookies. this is the browser's way of preventing csrf. other possible settings are `samesite=lax` and `samesite=strict`. `lax` generally allows the cookie on "safe" requests (like get requests navigating to the site). `strict` does not allow them at all in cross-site, so that's what you don't want. since we're dealing with a cross-site request we will be using `samesite=none` here.
*   `domain`: this *might* need to be set if you have subdomains. it usually defaults to the domain name the server is running on.

here is how you would configure this in a rails initializer (e.g., `config/initializers/session_store.rb` or a file you create for your cookie settings):

```ruby
# config/initializers/cookie_settings.rb
Rails.application.config.session_store :cookie_store,
                                       key: '_your_app_session',
                                       domain: '.example.com', # set to your domain e.g. .example.com or your subdomains
                                       same_site: :none,
                                       secure: true, # in production
                                       tld_length: 2 # add this to be explicit with top level domain length
                                       # and add this to allow http in development
                                       # secure: Rails.env.production?
```

**important notes on rails configuration:**

*   i've seen folks get caught out by not explicitly setting `tld_length: 2` in the session config, especially when dealing with subdomains and cookies, it is a good idea to add it, to avoid any possible issues with the browser not setting the cookie, it’s more explicit this way.
*   if you’re testing locally and do not have `https` set up you need to comment the `secure: true` line, and add the line `secure: Rails.env.production?`, because `samesite=none` needs `secure` to be true (unless you want to set it to `lax` or `strict` but that won't fix your cross-site cookie problem). it's one of those things that trips people up when switching environments.
*   always double-check your domain configuration. it's where many issues pop up. especially when you are using a shared hosting and sometimes you do not have access to create subdomains but can achieve the same goal with a more traditional domain and path url structure.

**2. react app setup (axios example):**

on the react side, you need to make sure your requests include the `withcredentials` option when you make a request using something like axios. this tells the browser to send cookies with the request to the backend.

here is how that would look using `axios` (or a similar library):

```javascript
import axios from 'axios';

const api = axios.create({
  baseURL: 'https://api.example.com', // your api rails backend url
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
});

// Example function to make an API request
const fetchSomething = async () => {
  try {
      const response = await api.get('/some-endpoint');
      console.log(response.data);
  } catch(error){
    console.error("Error fetching data: ", error);
  }
}

export default api;
```

**important notes on the react configuration:**

*   `withcredentials: true` is your friend. without it, cookies just won't be sent.
*   remember to include the correct headers for your requests, most of the time `content-type: 'application/json'` is enough for the rails api backend. but it is a good practice to add also `accept: 'application/json'` to be more explicit.
*   ensure that you have the correct url set on `baseurl`.
*   if you still have an issue, try using your browser dev tools network tab, and check if the cookies are being sent on the request. also make sure that the `response headers` from the rails backend are setting the cookie correctly using the `set-cookie` header.

**3. cors (cross-origin resource sharing):**

now, cors is the other side of the coin. your rails app also has to allow requests from your react domain using cors. you can configure cors either in the rails app itself using rack cors gem (this is the most common option), or a load balancer/reverse proxy like nginx or apache, or even aws api gateway (that is more rare to see). here is how to set up rack cors in rails:

first, you need to add the `rack-cors` gem to your `gemfile` and run `bundle install`:

```ruby
gem 'rack-cors'
```

then configure the cors settings in `config/initializers/cors.rb`:

```ruby
# config/initializers/cors.rb
Rails.application.config.middleware.insert_before 0, Rack::Cors do
  allow do
    origins 'https://app.example.com' # your react app url or add multiple urls
    resource '*',
             headers: :any,
             credentials: true,
             methods: [:get, :post, :put, :patch, :delete, :options, :head]
  end
end

```

**important notes on cors:**

*   the `origins` parameter must match the domain of your react app. if you have multiple environments you will have to add them here. remember that `localhost:3000` is a different origin from `localhost:3001`. if you need a dynamic solution based on the current rails environment you can set it using a loop.
*   `credentials: true` here is key, this lets the browsers know that the request is allowed to pass credentials, this includes cookies.
*   ensure your allowed `methods` are correctly configured.

**things i learned the hard way:**

i remember one time when i had a similar issue, i had forgotten to restart the rails server after changing the `session_store.rb` file. spent about an hour thinking my react app was the problem, only to find out the rails config was never loaded. this kind of thing really makes you re-evaluate your life choices. and the next time it was a missing `credentials: true` setting in cors that i missed. we all do these mistakes don't we? they are part of the learning process and that is why there is no shame to ask.

**useful resources:**

*   **rfc 6265 bis (http state management mechanism):** this is the specification for cookies and how they work. it's a very technical read but it is the source of the truth if you have any questions on cookie specifications.
*   **the 'web application hacker's handbook' by d. stuttard and m. pinto:** although this is a book mostly about web application security it covers in great detail browser security mechanisms. it explains well the importance of correct cookie configuration with the `samesite` and `secure` attributes, and how that relates with csrf and cross-site request forgery.
*   **mozilla developer network (mdn):** their documentation on cookies, cors, and http headers is quite useful and practical. it's usually the first place i go when i have a problem like this one.

**in summary:**

the "cross-site cookie not setting issue" is usually a combination of:

1.  incorrect rails cookie settings ( `samesite`, `secure`, `domain` ).
2.  missing `withcredentials: true` in your react http requests.
3.  incorrect cors configuration in rails.

remember to restart your servers after config changes, and always check the network tab of your browser's dev tools. if you get lost i find useful to read the specifications from time to time, and do some tests. do not take my word on faith. try it for yourself and then when you get it right you understand it better. hopefully these ideas help you fix your problem. good luck, let me know if you have any other questions. and remember it can be tricky sometimes, but you can do it. if i did, anyone can do it.
