---
title: "How can I refresh all the cookies, token and session after logout calling API using Rails Hotwire?"
date: "2024-12-15"
id: "how-can-i-refresh-all-the-cookies-token-and-session-after-logout-calling-api-using-rails-hotwire"
---

alright, so, you're hitting that classic problem with rails hotwire and needing to nuke all auth stuff on logout, huh? been there, messed that up, got the t-shirt, like, several times. it's not always as straightforward as it looks. let's break down the how and the why, because understanding the moving parts here is key to actually fixing it cleanly.

first off, i've found that the main issue isn't usually the hotwire part *per se*, but rather how we're handling the authentication state *and* how hotwire plays into that state. i remember once, early in my career i had this app that was like, 80% hotwire and 20% custom javascript (before i *really* learned stimulusjs. lol, what a mess). the logout button was a simple link_to that did a rails post request that cleared the cookies in the session, and rendered back with a 303 redirect, which worked sometimes! but then, users would try to hit the back button or the reload and would be "logged in" again. it felt like whack-a-mole. turns out, the browser was *really* eager to reuse cached pages. the session was empty in the backend, but the front end was completely unaware of it, because it still had the stale cookies and html. so frustrating.

anyway, the problem really comes from how hotwire intercepts turbo frames and streams. when you log out, and you are making a post to the backend, most probably you're clearing a cookie (or a session) on the server-side. hotwire, by default is not going to magically detect that the auth cookie changed. and your html might still show user details and stuff, cause it's cached or just rendered before the cookie was cleared.

the typical approach is to: 1. clear the session, tokens and cookies in the backend (we'll assume you have that sorted with something like devise or your own authentication logic. if not, that’s a whole other discussion.) 2. render an empty, neutral view that represents the "logged out" state, and to ensure the view is fresh, not cached by browser.

here's a basic example of the rails controller action for logout:

```ruby
  def destroy
    reset_session
    cookies.delete(:your_auth_token_cookie, domain: '.yourdomain.com')
    cookies.delete(:other_cookie, domain: '.yourdomain.com')
     # Optionally, add a flash message.
    flash[:notice] = "Logged out successfully."

    redirect_to root_path, status: :see_other
  end
```

the `reset_session` will invalidate the rails session on the backend, and we’re deleting the cookies in the response. setting the domain to `.yourdomain.com` will ensure that you remove it from all subdomains.  i use `:see_other` (http status 303) when i'm redirecting as it's the appropriate verb for this action. it's not very sexy, but it does what it needs to do.

now, let’s think hotwire. you want to avoid the full page reload as much as you can. that’s the point, isn't it? so, instead of a redirect (at least not a full page reload), you can use a turbo stream to replace parts of your page with fresh content that’s not tied to the logged-in user state. you might use turbo frames or broadcast to all clients.

here is a simple example how you could use a turbo stream to refresh components that rely on authentication. this is, of course, on top of clearing the cookies:

```ruby
def destroy
    reset_session
    cookies.delete(:your_auth_token_cookie, domain: '.yourdomain.com')
    cookies.delete(:other_cookie, domain: '.yourdomain.com')

    respond_to do |format|
      format.turbo_stream do
        render turbo_stream: [
          turbo_stream.replace("user-menu", partial: "shared/logged_out_menu"),
          turbo_stream.replace("main-content", partial: "shared/logged_out_content")
        ]
      end
      format.html { redirect_to root_path, status: :see_other } # fallback for no-js
    end
end
```

here, we're responding with `turbo_stream` if the request comes with that specific format (`format.turbo_stream do` ).  we're targeting elements with ids `user-menu` and `main-content`, and replacing their content with partials specific to the logged out state. your `_logged_out_menu.html.erb` and `_logged_out_content.html.erb` will hold the html for when users are not logged in. the `format.html` block, will just redirect as before if it's not a turbo request. for example, if the user is doing a ctrl-shift-r, which forces a full page reload.

the key here is that *you control* exactly which parts of the page are updated and you do it *only* after you've cleared the user session. that will prevent any stale data.

finally, let's think about js. sometimes, we might have some js state, or we might be doing some client-side caching (like using localstorage). i once had this horrible experience with a react app trying to sync the local state with the backend, and it turned out that i was caching a jwt token in the local storage. that was pain, and a lesson learned the hard way. we can handle this problem by adding a little bit of javascript to clear the client-side state upon logout. and that would help with those scenarios where users are hitting the back button in a really "aggressive" way.

you can attach a `turbo:before-stream-render` event listener in your `application.js` which is triggered before the turbo stream is rendered. it's a great place to clear some client-side storage, and clear state on the front end.

```javascript
document.addEventListener("turbo:before-stream-render", (event) => {
    if(event.detail.newStream.includes("logged_out_menu")){
        // clear all client side auth storage
        localStorage.removeItem('user_token');
        localStorage.removeItem('some_other_user_data');
        sessionStorage.removeItem('session_token');
    }
});
```

here we are listening for a turbo event that comes before the dom is modified and we check if the stream contains the string `"logged_out_menu"`. if it is, we assume that we are logging out, so we clear all the user related stuff. it's not super sophisticated, but it gets the job done. and it's better to over-clear than not to clear it at all.

to recap, here's the approach:

1.  on the backend, nuke the session and any cookies associated with authentication.
2.  send back a turbo stream that updates the view, replacing parts of your html with the logged-out version.
3.  optionally, in your js, add an event listener to ensure all client-side auth information is cleared.

it might seem like a lot of steps, but it will cover the vast majority of cases.

for resources, i'd recommend having a look at "rails 7 hotwire cookbook" by david b. copeland. it will go into a deeper dive into turbo streams and how they work under the hood. also, "working with javascript" by alex anderson can be helpful if you are trying to understand javascript events. it’s an old one, but it’s solid as a rock in its fundamentals.

don't get too caught up in the intricate details. just keep it simple, and if you find yourself getting lost in state management, it's time to step back, refactor and write better simpler code, and it's easier than you think, even if it's always a bit annoying.
