---
title: "How do Rails 6 sign_in and create sessions works in Devise?"
date: "2024-12-15"
id: "how-do-rails-6-signin-and-create-sessions-works-in-devise"
---

alright, so you're asking about how devise handles sign-in and session creation in rails 6, right? yeah, i've been down that rabbit hole plenty of times. it's one of those things that seems simple on the surface, but when you start needing to tweak it or debug weird behavior, things get interesting real fast.

first off, devise is a gem, a ruby gem, and it's not just a piece of code, it's a full authentication system. that's why it touches a bunch of stuff when you use it, not just models, or controllers. it basically automates the whole user authentication process, from registration to session management. when it comes to signing in, devise relies on a few core rails mechanisms, especially sessions.

the general flow goes something like this: user submits their login form (usually email and password), devise intercepts this request, finds the user record based on the email provided, and then checks if the password matches using bcrypt's hashing magic. if everything checks out, devise creates a session. this session is basically a hash stored on the server, usually identified by a unique cookie that's sent to the user's browser.

now, let's talk about some code examples. these are the kind of things i've had to deal with myself while trying to understand it better, or while doing some kind of debug in the middle of the night... you know, the good stuff.

first, devise doesn't really use a sign_in method that's exposed, at least directly. the logic is handled inside the devise controllers. but let's say we wanted to understand how the user is actually signed in and if the current user is logged in, we can explore some helpers that it gives us:

```ruby
# in a rails controller or view
def current_user
  @current_user ||= warden.authenticate(scope: :user)
end

def user_signed_in?
  !current_user.nil?
end

# how to sign a user in directly if you are doing a test.
def sign_in_test_user
   @user = User.create!(email: 'test@example.com', password: 'password', password_confirmation: 'password')
   sign_in @user
end
```

notice the `warden` object there? warden is a framework for authentication, devise uses this. it handles the heavy lifting of figuring out if a user is authenticated based on the session. the first time `current_user` is called, `warden.authenticate(scope: :user)` is called. devise uses sessions so, it checks if there is a valid session, finds the user_id inside the session, and returns the user. if there isn't a valid session it returns nil, because the user is not signed in. subsequent calls will retrieve the current user without hitting the database again since we are using the `@current_user ||= ...` mechanism to cache the user variable. the second helper `user_signed_in?` is just a simple check to see if that current user exists. i wrote that code a couple of times. once because i forgot it, once because i lost it and had to re write it. i should save it on github gist or something, but, well, i'm lazy.

when devise authenticates the user, it is actually calling the `sign_in` method. lets look at that method in an example test:

```ruby
require 'test_helper'

class UserSignInTest < ActionDispatch::IntegrationTest
  include Devise::Test::IntegrationHelpers

  setup do
    @user = User.create!(email: 'test@example.com', password: 'password', password_confirmation: 'password')
  end

  test "can sign in" do
    sign_in @user
    assert user_signed_in?
    get root_path
    assert_response :success
  end
end
```
the `sign_in` method is a devise's helper that is used to simulate the login flow of devise, meaning that is setting up the session as the user had just logged in. it is important to note here that this method is not directly related to a devise method with the same name, this method is from the `devise::test::integrationhelpers` module. this is very common when you deal with authentication, and there's plenty of methods and helpers that are specific to tests. so don't try to find it in the controller itself.

now, how does the session gets created? devise does that under the hood by saving the user id inside the session. here's what it might look like if you were inspecting the session object:

```ruby
session[:user_id] = user.id # this is what devise does internally
# after this the user is considered signed in
```
this is a simplified way, but it shows the main idea, devise saves the user id in the session. when devise checks for the current user, it looks for the session user id, if it finds it, then tries to find the user with that id, if it finds it, it returns the user, if not it returns nil.

now, if you are dealing with rails, you're mostly using cookies to manage sessions, by default, and that's usually enough. but, remember, the cookie itself does not contain any of the session information. it's like a key to the actual session on the server. if you need to store more data, it would be in the server memory, not in the cookies. if you're on a system that is dealing with thousands of users, a database session store would be a better approach for that.

it is important to mention that devise also supports rememberable functionality. when a user checks the remember me box, it uses a persistent cookie. this cookie stores a user id and a remember token. this allows the user to remain signed in across browser sessions. it's like having a key that works even when you restart your machine. but, keep in mind, it is good to protect that cookie using `http_only: true`. i had one of my apps hacked because of that. i learned that the hard way. you see those remember me cookies are also something to worry about, they are long lived and can be hijacked. never trust a cookie.

another thing is that devise also manages things like timeout and invalidations. if you are inactive for too long, your session expires, and you're logged out. this is important for security. also, there are options for session invalidation, which means that if you change your password, then all the old sessions are invalidated automatically. if you forget to implement that logic, that's bad. really bad. i also made that mistake one time... you never want to give any opportunity for a hacker, they will take it. i also had some problems when changing the default session timeout. it's not that hard but i missed it, took me a while. it was a long day.

if you want to dive deeper, i would suggest going through the devise gem itself. it's open source, you can see all the code in github, read the code is something that i do every once in a while. it really helps to know what's going on, under the hood. for a more general background in session handling and security concepts, i'd recommend looking into web security books like 'the web application hacker's handbook' by stefan esser or 'bulletproof web design' by dan cederholm. they are good books that explain this topic in detail and without too much jargon. but, you know, if reading code is not your cup of tea, those books will do the job for you.

one last thing, i heard that the average programmer spends half of their time debugging, and the other half of their time, coding bugs. just a little joke for you to remember how we usually spend our days. anyway, i hope this helped, and if you have any more questions, ask away. happy coding!
